import pandas as pd
import argparse
import pickle
import time
import math

import torch
import torch.nn
import torch.cuda
import torch.optim
import torch.utils.data
import torch.nn.functional


class Pathways(torch.nn.Module):

    def __init__(self, pathways_df, gene_ids):
        """
        Layer representing pathways.
        :param pathways: binary dataframe representing pathways
        :param gene_ids: gene ids used in datasets
        """
        super(Pathways, self).__init__()

        self.pathways = torch.tensor(pathways_df.loc[gene_ids, :].values).to(dtype=torch.float32)
        self.weight = torch.nn.Parameter(torch.Tensor(self.pathways.shape))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return torch.matmul(x, self.pathways * self.weight)

    def extra_repr(self):
        return f'in_features={self.pathways.shape[0]}, out_features={self.pathways.shape[1]}'


class NeuralNetwork(torch.nn.Module):

    def __init__(self, pathways, gene_ids, architecture):
        """
            pathways = filename of binary matrix representing pathways
            architecture = list of int with len() >= 2
        """

        super(NeuralNetwork, self).__init__()

        self.pathways = Pathways(pathways, gene_ids)

        self.architecture = architecture
        self.linears = torch.nn.ModuleList()

        current_size = self.architecture[0]
        for i in range(1, len(self.architecture) - 1):
            self.linears.append(torch.nn.Linear(current_size, self.architecture[i]))
            current_size = self.architecture[i]

        self.output = torch.nn.Linear(current_size, self.architecture[-1])
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.nn.functional.relu(self.pathways.forward(x))
        for i in range(0, len(self.architecture)-2):
            x = torch.nn.functional.relu(self.linears[i](x))
        x = self.output(x)
        return x


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, features, labels, pathways_df, device):
        features_df = pd.read_csv(features, sep='\t', header=0, index_col=0)
        labels_df   = pd.read_csv(labels,   sep='\t', header=0, index_col=0)

        names = [gene_id.split('.')[0] for gene_id in features_df.columns]
        features_df.columns = names

        self.gene_ids = list(set(names).intersection(set(pathways_df.index)))
        individuals = list(set(features_df.index).intersection(set(labels_df.index)))

        self.features = torch.tensor(features_df.loc[individuals, self.gene_ids].values).to(device=device, dtype=torch.float32)
        self.labels   = torch.tensor(labels_df.loc[individuals, :].values)              .to(device=device, dtype=torch.float32)

        self.input_size = self.features.shape[1]
        self.output_size = labels_df.shape[1]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx, :]


def train(model, dataset, loss_function, batch_size, optimizer):

    model.train()
    train_loss = 0
    correct = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()

        prediction = model(data)
        for i in range(prediction.shape[0]):
            if int(torch.argmax(prediction[i])) == int(torch.argmax(target[i])):
                correct += 1

        loss = loss_function(prediction, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_accuracy = correct/len(dataset)

    return train_loss, train_accuracy


def validate(model, dataset, loss_function):

    model.eval()
    validation_loss = 0
    correct = 0
    dataloader = torch.utils.data.DataLoader(dataset)

    for idx, (data, target) in enumerate(dataloader):

        prediction = model(data)
        for i in range(prediction.shape[0]):
            if int(torch.argmax(prediction[i])) == int(torch.argmax(target[i])):
                correct += 1

        loss = loss_function(prediction, target)
        validation_loss += loss.item()

    validation_accuracy = correct/len(dataset)

    return validation_loss, validation_accuracy


def main():

    parser = argparse.ArgumentParser(description="Training of pathways neural network")
    parser.add_argument('--train_features', required=True)
    parser.add_argument('--train_labels', required=True)
    parser.add_argument('--validate_features', required=True)
    parser.add_argument('--validate_labels', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--pathways', required=True)
    parser.add_argument('--linear_architecture', required=True)
    parser.add_argument('--max_seconds', type=int, default=3600)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Running on {device}.')

    architecture = [int(x) for x in args.linear_architecture.split('-')]
    pathways_df  = pd.read_csv(args.pathways, sep='\t', header=0, index_col=0)

    training_dataset = MyDataset(features=args.train_features, labels=args.train_labels,
                                 pathways_df=pathways_df, device=device)
    validation_dataset = MyDataset(features=args.validate_features, labels=args.validate_labels,
                                   pathways_df=pathways_df, device=device)

    # check for correctness
    if training_dataset.gene_ids != validation_dataset.gene_ids:
        print('Gene ids in datasets differ.')

    print(f'Training:\t{training_dataset.features.shape}\t{training_dataset.labels.shape}\n'
          f'Validation:\t{validation_dataset.features.shape}\t{validation_dataset.labels.shape}\n')

    model = NeuralNetwork(architecture=architecture,
                          pathways=pathways_df,
                          gene_ids=training_dataset.gene_ids).to(device=device)
    optimizer = torch.optim.Adam(model.parameters())
    print(f'Model architecture:\n{model}')

    loss_function = torch.nn.BCEWithLogitsLoss()

    start = time.time()

    epoch = 0
    seconds = time.time() - start

    while epoch < args.max_epochs and seconds < args.max_seconds:

        train_loss, train_acc = train(model, training_dataset, loss_function, args.batch_size, optimizer)
        val_loss, val_acc = validate(model, validation_dataset, loss_function)

        # save model
        model_file = f'{args.output_dir}/models/{epoch + 1}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

        # save stats
        loss_file = f'{args.output_dir}/loss.tsv'
        stats = f'{epoch + 1}\t{train_loss:.4f}\t{train_acc:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n'
        with open(loss_file, 'a') as f:
            f.write(stats)

        epoch += 1
        seconds = time.time() - start


if __name__ == '__main__':
    main()
