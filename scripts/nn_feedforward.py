import argparse
import pandas as pd
import numpy as np
import time
import pickle

import torch
import torch.nn
import torch.cuda
import torch.optim
# import torch.tensor
import torch.utils.data
import torch.nn.functional


class NeuralNetwork(torch.nn.Module):

    def __init__(self, architecture):
        """
            architecture = list of int with len() >= 2
        """

        super(NeuralNetwork, self).__init__()

        self.architecture = architecture
        self.linears = torch.nn.ModuleList()

        current_size = self.architecture[0]
        for i in range(1, len(self.architecture)-1):
            self.linears.append(torch.nn.Linear(current_size, self.architecture[i]))
            current_size = self.architecture[i]

        self.output = torch.nn.Linear(current_size, self.architecture[-1])
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        for i in range(0, len(self.architecture)-2):
            x = self.dropout(torch.nn.functional.relu(self.linears[i](x)))
        x = self.output(x)
        return x


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, features, labels, input_size, device):
        features_df = pd.read_csv(features, sep='\t', header=0, index_col=0)
        labels_df   = pd.read_csv(labels,   sep='\t', header=0, index_col=0)

        self.input_size = min(input_size, features_df.shape[1])
        self.output_size = labels_df.shape[1]

        individuals = list(set(features_df.index).intersection(set(labels_df.index)))

        # OPEN QUESTION: why pycharm can not find torch.tensor()?
        self.features = torch.tensor((features_df
                                      .iloc[:, 0:self.input_size]    # select columns
                                      .loc[individuals, :].values    # select rows
                                      )).to(device=device, dtype=torch.float32)

        self.labels = torch.tensor((labels_df
                                    .loc[individuals, :].values  # select rows
                                    )).to(device=device, dtype=torch.float32)

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
    parser = argparse.ArgumentParser(description="Feedforward neural network")
    parser.add_argument('--features_training', required=True)
    parser.add_argument('--labels_training', required=True)
    parser.add_argument('--features_validation', required=True)
    parser.add_argument('--labels_validation', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--architecture', required=True)
    parser.add_argument('--max_training_time', required=True, type=int)
    parser.add_argument('--max_epochs', required=True, type=int)
    parser.add_argument('--batch_size', required=True, type=int)
    args = parser.parse_args()

    # does torch.device() has the same problem as torch.tensor()? What is dynamic dispatch and duck typing?
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Running on {device}.')

    architecture = [int(x) for x in args.architecture.split('-')]

    training_dataset = MyDataset(features=args.features_training, labels=args.labels_training,
                                 input_size=architecture[0], device=device)
    validation_dataset = MyDataset(features=args.features_validation, labels=args.labels_validation,
                                   input_size=architecture[0], device=device)

    print(f'Training:\t{training_dataset.features.shape}\t{training_dataset.labels.shape}\n'
          f'Validation:\t{validation_dataset.features.shape}\t{validation_dataset.labels.shape}\n')

    model = NeuralNetwork(architecture).to(device=device)
    optimizer = torch.optim.Adam(model.parameters())
    print(f'Model architecture:\n{model}')

    loss_function = torch.nn.BCEWithLogitsLoss()

    start_time = time.time()
    for epoch in range(args.max_epochs):

        if start_time - time.time() > args.max_training_time:
            break

        train_loss, train_acc = train(model, training_dataset, loss_function, args.batch_size, optimizer)
        val_loss, val_acc = validate(model, validation_dataset, loss_function)

        # save model
        model_file = f'{args.out_dir}/models/{epoch+1}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

        # save stats
        loss_file = f'{args.out_dir}/loss.tsv'
        stats = f'{epoch+1}\t{train_loss:.4f}\t{train_acc:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n'
        with open(loss_file, 'a') as f:
            f.write(stats)


if __name__ == '__main__':
    main()
