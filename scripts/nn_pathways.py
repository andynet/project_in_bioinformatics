# libraries
import torch.nn.functional as F
import torch.optim as optim
from random import shuffle
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import pickle
import torch
import time

# define Pathways layer
class Pathways(nn.Module):

    def __init__(self, pathways):
        super(Pathways, self).__init__()
        self.pathways = pathways                                                # type: torch.Tensor
        self.weight = nn.Parameter(torch.Tensor(self.pathways.shape))           # type: torch.Tensor
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return torch.matmul(self.pathways * self.weight, x)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.pathways.shape[0], self.pathways.shape[1]
        )

# define neural network
class Net(nn.Module):

    def __init__(self, pathways, architecture, out_size):
        super(Net, self).__init__()
        self.pathways = Pathways(pathways)

        current_size = pathways.shape[1]
        if len(architecture) == 0:
            self.output = nn.Linear(current_size, out_size)
        else:
            self.linears = nn.ModuleList()
            for item in architecture:
                self.linears.append(nn.Linear(current_size, item))
                current_size = item
            self.output = nn.Linear(current_size, out_size)

    def forward(self, x):
        x = F.relu(self.pathways(x))
        if len(self.linears) == 0:
            x = self.output(x)
        else:
            for l in self.linears:
                x = F.relu(l(x))
            x = self.output(x)

        return x


def main():
    parser = argparse.ArgumentParser(description="Training of pathways neural network")

    # inputs
    parser.add_argument('--train_features', required=True)
    parser.add_argument('--train_labels', required=True)
    parser.add_argument('--validate_features', required=True)
    parser.add_argument('--validate_labels', required=True)

    # outputs
    parser.add_argument('--output_dir', required=True)

    # architecture related
    parser.add_argument('--pathways', required=True)
    parser.add_argument('--linear_architecture', required=True)

    # training related
    parser.add_argument('--max_seconds', type=int, default=3600)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)

    args = parser.parse_args()

    # GPU execution
    if torch.cuda.is_available():
        print("CUDA is available, running on gpu.")
        device = torch.device("cuda")
    else:
        print("CUDA not available, running on cpu.")
        device = torch.device("cpu")

    # load data
    train_features = torch.tensor(pd.read_csv(args.train_features, sep='\t', header=0, index_col=0).values).float().to(device)
    # TODO: 
    labels_train = pd.read_csv(args.samples_t, sep='\t', header=0, index_col=0)

    features_validate = pd.read_csv(args.counts_v, sep='\t', header=0, index_col=0)
    labels_validate = pd.read_csv(args.samples_v, sep='\t', header=0, index_col=0)


    # initialize neural network
    pathways = torch.tensor(pd.read_csv(args.pathways, sep='\t', header=0, index_col=0).values)
    architecture = [ int(item) for item in args.architecture.split('-') ]
    out_size = labels_train
    model = Net(pathways, architecture, out_size)
    print(model)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # prepare data
    features_train_tensor = torch.tensor(features_train.values).float()
    labels_train_tensor = torch.tensor(labels_train.values.argmax(axis=1)).long()

    features_validate_tensor = torch.tensor(features_validate.values).float()
    labels_validate_tensor = torch.tensor(labels_validate.values.argmax(axis=1)).long()

    # move data to GPU
    features_train_tensor = features_train_tensor.to(device)
    labels_train_tensor = labels_train_tensor.to(device)

    features_validate_tensor = features_validate_tensor.to(device)
    labels_validate_tensor = labels_validate_tensor.to(device)

    net.to(device)

    # start training
    start = time.time()
    loss_file = open(args.loss, 'w')

    # for each epoch
    for epoch in range(n_epochs):

        model_file = f'{args.model_dir}/{epoch}.pkl'
        prediction_file = open(f'{args.prediction_dir}/{epoch}.tsv', 'w')

        # train model
        net.train()
        train_loss = 0.0

        train_order = list(range(features_train_tensor.shape[0]))
        shuffle(train_order)

        for row in range(0, features_train_tensor.shape[0], batch_size):

            optimizer.zero_grad()
            rows = train_order[row:row+batch_size]           # which rows should be used in batch
            outputs = net(features_train_tensor[rows,:])
            loss = criterion(outputs, labels_train_tensor[rows])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss_avg = train_loss/features_train_tensor.shape[0]

        # save model
        with open(model_file, 'wb') as f:
            pickle.dump(net, f, pickle.HIGHEST_PROTOCOL)

        # test model
        net.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for row in range(0, features_validate_tensor.shape[0], 1):

                outputs = net(features_validate_tensor[row:row+1, :])
                loss = criterion(outputs, labels_validate_tensor[row:row+1])
                test_loss += loss.item()

                ground_truth = int(labels_validate_tensor[row])
                prediction = int(torch.argmax(outputs))

                if ground_truth == prediction:
                    correct += 1

                # save predictions
                print('{}\t{}\t{}\n'.format(ground_truth, prediction, list(outputs.detach().cpu().numpy().round(decimals=2)[0])), end='', file=prediction_file)

        test_loss_avg = test_loss/features_validate_tensor.shape[0]
        accuracy = correct/features_validate_tensor.shape[0]

        # print stats
        print('{}\t{}\t{}\t{}\n'.format(epoch, train_loss, test_loss, accuracy), end='', file=loss_file)

        prediction_file.close()

        if time.time() - start > train_seconds:
            print('Training stopped due to time limit.')
            break

    loss_file.close()


if __name__ == '__main__':
    main()
