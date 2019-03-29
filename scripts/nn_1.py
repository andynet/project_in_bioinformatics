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


# define neural network
class Net(nn.Module):

    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        self.hidden = nn.Linear(in_size, in_size)
        self.output = nn.Linear(in_size, out_size)

        # 50% Dropout here
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, _in):
        tmp = self.dropout(F.relu(self.hidden(_in)))
        out = self.output(tmp)
        return out


def main():
    parser = argparse.ArgumentParser(description="Filtering of TCGA")
    parser.add_argument('--counts_t', required=True)
    parser.add_argument('--samples_t', required=True)
    parser.add_argument('--counts_v', required=True)
    parser.add_argument('--samples_v', required=True)
    parser.add_argument('--loss', required=True)
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--prediction_dir', required=True)
    parser.add_argument('--seconds', required=True)
    parser.add_argument('--predictors', required=True)

    args = parser.parse_args()

    # parameters
    n_inputs = int(args.predictors)
    n_epochs = 200
    batch_size = 20
    train_seconds = int(args.seconds)

    # load data
    features_train = pd.read_csv(args.counts_t, sep='\t', header=0, index_col=0).iloc[:,0:n_inputs]
    labels_train = pd.read_csv(args.samples_t, sep='\t', header=0, index_col=0)

    features_validate = pd.read_csv(args.counts_v, sep='\t', header=0, index_col=0).iloc[:,0:n_inputs]
    labels_validate = pd.read_csv(args.samples_v, sep='\t', header=0, index_col=0)

    # initialize neural network
    net = Net(n_inputs, labels_train.shape[1])

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # prepare data
    features_train_tensor = torch.tensor(features_train.values).float()
    labels_train_tensor = torch.tensor(labels_train.values.argmax(axis=1)).long()

    features_validate_tensor = torch.tensor(features_validate.values).float()
    labels_validate_tensor = torch.tensor(labels_validate.values.argmax(axis=1)).long()

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
                print('{}\t{}\t{}\n'.format(ground_truth, prediction, list(outputs.detach().numpy().round(decimals=2)[0])), end='', file=prediction_file)

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
