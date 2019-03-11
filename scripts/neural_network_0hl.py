# libraries
import torch.optim as optim
from random import shuffle
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import pickle
import torch
import time


class Net(nn.Module):

    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size)         # performs y=x*A^T + b
        self.fc2 = nn.Softmax(dim=0)                    # apply softmax so output lies in [0,1] and sums to 1

    def forward(self, _in):
        out = self.fc2(self.fc1(_in))
        return out


def main():
    parser = argparse.ArgumentParser(description="Filtering of TCGA")
    parser.add_argument('--counts', required=True)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--predictors', required=True)
    parser.add_argument('--seconds', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--loss', required=True)
    parser.add_argument('--predictions', required=True)

    args = parser.parse_args()

    # parameters
    n_inputs = int(args.predictors)
    train_seconds = int(args.seconds)

    # load data
    features_df = pd.DataFrame(pd.read_csv(args.counts, sep='\t', header=0, index_col=0))
    labels_df = pd.DataFrame(pd.read_csv(args.samples, sep='\t', header=0, index_col=0))

    # initialize neural network
    net = Net(n_inputs, labels_df.shape[1])

    # define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters())

    # split to train set and test set
    features_train = features_df.sample(n=int(features_df.shape[0]*0.8)).iloc[:,0:n_inputs]
    features_test = features_df.drop(features_train.index).iloc[:,0:n_inputs]

    labels_train = labels_df.loc[features_train.index]
    labels_test = labels_df.loc[features_test.index]

    # train neural network
    features_train_tensor = torch.tensor(features_train.values).float()
    labels_train_tensor = torch.tensor(labels_train.values).float()

    start = time.time()
    n_train = 0
    epoch = 0
    running_loss = 0.0
    loss_file = open(args.loss, 'w')

    while time.time() - start < train_seconds:

        train_order = list(range(features_train_tensor.shape[0]))
        shuffle(train_order)

        for i, row in enumerate(train_order):

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(features_train_tensor[row,:])
            loss = criterion(outputs, labels_train_tensor[row,:])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if n_train % 1000 == 999:
                print('{}\t{}\t{:.2f}\n'.format(epoch+1, n_train+1, running_loss/1000),
                      end='', file=loss_file)
                running_loss = 0.0

            n_train += 1
        epoch += 1

    print('Finished Training')
    loss_file.close()

    # save model
    with open(args.model, 'wb') as f:
        pickle.dump(net, f, pickle.HIGHEST_PROTOCOL)

    # test neural network
    features_test_tensor = torch.tensor(features_test.values).float()
    labels_test_tensor = torch.tensor(labels_test.values).float()

    correct_prediction = 0
    predictions_file = open(args.predictions, 'w')

    for i in range(features_test_tensor.shape[0]):

        outputs = net(features_test_tensor[i, :])

        predicted = int(torch.argmax(outputs))
        truth = int(torch.argmax(labels_test_tensor[i, :]))

        print('{}\t{}\t{}\n'.format(truth, predicted, list(outputs.detach().numpy().round(decimals=2))),
              end='', file=predictions_file)

        if truth == predicted:
            correct_prediction += 1

    print('Correctly predicted', correct_prediction/features_test_tensor.shape[0], 'cases')
    predictions_file.close()


if __name__ == '__main__':
    main()
