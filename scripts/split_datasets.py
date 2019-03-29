# libraries
import pandas as pd
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="Split datasets to training, validation and testing.")
    parser.add_argument('--inf', required=True)
    parser.add_argument('--inl', required=True)
    parser.add_argument('--trf', required=True)
    parser.add_argument('--vaf', required=True)
    parser.add_argument('--tef', required=True)
    parser.add_argument('--trl', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--tel', required=True)

    args = parser.parse_args()

    # load features dataframe
    dff = pd.read_csv(args.inf, sep='\t', header=0, index_col=0)

    # load labels dataframe
    dfl = pd.read_csv(args.inl, sep='\t', header=0, index_col=0)

    # create training features, validation features and testing features
    trf, vaf, tef = np.split(dff.sample(frac=1), [int(.6*len(dff)), int(.8*len(dff))])

    # create corresponding labels
    trl = dfl.loc[trf.index]
    val = dfl.loc[vaf.index]
    tel = dfl.loc[tef.index]

    # save to tsv
    trf.to_csv(args.trf, sep='\t')
    vaf.to_csv(args.vaf, sep='\t')
    tef.to_csv(args.tef, sep='\t')
    trl.to_csv(args.trl, sep='\t')
    val.to_csv(args.val, sep='\t')
    tel.to_csv(args.tel, sep='\t')


if __name__ == '__main__':
    main()
