import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Create binary matrix sample vs. tissue")
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    # load samples
    samples_df = pd.DataFrame(pd.read_csv(args.input, sep='\t', header=0, index_col=0))

    # convert to binary matrix
    dummies_df = pd.get_dummies(samples_df.iloc[:, 0])

    # save dataframe
    dummies_df.to_csv(args.output, sep='\t')


if __name__ == '__main__':
    main()
