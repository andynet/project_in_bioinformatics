import pandas as pd
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="Scaling of recount experiment")
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    df = pd.DataFrame(pd.read_csv(args.input, sep='\t', header=0, index_col=0))

    df = df.apply(lambda x: ((x - np.mean(x)) / np.std(x)), axis=1)

    df.to_csv(args.output, sep='\t')


if __name__ == '__main__':
    main()
