import statistics as st
import pandas as pd
import argparse
import math


def main():
    parser = argparse.ArgumentParser(description="Get description of samples")
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    df = pd.DataFrame(pd.read_csv(args.input, sep='\t', header=0, index_col=0))

    desc = df.describe().T

    desc.to_csv(args.output, sep='\t')


if __name__ == '__main__':
    main()
