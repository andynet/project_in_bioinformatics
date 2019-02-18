import statistics as st
import pandas as pd
import argparse
import math


def main():
    parser = argparse.ArgumentParser(description="Normalization of recount experiment")
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    df = pd.DataFrame(pd.read_csv(args.input, sep='\t', header=0, index_col=0))

    df_med = df.apply(lambda x: x / st.median(x))
    df_log = df_med.applymap(lambda x: math.log(1 + x))

    df_log.to_csv(args.output, sep='\t')


if __name__ == '__main__':
    main()
