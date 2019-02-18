import statistics as st
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Scaling of recount experiment")
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    df = pd.DataFrame(pd.read_csv(args.input, sep='\t', header=0, index_col=0))

    df_scl = df.apply(lambda x: ((x - st.mean(x)) / st.stdev(x)), axis=1)
    df_drp = df_scl.dropna(axis=0, how='any')

    df_drp.to_csv(args.output, sep='\t')


if __name__ == '__main__':
    main()
