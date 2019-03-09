from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="Filtering of TCGA")
    parser.add_argument('--gtex_counts', required=True)
    parser.add_argument('--tcga_counts', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    # load GTEx data
    gtex_counts_df = pd.read_csv(args.gtex_counts, sep='\t', header=0, index_col=0).T

    # load TCGA data
    tcga_counts_df = pd.read_csv(args.tcga_counts, sep='\t', header=0, index_col=0).T

    # check if datasets are compatible
    if list(gtex_counts_df.columns) != list(tcga_counts_df.columns):
        print('In gtex, not in tcga:', set(gtex_counts_df.index) - set(tcga_counts_df.index))
        print('In tcga, not in gtex:', set(tcga_counts_df.index) - set(gtex_counts_df.index))

    # initialize PCA and fit GTEx data
    pca = PCA()
    pca.fit(gtex_counts_df)

    # transform tcga according to gtex
    columns = ['PC{}({:.2%})'.format(i, perc) for i, perc in enumerate(pca.explained_variance_ratio_, start=1)]
    indices = list(tcga_counts_df.index)

    tcga_pca_df = pd.DataFrame(pca.transform(tcga_counts_df), columns=columns, index=indices)

    # save dataframe
    tcga_pca_df.to_csv(args.output, sep='\t')


if __name__ == '__main__':
    main()
