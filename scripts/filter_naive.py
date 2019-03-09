import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Filtering of TCGA")
    parser.add_argument('--gtex_counts', required=True)
    parser.add_argument('--gtex_samples', required=True)
    parser.add_argument('--tcga_counts', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()

    # read GTEx data
    gtex_counts_df = pd.DataFrame(pd.read_csv(args.gtex_counts, sep='\t', header=0, index_col=0))
    gtex_samples_df = pd.DataFrame(pd.read_csv(args.gtex_samples, sep='\t', header=0, index_col=0))

    # process counts
    tmp = gtex_counts_df.reset_index()
    tmp = tmp.melt(id_vars=['index'])
    tmp.columns = ['gene', 'sample', 'expression']

    gtex_counts_df = tmp

    # process samples
    tmp = gtex_samples_df.reset_index()
    tmp.columns = ['sample', 'tissue_type']

    gtex_samples_df = tmp

    # merge, calculate stats and select genes
    tmp = pd.merge(gtex_counts_df, gtex_samples_df, on='sample')
    tmp = tmp.groupby(['gene', 'tissue_type']).mean().reset_index()
    tmp = tmp.groupby(['gene']).var().reset_index()
    tmp = tmp.sort_values(by=['expression'], ascending=False)

    sgn = list(tmp.iloc[:, 0])  # selected gene names
    columns = ['{:18}({:.2f})'.format(tmp.iloc[i, 0], tmp.iloc[i, 1]) for i in range(len(sgn))]

    # load TCGA data
    tcga_counts_df = pd.DataFrame(pd.read_csv(args.tcga_counts, sep='\t', header=0, index_col=0))

    if set(sgn) != set(tcga_counts_df.index):
        print('In sgn, not in tcga_counts_df:', set(sgn) - set(tcga_counts_df.index))
        print('In tcga_counts_df, not in sgn:', set(tcga_counts_df.index) - set(sgn))

    # select genes from tcga and transpose
    tmp = tcga_counts_df.loc[sgn]
    tmp.index = columns

    result_df = tmp.T

    # save dataframe
    result_df.to_csv(args.output, sep='\t')


if __name__ == '__main__':
    main()
