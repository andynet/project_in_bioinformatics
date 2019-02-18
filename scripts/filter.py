import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Normalization of recount experiment")
    parser.add_argument('-c', '--counts', required=True)
    parser.add_argument('-s', '--samples', required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()

    df = pd.DataFrame(pd.read_csv(args.counts, sep='\t', header=0, index_col=0))

    counts_df = df.copy()
    counts_df = counts_df.reset_index()
    counts_df = counts_df.melt(id_vars=['index'])
    counts_df.columns = ['gene', 'sample', 'expression']

    samples_df = pd.DataFrame(pd.read_csv(args.samples, sep='\t', header=0, index_col=0))
    samples_df = samples_df.reset_index()
    samples_df.columns = ['sample', 'tissue_type']

    main_df = pd.merge(counts_df, samples_df, on='sample')
    main_df = main_df.groupby(['gene', 'tissue_type']).mean().reset_index()
    main_df = main_df.groupby(['gene']).var().reset_index()
    main_df = main_df.sort_values(by=['expression'], ascending=False)

    selected_genes = list(main_df.iloc[0:main_df.shape[0] // 5, 0])

    df_final = df.loc[df.index.isin(selected_genes)].T.reset_index()
    df_final.columns = ['sample'] + df_final.columns.to_list()[1:]
    df_final = pd.merge(df_final, samples_df, on='sample')
    df_final['category'] = pd.Categorical(df_final['tissue_type']).codes
    col = ['sample', 'tissue_type', 'category'] + df_final.columns.to_list()[1:-2]
    df_final = df_final[col]

    df_final.to_csv(args.output, sep='\t')


if __name__ == '__main__':
    main()
