# libraries
import pandas as pd
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description="Split datasets to training, validation and testing.")
    parser.add_argument('--naive', required=True)
    parser.add_argument('--pca', required=True)
    parser.add_argument('--samples', required=True)
    parser.add_argument('--stages', required=True)
    parser.add_argument('--histogram_data', required=True)

    # params
    # naive = '/home/andy/Projects/pib/data/gene/full/naive/counts_20.tsv'
    # pca = '/home/andy/Projects/pib/data/gene/full/pca/counts_20.tsv'
    # samples = '/home/andy/Projects/pib/data/gene/full/TCGA/dummy/samples.tsv'
    # stages = '/home/andy/Projects/pib/data/gene/full/TCGA/dummy/stages.tsv'
    # histogram_data = '/home/andy/Projects/pib/data/gene/full/histogram_data'

    random_state = 0
    args = parser.parse_args()

    # load samples
    samples_df = pd.read_csv(args.samples, sep='\t', header=0, index_col=0)
    samples_list = samples_df.values.argmax(axis=1)
    samples_cat = pd.DataFrame(index = samples_df.index, data=samples_list, columns=['samples'])

    # load stages
    stages_df = pd.read_csv(args.stages, sep='\t', header=0, index_col=0)
    stages_list = stages_df.values.argmax(axis=1)
    stages_cat = pd.DataFrame(index = stages_df.index, data=stages_list, columns=['stages'])

    #
    joined_df = samples_cat.join(stages_cat, how='outer').dropna()

    NA_df = samples_cat.join(stages_cat, how='outer').drop(joined_df.index) # what to do with NA???
    NA_df.to_csv(f'{args.histogram_data}/na_df.tsv', sep='\t')

    full_df = joined_df.reset_index().groupby(['stages', 'samples']).count().reset_index()  # to create histogram
    full_df.to_csv(f'{args.histogram_data}/full_hist.tsv', sep='\t')

    # split
    print(f'Full set:\t{len(joined_df)}.')
    tmp_df = joined_df

    test_set = tmp_df.reset_index()\
                     .groupby(['samples', 'stages'])\
                     .apply(lambda x: x.sample(frac=0.15, random_state=random_state))

    test_list = list(test_set['index'])
    print(f'Test set:\t{len(test_list)}.\t({len(test_list)/len(joined_df)})')
    tmp_df = tmp_df.drop(test_list)

    validation_set = tmp_df.reset_index()\
                           .groupby(['samples', 'stages'])\
                           .apply(lambda x: x.sample(frac=0.175, random_state=random_state))

    validation_list = list(validation_set['index'])
    print(f'Validation set:\t{len(validation_list)}.\t({len(validation_list)/len(joined_df)})')
    tmp_df = tmp_df.drop(validation_list)

    train_set = tmp_df.reset_index()
    train_list = list(train_set['index'])
    print(f'Train set:\t{len(train_list)}.\t({len(train_list)/len(joined_df)})')

    len(set(test_list + validation_list + train_list))

    # write data for paper
    test_hist = test_set.reset_index(drop=True).groupby(['stages', 'samples']).count().reset_index()
    test_hist.to_csv(f'{args.histogram_data}/test_hist.tsv')

    validation_hist = validation_set.reset_index(drop=True).groupby(['stages', 'samples']).count().reset_index()
    validation_hist.to_csv(f'{args.histogram_data}/validation_hist.tsv')

    train_hist = train_set.reset_index(drop=True).groupby(['stages', 'samples']).count().reset_index()
    train_hist.to_csv(f'{args.histogram_data}/train_hist.tsv')

    # write split datasets
    files = [args.naive, args.pca, args.samples, args.stages]
    sets = {'training': train_list, 'validation': validation_list, 'testing': test_list}

    for file_name in files:
        print(f'Loading {file_name}.')
        df = pd.read_csv(file_name, sep='\t', header=0, index_col=0)
        for set_type in ['training', 'validation', 'testing']:
            new_file_name = file_name.replace('.tsv', f'.{set_type}.tsv')
            rows = list(set(sets[set_type]).intersection(set(df.index)))
            print(f'Creating {new_file_name} with {len(rows)} records.')
            new_df = df.loc[rows, :]

            new_df.to_csv(new_file_name, sep='\t')

if __name__ == '__main__':
    main()
