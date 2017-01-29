"""
Create all features and split the data into training and test sets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import logging

def process_data(df):
    df[['month', 'start_hour']] = df[['month', 'start_hour']].astype(np.dtype('i4'))

    for i in range(1, 13):
        df['month_{}'.format(i)] = df['month'].apply(lambda x: x == i)
    df = df.drop('month', axis=1)

    for i in [0] + [i for i in range(5, 24)]:
        df['hour_{}'.format(i)] = df['start_hour'].apply(lambda x: x == i)
    df = df.drop('start_hour', axis=1)

    for h in hexagons:
        df['hex_{}'.format(h)] = df['hexagon_id'].apply(lambda x: x == h)
    df = df.drop('hexagon_id', axis=1)

    return df


if __name__ == "__main__":

    logging.basicConfig(filename="output/split_data.log", level=logging.INFO)
    hexagons = pd.read_csv('data/hexagons.csv', header=None)[0].tolist()

    num_rows = int(sys.argv[1])
    if num_rows == -1:
        num_rows = None

    all_data = pd.read_csv('data/model_data.csv', nrows=num_rows)
    logging.info("Loaded data")
    all_data = all_data[all_data['hexagon_id'].notnull()]
    all_data = all_data.drop('id', axis=1)
    all_data = all_data.dropna()

    logging.info("Splitting into test and train subsets")
    indices = np.arange(len(all_data))
    np.random.shuffle(indices)
    train_top = int(np.floor(0.8 * len(all_data)))
    train_idx, test_idx = indices[:train_top], indices[train_top:]
    all_data.iloc[train_idx].to_csv('output/train.csv')
    all_data.iloc[test_idx].to_csv('output/test.csv')
    logging.info("Subsets saved to disk")

    # Process the training data
    logging.info("Loading training data")
    all_data = pd.read_csv('output/train.csv')
    logging.info("Processing training data")
    all_data = process_data(all_data)
    logging.info("Shuffling columns")
    new_cols = all_data.columns.tolist()
    new_cols.remove('duration')
    np.random.shuffle(new_cols)
    new_cols = ['duration'] + new_cols

    all_data = all_data[new_cols]
    logging.info("Saving training data to disk")
    all_data.to_hdf('output/train.h5', 'train', format='t')

    # Now process the test data
    logging.info("Loading test data")
    all_data = pd.read_csv('output/test.csv')
    logging.info("Processing test data")
    all_data = process_data(all_data)

    all_data = all_data[new_cols]
    logging.info("Saving test data to disk")
    all_data.to_hdf('output/test.h5', 'test', format='t')