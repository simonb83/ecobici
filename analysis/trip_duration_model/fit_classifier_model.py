"""
Train a machine learning model on Ecobici trips data for trips between 2014-01-01 AND 2016-07-31

The aim is to predict bicycle trip duration in seconds based on the following features:

- Gender
- Age
- Weekday vs. Weekend
- Hour of Day
- Month
- Start location based on hexagonal grid

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import logging
import argparse


def duration_classes(x):
    if x <= 60 * 10:
        return 1
    if x <= 60 * 20:
        return 2
    if x <= 60 * 30:
        return 3
    if x <= 60 * 60:
        return 4


def pre_process_data(data, hexagons):
    data['duration'] = data['duration'].apply(lambda x: duration_classes(x))
    data.drop('Unnamed: 0', axis=1, inplace=True)
    data.reset_index(inplace=True, drop=True)
    for h in hexagons:
        data.drop("hex_{}".format(h), axis=1, inplace=True)
    return data


def parse_data(data):
    y = np.ravel(data['duration'])
    data = data.drop('duration', axis=1)
    return data, y

if __name__ == "__main__":

    logging.basicConfig(
        filename="output/fit_classifier_model.log", level=logging.INFO)
    hexagons = pd.read_csv('data/hexagons.csv', header=None)[0].tolist()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--estimators", help="Number of estimators")
    parser.add_argument(
        "-f", "--features", help="Max features")
    parser.add_argument(
        "-s", "--sample", help="Training sample size")
    parser.add_argument(
        "-t", "--test", help="Test sample size")
    args = parser.parse_args()

    n_estimators = int(args.estimators)
    features = str(args.features)
    train_sample = int(args.sample)
    test_sample = int(args.test)

    class_weights = {1: 0.49, 2: 0.34, 3: 0.11, 4: 0.06}

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_features=features, class_weight=class_weights, random_state=1)

    logging.info("Start training")
    store = pd.HDFStore('output/train.h5')
    nrows = store.get_storer('train').nrows
    r = np.random.randint(0, nrows, size=train_sample)
    data = pd.read_hdf('output/train.h5', 'train', where=pd.Index(r))
    data = pre_process_data(data, hexagons)

    train_X, train_y = parse_data(data)
    clf.fit(train_X, train_y)

    logging.info("Start testing")
    store = pd.HDFStore('output/test.h5')
    nrows = store.get_storer('test').nrows
    r = np.random.randint(0, nrows, size=test_sample)
    data = pd.read_hdf('output/test.h5', 'test', where=pd.Index(r))
    data = pre_process_data(data, hexagons)

    test_X, test_y = parse_data(data)
    logging.info("Columns: \n")
    logging.info(test_X.columns.tolist())
    y_pred = clf.predict(test_X)

    logging.info("Model score: {}".format(clf.score(test_X, test_y)))

    df = pd.DataFrame(np.array([test_y, y_pred]).T,
                      columns=['True', 'Predicted'])
    df.to_hdf('output/predicted_class.h5',
              'predicted_class', append=True, format='t')


    logging.info("Save model to disk:\n")
    joblib.dump(clf, 'models/classifier.pkl') 
    logging.info("Detailed classification report:\n")
    df = pd.read_hdf('output/predicted_class.h5')
    logging.info(classification_report(df['True'], df[
                 'Predicted']))
