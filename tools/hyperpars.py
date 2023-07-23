from collections import Counter
from itertools import product

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

from tools.processes import ProcessHandler
from tools.utils import timer

METRICS = ('f1-score', 'precision', 'recall')
CLASSES = ('0', '1')
METRIC_NAMES = tuple(map(lambda x: '_'.join(list(x)), tuple(product(METRICS, CLASSES))))


df_final = pd.read_csv('../data/.local/processed_dataset.csv')
X = df_final.iloc[:, :-1].to_numpy()
Y = df_final.iloc[:, -1].to_numpy().reshape(df_final.shape[0])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
tr = np.unique(Y_train, return_counts=True)[1]
tst = np.unique(Y_test, return_counts=True)[1]
# print(f"\tTrain set\n\nbankrupt instances: {tr[1]},\nnon-bankrupt instances: {tr[0]}\n")
# print(f"\tTest set\n\nbankrupt instances: {tst[1]},\nnon-bankrupt instances: {tst[0]}")

rose = RandomOverSampler(random_state=31)
X_rose, Y_rose = rose.fit_resample(X_train, Y_train)
print(Counter(Y_rose))


params = {'n_estimators': list(np.arange(20, 40, 10)),
          'max_depth': list(np.arange(4, 25, 8)),
          'min_samples_split': list(np.power(2, np.arange(2, 1, -1)).astype(np.uint16)),
          'max_features': list(np.arange(0.4, 0.75, 0.1))}

param_prod_list = list(product(*params.values()))
param_sets = list(dict(zip(params.keys(), hs)) for hs in param_prod_list)
print('param set size:', len(param_sets))


def report_to_series(rp: dict) -> pd.Series:
    s = pd.Series(index=METRIC_NAMES)
    for m in METRICS:
        for c in CLASSES:
            s['_'.join([m, c])] = rp[c][m]
    return s


def train_clf(ps, return_list: bool = False):
    clf = RandomForestClassifier(**ps)
    clf.fit(X_rose, Y_rose)
    y_hat = clf.predict(X_test)
    report = classification_report(Y_test, y_hat, output_dict=True)
    series = report_to_series(rp=report)
    if return_list:
        return list(ps.values()) + list(series.values)
    return series


@timer
def iterative():
    df = pd.DataFrame(param_prod_list, columns=list(params.keys()))
    for i, ps in enumerate(param_sets):
        series = train_clf(ps)
        df.loc[i, series.index.values] = series
    df.to_csv('../data/.local/RF.csv', sep=';', index=False)


@timer
def processes():
    handler = ProcessHandler()
    handler.handle(func=train_clf, param_setups=param_sets,
                   output_file='../data/.local/RF2.csv',
                   header=list(params.keys()) + list(METRIC_NAMES))


if __name__ == "__main__":
    iterative()
    processes()

    # test
    df_it = pd.read_csv('../data/.local/RF.csv', sep=';')
    df_proc = pd.read_csv('../data/.local/RF2.csv', sep=';')

    print(df_it == df_proc)
    print('Done.')



