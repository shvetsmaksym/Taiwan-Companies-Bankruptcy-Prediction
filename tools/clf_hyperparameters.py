from collections import Counter
from itertools import product
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from tqdm import tqdm

from tools.processes import ProcessHandler
from tools.utils import timer

global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
global METRICS
global CLASSES
global METRIC_NAMES
global BASE_CLF
global HYPERPARAMETERS
global OUTFILE

HYPERPARAMETERS_HINT = Dict[str, List]


def report_to_series(rp: dict) -> pd.Series:
    s = pd.Series(index=METRIC_NAMES)
    for m in METRICS:
        for c in CLASSES:
            s['_'.join([m, c])] = rp[c][m]
    return s


def train_clf(ps):
    clf = BASE_CLF(**ps)
    clf.fit(X_TRAIN, Y_TRAIN)
    y_hat = clf.predict(X_TEST)
    report = classification_report(Y_TEST, y_hat, output_dict=True, zero_division=np.nan)
    series = report_to_series(rp=report)
    return series


@timer
def iterative(pp: List[Tuple], ps: List[Dict]):
    df = pd.DataFrame(pp, columns=list(HYPERPARAMETERS.keys()))
    for i, ps in tqdm(enumerate(ps), desc="Iterative method", total=len(ps)):
        series = train_clf(ps)
        df.loc[i, series.index.values] = series
    df.to_csv(OUTFILE, sep=';', index=False)


@timer
def processes(ps: List[Dict]):
    handler = ProcessHandler()
    handler.handle(func=train_clf, param_setups=ps,
                   output_file=OUTFILE,
                   header=list(HYPERPARAMETERS.keys()) + list(METRIC_NAMES))


def run_inference(data: Tuple[np.ndarray],
                  base_clf,
                  params: HYPERPARAMETERS_HINT,
                  metrics: tuple = ('f1-score', 'precision', 'recall'),
                  classes: tuple = (0, 1),
                  output_file: os.path = os.path.join('..', 'data', '.local', 'clf_results.csv'),
                  multiprocessing_mode: bool = False):
    global X_TRAIN, X_TEST, Y_TRAIN, Y_TEST
    global METRICS
    global CLASSES
    global METRIC_NAMES
    global BASE_CLF
    global HYPERPARAMETERS
    global OUTFILE

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = list(*data)
    METRICS = metrics
    CLASSES = [str(c) for c in classes]
    METRIC_NAMES = tuple(map(lambda x: '_'.join(list(x)), tuple(product(METRICS, CLASSES))))
    BASE_CLF = base_clf
    HYPERPARAMETERS = params
    OUTFILE = output_file

    pp = list(product(*HYPERPARAMETERS.values()))
    ps = list(dict(zip(HYPERPARAMETERS.keys(), hs)) for hs in pp)
    print("Number of classifiers to be trained:", len(pp))

    if multiprocessing_mode:
        processes(ps=ps)
    else:
        iterative(pp=pp, ps=ps)


def generate_hprs_report(df: pd.DataFrame, hprs: list, metrics=None):
    print("\t\tHyperparameters Report")
    if metrics is None:
        metrics = ['f1-score_1']

    print("\n\tCorrelation between given metrics and hyperparameters")
    for metric in metrics:
        print(f"\n\t---{metric}---\n")

        m_q50 = df[metric].quantile(0.5)
        r_ = pd.DataFrame(df[df[metric] >= m_q50][hprs + [metric]].corr()[metric])
        r_ = r_.drop(metric)
        print(r_)

    print("\n\n\tBest setups per metric")
    for metric in metrics:
        print(f"\n\t---{metric}---\n")

        m_q99 = df[metric].quantile(0.99)
        r_ = pd.DataFrame(df[df[metric] >= m_q99][hprs + [metric]])
        print(r_.head())


if __name__ == "__main__":
    df_final = pd.read_csv(os.path.join('..', 'data', '.local', 'processed_dataset.csv'))
    X = df_final.iloc[:, :-1].to_numpy()
    Y = df_final.iloc[:, -1].to_numpy().reshape(df_final.shape[0])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    tr = np.unique(Y_train, return_counts=True)[1]
    tst = np.unique(Y_test, return_counts=True)[1]
    print(f"\tTrain set\n\nbankrupt instances: {tr[1]},\nnon-bankrupt instances: {tr[0]}\n")
    print(f"\tTest set\n\nbankrupt instances: {tst[1]},\nnon-bankrupt instances: {tst[0]}")

    rose = RandomOverSampler(random_state=31)
    X_rose, Y_rose = rose.fit_resample(X_train, Y_train)
    print(Counter(Y_rose))

    params = {'n_estimators': [3, 5, 10],
              'criterion': ['gini', 'entropy'],
              'max_depth': [3, 4, 5, 6],
              'min_samples_leaf': [1, 2, 4, 6, 8],
              'max_features': [round(0.1 * x, 1) for x in range(3, 6)]}

    run_inference(data=(X_rose, X_test, Y_rose, Y_test),
                  base_clf=RandomForestClassifier,
                  params=params,
                  metrics=('f1-score', 'precision', 'recall'),
                  classes=(0, 1),
                  output_file=os.path.join('..', 'data', '.local', 'RF_it_results.csv'),
                  multiprocessing_mode=True)

    # test
    # df_it = pd.read_csv('../data/.local/RF.csv', sep=';')
    # df_proc = pd.read_csv('../data/.local/RF2.csv', sep=';')

    print('Done.')



