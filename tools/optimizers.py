import os
import csv
import time
from itertools import product

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from typing import Sequence, Union, Annotated, Literal
from multiprocessing import Manager, Pool, cpu_count
from tqdm import tqdm

from tools.utils import dtypes_fixer, timefn

ALLOWED_CLASSIFIERS = Union[RandomForestClassifier, AdaBoostClassifier]


class Optimizer:
    """
    Hyperparameter testing and optimization
    """
    def __init__(self, data: Annotated[Sequence[np.ndarray], 4], clf: ALLOWED_CLASSIFIERS, params: dict):
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.clf = clf
        self.setups = self.__unpack_params(p=params)
        self.__optimize_dtypes()
        self.history = pd.DataFrame()

        self.filepath = os.path.join('data', '.local', 'RF.csv')
        self.header = list(self.setups.columns) + list(SetupEvaluator().metrics.keys())

    @staticmethod
    @timefn
    def __unpack_params(p: dict) -> pd.DataFrame:
        d_ = list(product(*p.values()))
        return pd.DataFrame(data=d_, columns=list(p.keys()))

    def run_all_setups(self, processes: bool = False):
        if not processes:
            for stp in tqdm(self.setups.iterrows(), desc="training models..."):
                r = self.run_single_setup(setup=stp[1], out_type='Series')
                self.history = pd.concat([self.history, r], axis=1)
            self.history.to_csv(self.filepath, sep=';')
            print(f"Results writen into {self.filepath}.")
        else:
            handler = ProcessHandler()

            handler.handle(func=self.run_single_setup, setups=self.setups, output_file=self.filepath, header=self.header)
            self.history = pd.read_csv(self.filepath, encoding='utf-8')

    def run_single_setup(self, setup: pd.Series, out_type: str = 'dict'):
        """
        :param setup:
        :param out_type:
        :return:
        """
        recovered_dtypes = dtypes_fixer(setup.to_dict())
        self.clf.__dict__.update(recovered_dtypes)
        t1 = time.time()
        self.clf.fit(self.x_train, self.y_train)
        t2 = time.time()
        print(round(t2 - t1, 2))
        setup_evaluator = SetupEvaluator(model=self.clf, **recovered_dtypes)
        setup_evaluator.evaluate(self.x_test, self.y_test)

        return setup_evaluator.astype(t=out_type)

    @timefn
    def __optimize_dtypes(self):
        self.x_train, self.x_test = self.x_train.astype(np.float32), self.x_test.astype(np.float32)
        self.y_train, self.y_test = self.x_train.astype(np.uint8), self.x_test.astype(np.uint8)


class SetupEvaluator:
    def __init__(self, model=None, **params):
        self.model = model
        self.params = params

        self.weighted_balanced_accuracy = None
        self.precision_class_0 = None  # Non-bankrupts
        self.precision_class_1 = None  # Bankrupts

        self.recall_class_0 = None
        self.recall_class_1 = None

        self.f1_class_0 = None
        self.f1_class_1 = None

        self.f_beta = None

        self.metrics = {'weighted accuracy': self.weighted_balanced_accuracy,
                        'precision (class 0)': self.precision_class_0,
                        'precision (class 1)': self.precision_class_1,
                        'recall (class 0)': self.recall_class_0,
                        'recall (class 1)': self.recall_class_1,
                        'f1 (class 0)': self.f1_class_0,
                        'f1 (class 1)': self.f1_class_1,
                        'f-Beta': self.f_beta}
    @timefn
    def evaluate(self, x, y_true):
        y_predicted = self.model.predict(x)
        # y_proba = model.predict_proba(x)

        self.calculate_precision(y_true, y_predicted)
        self.calculate_recall(y_true, y_predicted)
        self.calculate_weighted_balanced_accuracy()
        self.calculate_f_score(beta=1)
        self.calculate_f_score(beta=2)  # False Negatives are more important,
                                        # because we don't want to misclassify actual bankrupts

    def calculate_weighted_balanced_accuracy(self):
        self.weighted_balanced_accuracy = (self.recall_class_0 + self.recall_class_1) / 2

    def calculate_f_score(self, beta=1):
        if beta == 1:
            self.f1_class_0 = 2 * self.precision_class_0 * self.recall_class_0 / (
                    self.precision_class_0 + self.recall_class_0)
            self.f1_class_1 = 2 * self.precision_class_1 * self.recall_class_1 / (
                    self.precision_class_1 + self.recall_class_1)
        else:
            self.f_beta = (1 + beta ** 2) * self.precision_class_1 * self.recall_class_1 / (
                    beta ** 2 * self.precision_class_1 + self.recall_class_1)

    def calculate_precision(self, true, predicted):
        self.precision_class_0 = (predicted == true)[predicted == 0].mean()
        self.precision_class_1 = (predicted == true)[predicted == 1].mean()

    def calculate_recall(self, true, predicted):
        self.recall_class_0 = (predicted == true)[true == 0].mean()
        self.recall_class_1 = (predicted == true)[true == 1].mean()

    def astype(self, t: Literal['dict', 'list', 'Series', 'str']):
        if t == 'dict': return self.to_dict()
        elif t == 'list': return self.to_list()
        elif t == 'Series': return self.to_series()
        elif t == 'str': return self.to_list_of_strings()
        else: return

    def to_dict(self):
        d = self.params.copy()
        d.update({'weighted accuracy': self.weighted_balanced_accuracy,
                  'precision (class 0)': self.precision_class_0,
                  'precision (class 1)': self.precision_class_1,
                  'recall (class 0)': self.recall_class_0,
                  'recall (class 1)': self.recall_class_1,
                  'f1 (class 0)': self.f1_class_0,
                  'f1 (class 1)': self.f1_class_1,
                  'f-Beta': self.f_beta})
        return d

    def to_list(self):
        return list(self.to_dict().values())

    def to_series(self):
        return pd.Series(self.to_dict())

    def to_list_of_strings(self):
        return list(map(lambda x: str(x), self.to_dict().values()))


class ProcessHandler:
    @classmethod
    def handle(cls, func, setups: pd.DataFrame, output_file: str, header):
        manager = Manager()
        queue = manager.Queue()
        pool = Pool(cpu_count() + 2)
        jobs = []

        pool.apply_async(cls.listener, (queue, output_file, header))

        for stp in setups.iterrows():
            job = pool.apply_async(cls.worker, (queue, func, stp[1]))
            jobs.append(job)

        for job in tqdm(jobs, desc="training models with multiprocessing..."):
            job.get()

        # now we are done, kill the listener
        queue.put('<KILL>')
        pool.close()
        pool.join()

    @classmethod
    def listener(cls, q, output_file, header: list):
        """Listen for messages from a queue. Then write them to csv."""

        with open(output_file, 'w', newline='\n', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(header)
            f.flush()

            while 1:
                m = q.get()
                if m == '<KILL>':
                    print(f"Results writen into {output_file}.")
                    break

                writer.writerow(m.split(','))
                f.flush()

    @classmethod
    def worker(cls, q, func, setup):
        res = func(setup, out_type='str')
        res = ",".join(res)
        q.put(res)


if __name__ == "__main__":
    t1 = time.time()
    params = {'n_estimators': list(np.arange(8, 25, 8)),
              'max_depth': list(np.arange(8, 25, 8)),
              'min_samples_split': list(np.power(2, np.arange(3, 1, -1)).astype(np.uint16)),
              'max_features': list(np.arange(0.4, 0.6, 0.1))}

    data_ = [pd.read_csv(os.path.join('data', '.local', 'x_train.csv')).to_numpy(),
             pd.read_csv(os.path.join('data', '.local', 'y_train.csv')).to_numpy(),
             pd.read_csv(os.path.join('data', '.local', 'x_test.csv')).to_numpy(),
             pd.read_csv(os.path.join('data', '.local', 'y_test.csv')).to_numpy()
             ]

    opt = Optimizer(data=data_, clf=RandomForestClassifier(), params=params)
    opt.run_all_setups(processes=False)
    opt = opt.history
    t2 = time.time()

    df = pd.read_csv('data/.local/RF.csv', sep=';')
    df.to_csv()
    with open('timings.txt', 'a', newline='\n') as f:
        f.write(f"processes=False, {len(list(product(*params.values())))} setups: " + str(round(t2 - t1, 2)))
    print('done.')

