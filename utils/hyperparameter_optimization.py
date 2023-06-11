import os
import re
import csv
from itertools import product

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from typing import Sequence, Union, Annotated
from multiprocessing import Manager, Pool, cpu_count
from tqdm import tqdm

ALLOWED_CLASSIFIERS = Union[RandomForestClassifier, AdaBoostClassifier]


# def __int__(self, data: Annotated[Sequence[np.ndarray], 4],
#             classifier: ALLOWED_CLASSIFIERS,
#             hyperparameter_setups: Union[dict, pd.DataFrame]):

class HyperparameterOptimizer:
    def __init__(self, data: Annotated[Sequence[np.ndarray], 4],
                 classifier: ALLOWED_CLASSIFIERS,
                 setups: Union[dict, pd.DataFrame]):
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.classifier = classifier()
        self.setups = setups
        self._optimize_dtypes()

    def run_all_setups(self, processes: bool = False):
        filepath = os.path.join('data', '.locals', 'file.csv')
        if not processes:
            for stp in tqdm(self.setups, desc="training forests..."):
                self.test_single_setup(setup=stp)
        else:
            mp_handler = MultiprocessHandler(func=self.test_single_setup, output_file=filepath, setups=self.setups)
            mp_handler.handle()

    def test_single_setup(self, setup: Union[dict, pd.Series], return_str=False):
        """
        :param setup:
        :param return_str:
        :return:
        """
        params = {'n_estimators': setup[0],
                  'criterion': 'entropy',
                  'max_depth': setup[1],
                  'min_samples_split': setup[2],
                  'max_features': setup[3],
                  'random_state': 35}

        self.classifier.__dict__.update(params)

        self.classifier.fit(self.x_train, self.y_train)
        setup_evaluator = SetupEvaluator(model=self.classifier,
                                         name="Random Forest",
                                         n_estimators=self.classifier.n_estimators,
                                         max_depth=self.classifier.max_depth,
                                         min_samples_split=self.classifier.min_samples_split,
                                         max_features=self.classifier.max_features
                                         )
        setup_evaluator.evaluate(self.x_test, self.y_test)
        if return_str:
            return list(map(lambda x: str(x), setup_evaluator.to_list()))
        else:
            return setup_evaluator.to_list()

    def _optimize_dtypes(self):
        self.x_train, self.x_test = self.x_train.astype(np.float32), self.x_test.astype(np.float32)
        self.y_train, self.y_test = self.x_train.astype(np.uint8), self.x_test.astype(np.uint8)


class SetupEvaluator:
    def __init__(self, model, name, **hyperparameters):
        self.model = model
        self.name = name
        self.hyperparameters = hyperparameters

        self.weighted_balanced_accuracy = None
        self.precision_class_0 = None  # Non-bankrupts
        self.precision_class_1 = None  # Bankrupts

        self.recall_class_0 = None
        self.recall_class_1 = None

        self.f1_class_0 = None
        self.f1_class_1 = None

        self.f_beta = None
        self.AUC = None

    def evaluate(self, x, y_true):
        y_predicted = self.model.predict(x)
        # y_proba = model.predict_proba(x)

        self.calculate_precision(y_true, y_predicted)
        self.calculate_recall(y_true, y_predicted)
        self.calculate_weighted_balanced_accuracy(y_true, y_predicted)
        self.calculate_f_score(beta=1)
        self.calculate_f_score(
            beta=2)  # False Negatives are more important, because we don't want to misclassify actual bankrupts
        # self.calculate_AUC(y_proba)

    def calculate_weighted_balanced_accuracy(self, true, predicted):
        self.weighted_balanced_accuracy = (self.recall_class_0 + self.recall_class_1) / 2

    def calculate_f_score(self, beta=1):
        if beta == 1:
            self.f1_class_0 = self.precision_class_0 * self.recall_class_0 / (
                    self.precision_class_0 + self.recall_class_0)
            self.f1_class_1 = self.precision_class_1 * self.recall_class_1 / (
                    self.precision_class_1 + self.recall_class_1)
        else:
            self.f_beta = (1 + beta ** 2) * self.precision_class_1 * self.recall_class_1 / (
                    beta ** 2 * self.precision_class_1 + self.recall_class_1)

    def calculate_precision(self, true, predicted):
        self.precision_class_0 = (predicted == true)[true == 0].mean()
        self.precision_class_1 = (predicted == true)[true == 1].mean()

    def calculate_recall(self, true, predicted):
        self.recall_class_0 = (predicted == true)[predicted == 0].mean()
        self.recall_class_1 = (predicted == true)[predicted == 1].mean()

    def calculate_auc(self, y_proba):
        raise NotImplementedError

    def to_dict(self):
        d = {'Name': self.name}
        d.update(self.hyperparameters)
        d.update({'Weighted accuracy': self.weighted_balanced_accuracy,
                  'Precision class 0': self.precision_class_0,
                  'Precision class 1': self.precision_class_1,
                  'Recall class 0': self.recall_class_0,
                  'Recall class 1': self.recall_class_1,
                  'F1 class 0': self.f1_class_0,
                  'F1 class 1': self.f1_class_1,
                  'F-Beta': self.f_beta})

        return d

    def to_list(self):
        return list(self.to_dict().values())


class MultiprocessHandler:
    def __init__(self, func, output_file: str, setups=Union[dict, pd.DataFrame]):
        self.func = func
        self.output_file = output_file
        self.setups = setups

        self.manager = Manager()
        self.queue = self.manager.Queue()
        self.pool = Pool(cpu_count() + 2)

        # put listener to work first
        self.pool.apply_async(self.listener, (self.queue,))

    def handle(self, ):
        # fire off workers
        jobs = []
        for stp in self.setups:
            job = self.pool.apply_async(self.worker, (stp, self.queue))
            jobs.append(job)

        # collect results from the workers through the pool result queue
        for job in tqdm(jobs, desc="training forests with multiprocessing..."):
            job.get()

        # now we are done, kill the listener
        self.queue.put('<KILL>')
        self.pool.close()
        self.pool.join()

    def listener(self):
        """Listen for messages from a queue. Then write them to csv."""

        filename = re.sub(r'\W', '_', self.func.__doc__).lower() + '.csv'
        header = []

        with open(self.output_file, 'w', newline='\n', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            # Write header
            writer.writerow(header)
            f.flush()

            while 1:
                m = self.queue.get()
                if m == '<KILL>':
                    print(f"Results have been writen into {filename}.")
                    break

                writer.writerow(m.split(','))
                f.flush()

    def worker(self, **kwargs):
        res = self.func(**kwargs)
        res = ",".join(res)
        self.queue.put(res)


if __name__ == "__main__":
    N_ESTIMATORS = np.arange(8, 25, 8)
    MAX_DEPTHS = np.arange(8, 25, 8)
    MIN_SAMPLES_SPLITS = np.power(2, np.arange(3, 1, -1)).astype(np.uint16)
    MAX_FEATURES = np.arange(0.4, 0.6, 0.1)

    data_ = [pd.read_csv(os.path.join('data', '.locals', 'x_train.csv')).to_numpy(),
             pd.read_csv(os.path.join('data', '.locals', 'y_train.csv')).to_numpy(),
             pd.read_csv(os.path.join('data', '.locals', 'x_test.csv')).to_numpy(),
             pd.read_csv(os.path.join('data', '.locals', 'y_test.csv')).to_numpy()
             ]
    hyperparameters_sets = list(product(N_ESTIMATORS, MAX_DEPTHS, MIN_SAMPLES_SPLITS, MAX_FEATURES))
    ho = HyperparameterOptimizer(data=data_, classifier=RandomForestClassifier, setups=hyperparameters_sets)
    ho.run_all_setups(processes=False)
