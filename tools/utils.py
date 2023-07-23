import re
import numpy as np
from itertools import combinations, product
from functools import wraps
from time import time

import pandas as pd
from typing import List


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time()
        result = fn(*args, **kwargs)
        t2 = time()
        print("@timefn:" + fn.__str__() + " took " + str(t2 - t1) + " seconds")
        return result
    return measure_time


def str_to_readable_title(text: str, max_lines: int = 3) -> str:
    space_count = text.count(" ")
    if space_count < max_lines:
        return re.sub(" ", "\n", text)
    else:
        words = np.array(text.split(" "))
        word_lens = np.vectorize(len)(words).astype(np.uint8)
        indexes = np.arange(1, word_lens.shape[0]).astype(np.uint8)
        combs = np.array(list(combinations(indexes, max_lines-1))).astype(np.uint8)
        combs = np.concatenate((np.zeros((combs.shape[0], 1), dtype=np.uint8),
                               combs,
                               np.array([word_lens.shape[0]] * combs.shape[0], dtype=np.uint8).reshape(combs.shape[0], 1)),
                               axis=1)

        arr_splits = np.apply_along_axis(subseq_sum, axis=1, arr=combs, seq=word_lens)  # possible solutions
        assert np.all(arr_splits.sum(axis=1) == word_lens.sum())
        optimal_split_id = arr_splits.max(axis=1).argmin()
        optimal_comb = combs[optimal_split_id, :]
        words_by_line = [words[optimal_comb[i]:optimal_comb[i+1]] for i in range(max_lines)]
        return "\n".join(list(map(lambda t: " ".join(t), words_by_line)))


def subseq_sum(comb: np.array, seq: np.array) -> np.array:
    """"
    :param comb: particular combination of indexes that point subsequence splits
    :param seq: sequence to calculate subsequence sums for
    """

    return np.array([seq[comb[x]:comb[x + 1]].sum() if comb[x+1]-comb[x] > 1 else seq[comb[x]]
                     for x in range(comb.shape[0] - 1)]).astype(np.uint8)


def arr2d_to_cubes(arr: np.array, cube_len=10) -> np.array:
    new_dim = np.array(arr.shape) // cube_len
    new_size = (new_dim[0] ** 2) * (cube_len ** 2)
    new_dim += 1 if new_size < arr.size else 0
    new_dim = list(new_dim) + [cube_len, cube_len]
    new_arr = np.zeros(shape=tuple(new_dim))

    for dim0 in range(new_dim[0]):
        for dim1 in range(new_dim[1]):
            new_arr[dim0, dim1, :, :] = arr[dim0 * cube_len: (dim0+1)*cube_len,
                                            dim1 * cube_len: (dim1+1)*cube_len]
    return new_arr


def pd_to_cubes(df: pd.DataFrame, cube_len=10) -> List[pd.DataFrame]:
    """
    :param df: initial DataFrame
    :param cube_len: side length of cubes
    :return: list of cubes of shape (cube_len x cube_len
    """
    new_dim = np.array(df.shape) // cube_len
    df_queue = list()
    for x, y in list(product(range(new_dim[0]), range(new_dim[1]))):
        df_queue.append(df.iloc[x*cube_len:(x+1)*cube_len, y*cube_len:(y+1)*cube_len])

    return df_queue


def filter_dataset(df: pd.DataFrame, t: int, label_col: str = 'bankrupt', return_df: bool = False):
    """
    :param df: initial DataFrame
    :param label_col: dependent column to exclude from filtering
    :param t: number of instances allowed to be irrelevant per particular column
    :param return_df:
    :return: filtered DataFrame if return_df=True. Otherwise, return (info_keep, bankrupt ratio),
    where info_keep is retained % of cells from df.
    """
    df_new = df.copy()

    # remove columns with significant number of irrelevant values
    cols_to_remove = df_new.columns[(df_new > 1).sum(axis=0) >= t]
    df_new = df_new.drop(columns=list(cols_to_remove))

    # remove rows with irrelevant values left
    df_new = df_new[~(df_new > 1).any(axis=1)]
    if return_df:
        return df_new
    else:
        info_keep = np.float32((df_new.shape[0] * df_new.shape[1]) / df.size)
        num_yes = df_new[df_new[label_col] == 1].shape[0]
        num_no = df_new[df_new[label_col] == 0].shape[0]
        bankruptcy_ratio = np.float32(num_yes / (num_yes + num_no))
        return info_keep, bankruptcy_ratio


@timefn
def dtypes_fixer(d: dict):
    return {k: (int(v) if not v % 1 else v) for k, v in d.items()}


if __name__ == "__main__":
    df = pd.DataFrame(np.random.random((80, 80)))
    cubes = pd_to_cubes(df)
    print(cubes)
