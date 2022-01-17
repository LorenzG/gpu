from ast import operator
from typing import Tuple
from itertools import accumulate
from functools import reduce
import random

import pandas as pd
import numpy as np

from decorators import on_gpu, gpus
from utils import coerce_iterable

random.seed(0)

def run_example(func, *args, **kwargs):
    now = pd.Timestamp.now()
    print(f'== Example {func.__name__} ==')
    res = func(*args, **kwargs)
    elapsed = pd.Timestamp.now() - now
    print(f'Elapsed: {elapsed}')
    res = coerce_iterable(res)
    for i,r in enumerate(res):
        print(f'Out {i}, type: {type(r)}:')
        print(r)
        print('')
    print('')
    print('')


@on_gpu
def dataframes_in_and_out(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[float, pd.Series]:
    res = df1 + df2
    last_column = res.iloc[:, -1]
    sum_last_column = last_column.sum()
    return sum_last_column, last_column


@on_gpu
def many_operations(df: pd.DataFrame, loops: int) -> float:
    res = df
    for _ in range(loops):
        res *= 3
        res -= df + df + df

    error = (res - df).abs().sum().sum()
    return error


@on_gpu
def db_operations(df: pd.DataFrame, groups, filters) -> float:
    masks = [(df[filter] == 0).values for filter in filters]
    # we can't easily mix cupy and cudf: https://stackoverflow.com/a/61667071
    # masks = np.array(masks)
    # mask = np.sum(masks, axis=0) == 0
    mask = reduce(lambda acc,x: acc & x, masks)
    df = df[mask]
    df = df.drop(filters, axis=1)
    return df.groupby(groups).sum()


@on_gpu(persist_cudf=True)
def persist_1(df: pd.DataFrame) -> pd.DataFrame:
    res = df
    res *= 3
    res -= df + df + df
    return df


@on_gpu
def persist(df, loops):
    df1 = persist_1(df)
    for _ in range(loops):
        df1 = persist_1(df1)
    return (df - df1).abs().sum().sum()


if __name__ == '__main__':

    ## 1
    df1 = pd.DataFrame({
        'a': np.arange(5) * 1,
        'b': np.arange(5) * 10,
    })
    df2 = pd.DataFrame({
        'a': np.arange(5) * 100,
        'b': np.arange(5) * 1000,
    })
    run_example(dataframes_in_and_out, df1, df2)
    with gpus(False):
        run_example(dataframes_in_and_out, df1, df2)

    ## 3
    big_df = pd.DataFrame({f'col_{i}':range(100000) for i in range(1000)})
    run_example(persist, big_df, 100)
    with gpus(False):
        run_example(persist, big_df, 100)

    ## 2
    big_df1 = pd.DataFrame({
        f'col_{i}': [random.choice(range(2)) for _ in range(100000)]
        for i in range(1000)
    })
    run_example(
        db_operations,
        big_df1,
        [f'col_{i}' for i in range(10)],
        [f'col_{i}' for i in range(100, 200)],
    )
    with gpus(False):
        run_example(
            db_operations,
            big_df1,
            [f'col_{i}' for i in range(50)],
            [f'col_{i}' for i in range(100, 200)],
        )

    ## 4
    big_df = pd.DataFrame({f'col_{i}':range(100000) for i in range(1000)})
    run_example(many_operations, big_df, 100)
    with gpus(False):
        run_example(many_operations, big_df, 100)
   