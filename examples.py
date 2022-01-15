from typing import Tuple
from itertools import accumulate

import pandas as pd
import numpy as np

from decorators import on_gpu, gpus
from utils import coerce_iterable



def run_example(func, *args, **kwargs):
    now = pd.Timestamp.now()
    print(f'== Test {func.__name__} ==')
    res = func(*args, **kwargs)
    elapsed = pd.Timestamp.now() - now
    print(f'Elapsed: {elapsed}')
    for i,r in enumerate(coerce_iterable(res)):
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
    for loop in range(loops):
        res *= 3
        res -= df + df + df

    error = (res - df).abs().sum().sum()
    return error


@on_gpu
def db_operations(df: pd.DataFrame, groups, filters) -> float:
    mask = accumulate([df[filter] == 1 for filter in filters], lambda x,y: x&y)
    df = df[mask]
    return df.groupby(groups).sum()


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

    ## 2
    big_df = pd.DataFrame({f'col_{i}':range(100000) for i in range(1000)})
    run_example(many_operations, big_df, 100)
    with gpus(False):
        run_example(many_operations, big_df, 100)

    ## 3
    big_df1 = pd.DataFrame({f'col_{i}':[1] * 10000 for i in range(1000)})
    run_example(
        db_operations,
        big_df1,
        [f'col_{i}' for i in range(50)],
        [f'col_{i}' for i in range(100, 200)],
    )
    with gpus(False):
        run_example(
            db_operations,
            big_df1,
            [f'col_{i}' for i in range(50)],
            [f'col_{i}' for i in range(100, 200)],
        )
        