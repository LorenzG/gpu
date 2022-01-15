from typing import Tuple

import pandas as pd
import numpy as np

from decorators import on_gpu



def run_example(func, *args, **kwargs):
    now = pd.Timestamp.now()
    res = func(*args, **kwargs)
    elapsed = pd.Timestamp.now() - now
    print(f'== Test {func.__name__} - elapsed: {elapsed} ==')
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
    for loop in loops:
        res *= 3
        res -= df + df + df

    error = (res - df).abs().sum().sum()
    return error


if __name__ == '__main__':

    df1 = pd.DataFrame({
        'a': np.arange(5) * 1,
        'b': np.arange(5) * 10,
    })

    df2 = pd.DataFrame({
        'a': np.arange(5) * 100,
        'b': np.arange(5) * 1000,
    })

    run_example(dataframes_in_and_out, df1, df2)

    run_example(many_operations, 1000)