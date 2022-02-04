import os
from functools import wraps, partial
from contextlib import contextmanager, ExitStack
from typing import Any, Tuple
import inspect

import pandas as pd
import numpy as np
import cudf
import cupy

np.random.seed(0)



class Timer:
    def __init__(self, now):
        self.now = now


@contextmanager
def timeit(*args):
    t = Timer(pd.Timestamp.now())
    try:
        yield t
    finally:
        t.elapsed = pd.Timestamp.now() - t.now
        print(f'{args}: {t.elapsed}')


def grid_search(ctor, op, *op_args, **op_kargs):
    res = {}
    for n_rows in np.logspace(1, 7, 5):
        for n_cols in np.logspace(1, 5, 5):
            n_rows = int(n_rows)
            n_cols = int(n_cols)
            data = np.random.rand(n_rows, n_cols) + 1.
            with timeit(n_rows, n_cols, 'ctor') as ctor_time:
                d = ctor(data)
            res[(n_rows, n_cols, 'ctor')] = ctor_time.elapsed
            with timeit(n_rows, n_cols, 'op') as op_time:
                op(d, *op_args, **op_kargs)
            res[(n_rows, n_cols, 'op')] = op_time.elapsed
    return res

if __name__ == '__main__':

    ## 1
    pandas_grid = grid_search(pd.DataFrame, np.log)
    cudf_grid = grid_search(cudf.DataFrame, np.log)

    pd.DataFrame(pandas_grid).to_csv('pandas_grid.csv')
    pd.DataFrame(cudf_grid).to_csv('cudf_grid.csv')
    print(pandas_grid)
    print(cudf_grid)
    