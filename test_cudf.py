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
    for n_rows in np.logspace(1, 5, 5):
        for n_cols in np.logspace(1, 4, 4):
            n_rows = int(n_rows)
            n_cols = int(n_cols)
            print('create data')
            data = np.random.rand(n_rows, n_cols) + 1.
            print('data created')
            with timeit(n_rows, n_cols, ctor.__name__) as ctor_time:
                d = ctor(data)
            res[(n_rows, n_cols, ctor.__name__)] = ctor_time.elapsed
            with timeit(n_rows, n_cols, op.__name__) as op_time:
                op(d, *op_args, **op_kargs)
            res[(n_rows, n_cols, op.__name__)] = op_time.elapsed
    return res

if __name__ == '__main__':

    ## 1
    print('pandas')
    pandas_grid = grid_search(pd.DataFrame, np.log)
    print('cudf')
    cudf_grid = grid_search(cudf.DataFrame, np.log)

    pandas_grid = pd.DataFrame(pandas_grid, index=['pandas']).T.applymap(lambda x: x.total_seconds() * 1000)
    pandas_grid.to_csv('pandas_grid.csv')
    cudf_grid = pd.DataFrame(cudf_grid, index=['cudf']).T.applymap(lambda x: x.total_seconds() * 1000)
    cudf_grid.to_csv('cudf_grid.csv')
    print(pandas_grid)
    print(cudf_grid)
    
    """
    In [22]: pd.concat([pandas_grid, cudf_grid], axis=1)
    Out[22]:
                             elapsed      cudf
    10     10    DataFrame     0.682  1410.141
                 log           0.390     1.201
           100   DataFrame     0.217     9.578
                 log           0.237     6.990
           1000  DataFrame     0.165    83.482
                 log           0.236    64.889
           10000 DataFrame     0.187   914.895
                 log           1.410   744.885
    100    10    DataFrame     0.165    46.204
                 log           0.170     0.984
           100   DataFrame     0.172     9.372
                 log           0.245     6.738
           1000  DataFrame     0.186    82.621
                 log           0.911    64.065
           10000 DataFrame     0.200   914.768
                 log           9.383   740.841
    1000   10    DataFrame     0.616    47.193
                 log           0.274     1.010
           100   DataFrame     0.189     9.456
                 log           0.939     7.120
           1000  DataFrame     0.199    87.079
                 log           9.088    67.663
           10000 DataFrame     0.745   973.454
                 log          65.722   682.010
    10000  10    DataFrame     0.485    49.407
                 log           1.329     1.019
           100   DataFrame     0.207    11.233
                 log           7.306     7.806
           1000  DataFrame     0.665   160.766
                 log          64.265    73.863
           10000 DataFrame     0.529  1427.856
                 log         628.365   753.564
    100000 10    DataFrame     2.164    78.463
                 log           7.652     2.048
           100   DataFrame     0.741    30.932
                 log          64.894    17.308
           1000  DataFrame     0.522   305.481
                 log         620.593   182.501
           10000 DataFrame     2.160  6885.347
                 log        6293.400  2239.256
    
    """
