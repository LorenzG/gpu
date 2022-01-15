from functools import wraps, partial
from typing import Tuple

import pandas as pd
import cudf


def pd_to_cudf(pd_item: Tuple[pd.DataFrame, pd.Series]):
    if isinstance(pd_item, pd.DataFrame):
        return cudf.DataFrame.from_pandas(pd_item)
    elif isinstance(pd_item, pd.Series):
        return cudf.Series.from_pandas(pd_item)
    else:
        raise ValueError(f'Type {type(pd_item)} not supported.')


def cudf_to_pd(pd_item: Tuple[cudf.DataFrame, cudf.Series]):
    if isinstance(pd_item, cudf.DataFrame):
        return pd_item.to_pandas()
    elif isinstance(pd_item, cudf.Series):
        return pd_item.to_pandas()
    else:
        raise ValueError(f'Type {type(pd_item)} not supported.')


def try_pd_to_cudf(maybe_pd_object):
    try:
        return pd_to_cudf(maybe_pd_object)
    except Exception:
        return maybe_pd_object


def try_cudf_to_pd(maybe_cu_object):
    try:
        return cudf_to_pd(maybe_cu_object)
    except Exception:
        return maybe_cu_object


def apply_to_nested_in_iterable(iterable):
    pass


def on_gpu(func):
    @wraps(func)
    def inner(*args, **kwargs):
        # chunksize = 3
        args_with_cudf = [try_pd_to_cudf(arg) for arg in args]
        kwargs_with_cudf = {k:try_pd_to_cudf(v) for k,v in kwargs.items()}
        res = func(*args_with_cudf, **kwargs_with_cudf)
        res = [try_cudf_to_pd(r) for r in res]
        return res
    inner.__name__ = func.__name__
    return inner
