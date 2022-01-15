import os
from functools import wraps, partial
from contextlib import contextmanager
from typing import Any, Tuple

import pandas as pd
import cudf


__USE_GPUS = True

def set_gpus(use_gpus=True):
    if use_gpus:
        print('GPUs ON')
    else:
        print('GPUs OFF')
    global __USE_GPUS
    __USE_GPUS = use_gpus


@contextmanager
def gpus(activate_gpus=True):
    previous_state = __USE_GPUS
    set_gpus(activate_gpus)
    try:
        yield None
    finally:
        set_gpus(previous_state)


def pd_to_cudf(pd_item: Tuple[pd.DataFrame, pd.Series]):
    out = None
    if isinstance(pd_item, pd.DataFrame):
        out = cudf.DataFrame.from_pandas(pd_item)
    elif isinstance(pd_item, pd.Series):
        out = cudf.Series.from_pandas(pd_item)
    else:
        raise ValueError(f'Type {type(pd_item)} not supported.')
    print('Transformed pandas to cudf.')
    return out


def cudf_to_pd(pd_item: Tuple[cudf.DataFrame, cudf.Series]):
    out = None
    if isinstance(pd_item, cudf.DataFrame):
        out = pd_item.to_pandas()
    elif isinstance(pd_item, cudf.Series):
        out = pd_item.to_pandas()
    else:
        raise ValueError(f'Type {type(pd_item)} not supported.')
    print('Transformed cudf to pandas.')
    return out


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
    raise NotImplemented()


def process_output(output: Any):
    if isinstance(output, list):
        return [try_cudf_to_pd(r) for r in output]
    elif isinstance(output, tuple):
        return tuple(try_cudf_to_pd(r) for r in output)
    elif isinstance(output, dict):
        return {k:try_cudf_to_pd(r) for k,r in output.items()}
    else:
        return try_cudf_to_pd(output)


def on_gpu(func, persist_cudf=False):
    @wraps(func)
    def inner(*args, **kwargs):
        if __USE_GPUS:
            args = [try_pd_to_cudf(arg) for arg in args]
            kwargs = {k:try_pd_to_cudf(v) for k,v in kwargs.items()}
        res = func(*args, **kwargs)
        if not persist_cudf:
            res = process_output(res)
        return res
    inner.__name__ = func.__name__
    return inner
