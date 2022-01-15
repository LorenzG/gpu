import os
from functools import wraps, partial
from contextlib import contextmanager, ExitStack
from typing import Any, Tuple
from mock import patch
import inspect

import pandas as pd
import numpy as np
import cudf
import cupy


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


def pd_to_cudf(cpu_obj: Tuple[pd.DataFrame, pd.Series]):
    out = None
    if isinstance(cpu_obj, pd.DataFrame):
        out = cudf.DataFrame.from_pandas(cpu_obj)
    elif isinstance(cpu_obj, pd.Series):
        out = cudf.Series.from_pandas(cpu_obj)
    elif isinstance(cpu_obj, np.ndarray):
        out = cupy.asarray(cpu_obj)
    else:
        raise ValueError(f'Type {type(cpu_obj)} not supported.')
    print('Transformed pandas to cudf.')
    return out


def cudf_to_pd(gpu_obj: Tuple[cudf.DataFrame, cudf.Series]):
    out = None
    if isinstance(gpu_obj, cudf.DataFrame):
        out = gpu_obj.to_pandas()
    elif isinstance(gpu_obj, cudf.Series):
        out = gpu_obj.to_pandas()
    elif isinstance(gpu_obj, cudf.ndarray):
        out = cupy.asnumpy(gpu_obj)
    else:
        raise ValueError(f'Type {type(gpu_obj)} not supported.')
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


def _create_patches(func):
    func_module = inspect.getmodule(func)
    module_name = func_module.__name__
    module_source = inspect.getsource(func_module)
    if 'import numpy as np' in module_source:
        yield patch(f'{module_name}.np', cupy)
    if 'import numpy\n' in module_source:
        yield patch(f'{module_name}.numpy', cupy)
    if 'import pandas as pd' in module_source:
        yield patch(f'{module_name}.pd', cudf)
    if 'import pandas\n' in module_source:
        yield patch(f'{module_name}.pandas', cudf)


def on_gpu(func, persist_cudf=False):
    @wraps(func)
    def inner(*args, **kwargs):
        if __USE_GPUS:
            args = [try_pd_to_cudf(arg) for arg in args]
            kwargs = {k:try_pd_to_cudf(v) for k,v in kwargs.items()}
        with ExitStack() as stack:
            for mgr in _create_patches(func):
                stack.enter_context(mgr)
            res = func(*args, **kwargs)
        if not persist_cudf:
            res = process_output(res)
        return res
    inner.__name__ = func.__name__
    return inner
