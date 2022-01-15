from collections.abc import Iterable

import pandas as pd
import numpy as np
import cudf
import cupy


def coerce_iterable(obj):
    if type(obj) in [pd.DataFrame, pd.Series, cudf.DataFrame, cudf.DataFrame, np.array, cupy.array]:
        return [obj]
    if not isinstance(obj, Iterable):
        return [obj]
    return obj