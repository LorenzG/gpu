from typing import Tuple

import pandas as pd
import numpy as np

from decorators import on_gpu


@on_gpu
def dataframes_in_and_out(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[float, pd.Series]:
    res = df1 + df2
    last_column = res.iloc[:, -1]
    sum_last_column = last_column.sum()
    return sum_last_column, last_column


if __name__ == '__main__':

    df1 = pd.DataFrame({
        'a': np.arange(5) * 1,
        'b': np.arange(5) * 10,
    })

    df2 = pd.DataFrame({
        'a': np.arange(5) * 100,
        'b': np.arange(5) * 1000,
    })

    res = dataframes_in_and_out(df1, df2)
    print(res)