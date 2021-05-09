import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Optional, NoReturn, Union

matrix = Union[np.ndarray, pd.DataFrame]
vector = Union[np.ndarray, pd.Series]


class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self) -> NoReturn:
        super().__init__()
        self.mean = 0
        self.std = 1

    def fit(self, x: matrix, y: Optional[vector] = None) -> "CustomStandardScaler":
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        return self

    def transform(self, x: matrix, y: Optional[vector] = None) -> np.ndarray:
        x = (x - self.mean) / self.std
        return x

    def fit_transform(self, x: matrix, y: Optional[vector] = None) -> np.ndarray:
        self.fit(x, y)
        return self.transform(x, y)
