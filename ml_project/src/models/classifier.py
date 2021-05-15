import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle
import os
import pandas as pd

from typing import NoReturn, Union

from src.entities import RfParams, LrParams

matrix = Union[np.ndarray, pd.DataFrame]
vector = Union[np.ndarray, pd.Series]
ModelParams = Union[RfParams, LrParams]

class Classifier:
    def __init__(self, params: ModelParams) -> NoReturn:
        if params.model_type == "Logistic Regression":
            self.model = LogisticRegression(
                C=params.C, penalty=params.penalty, solver=params.solver, random_state=params.random_state,
                n_jobs=params.n_jobs, max_iter=params.max_iter
            )
        if params.model_type == "Random Forest":
            self.model = RandomForestClassifier(
                n_estimators=params.n_estimators, max_depth=params.max_depth, n_jobs=params.n_jobs,
                random_state=params.random_state
            )

    def fit(self, x: matrix, y: vector) -> NoReturn:
        self.model.fit(x, y)

    def predict(self, x: matrix) -> vector:
        return self.model.predict(x)

    def dump(self, path: str) -> NoReturn:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)


def get_score(y_true: vector, y_score: vector) -> float:
    return roc_auc_score(y_true, y_score)
