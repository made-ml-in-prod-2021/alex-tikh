from dataclasses import dataclass, field
from typing import Optional


@dataclass()
class LrParams:
    model_type: str = field(default='Logistic Regression')
    solver: str = field(default='lbfgs')
    penalty: str = field(default='l2')
    C: float = field(default=1.0)
    max_iter: int = field(default=100)
    n_jobs: int = field(default=-1)
    random_state: int = field(default=42)

@dataclass()
class RfParams:
    model_type: str = field(default='Random Forest')
    max_depth: int = field(default=None)
    n_estimators: int = field(default=100)
    n_jobs: int = field(default=-1)
    random_state: int = field(default=42)
