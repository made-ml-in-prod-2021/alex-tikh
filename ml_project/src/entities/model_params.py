from dataclasses import dataclass, field
from typing import Optional


@dataclass()
class ModelParams:
    model_type: str
    max_depth: Optional[int]
    solver: str = field(default='lbfgs')
    penalty: str = field(default='l2')
    C: float = field(default=1.0)
    max_iter: int = field(default=100)
    n_estimators: int = field(default=100)
    n_jobs: int = field(default=-1)
    random_state: int = field(default=42)
