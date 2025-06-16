#trainer config dataclass
from dataclasses import dataclass

@dataclass
class TrainerConfig:
    N: int = 6
    alpha: float = 1
    iter: int = 1000
    tol: float = 1e-10
    epsilon: float = 0.01
    in_dim: int = 2
    out_dim: int = 1
    epochs: int = 100
    problem_type: str = 'D1'
    lr: float = 0.001
    show_plot: bool = False
    loss_compare: bool = False
    verbose: bool = False
