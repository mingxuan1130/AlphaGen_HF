from typing import Type
from data.snapshot_expression import *


MAX_EXPR_LENGTH = 15
MAX_EPISODE_LENGTH = 256

OPERATORS: List[Type[Operator]] = list(Operators)

DELTA_TIMES = [1, 5, 10, 20, 40]
LOOKBACK_TIMES = [100, 500, 1000, 5000]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

REWARD_PER_STEP = 0.
