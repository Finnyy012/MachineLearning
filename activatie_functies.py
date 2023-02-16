import math
from enum import Enum
import numpy as np


def step(x: np.array) -> np.array:
    return (x >= 0).astype(int)


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + math.e ** (-x))


class ActivatieFuncties(Enum):
    SIGMOID = staticmethod(sigmoid)
    STEP = staticmethod(step)
