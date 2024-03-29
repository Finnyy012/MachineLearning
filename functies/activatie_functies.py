import math
from abc import ABC
import numpy as np


class ActivatieFunctie(ABC):
    @staticmethod
    def function(x: np.array) -> np.array: pass

    @staticmethod
    def derivative(x: np.array) -> np.array: pass

    @staticmethod
    def toChar() -> chr: pass


class SIGMOID(ActivatieFunctie, ABC):
    @staticmethod
    def function(x: np.array) -> np.array:
        return 1 / (1 + math.e ** (-x))

    @staticmethod
    def derivative(x: np.array) -> np.array:
        return SIGMOID.function(x) * (-SIGMOID.function(x) + 1)

    @staticmethod
    def toChar() -> chr:
        return 'σ'


class STEP(ActivatieFunctie):
    @staticmethod
    def function(x: np.array) -> np.array:
        return (x >= 0).astype(int)

    @staticmethod
    def derivative(x: np.array) -> np.array:
        return 0

    @staticmethod
    def toChar() -> chr:
        return 'H'


class RELU(ActivatieFunctie):
    @staticmethod
    def function(x: np.array) -> np.array:
        return x * (x > 0)

    @staticmethod
    def derivative(x: np.array) -> np.array:
        return 1 * (x > 0)

    @staticmethod
    def toChar() -> chr:
        return 'R'


class LRELU(ActivatieFunctie):
    @staticmethod
    def function(x: np.array) -> np.array:
        return np.where(x > 0, x, x * 0.01)

    @staticmethod
    def derivative(x: np.array) -> np.array:
        return np.where(x > 0, 1, 0.01)

    @staticmethod
    def toChar() -> chr:
        return 'LR'


#
#
#     @staticmethod
#     def sigmoid(x: np.array) -> np.array:
#         return 1 / (1 + math.e ** (-x))
#
#     @staticmethod
#     def sigmoid_prime(x: np.array) -> np.array:
#         return ActivatieFuncties.sigmoid(x) * (-ActivatieFuncties.sigmoid(x) + 1)
#
#
# ActivatieFuncties.SIGMOID = {'function'   : ActivatieFuncties.sigmoid,
#            'derivative' : ActivatieFuncties.sigmoid_prime,
#            'string'     : 'H'}
#
# ActivatieFuncties.STEP =    {'function'   : ActivatieFuncties.step,
#            'derivative' : lambda x: 0,
#            'string'     : 'σ'}
#
