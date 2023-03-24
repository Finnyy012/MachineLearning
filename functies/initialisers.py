import math
from abc import ABC
import numpy as np


class Initialiser(ABC):
    @staticmethod
    def generate(in_size: np.array, out_size: np.array) -> np.array: pass


class Normal(Initialiser):
    @staticmethod
    def generate(in_size: np.array, out_size: np.array) -> np.array:
        return np.random.normal(loc=0.0, scale=1, size=(in_size, out_size))


class HE(Initialiser):
    @staticmethod
    def generate(in_size: np.array, out_size: np.array) -> np.array:
        return np.random.normal(loc=0.0, scale=math.sqrt(2 / in_size), size=(in_size, out_size))


class Glorot(Initialiser):
    @staticmethod
    def generate(in_size: np.array, out_size: np.array) -> np.array:
        limit = math.sqrt(6 / (in_size + out_size))
        return np.random.uniform(low=-limit, high=limit, size=(in_size, out_size))


class GlorotNormal(Initialiser):
    @staticmethod
    def generate(in_size: np.array, out_size: np.array) -> np.array:
        return np.random.normal(loc=0.0, scale=math.sqrt(2 / (in_size + out_size)), size=(in_size, out_size))