import math
from abc import abstractmethod, ABC
from functies import initialisers, activatie_functies
from scipy import signal
import numpy as np


class Layer(ABC):
    def __init__(self, f_act: activatie_functies.ActivatieFunctie, l_rate: float):
        self.f_act = f_act.function
        self.f_drv = f_act.derivative
        self.l_rate = l_rate
        self._z = None      # input of act function most recent ff pass
        self._a = None      # activation of layer l+1 most recent ff pass
        self._error = None  # error in weights current fb pass

    @abstractmethod
    def feed_forward(self, layer_in: np.array): pass

    @abstractmethod
    def feed_backward(self, dW: np.array): pass

    @abstractmethod
    def update(self): pass


class DenseLayer(Layer):
    def __init__(self,
                 in_size: int,
                 out_size: int,
                 f_act: activatie_functies.ActivatieFunctie,
                 f_init: initialisers.Initialiser = initialisers.Glorot,
                 l_rate: float = 0.01):
        super().__init__(f_act, l_rate)

        self._weights = f_init.generate(in_size, out_size)
        self._weights = np.vstack([self._weights, np.zeros(self._weights.shape[1])])

    def feed_forward(self, layer_in: np.array):
        self._a = layer_in
        self._z = np.matmul(np.c_[layer_in, np.ones(layer_in.shape[0])], self._weights)
        return self.f_act(self._z)

    def feed_backward(self, dW: np.array):
        self._error = self.f_drv(self._z) * dW
        return np.matmul(self._error, self._weights.T[:,:-1])

    def update(self):
        WD = self.l_rate * np.matmul(np.c_[self._a, np.ones(self._a.shape[0])].T, self._error)
        self._weights -= WD


class ConvLayer(Layer):
    def __init__(self,
                 in_shape: (int, int, int),
                 kernels: int,
                 kernel_size: int,
                 mode: str = "valid",
                 f_act: activatie_functies.ActivatieFunctie = activatie_functies.RELU,
                 f_init: initialisers.Initialiser = initialisers.Glorot,
                 l_rate: float = 0.01,
                 flatten: bool = False):
        super().__init__(f_act, l_rate)
        self._kernels = [[f_init.generate(kernel_size, kernel_size) for _ in range(in_shape[-1])] for _ in range(kernels)]
        self.in_shape = in_shape
        self.kernel_size = kernel_size
        self.mode = mode

    def feed_forward(self, m_in: np.array):
        self._a = m_in
        self._z = []
        maps = []

        for kernel in self._kernels:
            z = np.zeros(shape=(self.in_shape[0]-(self.kernel_size-1), self.in_shape[1]-(self.kernel_size-1), self.in_shape[-1]))
            for i in range(self.in_shape[-1]):
                z[:,:,i] += signal.convolve2d(m_in[:,:,i], kernel[i], mode=self.mode)
            self._z.append(z)
            maps.append(self.f_act(z.sum(-1)))
        return maps

    def feed_backward(self, dW: np.array):
        self._error = self.f_drv(self._z) * dW

