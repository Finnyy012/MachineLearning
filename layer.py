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
                 f_act: activatie_functies.ActivatieFunctie = activatie_functies.RELU,
                 f_init: initialisers.Initialiser = initialisers.Glorot,
                 l_rate: float = 0.01,
                 flatten: bool = False):
        super().__init__(f_act, l_rate)
        self._kernels = [np.array([f_init.generate(kernel_size, kernel_size) for _ in range(in_shape[-1])])
                         for _ in range(kernels)]
        self.in_shape = in_shape
        self.kernel_size = kernel_size
        self.flatten = flatten

    @staticmethod
    def conv3d(m_in, kernel, mode='valid'):
        z = np.zeros(shape=(m_in.shape[0]-(kernel.shape[0]-1), m_in.shape[1]-(kernel.shape[1]-1), m_in.shape[-1]))
        for i in range(m_in.shape[-1]):
            z[:, :, i] += signal.convolve2d(m_in[:, :, i], kernel[i], mode=mode)
        return z.sum(-1)

    def feed_forward(self, m_in: np.array):
        self._a = m_in
        self._z = np.zeros(shape=(self.in_shape[0]-(self.kernel_size-1),
                                  self.in_shape[1]-(self.kernel_size-1),
                                  len(self._kernels)))

        for i, kernel in enumerate(self._kernels):
            z = self.conv3d(m_in, kernel, mode='valid')
            self._z[:,:,i] += z

        res = self.f_act(self._z)
        if self.flatten:
            res = np.array([res.flatten])
        return res

    def feed_backward(self, dW: np.array):
        if self.flatten:
            dW = dW[0].reshape(shape=(self.in_shape[0]-(self.kernel_size-1),
                                      self.in_shape[1]-(self.kernel_size-1),
                                      len(self._kernels)))

        dLC2 = np.zeros(self._z.shape, dtype=np.float64)


        self._error = self.f_drv(self._z) * dW
        self.conv3d(self._error, np.rot90(np.rot90()), mode='full')



