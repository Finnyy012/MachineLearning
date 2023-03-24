import numpy as np


class Netwerk2:
    def __init__(self):
        self.layers = []

    def feed_forward(self, layer_in) -> np.array:
        if len(layer_in.shape) == 1:
            layer_in = np.array([layer_in])

        for layer in self.layers:
            layer_in = layer.feed_forward(layer_in)
        return layer_in

    def feed_backward(self, layer_in, target) -> None:
        a = self.feed_forward(layer_in)
        dC = (a - target)
        for layer in reversed(self.layers):
            dC = layer.feed_backward(dC)
            layer.update()

    def learn(self, epochs, x, t, printMSE = False):
        for i in range(epochs):
            print("epoch " + str(i))
            for layer_in, target in zip(x, t):
                self.feed_backward(layer_in, target)
            if printMSE:
                print(self.MSE(x, t))

    def MSE(self, x: np.array, target: np.array) -> np.array:
        if len(target.shape) == 1:
            target = np.array([target])
        return np.mean((target - self.feed_forward(x))**2, axis=0)