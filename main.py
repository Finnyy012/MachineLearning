from typing import Callable
import numpy as np
import graphviz
from netwerk import Netwerk
from activatie_functies import ActivatieFuncties
from sklearn.datasets import load_iris
import pandas as pd


def demo1():
    """
    demo voor P1
    """
    weights = []

    #verander deze voor andere input
    I = np.array([1,1])
    print(I.shape)

    weights.append(np.array([[ 1, 1,-2],
                             [ 1, 1,-1],
                             [-1,-1, 1]]))
    weights.append(np.array([[ 1, 0, 0,-1],
                             [ 0, 1, 1,-2]]))

    netw = Netwerk(0, 0, 0, 0, ActivatieFuncties.STEP)
    netw._weights = weights

    res = netw.evaluate(I)
    print("\nin: " + str(I))

    print("\nweights: ")
    for m in weights:
        print(m)
    print("\nout: " + str(res))

    g1 = netw.visualise_network(np.array(['x1', 'x2']), out_labels=['sum', 'carry'])
    g2 = netw.visualise_network(np.array([1, 1]), out_labels=['sum', 'carry'], evaluate=True)

    #g1 voor generic perceptron en g2 voor perceptron met input I
    g1.render(directory='graphviz_renders', view=True)


def demo2():
    """
    demo voor P2
    """
    np.random.seed(1)
    netw = Netwerk(0, 0, 0, 0, ActivatieFuncties.STEP, .8)

    netw._weights = [np.array([[-0.5, 0.5, -1.5]])]

    x_and = np.array([[1, 1],
                      [1, 0],
                      [0, 1],
                      [0, 0]])
    d_and = np.array([[1, 1], [0, 0], [0, 0], [0, 0]])

    print("\ninitial weights:")
    for w in netw._weights:
        print(w)
    print("initial MSE:")
    print(netw.loss_MSE(x_and, d_and))

    for i in range(4):
        netw.update_trivial(x_and, d_and, True)
        print("\nupdate " + str(i) + " weights: ")
        for w in netw._weights:
            print(w)
        print("update " + str(i) + " MSE: ")
        print(netw.loss_MSE(x_and, d_and))

    g1 = netw.visualise_network(np.array(['x1', 'x2']), mindiam=2.5, minlen=10)
    # g2 = netw.visualise_network(np.array([1, 0]), mindiam=2.5, minlen=10, evaluate=True)
    g1.render(directory='graphviz_renders', view=True)


def main():
    np.random.seed(1819772)
    netw = Netwerk(0, 0, 4, 1, ActivatieFuncties.STEP, .8)

    iris = load_iris()
    X = iris.data[:100]
    y = np.reshape(iris.target[:100],(-1,1))

    for i in range(10):
        netw.update_trivial(X, y, False)
        print(netw._weights)
        print(netw.loss_MSE(X, y))


if __name__ == '__main__':
    # demo1()
    # demo2()
    main()

