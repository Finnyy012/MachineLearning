import itertools
from typing import Callable
import numpy as np
import graphviz

import activatie_functies
from netwerk import Netwerk
from activatie_functies import ActivatieFunctie
from sklearn.datasets import load_iris
import pandas as pd


def demo1():
    """
    demo voor P1
    """
    weights = []

    #verander deze voor andere input
    I = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])

    # I = np.array([1,1])
    print(I.shape)

    weights.append(np.array([[ 1, 1,-1],
                             [ 1, 1,-1],
                             [-2,-1, 1]]))
    weights.append(np.array([[ 1, 0],
                             [ 0, 1],
                             [ 0, 1],
                             [-1,-2]]))
    netw = Netwerk(0, 0, 0, 0, activatie_functies.STEP())
    netw._weights = weights

    res = netw.evaluate(I)
    print("\nin: " + str(I))

    print("\nweights: ")
    for m in weights:
        print(m)
    print("\nout: " + str(res))

    g1 = netw.visualise_network(np.array(['x1', 'x2']), out_labels=['sum', 'carry'], mindiam=2, minlen=15)
    g2 = netw.visualise_network(np.array([1, 1]), out_labels=['sum', 'carry'], evaluate=True)

    #g1 voor generic perceptron en g2 voor perceptron met input I
    g1.render(directory='graphviz_renders', view=True)


def demo2():
    """
    demo voor P2
    """
    np.random.seed(1)
    netw = Netwerk(0, 0, 2, 1, activatie_functies.STEP(), .8)

    # netw._weights = [np.array([[-0.5, 0.5, -1.5],
    #                            [-0.5, 0.5, -1.5]])]

    # netw._weights = [np.array([[-0.5],
    #                            [ 0.5],
    #                            [-1.5]])]

    x_and = np.array([[1, 1],
                      [1, 0],
                      [0, 1],
                      [0, 0]])
    d_and = np.array([[1, 1],
                      [0, 0],
                      [0, 0],
                      [0, 0]])

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
    table_in_4 = np.array(list(itertools.product([0, 1], repeat=2)))

    netw = Netwerk(0, 0, 0, 0, activatie_functies.STEP())

    netw._weights = [np.array([[1],
                               [1],
                               [-2]])]
    print("truth table AND")
    for x in table_in_4:
        print(str(x) + " -> " + str(netw.evaluate(x)))


if __name__ == '__main__':
    # demo1()
    demo2()
    # main()

