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
    biases = []

    #verander deze voor andere input
    I = np.array([1,0])

    weights.append(np.array([[ 1, 1],
                             [ 1, 1],
                             [-1,-1]]))
    weights.append(np.array([[1,0,0],
                             [0,1,1]]))

    biases.append(np.array([-2,-1, 1]))
    biases.append(np.array([-1,-2]))

    netw = Netwerk(0, 0, 0, 0, ActivatieFuncties.STEP)
    netw._weights = weights
    netw._biases = biases

    res = netw.evaluate(I)
    print("in: " + str(I))

    print("\nweights: ")
    for m in weights:
        print(m)

    print("\nbiases: ")
    for v in biases:
        print(v)

    print("\nout: " + str(res))

    g1 = netw.visualise_network(np.array(['x1', 'x2']), out_labels=['sum', 'carry'])
    g2 = netw.visualise_network(I, out_labels=['sum', 'carry'], evaluate=True)

    #g1 voor generic perceptron en g2 voor perceptron met input I
    g2.render(directory='graphviz_renders', view=True)


def demo2():
    """
    demo voor P2
    """
    np.random.seed(1)
    netw = Netwerk(0, 0, 0, 0, ActivatieFuncties.STEP, .8)

    netw._weights = [np.array([[-0.5, 0.5]])]
    netw._biases = [np.array([1.5])]

    x_and = np.array([[1, 1],
                      [1, 0],
                      [0, 1],
                      [0, 0]])
    d_and = np.array([1, 0, 0, 0])

    print("\ninitial")
    print("weights:")
    for w in netw._weights:
        print(w)
    print("biases:")
    for t in netw._biases:
        print(t)

    for i in range(11):
        netw.update_trivial(x_and, d_and)
        print("\nupdate " + str(i) + ": ")
        print("weights:")
        for w in netw._weights:
            print(w)
        print("biases:")
        for t in netw._biases:
            print(t)

    g1 = netw.visualise_network(np.array(['x1', 'x2']), mindiam=2.5, minlen=10)
    g2 = netw.visualise_network(np.array([1, 0]), mindiam=2.5, minlen=10, evaluate=True)
    g2.render(directory='graphviz_renders', view=True)

    X_iris, y_iris = load_iris()


def demo3():
    pass


def main():
    data_iris = (load_iris())
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    print(data_iris)


if __name__ == '__main__':
    demo1()
    # demo2()
    # main()

