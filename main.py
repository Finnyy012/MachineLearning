from typing import Callable
import numpy as np
import graphviz
from netwerk import Netwerk
from activatie_functies import ActivatieFuncties


def demo1():
    """
    demo voor P1
    """
    weights = []
    thresholds = []

    #verander deze voor andere input
    I = np.array([1,1])

    weights.append(np.array([[ 1, 1],
                             [ 1, 1],
                             [-1,-1]]))
    weights.append(np.array([[1,0,0],
                             [0,1,1]]))

    thresholds.append(np.array([2,1,-1]))
    thresholds.append(np.array([1,2]))

    netw = Netwerk(0, 0, 0, 0, ActivatieFuncties.STEP)
    netw._weights = weights
    netw._thresholds = thresholds

    res = netw.evaluate(I)
    print("in: " + str(I))

    print("\nweights: ")
    for m in weights:
        print(m)

    print("\nthresholds: ")
    for v in thresholds:
        print(v)

    print("\nout: " + str(res))

    g1 = netw.visualise_network(np.array(['x1', 'x2']), out_labels=['sum', 'carry'])
    g2 = netw.visualise_network(I, out_labels=['carry', 'sum'], evaluate=True)

    #g1 voor generic perceptron en g2 voor perceptron met input I
    g1.render(directory='graphviz_renders', view=True)


def main():
    netw = Netwerk(3, 3, 2, 2, ActivatieFuncties.SIGMOID)

    print("\nweight:")
    for w in netw._weights:
        print(w)

    print("\nthreshold:")
    for t in netw._thresholds:
        print(t)

    g1 = netw.visualise_network(np.array(['x1', 'x2']), mindiam=2.5, minlen=10)
    g2 = netw.visualise_network(np.array([1,0]), mindiam=2.5, minlen=10, evaluate=True)
    g2.render(directory='graphviz_renders', view=True)


if __name__ == '__main__':
    demo1()
    # main()

