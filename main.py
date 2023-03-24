from datetime import datetime
import numpy as np
import layer
from functies import activatie_functies, initialisers
from deprecated.netwerk import Netwerk
from sklearn.datasets import load_iris, load_digits
from scipy import signal
from netwerk2 import Netwerk2


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

    g1 = netw.visualise_network(out_labels=['sum', 'carry'])
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
    d_and = np.array([[1],
                      [0],
                      [0],
                      [0]])

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

def bp_and():

    np.random.seed(100)
    netw = Netwerk(0, 0, 2, 1, activatie_functies.SIGMOID(), 1)

    # g1 = netw.visualise_network(np.array(['x1', 'x2']), mindiam=2.5, minlen=10)
    # g1.render(directory='graphviz_renders', view=True)
    #
    # for layer in netw._weights:
    #     print(layer)

    netw._weights = [np.array([[-0.5],
                               [ 0.5],
                               [ 1.5]])]

    I = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    # I = np.array([[0, 0]])

    target = np.array([[0],
                       [0],
                       [0],
                       [1]])

    # print(activatie_functies.SIGMOID.function(np.array([[ 1.5]])))
    # print(1 / (1 + pow(math.e, -1.5)))


    print("\nweights_curr: ")
    print(netw._weights[0])

    outc = netw.evaluate(I)
    print("\nout_curr: ")
    print(outc)

    for i in range(1000):
        netw.update_backprop(I, target)

    g1 = netw.visualise_network(np.array(['x1', 'x2']), mindiam=2.5, minlen=10)
    g1.render(directory='graphviz_renders', view=True)

    print("\nweights_new: ")
    print(netw._weights[0])

    print("\nout_new: ")
    print(netw.evaluate(I))
    netw.lo


def bp_halfadder():
    np.random.seed(1)
    netw = Netwerk(1, 3, 2, 2, activatie_functies.SIGMOID(), 1)
    I = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    t = np.array([[0, 0],
                  [0, 1],
                  [0, 1],
                  [1, 0]])

    start = datetime.now()

    for _ in range(10_000):
        netw.update_backprop(I, t)

    print(datetime.now() - start)
    print()
    print(netw.evaluate(I))

    g1 = netw.visualise_network(np.array(['x1', 'x2']), mindiam=2.5, minlen=10)
    g1.render(directory='graphviz_renders', view=True)


def main():
    netw = Netwerk(1, 2, 2, 1, activatie_functies.SIGMOID(), 1)
    netw._weights = [np.array([[ 0.2, 0.7],
                               [-0.4, 0.1],
                               [ 0.0, 0.0]]),
                     np.array([[0.6],
                               [0.9],
                               [0.0]])]

    g1 = netw.visualise_network(np.array(['x1', 'x2']), mindiam=2.5, minlen=10)
    g1.render(directory='graphviz_renders', view=True)

    # I = np.array([[1, 1]])

    I = np.array([[1, 1],
                  [0, 1]])

    target = np.array([[0],
                       [1]])

    for _ in range(1):
        netw.update_backprop(I, target)
        print(netw._weights)

    # print(netw._weights)


def bp_digit():
    np.random.seed(1)
    netw = Netwerk(2, 50, 64, 10, activatie_functies.LRELU, l_rate=0.0001, adaptive=0.99999)

    digits = load_digits()
    data = np.array(digits['data'])

    def tobin(x):
        res = np.zeros(10)
        res[x] = 1
        return res
    target = np.array(list(map(tobin, digits['target'])))

    data_train = data[:-360]
    data_test = data[-360:]

    target_train = target[:-360]
    target_test = target[-360:]


    # print(len(data_train))
    # print(len(data_test))
    # print((data_train))
    # print((data_test))
    print(data)
    print(digits['target'])
    # print(netw.evaluate(data))

    acc = (digits['target'][:-360] == list(map(np.argmax, netw.evaluate(data_train)))).astype(int).mean()
    print("\naccuracy initial: " + str(acc))

    for i in range(1000):
        netw.update_backprop(data_train, target_train)
        acc = (digits['target'][:-360] == list(map(np.argmax, netw.evaluate(data_train)))).astype(int).mean()
        print("\naccuracy epoch " + str(i) + ": " + str(acc))
        acc = (digits['target'][-360:] == list(map(np.argmax, netw.evaluate(data_test)))).astype(int).mean()
        print("\naccuracy test set: " + str(acc))

    # acc = (digits['target'][-360:] == list(map(np.argmax, netw.evaluate(data_test)))).astype(int).mean()
    # print("\naccuracy test set: " + str(acc))

    # g1 = netw.visualise_network(mindiam=2.5, minlen=10)
    # print('done!')
    # g1.render(directory='graphviz_renders', view=True)


def bp_mnist():
    np.random.seed(1)
    data = np.genfromtxt('./data/mnist_10k.csv', delimiter=',')
    # print(data)
    X = data[:, 1:]
    target = data[:, :1].flatten()
    # print(target)
    def tobin(x):
        res = np.zeros(10)
        # print(x)
        res[int(x)] = 1
        return res
    target_bin = np.array(list(map(tobin, target)))
    # print(X)
    cutoff = int(len(X)*0.8)

    X_train = X[:-cutoff]
    X_test = X[-cutoff:]

    y_train = target_bin[:-cutoff]
    y_test = target_bin[-cutoff:]

    # print(len(X_train[0]))

    netw = Netwerk(1, 50, 784, 10, activatie_functies.LRELU(), l_rate=0.0001, adaptive=0.99999)
    # 1, 10, 784, 10, activatie_functies.SIGMOID(), l_rate=0.01, adaptive=0.9999 : 0.552
    # 1, 10, 784, 10, activatie_functies.SIGMOID(), l_rate=0.01, adaptive=0.999  : 0.53325
    # 2, 10, 784, 10, activatie_functies.SIGMOID(), l_rate=0.01, adaptive=0.999  : 0.50325
    # 1, 50, 784, 10, activatie_functies.SIGMOID(), l_rate=0.01, adaptive=0.9999 : 0.634125
    # 1, 200, 784, 10, activatie_functies.SIGMOID(), l_rate=0.01, adaptive=0.9999


    for i in range(1000):
        netw.update_backprop(X_train, y_train)
        acc = (target[:-cutoff] == list(map(np.argmax, netw.evaluate(X_train)))).astype(int).mean()
        print("\naccuracy epoch " + str(i) + ": " + str(acc))

    acc = (target[-cutoff:] == list(map(np.argmax, netw.evaluate(X_test)))).astype(int).mean()
    print("\naccuracy test set: " + str(acc))


def bp_iris():
    np.random.seed(6)
    netw = Netwerk(1, 4, 4, 3, activatie_functies.SIGMOID(), .1)

    iris = load_iris()
    X = iris.data

    def tobin(x):
        res = np.zeros(3)
        res[x] = 1
        return res
    target = np.array(list(map(tobin, iris.target)))  # /2 zodat alle opties tussen de 0 en 1 zitten

    # print(X[0])
    # print(netw.evaluate(X[0]))
    #
    # print(target)

    # g1 = netw.visualise_network(mindiam=2.5, minlen=10)
    # g1.render(directory='graphviz_renders', view=True)

    for i in range(1):
        netw.update_backprop(X, target)
        print(netw.loss_MSE(X, target))

    print("out: ")
    print(netw.evaluate(X).round(3))
    print("\nMSE: ")
    print(netw.loss_MSE(X, target))


def test():
    a = np.array([[  0,  1,  1, -1],
                 [ -1,  0,  1, -1],
                 [  1,  0,  2,  0],
                 [ -1,  1,0.1,  0],
                 [  0, -1,0.6,0.810655]])

    k = np.array([[[1,2,1],[2,1,3],[3,1,7]],
                  [[1,1,2],[5,1,3],[6,5,9]],
                  [[1,2,1],[2,3,3],[7,1,1]]])

    print(k.sum(-1))

    l = np.array([k.flatten()])
    print(l)
    print(l.reshape(3,3,3))

    k[:, :, 1]+= np.array([[1,2,3],
                           [1,1,1],
                           [1,4,2]])
    print(k[:,:,1])
    # print(signal.convolve2d(a, k, 'same'))


def test2():
    np.random.seed(1)
    netw = Netwerk2()
    netw.layers.append(layer.DenseLayer(2, 3, activatie_functies.SIGMOID, initialisers.Glorot, 1))
    netw.layers.append(layer.DenseLayer(3, 2, activatie_functies.SIGMOID, initialisers.Glorot, 1))

    I = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    t = np.array([[0, 0],
                  [0, 1],
                  [0, 1],
                  [1, 0]])

    # for _ in range(1000):
    #     for x, target in zip(I,t):
    #         netw.feed_backward(x, target)

    netw.learn(1000, I, t, True)

    print(netw.feed_forward(I).round(3))




if __name__ == '__main__':

    # demo1()
    # demo2()
    # bp_and()
    # bp_halfadder()
    # bp_digit()
    # bp_mnist()
    # main()
    # bp_iris()
    test2()

