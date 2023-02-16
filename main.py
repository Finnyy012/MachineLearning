import numpy as np
import graphviz


def eval_perceptron(layer_in: np.array, weights: [np.array], thresholds: [np.array]) -> np.array:
    """
    evalueert een perceptron bestaande een lijst met matrices met weights en een lijst met vectoren met threshold.
    de functie multipliceert herhaaldelijk de input met de weights en vergelijkt dit met de tresholds van de
    respectievelijke layer.

    een half-adder met input [ 1  0] bijvoorbeeld werkt als volgt:
    (In = input layer n, Wn = weights layer n, Tn = threshold layer n)

         layer 0 → 1                        |   layer 1 → 2
       -------------------------------------|-----------------------------------------
        I0 → [ 1  0]              T0        |  I1 →  [ 0  1  1]            T1
                •                  ↓        |            •                  ↓
        W0 → [[ 1  1]     [ 1    [ 2        |  W1 → [[ 1  0  0]    [ 0    [ 1    [ 0
              [ 1  1]  →    1  ≥   1  = I1  |        [ 0  1  1]] →   2] ≥   2] =   1]
              [-1 -1]]     -1]    -1]       |

    :param layer_in: (np.array) array met input (dus layer 0)
    :param weights: ([np.array]) list met weight matrices
    :param thresholds:([np.array]) list met threshold vectors
    :return: (np.array) array met output
    """
    for i, m in enumerate(weights):
        layer_in = (np.matmul(m, layer_in) >= thresholds[i]).astype(int)
    return layer_in


def visualise_perceptron(layer_in: np.array,
                         weights: [np.array],
                         thresholds: [np.array],
                         out_labels: [str] = None,
                         evaluate: bool = False) -> graphviz.Digraph:
    """
    visualiseert de perceptron

    :param layer_in: (np.array) array met input; als evaluate = False kunnen hier de labels van de input-variabelen in
    :param weights: ([np.array]) list met weight matrices
    :param thresholds:([np.array]) list met threshold vectors
    :param out_labels: ([str]) list met labels voor de output; None voor geen label - default = None
    :param evaluate: (bool) evalueert de perceptron aan de hand van layer_in - default = False
    :return: (graphviz.Digraph) graph object van perceptron
    """

    res = graphviz.Digraph('perceptron', graph_attr={'splines': 'line', 'rankdir': 'LR', 'layout' : 'dot'})
    res.format = 'bmp'

    node_id = 0
    buffer = len(layer_in)

    if evaluate:
        result = eval_perceptron(layer_in, weights, thresholds)
    else:
        result = [" " for _ in range(len(thresholds[len(thresholds)-1]))]

    for i, n in enumerate(layer_in):
        res.node(str(node_id), str(n), shape = 'circle', fontname = 'Consolas', width='.8')
        for j in range(len(weights[0])):
            res.edge(str(node_id), str(buffer + j), taillabel=" "+str(weights[0][j][i]), minlen='2')
        node_id += 1

    for i, v in enumerate(thresholds):
        buffer += len(v)
        for j, t in enumerate(v):
            res.node(str(node_id), str(t), shape='circle', fontname='Consolas', width='.8')
            if i < len(thresholds)-1:
                for k in range(len(weights[i+1])):
                    res.edge(str(node_id), str(buffer + k), taillabel=" "+str(weights[i+1][k][j]), minlen='2')
            else:
                res.node(str(-j-1), str(result[j]), shape='none')
                if out_labels is not None:
                    label = out_labels.pop()
                else:
                    label = 'output'
                res.edge(str(node_id), str(-j-1), label=label)
            node_id += 1

    return res


def main():
    weights = []
    thresholds = []

    #verander deze voor andere input
    I = np.array([1,1])

    weights.append(np.array(  [[ 1, 1],
                               [ 1, 1],
                               [-1,-1]]))
    weights.append(np.array([[1,0,0],
                             [0,1,1]]))

    thresholds.append(np.array([2,1,-1]))
    thresholds.append(np.array([1,2]))

    res = eval_perceptron(I, weights, thresholds)
    print("in: " + str(I))
    print("\nweights: ")
    for m in weights:
        print(m)

    print("\nthresholds: ")
    for v in thresholds:
        print(v)

    print("\nout: " + str(res))

    g1 = visualise_perceptron(np.array(['x1', 'x2']), weights, thresholds, out_labels=['sum', 'carry'])
    g2 = visualise_perceptron(I, weights, thresholds, out_labels=['carry', 'sum'], evaluate=True)

    #g1 voor generic perceptron en g2 voor perceptron met input I
    g1.render(directory='graphviz_renders', view=True)



if __name__ == '__main__':
    main()

