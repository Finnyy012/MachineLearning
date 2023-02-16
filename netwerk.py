from activatie_functies import ActivatieFuncties
import numpy as np
import graphviz


class Netwerk:
    def __init__(self, n_layers: int, layer_size: int, in_size: int, out_size: int, f_act: ActivatieFuncties) -> None:
        self.f_act = f_act
        self._weights = []
        self._thresholds = []

        self._weights.append(np.random.normal(size=(layer_size, in_size)))
        self._weights.extend([np.random.normal(size=(layer_size, layer_size)) for _ in range(n_layers - 1)])
        self._weights.append(np.random.normal(size=[out_size, layer_size]))

        self._thresholds.extend([np.random.normal(size=layer_size) for _ in range(n_layers)])
        self._thresholds.append(np.random.normal(size=out_size))

    def evaluate(self, layer_in: np.array) -> np.array:
        """
        evalueert het netwerk aan de hand van input.
        de functie multipliceert de input met de weight, haalt hier de treshold vanaf en knalt dit door de
        activatiefunctie om een output te krijgen.

        een half-adder met input [ 1  0] en de step-activatiefunctie bijvoorbeeld werkt als volgt:
        (In = input layer n, Wn = weight layer n, Tn = threshold layer n)

             layer 0 → 1                                   |   layer 1 → 2
           ------------------------------------------------|-----------------------------------------------------
            I0 → [ 1  0]              T0                   |  I1 →  [ 0  1  1]            T1
                    •                  ↓                   |            •                  ↓
            W0 → [[ 1  1]     [ 1    [ 2         [-1       |  W1 → [[ 1  0  0]    [ 0    [ 1         [-1    [ 0
                  [ 1  1]  →    1  -   1  → STEP   0  = I1 |        [ 0  1  1]] →   2] -   2] → STEP   0] =   1]
                  [-1 -1]]     -1]    -1]          0]      |

        :param layer_in: (np.array) array met input (dus layer 0)
        :return: (np.array) array met output
        """
        for m, t in zip(self._weights, self._thresholds):
            layer_in = self.f_act(np.matmul(m, layer_in) - t)
        return layer_in

    def visualise_network(self,
                          layer_in: np.array,
                          # weights: [np.array],
                          # thresholds: [np.array],
                          out_labels: [str] = None,
                          evaluate: bool = False,
                          mindiam: float = .8,
                          minlen: float = 2) -> graphviz.Digraph:
        """
        visualiseert het netwerk

        :param layer_in: (np.array) array met input; als evaluate = False kunnen hier de labels van de input-variabelen in
        :param out_labels: ([str]) list met labels voor de output; None voor geen label - default = None
        :param evaluate: (bool) evalueert de perceptron aan de hand van layer_in - default = False
        :param minlen: (float) minimum pijl lengte (zet op minstens 10 voor gegenereerde netwerken)
        :param mindiam: (float) minimum node diameter
        :return: (graphviz.Digraph) graph object van perceptron
        """

        res = graphviz.Digraph('perceptron', graph_attr={'splines': 'line', 'rankdir': 'LR', 'layout': 'dot'})
        res.format = 'bmp'

        node_id = 0
        buffer = len(layer_in)

        if evaluate:
            result = self.evaluate(layer_in)
        else:
            result = [" " for _ in range(len(self._thresholds[len(self._thresholds) - 1]))]

        for i, n in enumerate(layer_in):
            res.node(str(node_id), str(n), shape='circle', fontname='Consolas', width=str(mindiam))
            for j in range(len(self._weights[0])):
                res.edge(str(node_id), str(buffer + j), taillabel=" " + str(self._weights[0][j][i]), minlen=str(minlen))
            node_id += 1

        for i, v in enumerate(self._thresholds):
            buffer += len(v)
            for j, t in enumerate(v):
                res.node(str(node_id), str(t), shape='circle', fontname='Consolas', width=str(mindiam))
                if i < len(self._thresholds) - 1:
                    for k in range(len(self._weights[i + 1])):
                        res.edge(str(node_id), str(buffer + k), taillabel=" " + str(self._weights[i + 1][k][j]),
                                 minlen=str(minlen))
                else:
                    res.node(str(-j - 1), str(result[j]), shape='none')
                    if out_labels is not None:
                        label = out_labels.pop()
                    else:
                        label = 'output'
                    res.edge(str(node_id), str(-j - 1), label=label)
                node_id += 1

        return res

