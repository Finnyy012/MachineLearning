import activatie_functies
from activatie_functies import ActivatieFuncties
import numpy as np
import graphviz


class Netwerk:
    def __init__(self,
                 n_layers: int,
                 layer_size: int,
                 in_size: int,
                 out_size: int,
                 f_act: ActivatieFuncties,
                 l_rate: float = .1) -> None:
        """
        initialiseert het netwerk met normaal-verdeelde, willekeurig-gegenereerde floats voor weights en biases.

        :param n_layers: (int) aantal hidden layers
        :param layer_size: (int) hidden layer lengte
        :param in_size: (int) input layer lengte
        :param out_size: (int) output layer lengte
        :param f_act: (ActivatieFuncties) activatiefunctie
        """
        self.f_act = f_act
        self.l_rate = l_rate
        self._weights = []

        if n_layers == 0:
            self._weights.append(np.random.normal(size=(out_size, in_size+1)))
        else:
            self._weights.append(np.random.normal(size=(layer_size, in_size+1)))
            self._weights.extend([np.random.normal(size=(layer_size, layer_size+1)) for _ in range(n_layers - 1)])
            self._weights.append(np.random.normal(size=(out_size, layer_size+1)))

    def evaluate(self, layer_in: np.array) -> np.array:
        """
        evalueert het netwerk aan de hand van input.
        de functie multipliceert voor elke layer de input met de weights, telt hier de biases bij op en knalt dit
        door de activatiefunctie om een output te krijgen, wat dus weer de input wordt voor de volgende layer.

        een half-adder met input [ 1  0] en de step-activatiefunctie bijvoorbeeld werkt als volgt:

             layer 0 → 1                                 |   layer 1 → 2
           ----------------------------------------------|-----------------------------------------------------
            I0|1  →  [ 1  0  1]                          |  I1|1  →  [ 0  1  1  1]
                         •                               |                 •
            W0|B0 → [[ 1  1 -2]         [-1    [ 0       |  W1|B1 → [[ 1  0  0 -1]         [-1    [ 0
                     [ 1  1 -1]  → STEP   0  =   1  = I1 |           [ 0  1  1 -2]] → STEP   0] =   1]
                     [-1 -1  1]]          0]     1]      |

         - In = input layer n
         - Wn = weight layer n
         - Bn = bias layer n
         - | is een concatenatie dus bijv. W0|B0 is matrix W0 met vector B0 eraan vastgeplakt

        :param layer_in: (np.array) array met input (dus layer 0)
        :return: (np.array) array met output
        """
        for m in self._weights:
            layer_in = self.f_act(np.matmul(m, np.append(layer_in, 1)))
        return layer_in

    def update_trivial(self, x: np.array, target: np.array, do_print=False) -> None:
        """
        functie die een netwerk met maar één laag updated

        de functie loopt door de features heen en updated de weights door er
         η • (target[i] – evaluate(x[i])) ⊗ (x[i]|1)
        bij op te tellen

        :param x: matrix met input values
        :param target: array met target values
        :param do_print: print weights etc. tussen elke stap wanneer True
        """
        for i, row in enumerate(x):
            d = self.l_rate*(target[i] - self.evaluate(row))
            if do_print:
                print("\nΔ         = " + str(d))
                print("in        = "   + str(row))
                print("Δ * in    = \n" + str(np.outer(d, np.append(row, 1))))
                print("updated W = \n" + str(self._weights))

            self._weights = self._weights + np.outer(d, np.append(row, 1))

    def loss_MSE(self, x: np.array, target: np.array):
        # TODO
        pass


    def f_act_to_char(self) -> chr:
        """
        tostring voor activatiefunctie
        :return: (chr) karakter representatie van activatiefunctie
        """
        if self.f_act == ActivatieFuncties.SIGMOID: return 'σ'
        if self.f_act == ActivatieFuncties.STEP: return 'H'

    def visualise_network(self,
                          layer_in: np.array,
                          out_labels: [str] = None,
                          evaluate: bool = False,
                          mindiam: float = .8,
                          minlen: float = 2,
                          titel: str = "",
                          filename: str = "netwerk") -> graphviz.Digraph:
        """
        visualiseert het netwerk

        :param layer_in: (np.array) array met input; als evaluate = False kunnen hier de labels van de input-variabelen in
        :param out_labels: ([str]) list met labels voor de output; None voor geen label - default = None
        :param evaluate: (bool) evalueert de perceptron aan de hand van layer_in - default = False
        :param mindiam: (float) minimum node diameter - default = .8
        :param minlen: (float) minimum pijl lengte (zet op minstens 10 voor gegenereerde netwerken) - default = 2
        :param titel: (str) afbeeldingstitel - default = ""
        :param filename: (str) bestandsnaam - default = "netwerk"

        :return: (graphviz.Digraph) graph object van perceptron
        """

        res = graphviz.Digraph(filename,
                               graph_attr={'splines'  : 'line',
                                           'rankdir'  : 'LR',
                                           'layout'   : 'dot',
                                           'ordering' : 'in',
                                           'label'    : titel})
        res.format = 'bmp'
        node_kwargs = {'shape'   : 'circle',
                       'fontname': 'Consolas',
                       'width'   : str(mindiam)}

        if evaluate:
            result = self.evaluate(layer_in)
        else:
            result = [" " for _ in range(self._weights[len(self._weights) - 1].shape[0])]

        layer_in = np.append(layer_in, 1)
        buffer = layer_in.shape[0]
        node_id = 0

        for i, n in enumerate(layer_in):
            res.node(str(node_id), str(n), **node_kwargs)
            for j in range(self._weights[0].shape[0]):
                res.edge(str(node_id),
                         str(buffer + j),
                         taillabel=" " + str(self._weights[0][j][i]),
                         minlen=str(minlen))
            node_id += 1

        for i, m in enumerate(self._weights):
            buffer += (m.shape[0] + 1)
            for j, node in enumerate(m):
                res.node(str(node_id), self.f_act_to_char(), **node_kwargs)
                if i < len(self._weights) - 1:
                    for k in range(self._weights[i + 1].shape[0]):
                        res.edge(str(node_id),
                                 str(buffer + k),
                                 taillabel=" " + str(self._weights[i + 1][k][j]),
                                 minlen=str(minlen))
                else:
                    res.node(str(-j - 1), str(result[j]), shape='none')
                    if out_labels is not None:
                        label = out_labels.pop()
                    else:
                        label = 'output'
                    res.edge(str(node_id), str(-j - 1), label=label)
                node_id += 1
            if i < len(self._weights) - 1:
                res.node(str(node_id), '1', **node_kwargs)
                for k in range(len(self._weights[i + 1])):
                    res.edge(str(node_id),
                             str(buffer + k),
                             taillabel=" " + str(self._weights[i + 1][k][buffer-4]),
                             minlen=str(minlen))
                node_id += 1

        return res

