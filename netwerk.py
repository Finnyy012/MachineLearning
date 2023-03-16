from activatie_functies import ActivatieFunctie
import numpy as np
import graphviz


class Netwerk:
    def __init__(self,
                 n_layers: int,
                 layer_size: int,
                 in_size: int,
                 out_size: int,
                 f_act: ActivatieFunctie,
                 l_rate: float = .1) -> None:
        """
        initialiseert het netwerk met normaal-verdeelde, willekeurig-gegenereerde floats voor weights en biases.

        :param n_layers: (int) aantal hidden layers
        :param layer_size: (int) hidden layer lengte
        :param in_size: (int) input layer lengte
        :param out_size: (int) output layer lengte
        :param f_act: (ActivatieFuncties) activatiefunctie
        """
        self.f_act = f_act.function
        self.f_drv = f_act.derivative
        self.f_str = f_act.toChar()
        self.l_rate = l_rate
        self._weights = []

        if n_layers == 0:
            self._weights.append(np.random.normal(size=(in_size+1, out_size)))
        else:
            self._weights.append(np.random.normal(size=(in_size+1, layer_size)))
            self._weights.extend([np.random.normal(size=(layer_size+1, layer_size)) for _ in range(n_layers - 1)])
            self._weights.append(np.random.normal(size=(layer_size+1, out_size)))

    def evaluate(self, layer_in: np.array) -> np.array:
        """
        evalueert het netwerk aan de hand van input.
        de functie multipliceert voor elke layer de input met de weights, telt hier de biases bij op en knalt dit
        door de activatiefunctie om een output te krijgen, wat dus weer de input wordt voor de volgende layer.

        de functie werkt ook met matrices als input, voor als je meerdere inputs tegelijk wilt evalueren

        een half-adder met input [ 1  0] en de step-activatiefunctie bijvoorbeeld werkt als volgt:
        (de matrices hieronder zijn getransponeerd t.o.v. de daadwerkelijke werking om de matrix-vector multiplicatie
        duidelijker te weergeven)

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
        if len(layer_in.shape) == 1:
            layer_in = np.array([layer_in])

        for m in self._weights:
            layer_in = self.f_act(np.matmul(np.c_[layer_in, np.ones(layer_in.shape[0])], m))
        return layer_in

    def loss_MSE(self, x: np.array, target: np.array):
        if len(target.shape) == 1:
            target = np.array([target])
        return np.mean((target - self.evaluate(x))**2, axis=0)

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

            self._weights[0] = self._weights[0] + np.outer(np.append(row, 1), d)
            if do_print:
                print("Δ ⊗ in    = \n" + str(np.outer(d, np.append(row, 1))))
                print("updated W = \n" + str(self._weights))
                print("MSE       = " + str(self.loss_MSE(x, target)) + "\n")

    def update_backprop(self, x: np.array, target: np.array) -> None:
        for x_row, target_row in zip(x, target):
            z = []
            x_row = np.array([x_row])
            a = [x_row]

            for i, m in enumerate(self._weights):
                z_curr = np.matmul(np.c_[x_row, np.ones(x_row.shape[0])], m)
                z.append(z_curr)
                x_row = self.f_act(z_curr)
                a.append(x_row)
            zl = z.pop()
            al = a.pop()
            D = self.f_drv(zl) * (al - target_row)

            WD = self.l_rate * np.matmul(np.c_[a[len(a)-1], np.ones(zl.shape[0])].T, D)
            for i in reversed(range(len(z))):
                zl = z.pop()
                a.pop()
                D = self.f_drv(zl) * np.matmul(D, self._weights[i+1].T[:,:-1])
                self._weights[i+1] -= WD
                WD = self.l_rate * np.matmul(np.c_[a[len(a)-1], np.ones(zl.shape[0])].T, D)
            self._weights[0] -= WD

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
            result = [[" " for _ in range(self._weights[len(self._weights) - 1].T.shape[0])]]

        layer_in = np.append(layer_in, 1)
        buffer = layer_in.shape[0]
        node_id = 0

        for i, n in enumerate(layer_in):
            res.node(str(node_id), str(n), **node_kwargs)
            for j in range(self._weights[0].T.shape[0]):
                res.edge(str(node_id),
                         str(buffer + j),
                         taillabel=" " + str(self._weights[0].T[j][i]),
                         minlen=str(minlen))
            node_id += 1

        for i, m in enumerate(self._weights):
            buffer += (m.T.shape[0] + 1)
            for j, node in enumerate(m.T):
                res.node(str(node_id), self.f_str, **node_kwargs)
                if i < len(self._weights) - 1:
                    for k in range(self._weights[i + 1].T.shape[0]):
                        res.edge(str(node_id),
                                 str(buffer + k),
                                 taillabel=" " + str(self._weights[i + 1].T[k][j]),
                                 minlen=str(minlen))
                else:
                    res.node(str(-j - 1), str(result[0][j]), shape='none')
                    if out_labels is not None:
                        label = out_labels.pop()
                    else:
                        label = 'output'
                    res.edge(str(node_id), str(-j - 1), label=label)
                node_id += 1
            if i < len(self._weights) - 1:
                res.node(str(node_id), '1', **node_kwargs)
                for k in range(len(self._weights[i + 1].T)):
                    res.edge(str(node_id),
                             str(buffer + k),
                             taillabel=" " + str(self._weights[i + 1].T[k][buffer-4]),
                             minlen=str(minlen))
                node_id += 1

        return res

