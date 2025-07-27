from typing import Callable

class Algorithm:
    def __init__(self):
        self.layers_dimensions = []
        self.weights = []
        self.bias = []
        self.activation_functions = []

    def add_layer(nodes: int, func: Callable[[float], float]):
        layers = len(self.layers_dimensions)
        self.layers_dimensions.append(nodes)
        if(layers != 1):
            self.weights.append(np.random.randn(len(layers_dimensions),len(layers_dimensions) - 1))
            self.bias.append(np.random.randn(len(layers_dimensions),1))
        self.activation_functions.append(func)

    def execute(A):
        for W,B,G in self.weights, self.bias, self.activation_functions:
            Z = W @ A + b
            A = G(Z)
        return A