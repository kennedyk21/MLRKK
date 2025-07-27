from typing import Callable
import numpy as np
class Algorithm:
    def __init__(self):
        self.layers_dimensions = []
        self.weights = []
        self.bias = []
        self.activation_functions = []

    def add_layer(self, nodes: int, func: Callable[[float], float]):
        self.layers_dimensions.append(nodes)
        if len(self.layers_dimensions) > 1:
            in_dim = self.layers_dimensions[-2]
            out_dim = self.layers_dimensions[-1]
            self.weights.append(np.random.randn(out_dim, in_dim))
            self.bias.append(np.random.randn(out_dim, 1))
        self.activation_functions.append(func)

    def execute(self,A):
        print(self.weights)
        for W,B,G in zip(self.weights, self.bias, self.activation_functions):
            Z = W @ A + B
            A = G(Z)
        return A