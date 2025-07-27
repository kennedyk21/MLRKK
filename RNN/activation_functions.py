import numpy as np
def sigmoid_function(x: float) -> float:
    return 1 / (1 + np.exp(-x))