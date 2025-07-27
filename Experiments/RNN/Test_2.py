from RNN.base import Algorithm
from RNN.activation_functions import sigmoid_function
import numpy as np
def prepare_data():
  X = np.array([
      [150, 70],
      [254, 73],
      [312, 68],
      [120, 60],
      [154, 61],
      [212, 65],
      [216, 67],
      [145, 67],
      [184, 64],
      [130, 69]
  ])
  y = np.array([0,1,1,0,0,1,1,0,1,0])
  m = 10
  A0 = X.T
  Y = y.reshape(1, m)

  return A0, Y

X, Y = prepare_data()
RNN = Algorithm()

RNN.add_layer(2, sigmoid_function)
RNN.add_layer(3, sigmoid_function)
RNN.add_layer(3, sigmoid_function)
RNN.add_layer(1, sigmoid_function)

y_hat = RNN.execute(X)

print(y_hat)