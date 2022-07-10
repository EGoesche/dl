import numpy as np
from Layers import Base


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        # input_size = dimension of the input vector
        # hidden_size = dimension of the hidden state
        super().__init__()
        self.trainable = True

    def forward(self, input_tensor):
        pass

    def backward(self, error_tensor):
        pass

