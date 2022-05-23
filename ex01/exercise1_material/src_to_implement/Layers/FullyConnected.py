import random

import Base
import numpy as np


class FullyConnected(Base):
    def __init__(self, input_size, output_size):
        super().trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.input_tensor = None
        self._optimizer = None
        self.weights = [random.uniform(0, 1) for _ in range(input_size)]
        self._gradient_weights = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor    # create a copy for backward pass
        return np.matmul(input_tensor, self.weights)

    def backward(self, error_tensor):
        # Calculate gradient
        self._gradient_weights = np.matmul(self.input_tensor.T, error_tensor)

        # Update weights
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        # TODO: return error tensor for prev layer

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights


