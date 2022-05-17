import Base


class FullyConnected(Base):
    def __init__(self, input_size, output_size):
        super().trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = None
        self.weights = None
        self._gradient_weights = None

    def forward(input_tensor):
        # TODO: Implement the forward method
        pass

    def backward(error_tensor):
        # TODO: Implement the backward method
        pass

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


