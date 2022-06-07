import numpy as np

from src_to_implement.Layers import Base


class Conv(Base.BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self._optimizer = None  # TODO: Maybe we need two optimizer (weights + bias)
        self.stride_shape = stride_shape  # allows for different strides in the spatial dimensions
        self.convolution_shape = convolution_shape  # determines whether this object provides a 1D or a 2D conv layer
        self.num_kernels = num_kernels

        # Initialize the parameters of this layer uniformly random in the range [0; 1)
        self.weights = np.random.uniform(0, 1, size=(self.num_kernels, self.convolution_shape))
        self.bias = np.random.uniform(0, 1, size=self.num_kernels)

        self._gradient_weights = np.zeros(self.weights.shape)
        self._gradient_bias = np.zeros(self.bias.shape)
        self.input_tensor = None

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, weights):
        self._gradient_weights = weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, bias):
        self._gradient_bias = bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, input_tensor):
        self.input_tensor = input_tensor  # create a copy for backward pass
        # TODO: Implement forward pass

    def backward(self, error_tensor):
        # TODO: Implement backward pass
        pass

    def initialize(self, weights_initializer, bias_initializer):
        # Get input and output dimensions
        if len(self.convolution_shape) == 2:
            # 1D signals
            fan_in = self.weights.shape[1] * self.weights.shape[2]
            fan_out = self.num_kernels * self.weights.shape[2]
        else:
            # 2D signals
            fan_in = self.weights.shape[1] * self.weights.shape[2] * self.weights.shape[3]
            fan_out = self.num_kernels * self.weights.shape[2] * self.weights.shape[3]

        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)
