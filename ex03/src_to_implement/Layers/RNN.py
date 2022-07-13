import numpy as np
from Layers import Base, FullyConnected, TanH, Sigmoid
import copy


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        # input_size = dimension of the input vector
        # hidden_size = dimension of the hidden state
        super().__init__()

        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self._memorize = False

        self.fc_hidden = FullyConnected.FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fc_output = FullyConnected.FullyConnected(self.hidden_size, self.output_size)
        self.last_input = None
        self.last_hidden = None
        self.last_o_t = None
        self._weights = self.fc_hidden.weights
        self._weights_output = self.fc_output.weights
        self._gradient_weights = np.zeros(self.weights.shape)

        self._weights_optimizer = None
        self._bias_optimizer = None

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def weights(self):
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, weights_hidden_state):
        self.fc_hidden.weights = weights_hidden_state

    @property
    def weights_output(self):
        return self._weights_output

    @weights_output.setter
    def weights_output(self, weights_output):
        self._weights_output = weights_output

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    @property
    def optimizer(self):
        return self._weights_optimizer, self._bias_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._weights_optimizer = copy.deepcopy(optimizer)
        self._bias_optimizer = copy.deepcopy(optimizer)

    def forward(self, input_tensor):
        output = None

        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)

        for time in range(len(input_tensor)):
            input_hidden_concatenated = np.array([np.append(input_tensor[time], self.hidden_state)])
            input_hidden_concatenated = self.fc_hidden.forward(input_hidden_concatenated)  #u_t

            self.hidden_state = TanH.TanH().forward(input_hidden_concatenated) #h_t

            o_t = self.fc_output.forward(self.hidden_state) #W_hy * h_t + b_y
            current_output = Sigmoid.Sigmoid().forward(o_t) # y_t

            if output is not None:
                output = np.concatenate((output, current_output))
            else:
                output = current_output

            # Saving the input at the last time (to use it in the backward pass)
            if time == len(input_tensor) - 1:
                self.last_input = input_tensor[time]
                self.last_o_t = o_t

        return output


    def backward(self, error_tensor):
        for time in range(len(error_tensor)-1, -1, -1):
            grad_b_y  = Sigmoid.Sigmoid().backward(self.last_o_t)
            grad_w_hy = grad_b_y * self.last_hidden




    def initialize(self, weights_initializer, bias_initializer):
        pass


    def calculate_regularization_loss(self):
        pass
