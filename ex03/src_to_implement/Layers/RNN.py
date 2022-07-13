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
        self.input_tensor = None
        self.hidden_values = []
        self.u_t_values = []
        self.o_t_values = []
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
        self.input_tensor = input_tensor  #Is there a better way than storing whole input?

        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)

        for time in range(len(input_tensor)):
            input_hidden_concatenated = np.array([np.append(input_tensor[time], self.hidden_state)])  #Is order correct? Pdf says the reverse.
            input_hidden_concatenated = self.fc_hidden.forward(input_hidden_concatenated)  #u_t
            self.u_t_values += input_hidden_concatenated

            self.hidden_state = TanH.TanH().forward(input_hidden_concatenated) #h_t
            self.hidden_values += self.hidden_state

            o_t = self.fc_output.forward(self.hidden_state) #W_hy * h_t + b_y
            self.o_t_values += o_t
            current_output = Sigmoid.Sigmoid().forward(o_t) # y_t

            if output is not None:
                output = np.concatenate((output, current_output))
            else:
                output = current_output

            # Saving the input at the last time (to use it in the backward pass)
            if time == len(input_tensor) - 1:
                self.last_o_t = o_t

        self.hidden_values = np.array(self.hidden_values)
        self.o_t_values = np.array(self.o_t_values)
        self.u_t_values = np.array(self.u_t_values)
        return output


    def backward(self, error_tensor):
        for time in range(len(error_tensor)-1, -1, -1):
            grad_b_y  = Sigmoid.Sigmoid().backward(self.o_t_values[time])  #grad_o_t in other words
            grad_w_hy = grad_b_y * self.hidden_values[time]
            input_hidden_concatenated = np.array([np.append(self.input_tensor[time + 1], self.hidden_state)]) # again, order???

            if time == len(error_tensor)-1:
                grad_h_t = self.fc_output.weights.transpose() * grad_b_y
            elif time == 0:
                grad_h_t = self.fc_hidden.weights.transpose() * TanH.TanH().backward(input_hidden_concatenated) * grad_h_t
            else:
                grad_h_t = self.fc_hidden.weights.transpose() * TanH.TanH().backward(
                    input_hidden_concatenated) * grad_h_t + self.fc_output.weights.transpose() * grad_b_y

            grad_bh = grad_h_t * TanH.TanH().backward(self.u_t_values[time])
            grad_hh = grad_bh * self.hidden_values[time-1].transpose()
            grad_xh = grad_bh * self.input_tensor[time]



    def initialize(self, weights_initializer, bias_initializer):
        pass


    def calculate_regularization_loss(self):
        pass
