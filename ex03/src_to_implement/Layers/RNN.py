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

        self._hidden_optimizer = None
        self._output_optimizer = None
        self.sigmoid = Sigmoid.Sigmoid()
        self.tanh = TanH.TanH()

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
        return self._hidden_optimizer, self._output_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._hidden_optimizer = copy.deepcopy(optimizer)
        self._output_optimizer = copy.deepcopy(optimizer)

    def forward(self, input_tensor):
        output = None
        self.input_tensor = input_tensor  #Is there a better way than storing whole input?

        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)

        for time in range(len(input_tensor)):
            input_hidden_concatenated = np.array([np.append(input_tensor[time], self.hidden_state)])
            input_hidden_concatenated = self.fc_hidden.forward(input_hidden_concatenated)  #u_t
            self.u_t_values += list(input_hidden_concatenated)

            self.hidden_state = self.tanh.forward(input_hidden_concatenated) #h_t
            self.hidden_values += list(self.hidden_state)

            o_t = self.fc_output.forward(self.hidden_state) #W_hy * h_t + b_y
            self.o_t_values += list(o_t)
            current_output = self.sigmoid.forward(o_t) # y_t

            if output is not None:
                output = np.concatenate((output, current_output))
            else:
                output = current_output

        self.hidden_values = np.array(self.hidden_values)
        self.o_t_values = np.array(self.o_t_values)
        self.u_t_values = np.array(self.u_t_values)
        return output

    def backward(self, error_tensor):
        grad_by_sum = 0; grad_w_hy_sum = 0; grad_ht_sum = 0; grad_hh_sum = 0; grad_xh_sum = 0

        for time in range(len(error_tensor)-1, -1, -1):
            grad_ot  = self.fc_output.backward(self.sigmoid.backward(error_tensor[time]))
            grad_by = grad_ot
            grad_w_hy = grad_ot * self.hidden_values[time].transpose()

            if time == len(error_tensor)-1:
                grad_ht = 0
            else:
                input_hidden_concatenated = np.array([np.append(self.input_tensor[time + 1], self.hidden_values[time])])
                grad_ht = self.fc_hidden.weights.transpose() * self.tanh.backward(
                    input_hidden_concatenated) * grad_ht + self.fc_output.weights.transpose() * grad_ot
            grad_middle = grad_ht + grad_ot # after the conjunction of two paths


            grad_ut = self.fc_hidden.backward(self.tanh.backward(grad_middle))
            grad_hh = grad_ut * self.hidden_values[time-1].transpose()
            grad_xh = grad_ut * self.input_tensor[time].transpose()

            grad_by_sum += grad_by; grad_w_hy_sum += grad_w_hy; grad_ht_sum += grad_ht
            grad_hh_sum += grad_hh; grad_xh_sum += grad_xh

        return error_tensor * grad_xh_sum #Think about this. I chose this gradient because it's the only one includes input

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def calculate_regularization_loss(self):
        return self._hidden_optimizer.regulizer.norm(self.weights) + \
               self._output_optimizer.regularizer.norm(self.weights_output)

