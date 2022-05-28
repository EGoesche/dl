import numpy as np
from exercise1_material.src_to_implement.Layers import Base


class SoftMax(Base.BaseLayer):
    def __init__(self):
        """
        Constructor of the SoftMax layer.
        """
        super().__init__()
        self.max_items = None
        self.out_y = None

    def forward(self, input_tensor):
        """
        The forward pass the SoftMax activation function is used to transform the logits (the output of the network)
        into a probability distribution.
        :param input_tensor: input on which the SoftMax function will get applied
        :return: estimated class probabilities for each row representing an element of the batch
        """
        self.find_max(input_tensor)
        input_tensor_new = input_tensor - self.max_items
        out_prob = np.zeros_like(input_tensor)

        for count, arr in enumerate(input_tensor_new):
            expo = np.exp(arr)
            denom = np.sum([expo])
            out_prob[count] = expo/denom

        self.out_y = out_prob
        return out_prob

    def backward(self, error_tensor):
        """
        The backward pass the SoftMax activation function.
        :param error_tensor: error tensor for current layer
        :return: error tensor for the previous layer
        """
        res = np.zeros_like(error_tensor)
        for count, arr in enumerate(error_tensor):
            sumo = np.sum(arr * self.out_y[count])
            res[count] = self.out_y[count] * (arr - sumo)
        return res

    def find_max(self, input_tensor):
        self.max_items = input_tensor.max(axis=1).reshape(input_tensor.shape[0], 1)




