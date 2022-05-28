import numpy as np
from exercise1_material.src_to_implement.Layers import Base


class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.max_items = None
        self.out_y = None

    def forward(self, input_tensor):
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
        res = np.zeros_like(error_tensor)
        for count, arr in enumerate(error_tensor):
            sumo = np.sum(arr * self.out_y[count])
            res[count] = self.out_y[count] * (arr - sumo)
        return res

    def find_max(self, input_tensor):
        self.max_items = input_tensor.max(axis=1).reshape(input_tensor.shape[0], 1)




