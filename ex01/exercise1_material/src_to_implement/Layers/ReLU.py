from exercise1_material.src_to_implement.Layers import Base
import numpy as np


class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        forward_tensor = np.array([rectified_vector(x) for x in input_tensor])
        return forward_tensor

    def backward(self, error_tensor):
        error_tensor[self.input_tensor <= 0] = 0
        return error_tensor

def rectified_vector(x):
    result = np.array([rectified(arr) for arr in x])
    return result

def rectified(x):
    return max(0, x)
