from exercise1_material.src_to_implement.Layers import Base
import numpy as np


class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        forward_tensor = np.array([rectified_vector(x) for x in input_tensor])
        return forward_tensor

    def backward(self, error_tensor):
        backward_error = np.array([rectified_vector(x) for x in error_tensor])
        return backward_error

def rectified_vector(x):
    result = np.array([rectified(arr) for arr in x])
    return result

def rectified(x):
    return max(0, x)
