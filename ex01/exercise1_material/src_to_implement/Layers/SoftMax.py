import Base
import numpy as np


class SoftMax(Base):
    def __init__(self):
        self.max_item = None
        self.out_y = None
        pass

    def forward(self, input_tensor):
        expo = np.exp(input_tensor - self.max_item)
        denom = np.sum([expo])
        self.out_y = expo/denom
        return self.out_y.copy()

    def backward(self, error_tensor):
        error_mul_y = np.sum(error_tensor * self.out_y)
        return self.out_y * (error_tensor - error_mul_y)

    def find_max(self, input_tensor):
        self.max_item = max(input_tensor)




