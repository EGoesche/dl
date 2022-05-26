import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        sum_all = np.sum(-1 * np.log(prediction_tensor + np.finfo(float).eps))
        return np.dot(sum_all, label_tensor)

    def backward(self, label_tensor):
        return -1 * label_tensor / [self.prediction_tensor + np.finfo(float).eps]
