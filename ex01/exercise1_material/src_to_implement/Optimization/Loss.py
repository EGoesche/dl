import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        loss = 0

        for count, arr in enumerate(prediction_tensor):
            yk_hat = np.dot(arr, label_tensor[count]) #this gives us the yk hat we shoudld use in the log
            loss += (-1 * np.log(yk_hat + np.finfo(float).eps))

        return loss

    def backward(self, label_tensor):
        return -1 * (label_tensor / np.array(self.prediction_tensor + np.finfo(float).eps))
