import numpy as np


class Flatten:
    def __init__(self):
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor[0].shape  # Store this so that we'll know the output shape when doing backward()
        batches = len(input_tensor)
        flatten_outputs = []

        # TODO: if there is enough time, we should try to kick out the loop
        for batch in range(0, batches):
            flatten_outputs.append(np.reshape(input_tensor[batch], -1))
        flatten_outputs = np.array(flatten_outputs)
        return flatten_outputs

    def backward(self, error_tensor):
        batches = len(error_tensor)
        return error_tensor.reshape(batches, self.shape[0], self.shape[1])
