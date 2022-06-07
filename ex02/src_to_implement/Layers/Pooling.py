import numpy as np


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        """
        Constructor for the Pooling layer.
        :param stride_shape: controls amount of downsampling
        :param pooling_shape:
        """
        self.input_tensor = None
        self.batch_size = None
        self.input_channels = None
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]
        self.input_channels = input_tensor.shape[1]

        output_height = int((self.input_tensor.shape[2] - self.pooling_shape[0]) / self.stride_shape[0]) + 1
        output_width = int((self.input_tensor.shape[3] - self.pooling_shape[1]) / self.stride_shape[1]) + 1

        output_tensor = np.zeros(shape=(self.batch_size, self.input_channels, output_height, output_width))

        # loop over every single value in every input channel in every batch
        for batch in range(self.batch_size):
            for input_channel in range(self.input_channels):
                for y in range(output_height):
                    # define the range of the rows in which pooling will be applied
                    first_row = y * self.stride_shape[0]
                    last_row = first_row + self.pooling_shape[0]

                    for x in range(output_width):
                        # define the range of the columns in which pooling will be applied
                        first_col = x * self.stride_shape[1]
                        last_col = first_col + self.pooling_shape[1]

                        # write the max value of the current pooling area into the output_tensor
                        output_tensor[batch, input_channel, y, x] = np.max(input_tensor[batch, input_channel,
                                                                           first_row:last_row, first_col:last_col])
        return output_tensor

    def backward(self, error_tensor):
        # TODO implement backward method of pooling
        pass
