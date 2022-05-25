import copy


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []  # contains the loss value for each iteration after calling train
        self.layers = []  # holds the architecture
        self.data_layer = None  # provide input data and labels
        self.loss_layer = None  # referring to the special layer providing loss and prediction
        self.label_tensor = None  # holds label_tensor for the backward pass

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            # Note: input_tensor is input for the next layer but also output of current layer
            input_tensor = layer.forward(input_tensor)
        output = self.loss_layer.forward(input_tensor, self.label_tensor)

        return output

    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)
        for layer in self.layers[::-1]:  # reverse layer list
            error = layer.backward(error)   # use error to update weights in backward method of each layer

    def append_layer(self, layer):
        if layer.trainable:
            layer._optimizer = copy.deepcopy(self.optimizer)    # create independent copy of optimizer object
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            output = self.forward()
            self.loss.append(output)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        output = input_tensor   # just to give it a proper name (it's no longer an input)

        return output
