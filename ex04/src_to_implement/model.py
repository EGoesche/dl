import torch

class ResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.training = True