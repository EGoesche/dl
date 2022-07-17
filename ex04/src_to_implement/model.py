import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.training = True




class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.training = True

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.outer_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) # may be wrong.
        self.outer_bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut            = self.shortcut(input)
        first_conv_output   = self.conv1(input)
        first_bn_output     = self.bn1(first_conv_output)
        first_relu_output   = nn.ReLU()(first_bn_output)
        sec_conv_output     = self.conv2(first_relu_output)
        sec_bn_output       = self.bn2(sec_conv_output)
        sec_bn_output       += shortcut   # Skip connection is added to the output of batch norm 2
        sec_relu_output     = nn.ReLU()(sec_bn_output)

        outer_conv_output   = self.outer_conv(input)
        outer_bn_output     = self.outer_bn(outer_conv_output)
        output              = sec_relu_output + outer_bn_output

        return output
