import torch
from torch import nn
import torchvision.transforms
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn.init import dirac_, xavier_normal_


class ReZero(nn.Module):
    def __init__(self, layer):
        super(ReZero, self).__init__()
        # Suppress pylint bug
        # pylint: disable=E1101
        self.layer = layer
        self.α = nn.Parameter(torch.zeros(1), requires_grad=True)
        # pylint: enable=E1101

    def forward(self, x):
        return x + self.α*self.layer(x)


class OntoF16(nn.Module):
    def __init__(self):
        super(OntoF16, self).__init__()
        self.f16_map = 2**12 - 1

    def forward(self, x):
        return x * self.f16_map


class ResnetHasher(nn.Module):
    def __init__(self):
        super(ResnetHasher, self).__init__()

        self.resnet = torch.hub.load(
            'pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.enc_layers = OrderedDict([
            ('resnet', self.resnet),
            ('relu_r', nn.ReLU(1000)),
            ('norm_r', nn.BatchNorm1d(1000)),

            ('full_1', nn.Linear(1000, 500)),
            ('relu_1', nn.ReLU()),
            ('norm_1', nn.BatchNorm1d(500)),

            ('full_2', nn.Linear(500, 250)),
            ('relu_2', nn.ReLU()),
            ('norm_2', nn.BatchNorm1d(250)),

            ('output', nn.Linear(250, 8)),
            ('tanh', nn.Tanh()),
            #('ontoF16', OntoF16())
        ])
        self.encoder = nn.Sequential(self.enc_layers)

    def forward(self, x):
        return self.encoder(x)

    def visualize(self):
        from torchsummary import summary
        summary(self, (3, 64, 64))


class HasherBlock(nn.Module):
    def __init__(self, block_num, in_channels, out_channels, kernel_size, kernel_stride, conv_padding, pool_num=0):
        super(HasherBlock, self).__init__()

        self.block_num = block_num

        # Set layers
        self.layers = OrderedDict()
        self.layers[f'conv{self.block_num}'] = nn.Conv2d(
            in_channels, out_channels, kernel_size, kernel_stride, padding=conv_padding)
        if pool_num != 0:
            self.layers[f'pool{self.block_num}'] = nn.MaxPool2d(pool_num)
        self.layers[f'relu{self.block_num}'] = nn.ReLU()
        self.layers[f'norm{self.block_num}'] = nn.BatchNorm2d(out_channels)

        self.model = nn.Sequential(self.layers)

    def init(self):
        dirac_(self.layers[f'conv{self.block_num}'].weight.data)

    def forward(self, x):
        return self.model(x)

    def visualize(self):
        from torchsummary import summary
        summary(self, (3, 64, 64))


class CustomHasher(nn.Module):
    def __init__(self):
        super(CustomHasher, self).__init__()

        self.layers = OrderedDict([
            # block_num, in_channels, out_channels, kernel_size, kernel_stride, conv_padding pool_num
            ('block_1', HasherBlock(1,  3, 12, 3, 1, 1, 2)),
            ('block_2', HasherBlock(2, 12, 24, 3, 1, 1, 2)),
            ('block_3', HasherBlock(3, 24, 48, 3, 1, 1, 2)),
            ('block_4', HasherBlock(4, 48, 96, 3, 1, 1, 2)),
            ('adapt', nn.AdaptiveMaxPool2d(1)),
            ('flat ', nn.Flatten()),
            ('full_1', nn.Linear(96, 36)),
            ('full_2', nn.Linear(36, 8)),
            ('tanh', nn.Tanh()),
            #('ontoF16', OntoF16())
        ])
        self.encoder = nn.Sequential(self.layers)

    def forward(self, x):
        return self.encoder(x)

    def visualize(self):
        from torchsummary import summary
        summary(self, (3, 64, 64))


class SmallHasher(nn.Module):
    def __init__(self):
        super(SmallHasher, self).__init__()

        self.layers = OrderedDict([
            # block_num, in_channels, out_channels, kernel_size, kernel_stride, conv_padding pool_num
            ('block_1', HasherBlock(1,  3, 16, 3, 2, 1, 1)),
            ('block_2', HasherBlock(2, 16, 20, 3, 2, 1, 1)),
            ('block_3', HasherBlock(3, 20, 28, 3, 2, 1, 1)),
            ('block_4', HasherBlock(4, 28, 32, 3, 2, 1, 1)),
            ('block_5', HasherBlock(5, 32, 40, 3, 2, 1, 1)),
            ('adapt', nn.AdaptiveMaxPool2d(1)),
            ('flat ', nn.Flatten()),
            ('full_1', nn.Linear(40, 24)),
            ('relu_1', nn.ReLU()),
            ('full_2', nn.Linear(24, 16)),
            ('tanh', nn.Tanh())
        ])

        # Careful initialization
        self.layers['block_1'].init()
        self.layers['block_2'].init()
        self.layers['block_3'].init()
        self.layers['block_4'].init()
        self.layers['block_5'].init()
        xavier_normal_(self.layers['full_1'].weight.data)
        xavier_normal_(self.layers['full_2'].weight.data)

        self.encoder = nn.Sequential(self.layers)

    def forward(self, x):
        return self.encoder(x)

    def visualize(self):
        from torchsummary import summary
        summary(self, (3, 64, 64))
