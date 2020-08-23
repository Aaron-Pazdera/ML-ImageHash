import torch
from torch import nn
import torchvision.transforms
import torch.nn.functional as F
from collections import OrderedDict

class OntoF16(nn.Module):
    def __init__(self):
        super(OntoF16, self).__init__()        
        self.f16_map = 2**12 - 1
    def forward(self, x):
        return x * self.f16_map

class ResnetHasher(nn.Module):
    def __init__(self):
        super(ResnetHasher, self).__init__()
        
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        self.enc_layers = OrderedDict([ 
            ('resnet', self.resnet),
            ('relu_r', nn.BatchNorm1d(1000)),
            ('norm_r', nn.BatchNorm1d(1000)),
                                       
            ('full_1', nn.Linear(1000, 500)), 
            ('relu_1', nn.ReLU()),
            ('norm_1', nn.BatchNorm1d(500)),
            
            ('full_2', nn.Linear(500, 250)),
            ('relu_2', nn.ReLU()),
            ('norm_2', nn.BatchNorm1d(250)),
            
            ('output', nn.Linear(250, 8)),
            ('sigmoid', nn.Sigmoid()),
            ('ontoF16', OntoF16())
        ])
        self.encoder = nn.Sequential(self.enc_layers)
    
    def forward(self, x):
        return self.encoder(x)
    
    def visualize(self):
        from torchsummary import summary
        summary(self, (3, 32, 32))

class HasherBlock(nn.Module):
    def __init__(self,block_num, in_channels, out_channels, kernel_size, kernel_stride, conv_padding, pool_num=0):
        super(HasherBlock, self).__init__()
        self.layers = OrderedDict()
        # Set layers
        self.layers[f'conv{block_num}'] = nn.Conv2d(in_channels,out_channels,kernel_size,kernel_stride,padding=conv_padding)     
        if pool_num!=0:
            self.layers[f'pool{block_num}'] = nn.MaxPool2d(pool_num)
        self.layers[f'relu{block_num}'] = nn.ReLU()
        self.layers[f'norm{block_num}'] = nn.BatchNorm2d(out_channels)
        
        self.model = nn.Sequential(self.layers)
        
    def forward(self, x):
        return self.model(x)
    
    def visualize(self):
        from torchsummary import summary
        summary(self, (3, 32, 32))
        
        
class CustomHasher(nn.Module):    
    def __init__(self):
        super(CustomHasher, self).__init__()
        
        self.layers = OrderedDict([
            # block_num, in_channels, out_channels, kernel_size, kernel_stride, conv_padding pool_num
            ('block_1', HasherBlock(1,  3, 12, 3, 1, 1, 2)),
            ('block_2', HasherBlock(2, 12, 24, 3, 1, 1, 2)),
            ('block_3', HasherBlock(3, 24, 48, 3, 1, 1, 2)),
            ('block_4', HasherBlock(3, 48, 96, 3, 1, 1, 2)),
            ('adapt', nn.AdaptiveMaxPool2d(1)),
            ('flat ', nn.Flatten()),
            ('full_1', nn.Linear(96 , 36)),
            ('full_2', nn.Linear(36, 8)),
            ('sigmoid', nn.Sigmoid()),
            ('ontoF16', OntoF16())
        ])
        self.encoder = nn.Sequential(self.layers)
        
    def forward(self, x):
        return self.encoder(x)
        
    def visualize(self):
        from torchsummary import summary
        summary(self, (3, 32, 32))
        