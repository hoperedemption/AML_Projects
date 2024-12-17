import torch 
import torch.nn as nn 
import torch.nn.functional as F 

conv1x1 = lambda in_channels, out_channels: nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), 
                                                      padding=(0, 0), bias=False)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), dilation=1, groups=1):
        super().__init__()
        
        padding = dilation * ((kernel_size[0] - 1) // 2), dilation * ((kernel_size[1] - 1) // 2)
        self.arch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()
        )
    def forward(self, x):
        return self.arch(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.lower_arch = nn.Sequential(
            ConvBlock(in_channels, out_channels), 
            ConvBlock(out_channels, out_channels)
        )
        
        self.upper_arch = conv1x1(in_channels, out_channels)
        
        self.out_arch = nn.Sequential(
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()        
        )
        
    def forward(self, x):
        return self.out_arch(self.lower_arch(x) + self.upper_arch(x))
    
class MidScope(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.arch = nn.Sequential(
            ConvBlock(in_channels, out_channels), 
            ConvBlock(out_channels, out_channels, dilation=2)
        )
        
    def forward(self, x):
        return self.arch(x)
    
class WideScope(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.arch = nn.Sequential(
            ConvBlock(in_channels, out_channels), 
            ConvBlock(out_channels, out_channels, dilation=2), 
            ConvBlock(out_channels, out_channels, dilation=3)
        )
        
    def forward(self, x):
        return self.arch(x)
    
class SeparatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_dim):
        super().__init__()
        
        self.arch = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=(1, n_dim)), 
            ConvBlock(out_channels, out_channels, kernel_size=(n_dim, 1))
        )
        
    def forward(self, x):
        return self.arch(x)
    
class DUCK(nn.Module):
    def __init__(self, in_channels, out_channels, n_dim=7):
        super().__init__()
        
        self.widescope = WideScope(in_channels, out_channels)
        self.midscope = MidScope(in_channels, out_channels)
        
        self.arch_list = nn.ModuleList([])
        for i in range(1, 3):
            l = []
            for j in range(i):
                if j == 0:
                    l.append(ResidualBlock(in_channels, out_channels))
                else:
                    l.append(ResidualBlock(out_channels, out_channels))
            l = nn.Sequential(*l)
            self.arch_list.append(l)
        
        self.sep = SeparatedBlock(in_channels, out_channels, n_dim)
        
        
    def forward(self, x):
        res = self.widescope(x)
        res = torch.add(res, self.midscope(x))
        for block in self.arch_list:
            res = torch.add(res, block(x))
        res = torch.add(res, self.sep(x))
        return res

class BlueDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.arch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2), bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()
        )
    def forward(self, x):
        return self.arch(x)    

class DUCKNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4):
        super().__init__()
        
        self.down_arch = nn.ModuleList([BlueDownsample(2 ** i * in_channels, 2 ** (i + 1) * in_channels) for i in range(depth)])
        self.mid_arch = nn.ModuleList([
            nn.ModuleList(
                [DUCK(2 ** i * in_channels, 2 ** i * in_channels), 
                BlueDownsample(2 ** i * in_channels, 2 ** (i + 1) * in_channels)]
            ) for i in range(depth)
        ])
        self.bneck = nn.Sequential(
            ResidualBlock(2 ** depth * in_channels, 2 ** depth * in_channels), 
            ResidualBlock(2 ** depth * in_channels, 2 ** (depth - 1) * in_channels)
        )
        self.up_arch = nn.ModuleList([
            nn.ModuleList([
                nn.ConvTranspose2d(2 ** i * in_channels, 2 ** i * in_channels, kernel_size=(2, 2), stride=(2, 2)), 
                DUCK(2 ** i * in_channels, 2 ** max(0, (i - 1)) * in_channels) 
            ])
            for i in range(depth - 1, -1, -1)
        ])
        self.final = conv1x1(in_channels, out_channels)
        
    def forward(self, x):
        down = x
        down_skip = []
        
        for block in self.down_arch:
            down = block(down)
            down_skip.append(down)
        
        mid = x
        mid_skip = []
        for skip, block in zip(down_skip, self.mid_arch):
            mid = block[0](mid)
            mid_skip.append(mid)
            mid = block[1](mid)
            mid = torch.add(skip, mid)
        
        bottom = self.bneck(mid)
                
        up = bottom 
        for skip, block in zip(mid_skip[::-1], self.up_arch):
            up = block[0](up)
            up = torch.add(skip, up)
            up = block[1](up)
            
        return F.sigmoid(self.final(up))
    
def test_resblock():
    res = ResidualBlock(5, 10)
    t = torch.randn(10, 5, 112, 112)
    print(res(t).shape)
    
def test_midblock():
    mid = MidScope(in_channels=1, out_channels=5)
    t = torch.randn(20, 1, 112, 112)
    print(mid(t).shape)
    
def test_wideblock():
    wide = WideScope(5, 15)
    t = torch.randn(25, 5, 112, 112)
    print(wide(t).shape)
    
def test_sep():
    sep = SeparatedBlock(4, 16, 5)
    t = torch.randn(22, 4, 112, 112)
    print(sep(t).shape)
    
def test_duck():
    duck = DUCK(5, 7)
    t = torch.randn(32, 5, 112, 112)
    print(duck(t).shape)

def test_ducknet():
    ducknet = DUCKNet(5, 1)
    t = torch.randn(32, 5, 112, 112)
    print(ducknet(t).shape)
    
    