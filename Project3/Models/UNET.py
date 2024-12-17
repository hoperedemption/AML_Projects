import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms.functional as TF

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()
        
        self.c = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.c(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
                
        self.c = nn.Sequential(
            ConvBlock(in_channels, inter_channels, kernel_size, stride, padding), 
            ConvBlock(inter_channels, out_channels, kernel_size, stride, padding)
        )
        
    def forward(self, x):
        return self.c(x)
    
class DoubleBaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        
        self.c = DoubleConv(in_channels, out_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        return self.c(x)
    
class UNET(nn.Module):
    def __init__(self, kernel_size, stride, padding, in_channels=3, out_channels=1, features=[64,128,256,512,1024], dropout=0.1):
        super().__init__()
        
        f = features[:-1]
        fd = [in_channels] + f
        fu = f[::-1]
        self.down_arch = nn.ModuleList([nn.ModuleList([DoubleBaseConv(inc, ouc, kernel_size, stride, padding), 
            nn.MaxPool2d((4, 4), (2, 2), (1, 1))]) for inc,ouc in zip(fd[:-1], fd[1:])])
        self.up_arch = nn.ModuleList([nn.ModuleList([nn.ConvTranspose2d(ouc, ouc, kernel_size, stride, padding), 
            DoubleConv(ouc * 2, ouc, ouc//2, kernel_size, stride, padding)]) for ouc in fu[:-1]])
        self.up_arch.append(nn.ModuleList([nn.ConvTranspose2d(features[0], features[0], (4, 4), (2, 2), (1, 1)), 
                                            DoubleBaseConv(features[1], features[0], kernel_size, stride, padding)]))
        self.bottleneck = DoubleConv(features[-2], features[-1], features[-2], kernel_size, stride, padding)
                
        self.final_project = nn.Conv2d(features[0], out_channels, kernel_size, stride, padding)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        up_list = []
        
        # descent 
        d = x
        for down_block in self.down_arch:
            d = self.dropout(down_block[0](d))
            up_list.append(d)
            d = down_block[1](d)

        
        # pass through the bottleneck layer        
        d = self.dropout(self.bottleneck(d))
        
        # go up
        u = d
        for up_block, skip in zip(self.up_arch, reversed(up_list)):
            u = up_block[0](u)
            
            # make sure the sizes match 
            if u.shape != skip.shape:
                u = TF.resize(u, size=skip.shape[-2:])
            
            u = torch.concat([skip, u], dim=1) 
            u = self.dropout(up_block[1](u))         

        # project 
        return torch.sigmoid(self.final_project(u))


def test():
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    
    x = torch.randn(5, 1, 224, 224)
    model = UNET(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding)
    
    
    preds = model(x)
    print(preds.shape)
    assert preds.shape == x.shape
