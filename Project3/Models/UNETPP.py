import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torchsummary import summary


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()

        self.c = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.c(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.c = nn.Sequential(
            ConvBlock(in_channels, inter_channels,
                      kernel_size, stride, padding),
            ConvBlock(inter_channels, out_channels,
                      kernel_size, stride, padding)
        )

    def forward(self, x):
        return self.c(x)


class DoubleBaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.c = DoubleConv(in_channels, out_channels,
                            out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.c(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.up(x)


class UNETPP(nn.Module):
    def __init__(self, kernel_size, stride, padding, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024], dropout=0.1):
        super().__init__()

        f = features[:-1]
        fd = [in_channels] + f
        fu = f[::-1]
        self.down_arch = nn.ModuleList([nn.ModuleList([DoubleBaseConv(inc, ouc, kernel_size, stride, padding),
                                                       nn.MaxPool2d((4, 4), (2, 2), (1, 1))]) for inc, ouc in zip(fd[:-1], fd[1:])])
        self.up_arch = nn.ModuleList([nn.ModuleList([nn.ConvTranspose2d(ouc, ouc, kernel_size, stride, padding),
                                                     DoubleConv(ouc * (idx+2), ouc, ouc//2, kernel_size, stride, padding)]) for idx, ouc in enumerate(fu[:-1])])
        self.up_arch.append(nn.ModuleList([nn.ConvTranspose2d(features[0], features[0], (4, 4), (2, 2), (1, 1)),
                                           DoubleBaseConv(features[0] * (len(fu) + 1), features[0], kernel_size, stride, padding)]))
        self.bottleneck = DoubleConv(
            features[-2], features[-1], features[-2], kernel_size, stride, padding)

        self.skip_pathways_convs = nn.ModuleDict()
        self.skip_upsampling = nn.ModuleDict()

        self.final_project = nn.Conv2d(
            features[0], out_channels, kernel_size, stride, padding)
        self.dropout = nn.Dropout(dropout)

        for i in range(len(f) - 1):
            for j in range(len(f) - i - 1):
                conv_key = f'{i}{j+1}'
                up_key = f'{i+1}{j}'
                up_sample_factor = 2
                self.skip_pathways_convs[conv_key] = nn.Conv2d(
                    (j+2) * f[i], f[i], kernel_size, stride, padding)
                self.skip_upsampling[up_key] = nn.ConvTranspose2d(
                    f[i+1], f[i], kernel_size, up_sample_factor, padding, 1)

    def forward(self, x):
        backbone = []
        inter_skips = {}

        # descent
        d = x
        for i, down_block in enumerate(self.down_arch):
            d = self.dropout(down_block[0](d))
            backbone.append(d)
            d = down_block[1](d)
            inter_skips[f'{i}0'] = backbone[-1]
            if i > 0:
                for j in range(min(i, 3)):
                    ck = f'{i-j-1}{j+1}'  # convolutional key
                    uk = f'{i-j}{j}'  # upsampling key
                    feature_map_stack = torch.concat(
                        [inter_skips[f'{i-j-1}{k}'] for k in range(j+1)], dim=1)
                    up_sampling_input = inter_skips[uk]
                    up_sampling_input = self.skip_upsampling[uk](
                        up_sampling_input)

                    inter_skips[ck] = torch.concat(
                        [feature_map_stack, up_sampling_input], dim=1)

                    inter_skips[ck] = self.skip_pathways_convs[ck](
                        inter_skips[ck])

        # pass through the bottleneck layer
        d = self.dropout(self.bottleneck(d))

        # go up
        u = d
        for idx, (up_block, skip) in enumerate(zip(self.up_arch, reversed(backbone))):
            u = up_block[0](u)

            path_level = len(self.up_arch) - idx - 1

            # make sure the sizes match
            if u.shape != skip.shape:
                u = TF.resize(u, size=skip.shape[-2:])
            path = []
            if idx > 0:
                path = [inter_skips[f'{path_level}{k}']
                        for k in range(1, 4 - (path_level))]
            pathway_stack = torch.concat(path + [skip], dim=1)

            u = torch.concat([pathway_stack] + [u], dim=1)
            u = self.dropout(up_block[1](u))
        # project
        return torch.sigmoid(self.final_project(u))


def test():
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)

    x = torch.randn(5, 1, 224, 224)
    model = UNETPP(in_channels=1, out_channels=1,
                   kernel_size=kernel_size, stride=stride, padding=padding)

    # summary(model, (1, 224, 224))

    preds = model(x)
    print(preds.shape)
    assert preds.shape == x.shape
