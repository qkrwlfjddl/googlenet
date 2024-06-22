import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
class Inception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = ConvBlock(in_channels, n1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, n3x3_reduce, kernel_size=1),
            ConvBlock(n3x3_reduce, n3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, n5x5_reduce, kernel_size=1),
            ConvBlock(n5x5_reduce, n5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([x1, x2, x3, x4], dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, aux_logits=False, num_classes=14):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = ConvBlock(1, 8, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv2 = ConvBlock(8, 16, kernel_size=1)
        self.conv3 = ConvBlock(16, 32, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.inception3a = Inception(32, 8, 8, 8, 2, 4, 4)
        self.inception3b = Inception(24, 12, 8, 12, 2, 4, 4)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(32, 12, 8, 12, 2, 4, 4)
        self.inception4e = Inception(32, 16, 12, 16, 4, 8, 8)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception(48, 16, 12, 16, 4, 8, 8)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear(x)
        return x
