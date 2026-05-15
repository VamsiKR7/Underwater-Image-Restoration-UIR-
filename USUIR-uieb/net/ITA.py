import torch

# Squeeze-and-Excitation Block
class SEBlock(torch.nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(channels, channels // reduction, kernel_size=1),
            torch.nn.ReLU(inplace=True), #r is inside this r = ReLU(Wa1 + b1)
            torch.nn.Conv2d(channels // reduction, channels, kernel_size=1),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        scale = self.se(x)
        return x * scale

# Depthwise Separable Convolution Block
class DSConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.depthwise = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(padding),
            torch.nn.Conv2d(
                in_channels, 
                in_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=0, 
                groups=in_channels
            )
        )
        self.pointwise = torch.nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# JDSNet with Depthwise Separable Convolution
class JDSNet(torch.nn.Module):
    def __init__(self, num=64):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            DSConv(3, num, 3, 1, 1),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.se1 = SEBlock(num)
        self.conv2 = torch.nn.Sequential(
            DSConv(num, num, 3, 1, 1),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            DSConv(num, num, 3, 1, 1),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            DSConv(num, num, 3, 1, 1),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(num, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )
    
    def forward(self, data):
        data = self.conv1(data)
        data = self.se1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        return self.final(data)

# TDNet with Depthwise Separable Convolution
class TDNet(torch.nn.Module):
    def __init__(self, num=64):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            DSConv(3, num, 3, 1, 1),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        # self.se1 = SEBlock(num)
        self.conv2 = torch.nn.Sequential(
            DSConv(num, num, 3, 1, 1),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            DSConv(num, num, 3, 1, 1),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            DSConv(num, num, 3, 1, 1),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(num, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )
    
    def forward(self, data):
        data = self.conv1(data)
        # data = self.se1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        return self.final(data)