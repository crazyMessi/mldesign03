import numpy
from model import *


class testConv2d:
    def __init__(self, in_size, out_size, ks=4, stride=2, padding=1):
        super(testConv2d, self).__init__()
        conv2d = [nn.Conv2d(in_size, out_size, kernel_size=ks, stride=stride, padding=1, bias=False)]
        self.model = nn.Sequential(conv2d)
        self.norm = nn.Sequential(nn.InstanceNorm2d(out_size))


# class AutoEncoder(nn.Module):
#     def __init__(self, in_channel=3, h=64, w=64, down_to=1, max_channel=512, first_out=64):
#         super(AutoEncoder, self).__init__()
#         self.Down = nn.ModuleList()
#         self.Up = nn.ModuleList()
#         out = first_out
#         dropout = 0
#         while h > down_to:
#             self.Down.append(UNetDown(in_channel, out, dropout=dropout, normalize=bool(h-2)))
#             self.Up.append(UNetUp(out, in_channel, dropout=dropout, normalize=bool(h-2)))
#             in_channel = out
#             h /= 2
#             if out < max_channel:
#                 out *= 2
#             else:
#                 dropout = 0.5
#
#     def forward(self, x):
#         times = len(self.Down)
#         for i in range(times):
#             x = self.Down[i](x)
#         for i in range(times):
#             x = self.Up[times-i-1](x)
#
#         return x

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(AutoEncoder, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)  # 不需要正规化了

        self.up1 = UNetUp(512, 1024, dropout=0.5)
        self.up2 = UNetUp(1024, 1024, dropout=0.5)
        self.up3 = UNetUp(1024, 512)
        self.up4 = UNetUp(512, 256)
        self.up5 = UNetUp(256, 128)
        # self.up6 = UNetUp(512, 128)
        # self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)  # x:[1, 3, 64, 64]  d1:[1, 64, 32, 32]
        d2 = self.down2(d1)  # d2:[1,128,16,16]
        d3 = self.down3(d2)  # d3:[1,256,8,8]
        d4 = self.down4(d3)  # d4:[1,512,4,4]
        d5 = self.down5(d4)  # d5:[1,512,2,2]
        d6 = self.down6(d5)  # d6:[1,512,1,1]
        # d7 = self.down7(d6)    #d7:[1,512,2,2]
        # d8 = self.down8(d7)    #d8:[1,512,1,1]
        # ipdb.set_trace()
        u1 = self.up1(d6)  # u1:[1,1024,2,2]
        u2 = self.up2(u1)  # u2:[1,1024,4,2]
        u3 = self.up3(u2)  # u3:[1,1024,8,2]
        u4 = self.up4(u3)  # u4:[1,1024,16,16]
        u5 = self.up5(u4)  # u5:[1,512,32,32]
        # u6 = self.up6(u5, d1)  #u6:[1,256,64,64]
        # u7 = self.up7(u6, d1)  #u7:[1,128,128,128]
        x = self.final(u5)
        return x


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, normalize=True):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


# x = torch.rand(8, 3, 64, 64)
# auto = AutoEncoder(x)
# auto(x)