import torch.nn as nn
import torch.nn.functional as F
import torch


# 为网络参数赋正态分布的初值
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, ks=4):

        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=ks, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, if_crop=True):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.InstanceNorm2d(out_size),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)
        self.if_crop = if_crop

    def forward(self, x, skip_input):
        # ipdb.set_trace()
        x = self.model(x)
        if self.if_crop > 0:
            x = torch.cat((x, skip_input), 1)
        else:
            x = torch.cat((x, x), 1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, if_crop=True):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)  # 不需要正规化了
        # self.down7 = UNetDown(512, 512, dropout=0.5, ks=1)
        # self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5, ks=1)

        self.up1 = UNetUp(512, 512, dropout=0.5, if_crop=if_crop)
        self.up2 = UNetUp(1024, 512, dropout=0.5, if_crop=if_crop)
        self.up3 = UNetUp(1024, 256, if_crop=if_crop)
        self.up4 = UNetUp(512, 128, if_crop=if_crop)
        self.up5 = UNetUp(256, 64, if_crop=if_crop)
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
        u1 = self.up1(d6, d5)  # u1:[1,1024,2,2]
        u2 = self.up2(u1, d4)  # u2:[1,1024,4,4]
        u3 = self.up3(u2, d3)  # u3:[1,1024,8,8]
        u4 = self.up4(u3, d2)  # u4:[1,1024,16,16]
        u5 = self.up5(u4, d1)  # u5:[1,512,32,32]
        # u6 = self.up6(u5, d1)  #u6:[1,256,64,64]
        # u7 = self.up7(u6, d1)  #u7:[1,128,128,128]

        return self.final(u5)


##############################
#        Discriminator
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image 
        # by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
