import torch
from torch import nn

generator_loss_fun = torch.nn.L1Loss()
discriminator_loss_fun = torch.nn.MSELoss()


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
class Encoder(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, ks=4):

        super(Encoder, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=ks, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0, normalize=True):
        super(Decoder, self).__init__()
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
    def __init__(self, in_channels=3, out_channels=3, if_crop=True, loss_fun=generator_loss_fun):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5, normalize=False)  # 不需要正规化了

        self.up1 = UNetUp(512, 512, dropout=0.5, if_crop=if_crop)
        self.up2 = UNetUp(1024, 512, dropout=0.5, if_crop=if_crop)
        self.up3 = UNetUp(1024, 256, if_crop=if_crop)
        self.up4 = UNetUp(512, 128, if_crop=if_crop)
        self.up5 = UNetUp(256, 64, if_crop=if_crop)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )
        self.loss_fun = loss_fun

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)  # x:[1, 3, 64, 64]  d1:[1, 64, 32, 32]
        d2 = self.down2(d1)  # d2:[1,128,16,16]
        d3 = self.down3(d2)  # d3:[1,256,8,8]
        d4 = self.down4(d3)  # d4:[1,512,4,4]
        d5 = self.down5(d4)  # d5:[1,512,2,2]
        d6 = self.down6(d5)  # d6:[1,512,1,1]

        u1 = self.up1(d6, d5)  # u1:[1,1024,2,2]
        u2 = self.up2(u1, d4)  # u2:[1,1024,4,4]
        u3 = self.up3(u2, d3)  # u3:[1,1024,8,8]
        u4 = self.up4(u3, d2)  # u4:[1,1024,16,16]
        u5 = self.up5(u4, d1)  # u5:[1,512,32,32]
        return self.final(u5)

    def loss(self, generate, target):
        return self.loss_fun(generate, target)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, loss_fun=discriminator_loss_fun):
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
        self.loss_fun = loss_fun

    def forward(self, img_A, img_B):
        # Concatenate image and condition image
        # by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

    def loss(self, pred, real):
        return self.loss_fun(pred, real)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, loss_fun=generator_loss_fun):
        super(AutoEncoder, self).__init__()
        self.down1 = Encoder(in_channels, 64, normalize=False)
        self.down2 = Encoder(64, 128)
        self.down3 = Encoder(128, 256)
        self.down4 = Encoder(256, 512, dropout=0.5)
        self.down5 = Encoder(512, 512, dropout=0.5)
        self.down6 = Encoder(512, 512, dropout=0.5, normalize=False)  # 不需要正规化了

        self.up1 = Decoder(512, 1024, dropout=0.5)
        self.up2 = Decoder(1024, 1024, dropout=0.5)
        self.up3 = Decoder(1024, 512)
        self.up4 = Decoder(512, 256)
        self.up5 = Decoder(256, 128)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )
        self.loss_fun = loss_fun

    def forward(self, x):
        # 上采样
        d1 = self.down1(x)  # x:[1, 3, 64, 64]  d1:[1, 64, 32, 32]
        d2 = self.down2(d1)  # d2:[1,128,16,16]
        d3 = self.down3(d2)  # d3:[1,256,8,8]
        d4 = self.down4(d3)  # d4:[1,512,4,4]
        d5 = self.down5(d4)  # d5:[1,512,2,2]
        d6 = self.down6(d5)  # d6:[1,512,1,1]
        # 下采样
        u1 = self.up1(d6)  # u1:[1,1024,2,2]
        u2 = self.up2(u1)  # u2:[1,1024,4,2]
        u3 = self.up3(u2)  # u3:[1,1024,8,2]
        u4 = self.up4(u3)  # u4:[1,1024,16,16]
        u5 = self.up5(u4)  # u5:[1,512,32,32]
        x = self.final(u5)
        return x

    def loss(self, x, y):
        return self.loss_fun(x, y)


class GAN(nn.Module):
    def __init__(self, train_opt=None, generator=GeneratorUNet(), discriminator=Discriminator()):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        if isinstance(train_opt, dict):
            self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=train_opt['lrG'],
                                                betas=(train_opt['b1'], train_opt['b2']))
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=train_opt['lrD'],
                                                betas=(train_opt['b1'], train_opt['b2']))

    def forward(self, source, target):
        generate = self.generator(source)
        source_generate = self.discriminator(generate, source)
        source_target = self.discriminator(target, source)
        source_generate2 = self.discriminator(generate.detach(), source)
        return generate, source_generate, source_target, source_generate2

    def loss(self, generate, target, source_generate, source_target, source_generate2):
        invalid = source_target.clone().detach() * 0
        valid = invalid + 1
        # 计算生成模型误差
        loss_pixel = self.generator.loss(generate, target)
        loss_GAN = self.discriminator.loss(source_generate, valid)
        loss_G = loss_GAN + 10 * loss_pixel
        # 分辨源图像和目标图像
        loss_real = self.discriminator.loss(source_target, valid)
        # 分辨源图像和生成图像
        loss_fake = self.discriminator.loss(source_generate2, invalid)
        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        return loss_G, loss_D, loss_GAN, loss_pixel

    def step(self, source, target):
        generate, source_generate, source_target, source_generate2 = self(source, target)
        loss_G, loss_D, loss_GAN, loss_pixel = self.loss(generate, target, source_generate,
                                                         source_target, source_generate2)
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

        self.optimizer_D.zero_grad()
        loss_D.backward()
        self.optimizer_D.step()
        return loss_G, loss_D, loss_GAN, loss_pixel
