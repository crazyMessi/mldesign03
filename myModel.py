from torch import nn
from utils.my_optimizer import *

generator_loss_fun = torch.nn.L1Loss()

discriminator_loss_fun = torch.nn.MSELoss()


# ===================================
#              网络单元
# ===================================


# class UNetDown(nn.Module):
#     def __init__(self, in_size, out_size, normalize=True, dropout=0.0, ks=4):
#
#         super(UNetDown, self).__init__()
#         layers = [nn.Conv2d(in_size, out_size, kernel_size=ks, stride=2, padding=1, bias=False)]
#         if normalize:
#             layers.append(nn.InstanceNorm2d(out_size))
#         layers.append(nn.LeakyReLU(0.2))
#         if dropout:
#             layers.append(nn.Dropout(dropout))
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x)


# class Decoder(nn.Module):
#     def __init__(self, in_size, out_size, dropout=0.0, normalize=True):
#         super(Decoder, self).__init__()
#         layers = [nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)]
#         if normalize:
#             layers.append(nn.InstanceNorm2d(out_size))
#         layers.append(nn.ReLU(inplace=True))
#         if dropout:
#             layers.append(nn.Dropout(dropout))
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.model(x)
#         return x


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, ks=4, if_res_block=False):
        # Unet和AutoEncoder的下采样是一致的
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
        x = self.model(x)
        if self.if_crop > 0:
            x = torch.cat((x, skip_input), 1)
        return x


# ===================================
#              子模型
# ===================================
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, loss_fun=generator_loss_fun, if_crop=True, dropout_rate=0.5):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=dropout_rate)
        self.down5 = UNetDown(512, 512, dropout=dropout_rate)
        self.down6 = UNetDown(512, 512, dropout=dropout_rate, normalize=False)  # 不需要正规化了

        out_channels_rate = 1
        if not if_crop:
            # 如果不使用上采样 则通道数翻倍
            # 曾经尝试直接复制一个 但loss存在差异
            out_channels_rate = 2

        self.up1 = UNetUp(512, 512 * out_channels_rate, dropout=dropout_rate, if_crop=if_crop)
        self.up2 = UNetUp(1024, 512 * out_channels_rate, dropout=dropout_rate, if_crop=if_crop)
        self.up3 = UNetUp(1024, 256 * out_channels_rate, if_crop=if_crop)
        self.up4 = UNetUp(512, 128 * out_channels_rate, if_crop=if_crop)
        self.up5 = UNetUp(256, 64 * out_channels_rate, if_crop=if_crop)

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
    def __init__(self, in_channels=3, out_channels=3, loss_fun=generator_loss_fun, dropout_rate=0.5):
        super(AutoEncoder, self).__init__()
        # AutoEncoder即不使用crop的Unet
        self.model = GeneratorUNet(in_channels=in_channels, out_channels=out_channels, loss_fun=loss_fun, if_crop=False,
                                   dropout_rate=dropout_rate)
        self.loss_fun = loss_fun

    def forward(self, x):
        y = self.model(x)
        return y

    def loss(self, x, y):
        return self.loss_fun(x, y)


# ===================================
#              顶级模型
# ===================================
class GAN(nn.Module):
    def __init__(self, train_opt=None, generator=GeneratorUNet(), discriminator=Discriminator()):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        if isinstance(train_opt, dict):
            self.optimizer_G = Adam_Optimizer(parameters=self.generator.parameters(), lr=train_opt['lrG'],
                                              betas=(train_opt['b1'], train_opt['b2']),
                                              freq=train_opt['lrG_d'] * train_opt['dataloader_length'])

            self.optimizer_D = Adam_Optimizer(parameters=self.discriminator.parameters(), lr=train_opt['lrD'],
                                              betas=(train_opt['b1'], train_opt['b2']),
                                              freq=train_opt['lrD_d'] * train_opt['dataloader_length'])
            self.train_opt = train_opt

    def forward(self, source, target):
        generate = self.generator(source)
        source_generate = self.discriminator(generate, source)
        source_target = self.discriminator(target, source)
        source_generate2 = self.discriminator(generate.detach(), source)
        return generate, source_generate, source_target, source_generate2

    def loss(self, generate, target, source_generate, source_target, source_generate2,
             if_G_backward=False, if_D_backward=False):
        invalid = source_target.clone().detach() * 0
        valid = invalid + 1
        # 计算生成模型误差
        loss_sVg = self.discriminator.loss(source_generate, valid)
        loss_pixel = self.generator.loss(generate, target)
        loss_G = loss_sVg + self.train_opt['weight_pic'] * loss_pixel
        if if_G_backward:
            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()
            return loss_G, 0, loss_sVg, loss_pixel
        # 分辨源图像和目标图像
        loss_real = self.discriminator.loss(source_target, valid)
        # 分辨源图像和生成图像
        loss_fake = self.discriminator.loss(source_generate2, invalid)
        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        if if_D_backward:
            self.optimizer_D.zero_grad()
            loss_D.backward()
            self.optimizer_D.step()
            return 0, loss_D, 0, 0
        return loss_G, loss_D, loss_sVg, loss_pixel

    def step(self, source, target):
        generate, source_generate, source_target, source_generate2 = self(source, target)
        loss_G, _, loss_sVg, loss_pixel = self.loss(generate, target, source_generate, source_target,
                                                    source_generate2, if_G_backward=True)
        _, loss_D, _, _ = self.loss(generate, target, source_generate, source_target,
                                    source_generate2, if_D_backward=True)

        loss_dic = {'loss_G': loss_G.item(), 'loss_D': loss_D.item(), 'loss_pixel': loss_pixel.item(),
                    'loss_sVg': loss_sVg.item()}
        return loss_dic


# 基于AutoEncoder的图片生成器
class AutoEncoderGen(nn.Module):
    def __init__(self, train_opt=None, generator=AutoEncoder()):
        super(AutoEncoderGen, self).__init__()
        self.generator = generator
        if isinstance(train_opt, dict):
            self.optimizer_G = Adam_Optimizer(parameters=self.generator.parameters(), lr=train_opt['lrG'],
                                              betas=(train_opt['b1'], train_opt['b2']),
                                              freq=train_opt['lrG_d'] * train_opt['dataloader_length'])

    def loss(self, x, y):
        return self.generator.loss_fun(x, y)

    def forward(self, x):
        return self.generator(x)

    def step(self, x, y):
        generate = self(x)
        loss_pixel = self.loss(y, generate)
        self.optimizer_G.zero_grad()
        loss_pixel.backward()
        self.optimizer_G.step()
        loss_dic = {'loss_pixel': loss_pixel.item()}
        return loss_dic
