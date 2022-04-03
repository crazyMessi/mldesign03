import re
from turtle import forward
import torch
from torch import nn
import torchvision
from utils.my_optimizer import *


# ===================================
#              网络单元
# ===================================
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, channels, padding_type = 'replicate', norm_layer = nn.BatchNorm2d, dropout = False, use_bias = False):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(channels, padding_type, norm_layer, dropout, use_bias)

    def build_conv_block(self, dim, padding_type = 'replicate', norm_layer = nn.BatchNorm2d, dropout = False, use_bias = False):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            dropout (float)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if dropout > 0:
            conv_block += [nn.Dropout(dropout)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0, ks=4):
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
    def __init__(self, in_size, out_size, dropout=0.0, if_crop=True, crop_weight = -0.99):
        super(UNetUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=False),
                  nn.InstanceNorm2d(out_size),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        self.if_crop = if_crop
        if crop_weight>=0:
            self.crop_weight = nn.Parameter(torch.tensor(crop_weight),requires_grad=True)
        else:
            self.crop_weight = nn.Parameter(torch.tensor(abs(crop_weight)),requires_grad=False)

    def forward(self, x, skip_input):
        x = self.model(x)
        if self.if_crop > 0:
            x = torch.cat((x, skip_input*self.crop_weight), 1)
        return x


# ===================================
#              子模型
# ===================================
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, if_crop=True, dropout_rate=0.5,crop_weight = -0.99):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=dropout_rate)
        self.down5 = UNetDown(512, 512, dropout=dropout_rate)
        self.down6 = UNetDown(512, 512, dropout=dropout_rate, normalize=False)  # 不需要正规化了

        out_channels_rate = 1
        if not if_crop:
            # 如果不使用下采样的信息补充 则反卷积层输出的通道数翻倍 这样这次上采样输出的层数就和原来保持一致了
            # 曾经尝试直接复制一个 但loss存在差异
            out_channels_rate = 2
        self.up1 = UNetUp(512, 512 * out_channels_rate, dropout=dropout_rate, if_crop=if_crop,crop_weight = crop_weight)
        self.up2 = UNetUp(1024, 512 * out_channels_rate, dropout=dropout_rate, if_crop=if_crop,crop_weight = crop_weight)
        self.up3 = UNetUp(1024, 256 * out_channels_rate, if_crop=if_crop,crop_weight = crop_weight)
        self.up4 = UNetUp(512, 128 * out_channels_rate, if_crop=if_crop,crop_weight = crop_weight)
        self.up5 = UNetUp(256, 64 * out_channels_rate, if_crop=if_crop,crop_weight = crop_weight)

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

        u1 = self.up1(d6, d5)  # u1:[1,1024,2,2]
        u2 = self.up2(u1, d4)  # u2:[1,1024,4,4]
        u3 = self.up3(u2, d3)  # u3:[1,512,8,8]
        u4 = self.up4(u3, d2)  # u4:[1,256,16,16]
        u5 = self.up5(u4, d1)  # u5:[1,128,32,32]
        return self.final(u5)

    # def loss(self, generate, target):
    #     return self.loss_fun(generate, target)



class UnetSkipConnectionBlock(nn.Module):
    """魔改后的UNet层.试图支持任意替换submodule替换成别的模型
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    def __init__(self, before_exit = 3, before_enter = 3, in_channels=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, crop_weight = -0.99):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if crop_weight>=0:
            self.crop_weight = nn.Parameter(torch.tensor(crop_weight),requires_grad=True)
        else:
            self.crop_weight = nn.Parameter(torch.tensor(abs(crop_weight)),requires_grad=False)

        if in_channels is None:
            in_channels = before_exit
        downconv = nn.Conv2d(in_channels, before_enter, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.ReLU(inplace=True)
        downnorm = norm_layer(before_enter)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(before_exit)

        if innermost:
            # 修改一：innermost中间也可能有子模型
            # 修改二：CBR结构
            upconv = nn.ConvTranspose2d(before_enter, before_exit,
                                        kernel_size=4, stride=2, padding=1)
            down = [downconv,downnorm,downrelu]
            up = [upconv, upnorm,uprelu]
            if submodule:
                model = down + [submodule] +  up
            else:
                model = down  +  up
        else:
            upconv = nn.ConvTranspose2d(before_enter * 2, before_exit,
                                        kernel_size=4, stride=2, padding=1)
            down = [downconv,downnorm,downrelu]
            up = [upconv, upnorm,uprelu]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # 如果不是最后一层,则返回的向量通道数不变
            return torch.cat([x, self.model(x)], 1)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dropout_rate=0.5):
        super(AutoEncoder, self).__init__()
        # AutoEncoder即不使用crop的Unet
        self.model = GeneratorUNet(in_channels=in_channels, out_channels=out_channels, if_crop=False,
                                   dropout_rate=dropout_rate)

    def forward(self, x):
        y = self.model(x)
        return y


# 傻瓜式生成器
class DumpGenerator(nn.Module):
    def __init__(self, loss_fun=torch.nn.L1Loss(), case=0):
        super(DumpGenerator, self).__init__()
        # 生成器类型
        # case = 0,返回图像每个像素点为x最大值(即空白图像)
        # case = 1,返回原图像
        self.case = case
        self.loss_fun = loss_fun

    def loss(self, x, y):
        return self.loss_fun(x, y)

    def forward(self, x):
        if self.case == 0:
            return torch.max(x) * torch.ones_like(x)
        else:
            return x


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, in_channels=3, out_channels=3, ngf=64, norm_layer=nn.BatchNorm2d, dropout_rate=0, n_blocks=6,
                 padding_type='reflect', n_downsampling = 2):
        """Construct a Resnet-based generator

        Parameters:
            in_channels (int)      -- the number of channels in input images
            out_channels (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout=dropout_rate,
                                  use_bias=use_bias)]
            # TODO 使用预训练resnet 

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


# 魔改版的UNet,将中间的三个up/down层去掉,换成resblock。UNet模型的构造器用的是network里的，但在靠经resblock做了一定修改以适配
class UResGen(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64, norm_layer=nn.BatchNorm2d, dropout_rate=0, n_blocks=6,
                 padding_type='reflect'):
        super(UResGen,self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
   
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        res_blocks = []
        n_downsampling = 2
        mult = 2 ** n_downsampling # 进入res blocks时图像层数增加的倍数 实际就是下采样的倍数
        if n_blocks>0:
            for i in range(n_blocks):  # 堆叠n个res block
                res_blocks += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, dropout=dropout_rate,
                                    use_bias=use_bias)]
            res_blocks = nn.Sequential(*res_blocks)
        else:
            if n_blocks == -18:
                res_blocks = torchvision.models.resnet18()
            if n_blocks == -34:
                res_blocks = torchvision.models.resnet34()


        unet_block = UnetSkipConnectionBlock(before_exit=ngf * 2, before_enter = ngf * 4, submodule=res_blocks, norm_layer=norm_layer, innermost=True)
    
        model += [UnetSkipConnectionBlock(before_exit=ngf, before_enter=ngf * 2, submodule=unet_block, norm_layer=norm_layer)]  # add the outermost layer

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf*2, out_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, source):
        return self.model(source)


class PixelDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PixelDiscriminator, self).__init__()

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


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, in_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):

        super(NLayerDiscriminator, self).__init__()
        
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image
        # by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)



# ===================================
#           面向训练、测试的模型
# ===================================
class GAN(nn.Module):
    def __init__(self, train_opt=None, generator=GeneratorUNet(), discriminator=PixelDiscriminator(),
                 g_loss_func=torch.nn.L1Loss(), d_loss_func=torch.nn.MSELoss()):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        if train_opt['model_mode'] == 'train':
            self.optimizer_G = Adam_Optimizer(parameters=self.generator.parameters(), lr=train_opt['lrG'],
                                              betas=(train_opt['b1'], train_opt['b2']),
                                              freq=train_opt['lrG_d'] * train_opt['dataloader_length'])

            self.optimizer_D = Adam_Optimizer(parameters=self.discriminator.parameters(), lr=train_opt['lrD'],
                                              betas=(train_opt['b1'], train_opt['b2']),
                                              freq=train_opt['lrD_d'] * train_opt['dataloader_length'])
            self.train_opt = train_opt

        self.g_loss_func = g_loss_func
        self.d_loss_func = d_loss_func

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
        loss_sVg = self.d_loss_func(source_generate, valid)
        loss_pixel = self.g_loss_func(generate, target)
        loss_G = loss_sVg + self.train_opt['weight_pic'] * loss_pixel
        if if_G_backward:
            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()
            return loss_G, 0, loss_sVg, loss_pixel, 0
        # 分辨源图像和目标图像
        loss_real = self.d_loss_func(source_target, valid)
        # 分辨源图像和生成图像
        loss_fake = self.d_loss_func(source_generate2, invalid)
        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        if if_D_backward:
            self.optimizer_D.zero_grad()
            loss_D.backward()
            self.optimizer_D.step()
            return 0, loss_D, 0, 0, 0
        return loss_G, loss_D, loss_sVg, loss_pixel,loss_fake

    def step(self, source, target):
        generate, source_generate, source_target, source_generate2 = self(source, target)
        loss_G, _, loss_sVg, loss_pixel, _ = self.loss(generate, target, source_generate, source_target,
                                                    source_generate2, if_G_backward=True)
        _, loss_D, _, _, loss_fake = self.loss(generate, target, source_generate, source_target,
                                    source_generate2, if_D_backward=True)

        loss_dic = {'loss_G': loss_G.item(), 'loss_D': loss_D.item(), 'loss_pixel': loss_pixel.item(),
                    'loss_sVg': loss_sVg.item(),'loss_fake':loss_fake}
        return loss_dic


# 基于单个generator的图片生成器
class AutoEncoderGen(nn.Module):
    def __init__(self, train_opt=None, generator=AutoEncoder(), g_loss_func=torch.nn.L1Loss()):
        super(AutoEncoderGen, self).__init__()
        self.generator = generator
        if train_opt['model_mode'] == 'train':
            self.optimizer_G = Adam_Optimizer(parameters=self.generator.parameters(), lr=train_opt['lrG'],
                                              betas=(train_opt['b1'], train_opt['b2']),
                                              freq=train_opt['lrG_d'] * train_opt['dataloader_length'])
        self.g_loss_func = g_loss_func

    def loss(self, x, y):
        return self.g_loss_func(x, y)

    def forward(self, x):
        return self.generator(x)

    def step(self, x, y):
        generate = self(x)
        loss_pixel = self.loss(generate, y)
        self.optimizer_G.zero_grad()
        loss_pixel.backward()
        self.optimizer_G.step()
        loss_dic = {'loss_pixel': loss_pixel.item()}
        return loss_dic


# 傻瓜式生成器
class Dump(nn.Module):
    def __init__(self, generator=DumpGenerator()):
        super(Dump, self).__init__()
        self.generator = generator

    def loss(self, x, y):
        return self.generator.loss_fun(x, y)

    def forward(self, x):
        return self.generator(x)

    def step(self, x, y):
        loss_pixel = self.loss(self.generator(x), y)
        loss_dic = {'loss_pixel': loss_pixel.item()}
        return loss_dic


pass