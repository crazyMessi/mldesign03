from myModel import *

valid_model_name = ['AutoEncoderGen', 'UNetGen', 'ResGen', 'UResGen', 'GAN', 'pic2pic', 'ResGAN', 'UResGAN']


# 修正的生成器loss
class fixed_loss_G(nn.Module):
    def __init__(self):
        super(fixed_loss_G, self).__init__()
        self.loss_G = torch.nn.L1Loss()
        return

    def forward(self, x, y):
        loss_G = self.loss_G(1 - x, 1 - y)
        return loss_G / (torch.mean(1 - y))

MSE = torch.nn.MSELoss()


# 为网络参数赋正态分布的初值
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# # 为网络参数赋零初值
def weights_init_zero(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.zeros_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.zeros_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


# def weights_init_one(m):
#     torch.nn.init.ones_

# 根据opt选取模型
def model_selector(opt):
    model_name = opt['model_name']
    g_loss_func_list = {'L1': torch.nn.L1Loss(), 'fixed_L1': fixed_loss_G(), 'MSE': torch.nn.MSELoss()}
    g_loss_func = g_loss_func_list[opt['g_loss_func']]
    dropout = 0.5 if opt['dropout'] > 0 else 0

    generator_list = {'AutoEncoder': AutoEncoder(dropout_rate=dropout, in_channels=opt['channels'],
                                                 out_channels=opt['channels']),
                      'ResGenerator': ResnetGenerator(dropout_rate=dropout, in_channels=opt['channels'],n_downsampling=opt['n_downsampling'],
                                                      out_channels=opt['channels'], n_blocks=opt['n_block']),
                      'UNet': GeneratorUNet(dropout_rate=dropout, in_channels=opt['channels'],
                                            out_channels=opt['channels'], crop_weight= opt['crop_weight']),
                      'UResGen': UResGen(dropout_rate=dropout, in_channels=opt['channels'], n_blocks=opt['n_block'],
                                            out_channels=opt['channels'], crop_weight= opt['crop_weight']),
                      'Dump': DumpGenerator()
                      }
    discriminator_list = {'pixel': PixelDiscriminator(in_channels=opt['channels']),'patch':NLayerDiscriminator(in_channels=opt['channels']*2)}

    founded = False
    while not founded:
        for n in valid_model_name:
            founded = founded | (model_name.find(n) >= 0)
        if not founded:
            print('未输入正确模型名 请输入正确模型名\n')
            print(valid_model_name)
            model_name = input()

    discriminator = generator = []
    if model_name.find('AutoEncoderGen') >= 0:
        generator = generator_list['AutoEncoder']

    if model_name.find('GAN') >= 0:
        generator = generator_list['AutoEncoder']
        discriminator = discriminator_list[opt['discriminator']]

    if model_name.find('UNet') >= 0:
        generator = generator_list['UNet']

    if model_name.find('pic2pic') >= 0:
        generator = generator_list['UNet']
        discriminator = discriminator_list[opt['discriminator']]

    if model_name.find('ResGen') >= 0:
        if model_name.find('UResGen') >= 0:
            generator = generator_list['UResGen']        
        else:
            generator = generator_list['ResGenerator']

    if model_name.find('ResGAN') >= 0:
        if model_name.find('UResGAN') >= 0:
            generator = generator_list['UResGen']        
        else:
            generator = generator_list['ResGenerator']
        discriminator = discriminator_list[opt['discriminator']]

    if not discriminator and generator:
        # 这是一个自编码器
        model = AutoEncoderGen(train_opt=opt, generator=generator, g_loss_func=g_loss_func)
    else:
        # 这是一个对抗神经网络
        model = GAN(train_opt=opt, generator=generator, g_loss_func=g_loss_func, discriminator=discriminator)

    # if model_name == 'Dump':
    #     model = Dump(generator_list['Dump'])

    return model

