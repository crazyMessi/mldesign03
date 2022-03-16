from myModel import *

valid_model_name = ['GAN', 'AutoEncoderGen', 'pic2pic']


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
        torch.nn.init.zeros_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.zeros_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# 根据opt选取模型
def model_selector(opt):
    if isinstance(opt, dict):
        model_name = opt['model_name']
    else:
        model_name = opt

    generator_list = {'UNet': GeneratorUNet(), 'GAN': GeneratorUNet(if_crop=False), 'AutoEncoder': AutoEncoder()}
    discriminator_list = {'Discriminator': Discriminator()}

    while model_name not in valid_model_name:
        print('未输入正确模型名 请输入正确模型名\n')
        print(valid_model_name)
        model_name = input()

    if model_name == 'AutoEncoderGen':
        model = AutoEncoderGen(train_opt=opt, generator=generator_list['AutoEncoder'])

    if model_name == 'GAN':
        generator = generator_list['GAN']
        discriminator = discriminator_list['Discriminator']
        model = GAN(train_opt=opt, generator=generator, discriminator=discriminator)

    if model_name == 'pic2pic':
        generator = generator_list['UNet']
        discriminator = discriminator_list['Discriminator']
        model = GAN(train_opt=opt, generator=generator, discriminator=discriminator)

    return model
