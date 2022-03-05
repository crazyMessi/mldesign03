import myModel
from myModel import *

valid_model_name = ['GAN', 'AutoEncoder', 'pic2pic']


# 为网络参数赋正态分布的初值
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def model_selector(opt):
    if isinstance(opt, dict):
        model_name = opt['model_name']
    else:
        model_name = opt

    generator_list = {'UNet': GeneratorUNet(), 'AutoEncoder': AutoEncoder(), 'GAN': GeneratorUNet(if_crop=False)}
    discriminator_list = {'Discriminator': Discriminator()}

    while model_name not in valid_model_name:
        print('未输入正确模型名 请输入正确模型名\n')
        print(valid_model_name)
        model_name = input()

    if model_name == 'pic2pic':
        generator = generator_list['UNet']
        discriminator = discriminator_list['Discriminator']
        model = myModel.GAN(opt, generator=generator, discriminator=discriminator)

    if model_name == 'AutoEncoder':
        generator = generator_list['AutoEncoder']
        discriminator = discriminator_list['Discriminator']
        model = myModel.GAN(opt, generator=generator, discriminator=discriminator)

    if model_name == 'GAN':
        generator = generator_list['GAN']
        discriminator = discriminator_list['Discriminator']
        model = myModel.GAN(opt, generator=generator, discriminator=discriminator)

    return model
