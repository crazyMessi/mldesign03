import os

ep = 100
lrG = 0.0001

# 初始化fitlog 只有在第一次运行的时候用
os.system('fitlog init')

os.system('python my_train.py --model_name AutoEncoderGen --ep %d --lrG %f' % (ep, lrG))
os.system('python my_train.py --model_name GAN --ep %d --lrG %f' % (ep, lrG))
os.system('python my_train.py --model_name pic2pic --ep %d --lrG %f' % (ep, lrG))

os.system('fitlog log logs')