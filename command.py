import os

ep = 100
lrG = 0.0001

os.system('python my_train.py --model_name AutoEncoderGen --ep %d --lrG %f' % (ep, lrG))
os.system('python my_train.py --model_name GAN --ep %d --lrG %f' % (ep, lrG))
os.system('python my_train.py --model_name pic2pic --ep %d --lrG %f' % (ep, lrG))
