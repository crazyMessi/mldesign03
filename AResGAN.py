import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ep = 202
lrGs = [0.001]
bss = [8]
lrD_rate_s = [1]
#autogen_name = ['AutoEncoderGen', 'UNetGen', 'ResGen']
gan_name = ['UResGAN']
g_loss_func = ['fixed_L1']
dropout = [1]
channels = [1]
wight_pic = [10]


data_path = 'fontdata'
script_path = 'my_train.py'

for n in gan_name:
    for lrG in lrGs:
        for bs in bss:
            for d in dropout:
                for c in channels:
                    for lo in g_loss_func:
                        for lamb in wight_pic:
                            name = '%s'%(n)
                            os.system('python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" '
                                    '--g_loss_func %s --dropout %d --channels %d --weight_pic %d'
                                    % (script_path, name, ep, lrG, bs, data_path, lo, d, c, lamb))
