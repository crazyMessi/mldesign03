import os
ep = 1
lrGs = [0.001]
bss = [8]
n_block = [6,6,6]
model_name = ['UResGen']
g_loss_func = ['fixed_L1']
crop_wight = [-0.99,0,0.99]

data_path = 'fontdata'
script_path = 'my_train.py'

for n in model_name:
    for lrG in lrGs:
        for bs in bss:
            for lo in g_loss_func:
                for cpw in crop_wight:
                    name = '%s_%d'%(n,cpw)
                    os.system('python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" '
                                '--g_loss_func %s --crop_weight %d'
                                % (script_path, name, ep, lrG, bs, data_path, lo,cpw))

