import os
eps = [1]
lrGs = [0.001]
bss = [8]
n_block = [6]
g_loss_func = ['fixed_L1']
crop_wight = [-0.99]
dropout = 0
n_downsampling = [5]
if_save = 0
ad_res = ['10101']
residual_unet = ['10101']

model_name = ['UResGen','ResGen']

data_path = 'fontdata'
script_path = 'my_train.py'

for ep in eps:
    for n in model_name:
        for lrG in lrGs:
            for bs in bss:
                for lo in g_loss_func:
                    for cpw in crop_wight:
                        for nb in n_block:
                            for nd in n_downsampling:
                                for ar in ad_res:
                                    for ru in residual_unet:
                                        fnb = nb + 2*(2-nd)# 调整block数量 每增加一个下采样层增加一定数量的block
                                        name = '%s_d%d_%d'%(n,nd,fnb)
                                        com = 'python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" --g_loss_func %s --dropout %d --n_block %d --if_save %d --ad_res %s --residual_unet %s --n_downsampling %d' % (script_path, name, ep, lrG, bs, data_path, lo, dropout, fnb, if_save, ar, ru, nd)                         
                                        os.system(com)



