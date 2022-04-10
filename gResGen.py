import os
eps = [209]
lrGs = [0.001,0.0005]
bss = [4,8]
g_loss_func = ['fixed_L1']
crop_wight = [-0.99]
dropout = 1
if_save = 0
ad_res = ['00000']
residual_unet = ['11111']
data_path = 'fontdata'
script_path = 'my_train.py'

model_name = ['ResGen']
n_downsampling = [2]
n_block = [22,34]



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
                                        fnb = 2*(2-nd) + nb
                                        name = '%s_d%d_%d'%(n,nd,fnb)
                                        com = 'python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" --g_loss_func %s --dropout %d --n_block %d --if_save %d --ad_res %s --residual_unet %s --n_downsampling %d' % (script_path, name, ep, lrG, bs, data_path, lo, dropout, fnb, if_save, ar, ru, nd)                         
                                        os.system(com)



