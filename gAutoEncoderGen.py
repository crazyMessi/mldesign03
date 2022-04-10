import os
eps = [201] #1表示g
lrGs = [0.0001]
bss = [40]
autogen_name = ['AutoEncoderGen_origin']
g_loss_func = ['fixed_L1']
residual_learning = ['00000']
ad_res = ['00000']


data_path = 'fontdata'
script_path = 'my_train.py'

for ep in eps:
    for n in autogen_name:
        for lrG in lrGs:
            for bs in bss:
                for lo in g_loss_func:
                    for res_l in residual_learning:
                        for ar in ad_res:
                            name = '%s'%(n)
                            os.system('python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" '
                                        '--g_loss_func %s --residual_unet %s --ad_res %s'
                                        % (script_path, name, ep, lrG, bs, data_path, lo,res_l,ar))