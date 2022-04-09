import os
eps = [1] #1表示g
lrGs = [0.002]
bss = [8]
autogen_name = ['resAutoEncoderGen']
g_loss_func = ['fixed_L1']
residual_learning = ['11111','11000']
ad_res = ['13463']


data_path = 'fontdata'
script_path = 'my_train.py'

for ep in eps:
    for n in autogen_name:
        for lrG in lrGs:
            for bs in bss:
                for lo in g_loss_func:
                    for res_l in residual_learning:
                        for ar in ad_res:
                            name = '%s_u%s_ad%s'%(n,res_l,ar)
                            os.system('python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" '
                                        '--g_loss_func %s --residual_unet %s --ad_res %s'
                                        % (script_path, name, ep, lrG, bs, data_path, lo,res_l,ar))