import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ep = 200 #1表示g
lrGs = [0.0001]
bss = [80]
autogen_name = ['AutoEncoderGen_origin']
g_loss_func = ['fixed_L1']


data_path = 'fontdata'
script_path = 'my_train.py'

for n in autogen_name:
    for lrG in lrGs:
        for bs in bss:
            for lo in g_loss_func:
                name = '%s'%(n)
                os.system('python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" '
                            '--g_loss_func %s'
                            % (script_path, name, ep, lrG, bs, data_path, lo))

