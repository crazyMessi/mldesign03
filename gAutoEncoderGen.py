import os
eps = [203] #1表示g
lrGs = [0.002]
bss = [8]
autogen_name = ['resAutoEncoderGen']
g_loss_func = ['fixed_L1']


data_path = 'fontdata'
script_path = 'my_train.py'

for ep in eps:
    for n in autogen_name:
        for lrG in lrGs:
            for bs in bss:
                for lo in g_loss_func:
                    name = '%s'%(n)
                    os.system('python \"%s\" --model_name %s --ep %d --lrG %f --bs %d --data_path \"%s\" '
                                '--g_loss_func %s'
                                % (script_path, name, ep, lrG, bs, data_path, lo))

