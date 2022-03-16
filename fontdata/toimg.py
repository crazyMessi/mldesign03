import numpy as np
from imageio import imwrite
import os

os.makedirs('imgs/trainA', exist_ok=True)
os.makedirs('imgs/trainB', exist_ok=True)
val = np.load('val/val.npy')
print(val.shape)
imwrite('1.png',val[0,:,:64,1])


for i in range(len(val)):
    imwrite('imgs/trainA/val_%dA.png'%i,val[i,:,64:,1])
    imwrite('imgs/trainB/val_%dB.png'%i,val[i,:,:64,1])
