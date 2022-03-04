[TOC]

## Font_pix_fast

### 数据批处理

原数据为400个大小为3\*128\*64的图片，图片左侧为京黑，右侧为黑体。

**ImageDataset**

将每批图片变成一个字典，其中‘A’表示京黑，‘B’表示黑体



### 损失函数

**lose function**

* `criterion_GAN`: 均方误差函数，用于计算判别器表现与正确值的均方误差
* `criterion_pixelwise`: L1误差。用于计算两张图片的L1误差

<font color='cornflowerblue'>我们要不要更改lose function？</font>



**loss的算法**

* 生成器loss
  $$
  loss_G = loss_{GAN} + \lambda_{pixel} * loss_{pixel}
  $$
  

  * $loss_{pixel}$：生成器生成的图片与新风格的误差
  * $loss_{GAN}$：判别器是否将生成图认成了原图。如果全为1则$loss_{GAN}$为

  * 其中$\lambda_{pixel}$为超参。



* 判别器loss
  $$
  loss_D = 0.5 * (loss_{real} + loss_{fake})
  $$
  

  * $loss_{real}$: 判别器是否认为原图和新风格图为同一张图
  * $loss_{fake}$:判别器是否将生成图认成了原图。不同于$loss_{GAN}$，若全为0则$loss_{fake}$为0









### 参数输入

使用argparse。考虑改成文件输入便于调参



### GeneratorUNet

生成器是一个表示模型，通过正反编码完成对图片的表征。在这里被当作了一个生成器，其学习的目标不再是尽可能保留原有特征，而是与新的风格尽可能相似



**层设置**





### Discriminator

判别器，输入两张图片，判断这两张图片是否是同一张图片。在本例中将生成器生成的图片与原图片进行比对，判断是否是同一张图片



```python
Discriminator(
  (model): Sequential(
    (0): Conv2d(6, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2, inplace=True)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (4): LeakyReLU(negative_slope=0.2, inplace=True)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (7): LeakyReLU(negative_slope=0.2, inplace=True)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (9): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
    (10): LeakyReLU(negative_slope=0.2, inplace=True)
    (11): ZeroPad2d(padding=(1, 0, 1, 0), value=0.0)
    (12): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
  )
)
```



















