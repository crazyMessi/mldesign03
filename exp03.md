[TOC]



### 注意事项



**网络层可视化**



[简易版教程](https://zhuanlan.zhihu.com/p/220403674)



[或者使用tensorboard](https://zhuanlan.zhihu.com/p/58961505)



**代码注释**



读代码的时候注意多写注释 尤其是关键函数和一些比较难理解的函数



**结果分析**



重点回答任务书里面的几个问题



- 分析并列出影响各个模型实验结果的主要原因
- 列出不同算法的优劣 
  - FLOPs与时间复杂度
  - 对超参的敏感程度
  - 对数据量的需求





### 数据格式

由于这次字体迁移任务中的源图像和目标图像均可用灰度表示。base_code中使用了三通道，相比单通道输入输出的模型，由于值空间更大了，训练难度很可能也会更大。我们在验证了这一点后，决定采取单通道的输入输出。



### 网络单元

**Encoder**

下采样层。

首先，使用步长为2的卷积层对图像进行步长为2的卷积运算。每次卷积后的图像通道数翻倍，大小减半。

这实际上是普通的卷积过程的简化版。可以考虑换成步长为1的卷积+一次池化。

之后按情况对数据进行InstanceNorm操作，我们了解到这个操作能使得每个batch的分布合理。

最后使用Dropout丢弃一部分输出（置零），达到正则化网络层的效果。我们验证了这一操作带来的效果



**Decoder**

上采样层。

首先是步长为2的转置卷积层，每次转置卷积后图像通道数减半，大小翻倍

此后的操作同样是归一化、Dropout正则化



**UpDown**

**Resblock**



### AutoEncoderGen

#### 背景

自编码器（AutoEncoder）是**表示学习**的经典算法，本是一种无监督学习模型。其由一个编码器和一个解码器组成。前者将数据转化为一种不同的表现形式，后者将这个新的表示转化为原来的形式，转化的目的是让新的表示尽可能留有原来特性的同时，拥有各种好的特性。

我们将AutoEncoder应用于字体迁移任务中，构建了以此为基础的监督学习模型AutoEncoderGen。这一应用具有一定的合理性：目标字体和源字体整体变动不大，如果将目标字体理解为源字体的一种表示，在loss函数的引导下，生成器表示的源字体将朝目标字体靠近

#### 原理分析

**模型架构**

AutoEncoder模型对于输入的图片，进行了如下操作：

连续若干次Encoder操作，得到一个向量；其中有选择地在几个阶段的Encoder时进行概率为0.5dropout处理，以实现对网络层的正则化。我们了解到dropout带来的效果可以用集成学习来解释

之后，再连续若干次Decoder，重新得到一个长宽为原来两倍的矩阵；最后的final层通过下采样将通道数、图像大小变回原样，再经过一个非线性的输出层（例如Tanh函数）得到最终结果。

![img](https://cdn.nlark.com/yuque/0/2022/png/26738067/1647400791944-e1181fc7-f98b-4ccd-b967-64dc20ebc391.png)



**AutoEncoderGen**

AutoEncoderGen即基于单个生成器的有监督生成模型。我们遵循任务书，将pix2pix的上采样信息补充（skip connection）以及判别器（Discriminator）删去之后，就得到了一种最简单的单loss函数AutoEncoderGen，我们称其为basic AutoEncoder。显然这不是AutoEncoderGen的唯一形式——除此以外，我们还分别尝试了基于UNet、Resnet的AutoEncoderGen，试图更进一步地探究生成器能够做什么。其中UNet即Pix2pix所使用的生成器，Resnet则为用resblock构建正反编码器的AutoEncoder。





#### loss函数

loss函数是引导生成器“重表示”图像的关键。图像领域中L1误差较为常见。我们认为loss函数应该考虑如下几个因素：

* 图片灰度值两极分化（归一化后多为0和1），并且1的数量远大于0（图片以白为底）
* 字体迁移任务本身对结果的整体性要求较高
* 汉字的字体迁移具有一定的特殊性：不同的汉字之间的笔画数的差异可能很大，因此不同汉字与空白图像的L1距离的差异也会比较大。



**L1 loss**

$L1 \ loss$在图像处理领域非常常见，也是最简单的一类loss。`torch.nn.L1_loss`公式如下：
$$
L1\ {loss}(x,y) = \frac{\sum_n|x_i - y_i|}{n}
$$
我们认为其在汉字的字体迁移任务中可能存在不合理之处：

* 在大部分情况下，$mean(y)\approx 1$，在模型较差的情况下，生成器得到的$L1 \ loss$会小于空白图像。实际上我们在调整AutoEncoder的参数时，在学习率较高时，得到了一些输出空白图像的模型。
* loss值受笔划数影响较大



**fixed L1 loss**

针对L1 loss可能存在的局限性，我们尝试了修正的loss函数的计算方式$\text{fixed} \ L_1 \ loss$:
$$
\text{fixed} \ L1\ loss = \frac{L1\ loss(x,y)}{\text{mean} |1-y|+\alpha}
$$
其中$\alpha$是防止除以零错误的极小量。由于我们的输入数据中没有空白图像，$\alpha$ 设为0即可。$\text{fixed} \ L_1 \ loss$对于任何空白的输出的结果都为1；此外，该$\text{fixed} \ L_1 \ loss$下笔划较少的汉字的loss更敏感，我们期待它能够平衡不同汉字的loss。



#### dropout

理论上一个容量足够的自编码器的表示能力是很强的，因此我们的模型中有较多的Dropout操作防止矩阵过拟合。为了验证Dropout的效果，我们将尝试除去模型中的Dropout层。在我们的预期中，模型的train_loss将会下降，而test_loss将会上升。



#### 超参

我们调参顺序如下

* lrG 生成器学习率
* bs batch_size
* lrG_d 生成器学习率衰减率
* 优化器的b1、b2不作调整



### 运行结果

**最简单的AutoEncoderGen**

AutoEncoder在参数合适的情况下能够在训练集上取得较好的成效，但它的test_loss却持续处于较高的值，但在大量的参数尝试下，AutoEncoder的test_loss依旧无法收敛，并一致持续在较高的值。

![img](https://cdn.nlark.com/yuque/0/2022/png/26738067/1647829444726-63b95746-b2da-4174-9055-65232d2b0cea.png)

如表所示，lr=0.0001，bs=8；lr=0.0005，bs=40时，训练期间的loss均可下降到较低的值；但test_loss几乎都维持在1以上。这表明模型在测试集上几乎毫无作为。

![image-20220325141604632](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325141604632.png)

我们还对使用dropout层（dropout=1）与不使用dropout层（dropout=0）的模型进行了对比（表中取了各指标下的均值）。使用dropout的生成器的train_loss明显高于不使用dropout的loss。这是符合预期的，但遗憾的是，其test_loss依旧没有降低。

basic AutoEncoderGen在训练集上的表现十分不错。

<img src="E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325143835961.png" alt="image-20220325143835961" style="zoom:25%;" />

lrG=0.001，bs=40，在ep=80时生成的图像就已经真假难辨。但测试集的效果却很差

* basic AutoEncoderGen最好的一期，test_loss = 0.98(fixed_loss)

  ![image-20220325143246137](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325143246137.png)

* dropout=1最好的一期，test_loss = 0.95 (fixed_loss)

  ![image-20220325143140248](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325143140248.png)

  

可以看到，虽然从loss上看后者有所下降，但其已经看不出字型了。训练集之所以能完美复原，应该是因为12个卷积层产生了足够的函数空间模型在训练集上有严重的过拟合。下图很好地说明了这样的过拟合现象：

![image-20220325154829247](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325154829247.png)

这个字的绞丝旁被成功迁移，但“四”却与目标、原字体完全不一致。

并且这种过拟合难以用正则化解决，这是模型或许本身的局限性。我们猜测这是在下采样过程中，一个二维矩阵被压缩为一个向量，其中的空间信息在压缩期间丢失了，因而上采样是无法复原其空间信息的。

很自然地想到。如果能够避免采样期间的压缩，或者在上采样期间找回原来的空间信息，那么空间信息的丢失或许就能得以缓解。

UNet试图缓解下采样中空间信息的丢失：在向量通过第k个上采样层后，将输出向量使用了第n-k次下采样时得到的向量进行拼接，也就是跳层连接（skip connection）。我们将UNet作为生成器得到了UNetGen（Gen表示Generator，下同）。我们还尝试了使用输入前后大小不变的Resblock作为编解码器的生成器ResGen。



**UNetGenerator&ResGenerator**

新的生成器带来的效果提升是显著的，其中在我们调整了lrG、bs、dropout层买得到了一共2*30个超参组合，其中每个参数组合固定训练1001个epoch；将这60个模型的test_loss（取最小值）从小到大排列得到一个榜单，ResGen以压倒性的优势占据了这份榜单的头部；UNet也以绝对的优势领先了basic AutoEncoder。其中ResGen的fixed_L1_loss已经超过后面的很多pix2pix

![image-20220325160527986](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325160527986.png)

![image-20220325161320548](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325161320548.png)

ResGen，lr=0.001，bs=8，不使用dropout操作

![image-20220325161551086](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325161551086.png)

UNet，lr=0.002，bs=20，使用了dropout

**结论**

* 模型对学习率较为敏感，对dropout不敏感

  









## GAN

### 模型架构



尝试可视化一下整体架构(这里不是指网络层 而是两个模型相互的关系)



**AutoEncoder**



基于AutoEncoder 但loss函数不一样 (为什么要这么改)



**Discriminator**



loss计算

 

是怎么想到loss的



### 原理分析



为什么有用(尝试从理论的角度)



### 运行结果



对比AutoEncoder



## Pix2pix



### 模型架构



skip connection是怎么做的

 生成器G用的是Unet结构。

 跳层连接是 ![img](https://cdn.nlark.com/yuque/0/2022/svg/22408135/1647822950580-312bd254-f2d1-4733-a153-923631ee7d2f.svg) 层直接与 ![img](https://cdn.nlark.com/yuque/0/2022/svg/22408135/1647822950567-f664eec1-81da-4567-b512-7cc8f4614da7.svg) 层相加

 

### 原理分析



为什么要skip connection



### 运行结果



### 尝试



- 对结果进行适当的低级图像处理，比如一些滤波操作
- 把图片通道数量改成1。因为我们**主要关注字体变形，不关注颜色**。
- 在原有模型上再加一个unet（和原来模型不一起训练），因为目前的pix2pix形状已经比较形似了，可以尝试单独训练一个unet让成像更清晰的模型。即**一个模型学习形态的变化，另一个模型对结果进行降噪处理**
- 修改网络层的设置。 
- - 当前的训练loss似乎还有降低的空间，或许可以让网络层更深一点?
  - **在网络尾部增加一个softmax**。因为我们看了一下经过生成器前后的图片的灰度直方图，发现映射后的图片的直方图多了很多”杂灰度“，即多了一些半黑不白的地方，而这类灰度是原图片以及目标图片所没有的
- 我们查阅了一下相关资料，很多人说Unet的skip connection的作用主要是在上采样时还原图片的一些空间特征。据了解，resnet中也有一个skip connection，用于缓解网络过深导致的优化困难。这两个skip connection之间存在联系吗？我们的Unet有12层，**是否也需要引进残差学习**?
- 使用cycleGAN
- 损失函数



## cycleGAN
