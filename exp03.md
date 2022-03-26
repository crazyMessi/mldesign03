[toc]

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

## AutoEncoderGen

### 背景

自编码器（AutoEncoder）是**表示学习**的经典算法，本是一种无监督学习模型。其由一个编码器和一个解码器组成。前者将数据转化为一种不同的表现形式，后者将这个新的表示转化为原来的形式，转化的目的是让新的表示尽可能留有原来特性的同时，拥有各种好的特性。

我们将AutoEncoder应用于字体迁移任务中，构建了以此为基础的监督学习模型AutoEncoderGen。这一应用具有一定的合理性：目标字体和源字体整体变动不大，如果将目标字体理解为源字体的一种表示，在loss函数的引导下，生成器表示的源字体将朝目标字体靠近

#### **模型架构**

**AutoEncoder**

一个典型的AutoEncoder模型将对输入的图片进行如下操作：

连续若干次Encoder操作，得到一个向量；其中有选择地在几个阶段的Encoder时进行概率为0.5dropout处理，以实现对网络层的正则化。我们了解到dropout带来的效果可以用集成学习来解释

之后，再连续若干次Decoder，重新得到一个长宽为原来两倍的矩阵；最后的final层通过下采样将通道数、图像大小变回原样，再经过一个非线性的输出层（例如Tanh函数）得到最终结果。

![img](https://cdn.nlark.com/yuque/0/2022/png/26738067/1647400791944-e1181fc7-f98b-4ccd-b967-64dc20ebc391.png)

**basic AutoEncoderGen**

AutoEncoderGen即基于单个生成器的有监督生成模型。我们遵循任务书，将pix2pix的上采样信息补充（skip connection）以及判别器（Discriminator）删去之后，就得到了一种最简单的单loss函数AutoEncoderGen，我们称其为basic AutoEncoder。显然这不是AutoEncoderGen的唯一形式——除此以外，我们还分别尝试了基于UNet、Resnet的AutoEncoderGen，试图更进一步地探究生成器能够做什么。其中UNet即Pix2pix所使用的生成器，Resnet则为用resblock构建正反编码器的AutoEncoder。



**UNetGen**

UNet的深度和basic AutoEncoderGen一致，但增加了skip connection。



**ResGen**

ResGen使用的生成器也有Encoder、Decoder的过程，但其只有两层下采样层和两层上采样层；而采样前后各有一个核大小为7的大卷积层。采样层中间中间有6个res block，以保持总卷积层数与UNet一致。我们后续会对res block的数量进行探究





#### 感受野

我们在修改网络层层数、核大小时，以底层感受野不变作为约束。记第n层网络感受野为$RF_n$
$$
r_0 = \sum_{l=1}^L\bigg((k_l-1)\prod_{i=1}^{l-1}s_i\bigg)+1
$$
感受野的计算很简单，但影响网络层效果的重要因素还包括特征层的深度，感受野也不一定是有效的。这里只考虑了感受野的大小。

从pix2pix简化出来的自编码器在下采样底层处感受野为**192**。由于上采样主要工作是对图像进行复原，因此只考虑底层处的感受野。



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



#### 超参

我们调参顺序如下

* lrG 生成器学习率

* bs batch_size

* dropout

  理论上一个容量足够的自编码器的表示能力是很强的，因此我们的模型中有较多的Dropout操作防止矩阵过拟合。为了验证Dropout的效果，我们将尝试除去模型中的Dropout层。在我们的预期中，模型的train_loss将会下降，而test_loss将会上升。

* lrG_d 生成器学习率衰减率





### 模型调整 period1

仅仅在生成器模型中就用大量的参数需要调节。刚开始时，我们采用的是**网格调参**；但随着我们研究的推进，需要手动调整的参数、模型架构越来越多，参数的组合呈指数级增长。**随机调参**是代替网格调参的一种方式，但我们意识到我们并不是在寻找一个模型的最优参数。因此，我们决定先在基本模型下确定几个主要参数，再在此参数下对模型进行探索。

#### basic AutoEncoderGen

AutoEncoder在参数合适的情况下能够在训练集上取得较好的成效，但它的test_loss却持续处于较高的值，但在大量的参数尝试下，AutoEncoder的test_loss依旧无法收敛，并一致持续在较高的值。

![img](https://cdn.nlark.com/yuque/0/2022/png/26738067/1647829444726-63b95746-b2da-4174-9055-65232d2b0cea.png)

如表所示，lr=0.0001，bs=8；lr=0.0005，bs=40时，训练期间的loss均可下降到较低的值；但test_loss几乎都维持在1以上。这表明模型在测试集上几乎毫无作为。

![image-20220325141604632](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325141604632.png)

我们还对使用dropout层（dropout=1）与不使用dropout层（dropout=0）的模型进行了对比（表中取了各指标下的均值）。使用dropout的生成器的train_loss明显高于不使用dropout的loss。这是符合预期的，但遗憾的是，其test_loss依旧没有降低。

basic AutoEncoderGen在训练集上的表现十分不错，生成图像和目标图像基本一致

lrG=0.0001，bs=40，在ep=80时生成的图像就已经真假难辨。但测试集的效果却很差

* basic AutoEncoderGen最好的一期，test_loss = 0.98(fixed_loss)

  ![image-20220325143246137](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325143246137.png)
* dropout=1最好的一期，test_loss = 0.95 (fixed_loss)

  ![image-20220325143140248](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325143140248.png)

可以看到，虽然从loss上看后者有所下降，但其已经看不出字型了。训练集之所以能完美复原，应该是因为12个卷积层产生了足够的函数空间模型在训练集上有严重的过拟合。下图很好地说明了这样的过拟合现象：

![image-20220325154829247](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325154829247.png)

这个字的绞丝旁被成功迁移，但“四”却与目标、原字体完全不一致。

并且这种过拟合难以用正则化解决，这是模型或许本身的局限性。我们猜测这是在下采样过程中，一个二维矩阵被压缩为一个向量，其中的空间信息在压缩期间丢失了，因而上采样是无法复原其空间信息的。

很自然地想到。如果能够避免采样期间的压缩，或者在上采样期间找回原来的空间信息，那么空间信息的丢失或许就能得以缓解。

UNet试图使用信息补充的方式缓解下采样中空间信息的丢失：在向量通过第k个上采样层后，将输出向量使用了第n-k次下采样时得到的向量进行拼接，也就是跳层连接（skip connection）。我们将UNet作为生成器得到了UNetGen。我们还尝试了使用输入前后大小不变的Resblock作为编解码器的生成器ResGen。ResGen在最开始时使用两次stride为2的下采样，通过一定数量的res block后再通过两次同样步长的上采样得到输出。



**结果概览**

* lrG = 0.0001，影响较大

* bs 影响不大

* dropout 不使用dropout

  



#### UNetGen&ResGen

新的生成器带来的效果提升是显著的，其中在我们调整了lrG、bs、dropout层买得到了一共2*30个超参组合，其中每个参数组合固定训练1001个epoch；将这60个模型的test_loss（取最小值）从小到大排列得到一个榜单，ResGen以压倒性的优势占据了这份榜单的头部；UNet也以绝对的优势领先了basic AutoEncoder。值得一提的是，ResGen的fixed_L1_loss已经超过大多数的pix2pix

![image-20220325160527986](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325160527986.png)

![image-20220325161320548](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325161320548.png)

ResGen，lr=0.001，bs=8，不使用dropout操作

![image-20220325161551086](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220325161551086.png)

UNet，lr=0.001，bs=20，使用了dropout

![image-20220326154801991](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326154801991.png)

如图，前100个epochUNet和ResGen的走势对比。其中绿色为UNet，蓝色为ResNet





**结论**

* lrG UNet和ResGen都在lrG = 0.001时表现最好。
* bs UNet取bs = 20 ResGen取bs = 8
* ep ep=200 模型即已经很好地收敛了

* dropout

  * UNetGen中的dropout操作对生成器有略微的提升效果；此外，学习率不合适（lrG=0.01）时dropout能为模型“止损”，即**dropout下UNet对于学习率会更鲁棒一些**。

  ![image-20220326141555718](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326141555718.png)

  ​	

  <img src="E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326142716751.png" alt="image-20220326142716751" style="zoom:50%;" />

  <img src="E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326142914494.png" alt="image-20220326142914494" style="zoom:50%;" />	

  <img src="E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326143009518.png" alt="image-20220326143009518" style="zoom:50%;" />

   	  如图，橙线即加入了dropout层；灰色线为未加dropout

  * ResGen

    我们得到的最好的ResGen模型是不使用dropout的，但领先幅度较小。

    ![image-20220326144619573](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326144619573.png)

    ResGen的dropout层同样能够起到**增强参数鲁棒性**的作用。

  综合考量，我们决定在后续探究中使用dropout

* loss_fun

  loss_fun对生成图片的影响不大，但fixed_L1_loss计算的loss在数值具有更大的变化范围，故我们后续主要采用fixed_L1_loss

  

### 模型调整 period2

#### 下采样

不论是指标上还是效果上，ResGen都较明显地优于UNetGen；而二者又远远优于basic AutoEncoder。三者最大的差别在于卷积核步长以及卷积核大小。

UNet希望能够既保持较大的感受野，又可以避免大卷积核或者深层网络带来的大量参数，因此采用了连续的几个stride = 2 的卷积层对图像进行采样，并在上采样复原时用skip connection来缓解分辨率降低导致的信息丢失。这确实起到了一定作用：模型在测试集上能够生成较清晰的简单图像了。但在笔划较密集处，依旧会生成一片糊状区域。此前我们提到过，一个6次kenerl为4、stride为2的下采样的UNet，具有大小为192的感受野，其感知面积是是图像大小的16倍，这意味感受野内很多信息是来自零填充的。这样大的感受野是有必要的吗？

首先我们尝试减少采样了一次采样次数，此时UNet底层感受野变成了94；又尝试了四次下采样，UNet感受野变成了46，已经小于64。

<img src="E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326170335419.png" alt="image-20220326170335419" style="zoom:50%;" />

![image-20220326193703187](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326193703187.png)





从loss上看模型五次或六次下采样的模型基本一致；但只使用四次下采样会导致模型性能有一定下降。

<img src="E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326194105899.png" alt="image-20220326194105899" style="zoom:150%;" />



上图，使用4次下采样

<img src="E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326194131316.png" alt="image-20220326194131316"  />

使用5次下采样

<img src="E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326194226189.png" alt="image-20220326194226189"  />

使用6次下采样



需要注意的是，只使用四次下采样的模型无法应对像“勹”这样的字体，我们推测其有效感受野无法覆盖整个字型。

在意识到这个问题以后，我们重新审视了表现较好的ResGen。

在此前的分析中，ResNet对于密集笔划区域的迁移表现更好，很可能是因为其下采样层较浅、分辨率损失较小；但我们忽略了“勹”这类字的特殊情况：ResGen的“勹”字和4层下采样层的UNet产生的“勹”字具有十分相似的问题！

<img src="E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326204004343.png" alt="image-20220326204004343" style="zoom: 200%;" />

经过计算，ResGen最后一个res_block的感受野为109，要大于UNet五次下采样后的感受野（94），理论上，似乎是不应该出现这样的情况的。

记过思考后，我们发现我们忽略了一个问题，即我们的先验是：ResGen也是一个对称的AutoEncoder，只是其编码、解码层有一部分被替换成了res block。因此我们比较编码器底层感受野大小时，应当使用第三个res block的感受野，而不是最后一个。而**第三个res block的感受野为61**，略小于图片大小。值得强调的是，堆叠6个res block的ResGen和应用了6个下采样层的UNet的卷积层数是一致的，实际上我们允许的时间也是一致的。而如果希望通过只增加res block的方式，将ResGen的感受野扩大到与UNet相近，那么我们至少需要堆叠22（或24）个res block，此时总的卷积层数为28（30），第11（12）个block的感受野大小为189（205）。

















## GAN

![image-20220326140633970](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326140633970.png)

![image-20220326141233903](E:\ZHUODONG LI\Work\AI\mldesign03\exp03.assets\image-20220326141233903.png)



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
