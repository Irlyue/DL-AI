> Separable-Convolution
- 出发点：Convolutional Filters are Redundant

1. 普通的卷积

   ![](images/0-img0.png)

2. separable-conv

   ！[](images/0-img1.png)

简单地来说，就是对输入的每个通道独立使用DM(Depth Multiplier)个卷积核，这样可以得到in_channels\*DM个输出通道，这些个输出按通道连接在一起，然后在使用1\*1的卷积核得到out_channels的输出。

主要的好处是减少了参数的使用量，当然相应地会减少计算量，并且不会导致性能的明显变化（甚至有些微提升）。
- Converges in 20% fewer steps on ImageNet.
- Faster inference.
- Identical to slightly better final accuracy.
- Very easy to implement.
- No benefits on smaller tasks (e.g. Cifar10)

![](images/0-img2.png)

## Implementation in Popular Framework
- Tensorflow: [`tf.nn.separable_conv2d`](https://www.tensorflow.org/api_docs/python/tf/nn/separable_conv2d)
- PyTorch: [`torch.nn.Conv2d`](https://pytorch.org/docs/0.3.1/nn.html#torch.nn.Conv2d), see the description for the `groups` parameter.
