"""
Please refer to blog
https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
for detailed information.
"""
import math

from collections import namedtuple


class Kernel(namedtuple('Kernel', 'k s p name')):
    def __repr__(self):
        return '{}(kernel size={:<2}, stride={}, padding={}, name={})'.format(self.__class__.__name__, *self)


class LayerInfo(namedtuple('LayerInfo', 'n j r start')):
    """
    Information about the output of one layer, including the spatial dimension `n`, jump `j`, receptive field `r` and
    the starting position of the upper left feature `start`. The "jump" is actually the stride distance in the input
    image and it's usually the power of 2, say 2, 4, 8, etc.

    Supposed the output tensor of this layer is X with shape(batch_size, height, width, n_channels), then the upper
    left feature or the first feature as mentioned in the blog is X[:, 0, 0, :]. The starting position of one specific
    feature value is the center location of the receptive field.
    The starting position may or may not move, depending on the padding strategy and the kernel size. Take VGG for
    example, the starting position moves due to the pooling layers and fc layers. Yet in ResNet, the starting
    position remains unchanged.
    """
    def __repr__(self):
        return '{}(size={:<3}, jump={:<3}, receptive field={:<3}, start={:<5})'.format(self.__class__.__name__, *self)


class Layer:
    def __init__(self, info=None, kernel=None):
        self.info = info
        self.kernel = kernel

    def __repr__(self):
        return '{} {}'.format(self.info, 'Input layer' if self.kernel is None else self.kernel)


class Net:
    def __init__(self, kernels, name='Net'):
        self.name = name
        self.kernels = kernels
        self.layers = []

    def display_network(self, input_layer=None):
        input_layer = input_layer or Layer(info=LayerInfo(224, 1, 1, 0.5))
        layers = [input_layer]
        for kernel in self.kernels:
            layer_info = Net.out_from_in(kernel, layers[-1].info)
            layers.append(Layer(layer_info, kernel))
        self.layers = layers
        print(self)

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        fmt = '{}\n' * len(self)
        return '{}{}{}\n'.format('*'*10, self.name, '*'*10) + fmt.format(*self.layers)

    @staticmethod
    def out_from_in(kernel, layer):
        n = math.floor((layer.n + 2*kernel.p - kernel.k) / kernel.s) + 1
        j = layer.j * kernel.s
        r = layer.r + (kernel.k - 1) * layer.j
        start = layer.start + ((kernel.k - 1)/2 - kernel.p) * layer.j
        return LayerInfo(n, j, r, start)


