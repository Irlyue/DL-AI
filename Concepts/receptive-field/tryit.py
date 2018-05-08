from calc_receptive_field import Net, Kernel, LayerInfo, Layer


def test_alexnet():
    convnet = [
        Kernel(11, 4, 0, 'conv1'),
        Kernel(3, 2, 0, 'pool1'),
        Kernel(5, 1, 2, 'conv2'),
        Kernel(3, 2, 0, 'pool2'),
        Kernel(3, 1, 1, 'conv3'),
        Kernel(3, 1, 1, 'conv4'),
        Kernel(3, 1, 1, 'conv5'),
        Kernel(3, 2, 0, 'pool5'),
        Kernel(6, 1, 0, 'fc6'),
        Kernel(1, 1, 0, 'fc7')
    ]
    net = Net(convnet, name='alexnet')
    net.display_network(Layer(info=LayerInfo(227, 1, 1, 0.5)))


def test_vggnet():
    convnet = [
        Kernel(3, 1, 1, 'conv1_1'),
        Kernel(3, 1, 1, 'conv1_2'),
        Kernel(2, 2, 0, 'pool1'),
        Kernel(3, 1, 1, 'conv2_1'),
        Kernel(3, 1, 1, 'conv2_2'),
        Kernel(2, 2, 0, 'pool1'),
        Kernel(3, 1, 1, 'conv3_1'),
        Kernel(3, 1, 1, 'conv3_2'),
        Kernel(3, 1, 1, 'conv3_3'),
        Kernel(2, 2, 0, 'pool3'),
        Kernel(3, 1, 1, 'conv4_1'),
        Kernel(3, 1, 1, 'conv4_2'),
        Kernel(3, 1, 1, 'conv4_3'),
        Kernel(2, 2, 0, 'pool4'),
        Kernel(3, 1, 1, 'conv5_1'),
        Kernel(3, 1, 1, 'conv5_2'),
        Kernel(3, 1, 1, 'conv5_3'),
        Kernel(2, 2, 0, 'pool5'),
        Kernel(7, 1, 0, 'fc6'),
        Kernel(1, 1, 0, 'fc7'),

    ]
    net = Net(convnet, name='VGG16')
    net.display_network()


def test_resnet():
    def _bottleneck(name, stride=1):
        kernels = [
            Kernel(1, 1, 0, name + '/' + 'conv1'),
            Kernel(3, stride, 1, name + '/' + 'conv2'),
            Kernel(1, 1, 0, name + '/' + 'conv3')
        ]
        return kernels

    def _resnet_block(n_uints, name, first_stride=2):
        kernels = []
        for i in range(n_uints):
            kernels.extend(_bottleneck(name + '/' + 'unit' + str(i+1), stride=first_stride if i == 0 else 1))
        return kernels

    convnets = [
        Kernel(7, 2, 3, 'conv1'),
        Kernel(3, 2, 1, 'pool1')
    ]
    convnets.extend(_resnet_block(3, 'block1', first_stride=1))
    convnets.extend(_resnet_block(4, 'block2'))
    convnets.extend(_resnet_block(6, 'block3'))
    convnets.extend(_resnet_block(3, 'block4'))
    net = Net(convnets, name='ResNet-50')
    net.display_network()


if __name__ == '__main__':
    test_alexnet()
    test_vggnet()
    test_resnet()
