import math
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable

class ResidualBlock(chainer.Chain):

    def __init__(self, ch):
        initialW = chainer.initializers.Normal(0.02)
        super(ResidualBlock, self).__init__(
            conv1=L.Convolution2D(ch, ch, 3, pad=1, initialW=initialW),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, pad=1, initialW=initialW),
            bn2=L.BatchNormalization(ch),
        )

    def __call__(self, x, train=True):
        h1 = F.relu(self.bn1(self.conv1(x), test=not train))
        h2 = self.bn2(self.conv2(h1), test=not train)
        return F.relu(x + h2)

class UpSamplingBlock(chainer.Chain):

    def __init__(self, in_ch, out_ch, activate=True):
        initialW = chainer.initializers.Normal(0.02)
        super(UpSamplingBlock, self).__init__(
            conv=L.Convolution2D(in_ch, out_ch, 3, pad=1, initialW=initialW),
        )
        if activate:
            self.add_link('bn', L.BatchNormalization(out_ch))
        self.activate = activate

    def __call__(self, x, train=True):
        b, c, height, width = x.shape
        h = self.conv(F.unpooling_2d(x, 2, outsize=(height * 2, width * 2)))
        if self.activate:
            h = F.relu(self.bn(h, test=not train))
        return h


class DownSamplingBlock(chainer.Chain):

    def __init__(self, in_ch, out_ch):
        initialW = chainer.initializers.Normal(0.02)
        super(DownSamplingBlock, self).__init__(
            conv=L.Convolution2D(in_ch, out_ch, 4, stride=2, pad=1, initialW=initialW),
            bn=L.BatchNormalization(out_ch),
        )

    def __call__(self, x, train=True):
        return F.leaky_relu(self.bn(self.conv(x), test=not train))


class UpSampling(chainer.Chain):

    def __init__(self, in_size, in_ch, out_size):
        super(UpSampling, self).__init__()
        size = in_size
        ch = in_ch
        i = 1
        while size < out_size // 2:
            self.add_link('block{}'.format(i), UpSamplingBlock(ch, ch // 2))
            size *= 2
            ch //= 2
            i += 1
        self.add_link('block{}'.format(i), UpSamplingBlock(ch, 3, activate=False))

    def __call__(self, x, train=True):
        h = x
        links = self.children()
        for link in links:
            h = link(h, train=train)
        return h

class DownSampling(chainer.Chain):

    def __init__(self, in_size, in_ch, out_size, out_ch):
        super(DownSampling, self).__init__()
        channel_pairs = []
        size = out_size * 2
        ch = out_ch
        while size < in_size:
            channel_pairs.append(ch)
            size *= 2
            ch //= 2
        initialW = chainer.initializers.Normal(0.02)
        self.add_link('conv1', L.Convolution2D(in_ch, ch, 4, stride=2, pad=1, initialW=initialW))
        for i, ch in enumerate(channel_pairs[::-1]):
            self.add_link('block{}'.format(i + 2), DownSamplingBlock(ch // 2, ch))

    def __call__(self, x, train=True):
        links = self.children()
        h = F.leaky_relu(next(links)(x))
        for link in links:
            h = link(h, train)
        return h

class MinibatchDiscriminator(chainer.Chain):
    def __init__(self, in_size, knum=50, ksize=5):
        super(MinibatchDiscriminator, self).__init__(
            trans=L.Linear(in_size, knum * ksize, nobias=True)
        )
        self.knum = knum
        self.ksize = ksize

    def __call__(self, x):
        x_flat = F.reshape(x, (x.shape[0], -1))
        m = F.reshape(self.trans(x_flat), (-1, self.knum, self.ksize))
        m = F.expand_dims(m, 3)
        m_t = F.transpose(m, (3, 1, 2, 0))
        m, m_t = F.broadcast(m, m_t)

        norm = F.sum(abs(m - m_t), axis=2)
        c_b = F.exp(-norm)
        o_b = F.sum(c_b, axis=2)
        return F.concat((x_flat, o_b), axis=1)

class Generator1(chainer.Chain):
    def __init__(self):
        initialW = chainer.initializers.Normal(0.02)
        super(Generator1, self).__init__(
            conv1=L.Deconvolution2D(100, 1024, 4, initialW=initialW),
            bn1=L.BatchNormalization(1024),
            up=UpSampling(4, 1024, 64),
        )

    def __call__(self, x, train=True):
        h = F.reshape(x, x.shape + (1, 1))
        h = F.relu(self.bn1(self.conv1(h), test=not train))
        h = self.up(h, train)
        return F.tanh(h)

class Discriminator1(chainer.Chain):
    def __init__(self):
        initialW = chainer.initializers.Normal(0.02)
        super(Discriminator1, self).__init__(
            down=DownSampling(64, 3, 4, 1024),
            fc=L.Linear(4 * 4 * 1024, 1, initialW=initialW),
        )

    def __call__(self, x, train=True):
        h = self.down(x, train)
        y = self.fc(h)
        return y

class Generator2(chainer.Chain):
    def __init__(self):
        super(Generator2, self).__init__(
            down=DownSampling(64, 3, 16, 512),
            block1=ResidualBlock(512),
            block2=ResidualBlock(512),
            up=UpSampling(16, 512, 128),
        )

    def __call__(self, x, train=True):
        h = self.down(x, train)
        h = self.block1(h, train)
        h = self.block2(h, train)
        h = self.up(h, train)
        return F.tanh(h)

class Discriminator2(chainer.Chain):
    def __init__(self):
        initialW = chainer.initializers.Normal(0.02)
        super(Discriminator2, self).__init__(
            down=DownSampling(128, 3, 4, 1024),
            fc=L.Linear(4 * 4 * 1024, 1, initialW=initialW),
        )

    def __call__(self, x, train=True):
        h = self.down(x, train)
        y = self.fc(h)
        return y
