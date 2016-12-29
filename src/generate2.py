import argparse
import numpy as np
import io
import os
import six
from PIL import Image
from six.moves import cPickle as pickle

import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F
import chainer.links as L
import net


latent_size = 100

def parse_args():
    parser = argparse.ArgumentParser(description='Stack GAN Generator')
    parser.add_argument('model_path1', type=str,
                        help='input stack-I model file path')
    parser.add_argument('model_path2', type=str,
                        help='input stack-II model file path')
    parser.add_argument('output', type=str,
                        help='output file path without extension')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--vector_file1', '-v1', type=str,
                        help='input vector file path 1')
    parser.add_argument('--vector_index1', '-i1', type=int, default=-1,
                        help='input vector index 1')
    parser.add_argument('--vector_file2', '-v2', type=str,
                        help='input vector file path 2')
    parser.add_argument('--vector_index2', '-i2', type=int, default=-1,
                        help='input vector index 2')
    return parser.parse_args()

def main():
    args = parse_args()
    gen1 = net.Generator1()
    gen2 = net.Generator2()
    chainer.serializers.load_npz(args.model_path1, gen1)
    chainer.serializers.load_npz(args.model_path2, gen2)
    device_id = None
    if args.gpu >= 0:
        device_id = args.gpu
        cuda.get_device(device_id).use()
        gen1.to_gpu(device_id)
        gen2.to_gpu(device_id)

    out_vector_path = None
    if args.vector_file1 and args.vector_index1 >= 0 and args.vector_file2 and args.vector_index2 >= 0:
        with open(args.vector_file1, 'rb') as f:
            z = np.load(f)
        z1 = z[args.vector_index1]
        with open(args.vector_file2, 'rb') as f:
            z = np.load(f)
        z2 = z[args.vector_index2]
        w = np.arange(10).astype(np.float32).reshape((-1, 1)) / 9
        z = (1 - w) * z1 + w * z2
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    else:
        z = np.random.normal(0, 1, (100, latent_size)).astype(np.float32)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        out_vector_path = '{}.npy'.format(args.output)


    with chainer.no_backprop_mode():
        if device_id is None:
            x1 = gen1(z, train=False)
        else:
            x1 = gen1(cuda.to_gpu(z, device_id), train=False)
        x2 = gen2(x1, train=False)
    x1 = cuda.to_cpu(x1.data)
    x2 = cuda.to_cpu(x2.data)
    batch, ch, h, w = x1.shape
    x1 = x1.reshape((-1, 10, ch, h, w)).transpose((0, 3, 1, 4, 2)).reshape((-1, 10 * w, ch))
    x1 = ((x1 + 1) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(x1).save('{}_stack1.jpg'.format(args.output))
    batch, ch, h, w = x2.shape
    x2 = x2.reshape((-1, 10, ch, h, w)).transpose((0, 3, 1, 4, 2)).reshape((-1, 10 * w, ch))
    x2 = ((x2 + 1) * 127.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(x2).save('{}_stack2.jpg'.format(args.output))
    if out_vector_path:
        with open(out_vector_path, 'wb') as f:
            np.save(f, z)

if __name__ == '__main__':
    main()
