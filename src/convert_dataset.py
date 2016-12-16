import argparse
import os
import sys
from six.moves import cPickle as pickle

def parse_args():
    parser = argparse.ArgumentParser('Convert dataset')
    parser.add_argument('image_dir', type=str, help='Image directory path')
    parser.add_argument('out_path', type=str, help='Output file path')
    parser.add_argument('--num', '-n', type=int, default=1000000, help='Maximum number of images')
    return parser.parse_args()

def main():
    args = parse_args()
    files = os.listdir(args.image_dir)
    images = []
    for file_name in files[:args.num]:
        name, ext = os.path.splitext(file_name)
        if not ext in ['.jpg', '.jpeg', '.png', '.gif']:
            continue
        with open(os.path.join(args.image_dir, file_name), 'rb') as f:
            images.append(f.read())
    with open(args.out_path, 'wb') as f:
        pickle.dump(images, f)

if __name__ == '__main__':
    main()
