Stack GAN
====

Implementation of [StackGAN](https://arxiv.org/abs/1612.03242) except text to image synthesis.

# Usage

## Convert images to pickle file

```
$ python src/convert_dataset.py image_dir out_path [-n max_image_num]
```

Parameters:

* `image_dir`: (Required) Image file directory path
* `out_path`: (Required) Output pickle file path
* `-n max_image_num`: (Optional) The maximum number of images to be stored (default: 1000000)

Example:

```
$ python src/convert_dataset.py ../images dataset/images.pkl
```

## Train Stage-I model

Example:

```
$ python src/train.py -d dataset/images.pkl -o model/stage1 -g 0 -e 5 --lr-decay 2 --out-image-dir image/stage1 --clip-rect 17,47,144,144
```

## Train Stage-II model

Example:

```
$ python src/train.py model/stage1_005 -d dataset/images.pkl -o model/stage2 -g 0 -e 5 --lr-decay 1 --out-image-dir image/stage2 --clip-rect 17,47,144,144
```

## Generate image

Example:

```
$ python src/generate2.py model/stage1_005.gen.model model\stage2_005.gen.model image\generated -g 0
```

# Differences from original paper

* Text to image synthesis is not implemented.  
Input of generator is 100-dimension latent vector.
* Use Minibatch discriminator described in [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498).
* Output image size of Stage-II GAN is only 128 x 128.

# License

MIT License
