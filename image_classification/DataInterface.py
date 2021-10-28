import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia

class DataInterface:

    def __init__(self, config):
        self.config = config
        self.buffer_size = self.config['buffer_size']
        self.batch_size = self.config['batch_size']
        self.image_size = self.config['image_size']

        self.min_val = 100000.0
        self.max_val = -100000.0

        self.rand_aug = iaa.RandAugment(n=2, m=7)

    # load the demanded dataset
    def load(self, dataset_name):

        train_ds, test_ds, num_classes, split = None, None, None, None

        if dataset_name == 'mnist':
            self.num_classes = 10
            split=["train", "test"]
        elif dataset_name == 'fashion_mnist':
            self.num_classes = 10
            split = ["train", "test"]
        elif dataset_name == 'cifar10':
            self.num_classes = 10
            split=["train", "test"]
        elif dataset_name == 'cifar100':
            self.num_classes = 100
            split=["train", "test"]
        elif dataset_name == 'food101':
            self.num_classes = 101
            split = ["train", "test"]
        elif dataset_name == 'rock_paper_scissors':
            self.num_classes = 3
            split = ["train", "test"]
        elif dataset_name == 'imagenet2012':
            self.num_classes = 1000
            split=["train", "validation"]
        elif dataset_name == 'patch_camelyon':
            self.num_classes = 2
            split = ["train", "test"]
        elif dataset_name == 'imagenet':
            dataset_name = 'imagenet_resized/64x64'
            self.num_classes = 1000
            split=["train", "validation"]


        train_ds, test_ds = tfds.load(dataset_name, split=split)

        self.train_ds = train_ds.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)\
            .map(lambda feats: (tf.image.resize(feats['image'], self.image_size[:-1]), feats['label'])) \
            .map(lambda x, y: (tf.py_function(self.augment, [x], [tf.float32])[0], y), num_parallel_calls=tf.data.AUTOTUNE) \
            .map(lambda x, y: (x, tf.one_hot(y, self.num_classes))) \
            .prefetch(tf.data.AUTOTUNE)

        self.test_ds = test_ds.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True) \
            .map(lambda feats: (tf.image.resize(feats['image'], self.image_size[:-1]), feats['label'])) \
            .map(lambda x, y: (x, tf.one_hot(y, self.num_classes))) \
            .prefetch(tf.data.AUTOTUNE)

    def augment(self, images):
        # Input to `augment()` is a TensorFlow tensor which
        # is not supported by `imgaug`. This is why we first
        # convert it to its `numpy` variant.
        images = tf.cast(images, tf.uint8)
        return self.rand_aug(images=images.numpy())

    # preprocess a batch and get the images and the sparse labels
    def preprocess_batch(self, feats):
        x = feats['image']
        y = feats['label']
        x = tf.cast(x, dtype=tf.float32)
        x = tf.divide(x, 255.0)

        # convert grayscale to rgb
        if x.shape[-1] == 1:
            x = tf.image.grayscale_to_rgb(x)

        # resize all images to a pre-defined resolution
        x = tf.image.resize(x, self.image_size[:-1])

        # normalize
        x = tf.image.per_image_standardization(x)

        # one-hot labels
        y = tf.one_hot(y, self.num_classes)

        return x, y

