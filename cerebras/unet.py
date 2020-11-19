### Adapted from PyTorch code at https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py

from collections import OrderedDict

import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow import keras
from gis_preprocess import tf_gis_test_train_split
import os

class UNet(keras.Model):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = keras.layers.MaxPool2D(pool_size=2, strides=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = keras.layers.Conv2DTranspose(features * 8, kernel_size=2, strides=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = keras.layers.Conv2DTranspose(features * 4, kernel_size=2, strides=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = keras.layers.Conv2DTranspose(features * 2, kernel_size=2, strides=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = keras.layers.Conv2DTranspose(features, kernel_size=2, strides=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = keras.layers.Conv2D(out_channels, kernel_size=1)

    def call(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        print(dec4.shape, enc4.shape)
        dec4 = tf.concat((dec4, enc4), axis=-1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = tf.concat((dec3, enc3), axis=-1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = tf.concat((dec2, enc2), axis=-1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = tf.concat((dec1, enc1), axis=-1)
        dec1 = self.decoder1(dec1)
        return keras.activations.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        block = keras.models.Sequential()
        block.add(keras.layers.Conv2D(
                            features,
                            kernel_size=3,
                            padding="valid"
                        ))
        block.add(keras.layers.BatchNormalization())
        block.add(keras.layers.ReLU())
        block.add(keras.layers.Conv2D(
                            features,
                            kernel_size=3,
                            padding="valid"
                        ))
        block.add(keras.layers.BatchNormalization())
        block.add(keras.layers.ReLU())
        return block

# test on lambda system
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '[7]'
    config = {'batch_size': 1, 'learning_rate': .001, 'epochs': 1}
    # load GIS data
    tf.random.set_seed(0)
    keras.backend.set_image_data_format('channels_last')
    b = int(config['batch_size'])
    model = UNet()
    opt = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    # fit model on gis data
    (x_train, y_train), (x_test, y_test) = tf_gis_test_train_split()
    train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(b)
    test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(b)
    res = model.fit(train, epochs=config['epochs'], batch_size=b)
    res_test = model.evaluate(test)