### Adapted from PyTorch code at https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py

from collections import OrderedDict

import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow import keras


class UNet(keras.Model):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = keras.layers.MaxPool2D(kernel_size=2, strides=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = keras.layers.MaxPool2D(kernel_size=2, strides=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = keras.layers.MaxPool2D(kernel_size=2, strides=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = keras.layers.MaxPool2D(kernel_size=2, strides=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = keras.layers.Conv2DTranspose(features * 8, kernel_size=2, strides=2)
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = keras.layers.Conv2DTranspose(features * 4, kernel_size=2, strides=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 2, kernel_size=2, strides=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features, kernel_size=2, strides=2)
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = keras.layers.Conv2D(out_channels, kernel_size=1)

    def call(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = tf.concat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = tf.concat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = tf.concat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = tf.concat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return keras.activations.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return keras.models.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        keras.layers.Conv2D(
                            features,
                            kernel_size=3,
                            padding="valid"
                        ),
                    ),
                    (name + "norm1", keras.layers.BatchNormalization()),
                    (name + "relu1", keras.layers.ReLU()),
                    (
                        name + "conv2",
                        keras.layers.Conv2D(
                            features,
                            kernel_size=3,
                            padding="valid"
                        ),
                    ),
                    (name + "norm2", keras.layers.BatchNormalization()),
                    (name + "relu2", keras.layers.ReLU()),
                ]
            )
        )

# test on lambda system
if __name__ == "__main__":
    # load GIS data
