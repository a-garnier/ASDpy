# -*- coding: utf-8 -*-

"""
    Fit a auto encoder: not yet used in the project
    for one machine e.g. 'valve' (list_datasets[0])
    from: https://blog.keras.io/building-autoencoders-in-keras.html
"""

from utils import list_datasets, folders_train_test, rootFolder
import joblib
import sys
import datetime

machine = list_datasets[0]
print('machine:', machine) # use the first folder: e.g. 'valve'

# Part 1 - Building the CNN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

image_size = (432, 288, 3)
folder_pngs = 'data/' + machine + '/png_v4'

import keras
from keras import layers
from keras.callbacks import TensorBoard


# input_img = keras.Input(shape=(28, 28, 1))
input_img = keras.Input(shape=image_size)

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# At this point the representation is (7, 7, 32)

x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])





sys.exit()

 






