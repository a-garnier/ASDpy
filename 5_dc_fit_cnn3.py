# -*- coding: utf-8 -*-

"""
    Fit a CNN on the pngs files in data_v5/train/[machine] ---------> (with image_dataset_from_directory)
    from: https://blog.jovian.ai/using-resnet-for-image-classification-4b3c42f2a27e
"""

# REFAIRE !! avec https://blog.jovian.ai/using-resnet-for-image-classification-4b3c42f2a27e 
# https://keras.io/examples/


from utils import list_datasets, rootFolder
import datetime
import sys
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path


use_folder = rootFolder + 'data_v5/train/'
data_path = Path(use_folder)
folders_classes = [f for f in data_path.iterdir() if f.is_dir()]
# print('foldersClasses: ', folders_classes)
# for p in folders_classes:
#     print('p: ', p)
num_classes = len(folders_classes)
print('num_classes: ', num_classes)

# sys.exit()

# Part 1 - Building the CNN
image_size = (333, 216)
batch_size = 32
data_augmentation = keras.Sequential(
    [
        # layers.experimental.preprocessing.RandomFlip("horizontal"),
        # layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

# Initialising the CNN v4: make model
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)
    
    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256]: #, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=num_classes)
epochs = 8 # 50

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    # loss="binary_crossentropy", # 2 classes 
    # loss="categorical_crossentropy", # n classes - 2D shape (n_samples, n_class)
    loss="sparse_categorical_crossentropy", # n classes - 1D integer encoded target (n_class)
    metrics=["accuracy"],
)

# Found 17858 files belonging to 6 classes.
# Using 14287 files for training.
# Using 3571 files for validation.

# Part 2 - Fitting the CNN to the images
# for folder_machine in list_datasets: 
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    use_folder,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    use_folder,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
model.fit(
    train_ds, epochs=epochs, validation_data=val_ds,
)

# save the model to disk  ----------------------- ' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_
filename_classifier = rootFolder + '5_dc_classifiers/global_cnn.h5'
model.save(filename_classifier)

print('model saved: ', filename_classifier)



