# -*- coding: utf-8 -*-

"""
    Fit a CNN on the pngs files ---------> (with image_dataset_from_directory)
    for one machine e.g. 'valve' (list_datasets[0])
    from: https://keras.io/examples/vision/image_classification_from_scratch/
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

image_size = (313, 128)
batch_size = 32
folder_pngs = 'data/' + machine + '/png_v3'

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

# Initialising the CNN v3: make model
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


model = make_model(input_shape=image_size + (3,), num_classes=2)
epochs = 15 # 50

# callbacks = [
#     keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
# ]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Part 2 - Fitting the CNN to the images
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    folder_pngs,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    folder_pngs,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


# Visualize the data 9 first images
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")


# sys.exit()



model.fit(
    train_ds, epochs=epochs, validation_data=val_ds,
)


# save the cnn to disk  -----------------------
filename_classifier = rootFolder + '3_dc_classifiers/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_' + machine + '_cnn.h5'
model.save(filename_classifier)


print('model saved:', filename_classifier)
 



sys.exit()

 






