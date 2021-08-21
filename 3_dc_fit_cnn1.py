# -*- coding: utf-8 -*-

"""
    Fit a CNN on the pngs files
    for one machine e.g. 'valve' (list_datasets[0])
"""

from utils import list_datasets, folders_train_test, rootFolder
import joblib
import sys
import datetime

machine = list_datasets[0]
print('machine:', machine) # use the first folder: e.g. 'valve'

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 1), activation = 'relu'))
# classifier.add(Conv2D(32, (3, 3), input_shape = (313, 128, 1), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('data/' + machine + '/png_train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('data/' + machine+ '/png_validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 80, # 8000
                         epochs = 10,   # 25
                         validation_data = test_set,
                         validation_steps = 2000)

# save the cnn to disk  -----------------------
filename_model = rootFolder + 'dc_classifiers/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_' + machine + '_cnn.h5'
# filename_model = 'cnn.sav'
classifier.save(filename_model)
print('model saved : ', filename_model)
# joblib.dump(classifier, filename_model + ".sav")  


# # serialize model to JSON
# model_json = classifier.to_json()
# with open(filename_model + ".json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# classifier.save_weights(filename_model + ".h5")
# print("model saved to disk")
 


 



sys.exit()

 





