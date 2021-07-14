# -*- coding: utf-8 -*-

"""
    Fit a CNN on the pngs files (like course 151)
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
model = Sequential()

first_layer = Conv2D(filters = 32,
                     kernel_size = (5, 5),
                     padding = 'valid',
                     input_shape = (28, 28, 1),
                     activation = 'relu')

second_layer = MaxPooling2D(pool_size = (2, 2))


third_layer = Dropout(rate = 0.2)

fourth_layer = Flatten()

fifth_layer = Dense(units = 128,
                    activation = 'relu')

output_layer = Dense(units = 10,
                     activation='softmax')
model.add(first_layer)
model.add(second_layer)
model.add(third_layer)
model.add(fourth_layer)
model.add(fifth_layer)
model.add(output_layer)

model.compile(loss='categorical_crossentropy', # fonction de perte
              optimizer='adam',                # algorithme d'optimisation
              metrics=['accuracy'])            # métrique d'évaluation

training_history = model.fit(X_train, y_train,
                             validation_split = 0.2,
                             epochs = 10,
                             batch_size = 200)

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
filename_model = rootFolder + 'dc_models/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_' + machine + '_cnn.h5'
# filename_model = 'cnn.sav'
classifier.save(filename_model)
# joblib.dump(classifier, filename_model + ".sav")  


# # serialize model to JSON
# model_json = classifier.to_json()
# with open(filename_model + ".json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# classifier.save_weights(filename_model + ".h5")
# print("model saved to disk")
 


 



sys.exit()

 





