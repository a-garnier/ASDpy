# -*- coding: utf-8 -*-

"""
    Use the CNN model in order to mak predictions
    Use the first image in png_test folder of the machine (data/machine/png_test)
"""

from utils import list_datasets, folders_train_test, rootFolder
import sys
import datetime
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
from keras.models import load_model

# config
machine = 'valve'
modelName = '2021-07-14-16-41-26_valve_cnn.h5'

# ---
png_folder = rootFolder + 'data/' + machine + '/png_test/'
wavfiles = [f for f in listdir(png_folder) if isfile(join(png_folder, f))]
nameFilePngTotest = wavfiles[0]
print('nameFilePngTotest: ', nameFilePngTotest)
#sys.exit()

# load the model from disk
pathModel = rootFolder + 'dc_models/' + modelName
model = load_model(pathModel)

# loaded_model = joblib.load(rootFolder + 'dc_models/' + modelName)
test_image = image.load_img(png_folder + nameFilePngTotest, target_size = (64, 64)) # , target_size = (64, 64)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image) # bug ici    ValueError: Input 0 of layer dense_16 is incompatible with the layer: expected axis -1 of input shape to have value 6272 but received input with shape (None, 72960)
# training_set.class_indices
# if result[0][0] == 1:
#     prediction = '1'
# else:
#     prediction = '2'
# loaded_model = pickle.load(open(filename_model, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
 
# evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# Part 3 - Making new predictions


# test_image = image.load_img('data/' + machine + '/test_which_class_png', target_size = (64, 64))
# test_image = image.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = classifier.predict(test_image)
# training_set.class_indices
# if result[0][0] == 1:
#     prediction = 'dog'
# else:
#     prediction = 'cat'


