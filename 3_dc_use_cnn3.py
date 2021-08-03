# -*- coding: utf-8 -*-

"""
    Use the CNN model in order to mak predictions
    Use the first image in png_test folder of the machine (data/machine/png_test)
    from: https://keras.io/examples/vision/image_classification_from_scratch/
"""

from utils import list_datasets, folders_train_test, rootFolder
import sys
import datetime
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
from keras.models import load_model

# config
machine = 'valve'
modelName = '2021-08-02-21-42-25_valve_cnn.h5'
indiceFile = 3
# ---
# load the model from disk
pathModel = rootFolder + 'dc_classifiers/' + modelName
model = load_model(pathModel)


# browse test files
df_result = pd.DataFrame(columns=['file', 'score'])

png_folder = rootFolder + 'data/' + machine + '/png_test/'
wavfiles = [f for f in listdir(png_folder) if isfile(join(png_folder, f))]
for nameFilePngTotest in wavfiles:
    if nameFilePngTotest[-4:] != '.png': # ignore non .png files
        continue
    # print('test nameFilePngTotest: ', nameFilePngTotest)
    test_image = image.load_img(png_folder + nameFilePngTotest, target_size = (313, 128)) # , target_size = (64, 64)
    img_array = image.img_to_array(test_image)
    img_array = np.expand_dims(test_image, axis = 0)
    predictions = model.predict(img_array)
    score = predictions[0]
    df_result = df_result.append({'file': nameFilePngTotest, 'score': score}, ignore_index=True)
    print("This file %s is %.2f %% normal." % (nameFilePngTotest, 100 * score))

df_result = df_result.sort_values(by = ['score'], ascending = False)

