# -*- coding: utf-8 -*-

"""
    Use the CNN model in order to make predictions on files in data_v5/all_png_test_v5
    from: https://keras.io/examples/vision/image_classification_from_scratch/
"""

from utils import list_datasets, rootFolder
import sys
import datetime
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
from keras.models import load_model


# machine = 'valve'
now = datetime.datetime.now()
print("*************** Start... ******************", now.strftime("%Y-%m-%d %H:%M:%S"))



image_size = (333, 216)
png_test_folder = rootFolder + 'data_v5/all_png_test_v5/'
model_folder = rootFolder + '5_dc_classifiers/global_cnn.h5'
threshold = 0.6 # if > , change class

# load the model from disk
model = load_model(model_folder)

df_result = pd.DataFrame(columns=['file', 'score']) # df result  with files and scores
indiceFile = 0
nbPredictionOK = 0
# list_machine = ['fan', 'pump', 'slider', 'ToyCar',  'ToyConveyor', 'valve']  
list_machine = ['ToyCar', 'ToyConveyor', 'slider', 'pump', 'fan', 'valve']  # ok test
# browse test files
wavfiles = [f for f in listdir(png_test_folder) if isfile(join(png_test_folder, f))] # 7730 pngs
for nameFilePngTotest in wavfiles:
    if nameFilePngTotest[-4:] != '.png': # ignore non .png files
        continue
    arrName = nameFilePngTotest.split("_") # normal_id_06_00000451_pump.png
    classPrefixReal = arrName[0] # 'normal' or 'anomaly'
    classNameReal = arrName[4][:-4] # 'pump' / [:-4] remove the '.png'
    # print(className)
    # if classPrefixReal == 'anomaly':
    #     continue
    # print('test nameFilePngTotest: ', nameFilePngTotest)
    test_image = image.load_img(png_test_folder + nameFilePngTotest, target_size = image_size) # , target_size = (64, 64)
    img_array = image.img_to_array(test_image)
    img_array = np.expand_dims(test_image, axis = 0)
    prediction = model.predict(img_array)
    # print(nameFilePngTotest, prediction)
    # print(nameFilePngTotest, np.rint(prediction)) # [[0. 0. 0. 1. 0. 0.]]
    # print(nameFilePngTotest, np.rint(prediction).argmax(axis = 1)) # 
    indiceClassNamePredict = np.rint(prediction).argmax(axis = 1)[0]
    classOK = False
    predictionOK = False
    #todo : test aussi que la somme du tableau = 1 (un seul 1 dans le tableau)
    # if the prediction is right, sound is normal otherwise, it's anormal
    if classNameReal == list_machine[indiceClassNamePredict]:
        classOK = True
    if classOK and classPrefixReal == 'normal' or not classOK and classPrefixReal == 'anomaly':
        predictionOK = True
        nbPredictionOK+= 1
    # print(nameFilePngTotest, result) # [[0. 0. 0. 1. 0. 0.]]
    # score = prediction[0]
    print(nameFilePngTotest, np.rint(prediction), indiceClassNamePredict, list_machine[indiceClassNamePredict], predictionOK)
    # df_result = df_result.append({'file': nameFilePngTotest, 'score': score}, ignore_index=True)
    indiceFile += 1
    if indiceFile == 18:
        break
    # normal if list_machine
# df_result = df_result.sort_values(by = ['score'], ascending = False)
# df_result.head(40)

accuracy = nbPredictionOK / indiceFile
print('accuracy: ', accuracy)
# countCorrectPrediction = 0
# for index, row in df_result.iterrows():
#     # print("file %s : %.2f%%" % (row['file'], row['score']))
#     arrName = row['file'].split("_") # anomaly_id_00_00000001.wav
#     classPrefix = arrName[0] # 'normal' or 'anomaly'
#     isNormalPredict = 1 if row['score'] > threshold else 0
#     isNormalReal = 1 if classPrefix == "normal" else 0
#     correctPrediction = 'OK' if isNormalReal == isNormalPredict else 'NOK'
#     if isNormalReal == isNormalPredict:
#         countCorrectPrediction += 1
#     if indiceFile % 100 == 0:
#         print('countImages generated...: ', indiceFile, ' / ', nbWavs, ')')

#     print("File %s predict %s (score: %.2f%%)." % (row['file'], correctPrediction, 100 * row['score']))

# print("%s: accuracy:  %.2f" % (machine, countCorrectPrediction / len(df_result)))

# anomaly_id_02_00000078_ToyCar.png [[0. 1. 0. 0. 0. 0.]] 1 ToyConveyor True
# anomaly_id_04_00000046_slider.png [[0. 0. 0. 0. 1. 0.]] 4 fan True
# anomaly_id_02_00000097_slider.png [[0. 0. 0. 0. 1. 0.]] 4 fan True
# normal_id_06_00000451_pump.png [[0. 0. 0. 1. 0. 0.]] 3 pump True
# anomaly_id_00_00000119_pump.png [[0. 1. 0. 0. 0. 0.]] 1 ToyConveyor True
# anomaly_id_02_00000254_fan.png [[0. 0. 0. 0. 1. 0.]] 4 fan False
# normal_id_06_00000637_pump.png [[0. 0. 0. 1. 0. 0.]] 3 pump True
# normal_id_04_00000546_fan.png [[0. 0. 0. 0. 1. 0.]] 4 fan True
# normal_id_02_00000492_valve.png [[0. 0. 0. 0. 0. 1.]] 5 valve True
# normal_id_02_00000347_valve.png [[0. 0. 0. 0. 0. 1.]] 5 valve True
# anomaly_id_00_00000024_valve.png [[0. 1. 0. 0. 0. 0.]] 1 ToyConveyor True
# anomaly_id_03_00000031_ToyCar.png [[1. 0. 0. 0. 0. 0.]] 0 ToyCar False
# anomaly_id_00_00000118_pump.png [[0. 0. 0. 1. 0. 0.]] 3 pump False
# anomaly_id_02_00000244_fan.png [[0. 0. 0. 0. 1. 0.]] 4 fan False
# anomaly_id_04_00000087_valve.png [[0. 0. 0. 0. 0. 1.]] 5 valve False
# anomaly_id_02_00000205_ToyCar.png [[1. 0. 0. 0. 0. 0.]] 0 ToyCar False
# anomaly_id_02_00000232_ToyCar.png [[1. 0. 0. 0. 0. 0.]] 0 ToyCar False
# anomaly_id_00_00000135_fan.png [[0. 0. 1. 0. 0. 0.]] 2 slider True
# accuracy:  0.61