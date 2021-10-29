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


# normal and scorePredict of the class > thresholdPrediction -> accuracy ok
# normal and scorePredict of the class <= thresholdPrediction -> accuracy nok

# anomaly and scorePredict of the class > thresholdPrediction -> accuracy nok
# anomaly and scorePredict of the class <= thresholdPrediction -> accuracy ok
# anomaly if it doesn't ptedict the correct machine or the score < thresholdPrediction
thresholdPrediction = 0.95 
# thresholdPredictionMin = 0.10 

dict_machine = {'ToyCar': {'err': 0, 'total': 0}, 'ToyConveyor': {'err': 0, 'total': 0}, 'slider': {'err': 0, 'total': 0}, 'pump': {'err': 0, 'total': 0}, 'fan': {'err': 0, 'total': 0}, 'valve': {'err': 0, 'total': 0}} # stats accuracy by machine
list_machine = ['ToyCar', 'ToyConveyor', 'fan', 'pump', 'slider', 'valve'] # ok 
# browse test files
wavfiles = [f for f in listdir(png_test_folder) if isfile(join(png_test_folder, f))] # 7730 pngs
nbWavs = len(wavfiles)
for nameFilePngTotest in wavfiles:
    if nameFilePngTotest[-4:] != '.png': # ignore non .png files 
        continue
    arrName = nameFilePngTotest.split("_") # normal_id_06_00000451_pump.png
    classPrefixReal = arrName[0] # 'normal' or 'anomaly'
    classNameReal = arrName[4][:-4] # 'pump' / [:-4] remove the '.png'
    # print(className)
    if classPrefixReal != 'anomaly': # anomaly only  accuracy:  0.615
    # if classPrefixReal != 'normal': # normal only  accuracy:  0.568
        continue
    # print('test nameFilePngTotest: ', nameFilePngTotest)
    test_image = image.load_img(png_test_folder + nameFilePngTotest, target_size = image_size) # , target_size = (64, 64)
    img_array = image.img_to_array(test_image)
    img_array = np.expand_dims(test_image, axis = 0)
    arr_predictions = model.predict(img_array)
    arr_predictions = arr_predictions[0] # prediction one by one so use the [0]
    # print(nameFilePngTotest, np.rint(arr_predictions)) # [[0. 0. 0. 1. 0. 0.]]
    # print(nameFilePngTotest, np.round(arr_predictions, 2)) # [[0.   0.03 0.   0.95 0.03 0.  ]] 
    # print(nameFilePngTotest, np.rint(arr_predictions).argmax(axis = 1)) # 
    
    predictionOK = False
    indiceClassNamePredict = list_machine.index(classNameReal)
    scorePredict = arr_predictions[indiceClassNamePredict]
    if scorePredict > thresholdPrediction and classPrefixReal == 'normal' or scorePredict <= thresholdPrediction and classPrefixReal == 'anomaly' :
        predictionOK = True
        nbPredictionOK+= 1
    
    dict_machine[classNameReal]['total'] =  dict_machine[classNameReal]['total'] + 1
    # count errors by machine 
    if predictionOK == False:
        dict_machine[classNameReal]['err'] =  dict_machine[classNameReal]['err'] + 1
    # indiceClassNamePredict = np.rint(arr_predictions).argmax(axis = 1)[0]
    # if the prediction is right, sound is normal otherwise, it's anormal
    # classOK = True if classNameReal == list_machine[indiceClassNamePredict] else False
    # # if classNameReal == list_machine[indiceClassNamePredict]:
    # #     classOK = True
    # if classOK and classPrefixReal == 'normal' or not classOK and classPrefixReal == 'anomaly':
    #     predictionOK = True
    #     nbPredictionOK+= 1
    # print(nameFilePngTotest, result) # [[0. 0. 0. 1. 0. 0.]]
    # score = arr_predictions[0]

    print(nameFilePngTotest, np.round(arr_predictions, 2), list_machine[indiceClassNamePredict], predictionOK) #ok
    # df_result = df_result.append({'file': nameFilePngTotest, 'score': score}, ignore_index=True)
    indiceFile += 1
    # if indiceFile % 100 == 0:
    #     print('indiceFile: ', indiceFile, ' / ', nbWavs, ' nbPredictionOK: ', nbPredictionOK, ' accuracy: ', nbPredictionOK / indiceFile)
    if indiceFile == 200:
        break
# df_result = df_result.sort_values(by = ['score'], ascending = False)
# df_result.head(40)

accuracy = nbPredictionOK / indiceFile
print('indiceFile: ', indiceFile, ' / ', nbWavs, ' nbPredictionOK: ', nbPredictionOK, ' accuracy: ', accuracy)
print('dict_machine: ', dict_machine) 

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


#                ['ToyCar', 'ToyConveyor', 'slider', 'pump', 'fan', 'valve'] 
# list_machine = ['ToyCar', 'ToyConveyor', 'fan', 'pump', 'slider', 'valve'] # ok 

# anomaly only
# indiceFile:  200  /  7731  nbPredictionOK:  83  accuracy:  0.415
# dict_machine:  {'ToyCar': {'err': 14, 'total': 38}, 
#                 'ToyConveyor': {'err': 45, 'total': 45}, 
#                 'slider': {'err': 28, 'total': 28}, 'pump': {'err': 2, 'total': 22}, 
#                 'fan': {'err': 17, 'total': 51}, 'valve': {'err': 11, 'total': 16}}
# ToyConveyor, slider always found (valve as well)



# normal only
# indiceFile:  200  /  7731  nbPredictionOK:  152  accuracy:  0.76
# dict_machine:  {'ToyCar': {'err': 5, 'total': 19}, 'ToyConveyor': {'err': 0, 'total': 15}, 
#                 'slider': {'err': 0, 'total': 36}, 'pump': {'err': 29, 'total': 29}, 
#                 'fan': {'err': 10, 'total': 14}, 'valve': {'err': 4, 'total': 87}}
# pump never found

sortir ToyConveyor du lot 
ou bien lui changer les images 
changer de lib pour generer les images 
