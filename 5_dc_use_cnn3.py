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

list_machine = ['fan', 'pump', 'slider', 'ToyCar',  'ToyConveyor', 'valve']  
list_machine = ['fan', 'ToyCar', 'ToyConveyor', 'pump',  'slider', 'valve']  # ok ?
# browse test files
wavfiles = [f for f in listdir(png_test_folder) if isfile(join(png_test_folder, f))] # 7730 pngs
for nameFilePngTotest in wavfiles:
    if nameFilePngTotest[-4:] != '.png': # ignore non .png files
        continue
    # print('test nameFilePngTotest: ', nameFilePngTotest)
    test_image = image.load_img(png_test_folder + nameFilePngTotest, target_size = image_size) # , target_size = (64, 64)
    img_array = image.img_to_array(test_image)
    img_array = np.expand_dims(test_image, axis = 0)
    prediction = model.predict(img_array)
    print(nameFilePngTotest, prediction)
    # anomaly_id_02_00000078_ToyCar.png [[1.0803374e-02 9.8868352e-01 2.1100255e-12 2.1557459e-12 5.1306013e-04 9.0587385e-11]]
    # anomaly_id_04_00000046_slider.png [[8.1691342e-26 3.0578049e-14 2.0197277e-34 6.1783719e-33 1.0000000e+00 1.3286750e-25]]
    # normal_id_06_00000451_pump.png [[8.7345477e-08 2.6160514e-02 4.6769547e-04 9.4642216e-01 2.6945930e-02 3.5615378e-06]]
    # score = prediction[0]
    # df_result = df_result.append({'file': nameFilePngTotest, 'score': score}, ignore_index=True)
    indiceFile += 1
    if indiceFile == 5:
        break
    
# df_result = df_result.sort_values(by = ['score'], ascending = False)
# df_result.head(40)

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
