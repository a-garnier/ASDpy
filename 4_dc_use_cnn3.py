# -*- coding: utf-8 -*-

"""
    Use the CNN model in order to mak predictions on files in png_test_v4
    Use the first image in png_test folder of the machine (data/machine/png_test)
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

# config one machine 
machine = list_datasets[0]

# modelName = '2021-08-21-20-23-40_valve_cnn.h5'
modelName = '2021-09-06-20-44-06_slider_cnn.h5' 
image_size = (333, 216)
png_folder = rootFolder + 'data/' + machine + '/png_test_v4/'
model_folder = rootFolder + 'dc_classifiers/' + modelName
threshold = 0.6 # if > , change class

# load the model from disk
model = load_model(model_folder)

df_result = pd.DataFrame(columns=['file', 'score']) # df result  with files and scores
indiceFile = 0

# browse test files
wavfiles = [f for f in listdir(png_folder) if isfile(join(png_folder, f))]
for nameFilePngTotest in wavfiles:
    if nameFilePngTotest[-4:] != '.png': # ignore non .png files
        continue
    # print('test nameFilePngTotest: ', nameFilePngTotest)
    test_image = image.load_img(png_folder + nameFilePngTotest, target_size = image_size) # , target_size = (64, 64)
    img_array = image.img_to_array(test_image)
    img_array = np.expand_dims(test_image, axis = 0)
    predictions = model.predict(img_array)
    score = predictions[0]
    df_result = df_result.append({'file': nameFilePngTotest, 'score': score}, ignore_index=True)

df_result = df_result.sort_values(by = ['score'], ascending = False)
df_result.head(40)

countCorrectPrediction = 0
for index, row in df_result.iterrows():
    # print("file %s : %.2f%%" % (row['file'], row['score']))
    arrName = row['file'].split("_") # anomaly_id_00_00000001.wav
    classPrefix = arrName[0] # 'normal' or 'anomaly'
    isNormalPredict = 1 if row['score'] > threshold else 0
    isNormalReal = 1 if classPrefix == "normal" else 0
    correctPrediction = 'OK' if isNormalReal == isNormalPredict else 'NOK'
    if isNormalReal == isNormalPredict:
        countCorrectPrediction += 1
    print("File %s predict %s (score: %.2f%%)." % (row['file'], correctPrediction, 100 * row['score']))

print("%s: accuracy:  %.2f" % (machine, countCorrectPrediction / len(df_result)))

"""  --- valve: Accuracy:  0.97
    # This file normal_id_00_00000372.png predict OK (score: 99.99%).
    # This file normal_id_00_00000470.png predict OK (score: 99.97%).
    # This file normal_id_04_00000719.png predict OK (score: 99.96%).
    # This file normal_id_00_00000434.png predict OK (score: 99.92%).
    # This file normal_id_02_00000597.png predict OK (score: 99.90%).
    # This file normal_id_02_00000259.png predict OK (score: 99.77%).
    # This file normal_id_04_00000417.png predict OK (score: 99.73%).
    # This file normal_id_04_00000592.png predict OK (score: 99.46%).
    # This file normal_id_06_00000329.png predict OK (score: 99.23%).
    # This file normal_id_06_00000213.png predict OK (score: 98.68%).
    # This file normal_id_00_00000614.png predict OK (score: 98.68%).
    # This file normal_id_00_00000616.png predict OK (score: 98.10%).
    # This file normal_id_06_00000377.png predict OK (score: 97.18%).
    # This file normal_id_02_00000547.png predict OK (score: 96.56%).
    # This file normal_id_02_00000165.png predict OK (score: 96.35%).
    # This file normal_id_00_00000795.png predict OK (score: 96.05%).
    # This file normal_id_04_00000810.png predict OK (score: 95.30%).
    # This file normal_id_06_00000524.png predict OK (score: 94.38%).
    # This file normal_id_02_00000557.png predict OK (score: 85.97%).
    # This file anomaly_id_04_00000080.png predict NOK (score: 80.01%).
    # This file normal_id_04_00000584.png predict OK (score: 63.61%).
    # This file anomaly_id_00_00000097.png predict OK (score: 56.93%).
    # This file anomaly_id_04_00000095.png predict OK (score: 44.31%).
    # This file anomaly_id_04_00000001.png predict OK (score: 34.34%).
    # This file anomaly_id_00_00000065.png predict OK (score: 32.46%).
    # This file anomaly_id_00_00000094.png predict OK (score: 30.45%).
    # This file anomaly_id_00_00000082.png predict OK (score: 21.78%).
    # This file anomaly_id_00_00000048.png predict OK (score: 12.97%).
    # This file anomaly_id_02_00000024.png predict OK (score: 10.93%).
    # This file anomaly_id_04_00000004.png predict OK (score: 10.82%).
    # This file anomaly_id_00_00000076.png predict OK (score: 10.77%).
    # This file anomaly_id_04_00000098.png predict OK (score: 5.67%).
    # This file anomaly_id_02_00000017.png predict OK (score: 5.62%).
    # This file anomaly_id_00_00000056.png predict OK (score: 5.03%).
    # This file anomaly_id_02_00000108.png predict OK (score: 4.58%).
    # This file anomaly_id_04_00000065.png predict OK (score: 3.35%).
    # This file anomaly_id_02_00000061.png predict OK (score: 0.19%).
    # This file anomaly_id_02_00000119.png predict OK (score: 0.12%).
    # This file anomaly_id_02_00000103.png predict OK (score: 0.06%).
    # This file anomaly_id_04_00000110.png predict OK (score: 0.01%).
"""

"""  --- slider: accuracy:  0.80
    File normal_id_00_00000124.png predict OK (score: 99.99%).
    File normal_id_00_00000077.png predict OK (score: 99.99%).
    File normal_id_00_00000068.png predict OK (score: 99.99%).
    File normal_id_00_00000083.png predict OK (score: 99.99%).
    File normal_id_00_00000074.png predict OK (score: 99.99%).
    File normal_id_02_00000400.png predict OK (score: 99.99%).
    File normal_id_00_00000690.png predict OK (score: 99.99%).
    File normal_id_00_00000132.png predict OK (score: 99.98%).
    File normal_id_00_00000054.png predict OK (score: 99.98%).
    File normal_id_00_00000511.png predict OK (score: 99.98%).
    File normal_id_02_00000547.png predict OK (score: 99.97%).
    File normal_id_02_00000025.png predict OK (score: 99.97%).
    File normal_id_02_00000831.png predict OK (score: 99.97%).
    File normal_id_00_00000046.png predict OK (score: 99.97%).
    File normal_id_00_00000059.png predict OK (score: 99.97%).
    File normal_id_02_00000793.png predict OK (score: 99.96%).
    File normal_id_00_00000081.png predict OK (score: 99.96%).
    File normal_id_02_00000395.png predict OK (score: 99.95%).
    File normal_id_04_00000261.png predict OK (score: 99.95%).
    File normal_id_06_00000377.png predict OK (score: 99.95%).
    File normal_id_00_00000818.png predict OK (score: 99.94%).
    File normal_id_00_00000062.png predict OK (score: 99.93%).
    File normal_id_04_00000239.png predict OK (score: 99.92%).
    File anomaly_id_02_00000078.png predict NOK (score: 99.88%).
    File normal_id_02_00000144.png predict OK (score: 99.87%).
    File anomaly_id_06_00000037.png predict NOK (score: 99.87%).
    File normal_id_00_00000216.png predict OK (score: 99.83%).
    File normal_id_00_00000070.png predict OK (score: 99.80%).
    File normal_id_00_00000067.png predict OK (score: 99.78%).
    File normal_id_04_00000009.png predict OK (score: 99.76%).
    File normal_id_06_00000155.png predict OK (score: 99.73%).
    File anomaly_id_06_00000030.png predict NOK (score: 99.62%).
    File normal_id_00_00000611.png predict OK (score: 99.51%).
    File anomaly_id_00_00000241.png predict NOK (score: 96.88%).
    File anomaly_id_02_00000051.png predict NOK (score: 96.73%).
    File anomaly_id_06_00000003.png predict NOK (score: 92.12%).
    File anomaly_id_04_00000173.png predict NOK (score: 87.74%).
    File anomaly_id_00_00000221.png predict NOK (score: 86.01%).
    File anomaly_id_02_00000192.png predict NOK (score: 85.69%).
    File anomaly_id_06_00000027.png predict NOK (score: 85.00%).
    File anomaly_id_06_00000079.png predict NOK (score: 76.36%).
    File anomaly_id_02_00000116.png predict NOK (score: 73.55%).
    File anomaly_id_04_00000101.png predict OK (score: 51.38%).
    File anomaly_id_00_00000005.png predict OK (score: 48.80%).
    File anomaly_id_06_00000008.png predict OK (score: 41.11%).
    File anomaly_id_00_00000064.png predict OK (score: 27.01%).
    File anomaly_id_00_00000353.png predict OK (score: 22.96%).
    File anomaly_id_06_00000033.png predict OK (score: 22.80%).
    File anomaly_id_00_00000208.png predict OK (score: 20.74%).
    File anomaly_id_00_00000042.png predict OK (score: 13.35%).
    File anomaly_id_00_00000234.png predict OK (score: 7.15%).
    File anomaly_id_02_00000153.png predict OK (score: 6.48%).
    File anomaly_id_04_00000133.png predict OK (score: 5.81%).
    File anomaly_id_02_00000151.png predict OK (score: 5.26%).
    File anomaly_id_04_00000171.png predict OK (score: 0.83%).
    File anomaly_id_00_00000319.png predict OK (score: 0.13%).
    File anomaly_id_04_00000088.png predict OK (score: 0.11%).
    File anomaly_id_04_00000167.png predict OK (score: 0.03%).
    File anomaly_id_00_00000243.png predict OK (score: 0.00%).
    File anomaly_id_04_00000177.png predict OK (score: 0.00%).
    
"""