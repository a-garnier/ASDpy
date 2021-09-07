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
# machine = list_datasets[0]
# machine = 'valve'
# machine = 'slider'
# machine = 'pump'
# machine = 'fan'
# machine = 'ToyCar'
machine = 'ToyConveyor'

modelName = machine + '_cnn.h5' 

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
"""  --- pump: accuracy:  0.73
    File normal_id_06_00000935.png predict OK (score: 100.00%).
    File normal_id_00_00000304.png predict OK (score: 99.99%).
    File normal_id_02_00000530.png predict OK (score: 99.99%).
    File normal_id_00_00000099.png predict OK (score: 99.98%).
    File normal_id_02_00000124.png predict OK (score: 99.98%).
    File normal_id_02_00000538.png predict OK (score: 99.96%).
    File normal_id_00_00000158.png predict OK (score: 99.96%).
    File anomaly_id_00_00000136.png predict NOK (score: 99.95%).
    File normal_id_00_00000108.png predict OK (score: 99.95%).
    File normal_id_00_00000020.png predict OK (score: 99.92%).
    File normal_id_00_00000018.png predict OK (score: 99.85%).
    File normal_id_00_00000379.png predict OK (score: 99.84%).
    File normal_id_00_00000162.png predict OK (score: 99.79%).
    File normal_id_02_00000551.png predict OK (score: 99.73%).
    File normal_id_06_00000683.png predict OK (score: 99.63%).
    File normal_id_00_00000774.png predict OK (score: 99.60%).
    File normal_id_06_00000248.png predict OK (score: 99.58%).
    File normal_id_00_00000024.png predict OK (score: 99.58%).
    File normal_id_04_00000125.png predict OK (score: 99.35%).
    File normal_id_06_00000130.png predict OK (score: 99.34%).
    File normal_id_02_00000478.png predict OK (score: 99.10%).
    File anomaly_id_00_00000106.png predict NOK (score: 99.08%).
    File normal_id_06_00000211.png predict OK (score: 98.70%).
    File anomaly_id_02_00000071.png predict NOK (score: 98.54%).
    File normal_id_04_00000039.png predict OK (score: 98.47%).
    File anomaly_id_00_00000064.png predict NOK (score: 98.30%).
    File normal_id_00_00000004.png predict OK (score: 98.12%).
    File normal_id_00_00000048.png predict OK (score: 96.68%).
    File anomaly_id_06_00000065.png predict NOK (score: 96.13%).
    File normal_id_02_00000464.png predict OK (score: 93.73%).
    File anomaly_id_00_00000097.png predict NOK (score: 92.68%).
    File anomaly_id_00_00000073.png predict NOK (score: 91.57%).
    File anomaly_id_06_00000094.png predict NOK (score: 84.36%).
    File anomaly_id_02_00000001.png predict NOK (score: 81.25%).
    File normal_id_06_00000065.png predict OK (score: 78.30%).
    File anomaly_id_00_00000092.png predict NOK (score: 77.05%).
    File anomaly_id_02_00000083.png predict NOK (score: 76.86%).
    File normal_id_00_00000015.png predict OK (score: 72.62%).
    File normal_id_06_00000405.png predict OK (score: 71.46%).
    File anomaly_id_06_00000044.png predict NOK (score: 68.84%).
    File anomaly_id_00_00000114.png predict NOK (score: 68.61%).
    File anomaly_id_00_00000005.png predict NOK (score: 68.29%).
    File anomaly_id_02_00000075.png predict NOK (score: 61.68%).
    File normal_id_02_00000522.png predict OK (score: 61.13%).
    File anomaly_id_02_00000040.png predict OK (score: 57.74%).
    File anomaly_id_06_00000073.png predict OK (score: 44.85%).
    File anomaly_id_04_00000084.png predict OK (score: 43.27%).
    File anomaly_id_00_00000108.png predict OK (score: 42.07%).
    File normal_id_00_00000103.png predict NOK (score: 33.83%).
    File anomaly_id_02_00000013.png predict OK (score: 28.45%).
    File anomaly_id_06_00000036.png predict OK (score: 24.53%).
    File anomaly_id_02_00000092.png predict OK (score: 23.76%).
    File anomaly_id_02_00000101.png predict OK (score: 18.76%).
    File anomaly_id_00_00000035.png predict OK (score: 18.76%).
    File anomaly_id_00_00000102.png predict OK (score: 17.46%).
    File anomaly_id_04_00000080.png predict OK (score: 15.40%).
    File anomaly_id_00_00000013.png predict OK (score: 6.48%).
    File anomaly_id_04_00000047.png predict OK (score: 4.19%).
    File anomaly_id_00_00000032.png predict OK (score: 1.80%).
    File anomaly_id_04_00000072.png predict OK (score: 0.06%).
"""
""" --- pump: accuracy:  0.58
    File normal_id_02_00000124.png predict OK (score: 99.98%).
    File anomaly_id_06_00000036.png predict NOK (score: 99.63%).
    File normal_id_00_00000158.png predict OK (score: 99.48%).
    File normal_id_06_00000683.png predict OK (score: 98.57%).
    File anomaly_id_06_00000094.png predict NOK (score: 98.36%).
    File normal_id_02_00000530.png predict OK (score: 97.76%).
    File anomaly_id_02_00000071.png predict NOK (score: 95.15%).
    File anomaly_id_02_00000075.png predict NOK (score: 92.85%).
    File normal_id_00_00000018.png predict OK (score: 87.82%).
    File normal_id_06_00000211.png predict OK (score: 74.88%).
    File normal_id_00_00000108.png predict OK (score: 67.46%).
    File normal_id_04_00000125.png predict OK (score: 65.74%).
    File normal_id_06_00000248.png predict OK (score: 61.96%).
    File normal_id_00_00000099.png predict NOK (score: 58.96%).
    File anomaly_id_02_00000083.png predict OK (score: 58.39%).
    File anomaly_id_00_00000005.png predict OK (score: 51.69%).
    File normal_id_02_00000522.png predict NOK (score: 44.72%).
    File normal_id_02_00000538.png predict NOK (score: 38.62%).
    File normal_id_00_00000004.png predict NOK (score: 33.95%).
    File normal_id_02_00000464.png predict NOK (score: 28.73%).
    File anomaly_id_02_00000101.png predict OK (score: 20.53%).
    File normal_id_00_00000379.png predict NOK (score: 20.51%).
    File normal_id_02_00000551.png predict NOK (score: 19.65%).
    File anomaly_id_00_00000064.png predict OK (score: 17.80%).
    File anomaly_id_00_00000136.png predict OK (score: 16.33%).
    File anomaly_id_00_00000102.png predict OK (score: 14.86%).
    File anomaly_id_02_00000013.png predict OK (score: 14.12%).
    File anomaly_id_00_00000032.png predict OK (score: 12.40%).
    File normal_id_00_00000304.png predict NOK (score: 10.99%).
    File anomaly_id_00_00000092.png predict OK (score: 9.65%).
    File normal_id_00_00000020.png predict NOK (score: 8.87%).
    File anomaly_id_06_00000044.png predict OK (score: 7.56%).
    File anomaly_id_04_00000072.png predict OK (score: 6.61%).
    File anomaly_id_02_00000001.png predict OK (score: 6.53%).
    File normal_id_06_00000405.png predict NOK (score: 6.40%).
    File normal_id_06_00000130.png predict NOK (score: 4.69%).
    File normal_id_02_00000478.png predict NOK (score: 3.88%).
    File anomaly_id_00_00000073.png predict OK (score: 2.71%).
    File normal_id_06_00000065.png predict NOK (score: 2.68%).
    File normal_id_00_00000015.png predict NOK (score: 2.63%).
    File anomaly_id_00_00000035.png predict OK (score: 2.28%).
    File anomaly_id_04_00000084.png predict OK (score: 2.03%).
    File anomaly_id_00_00000097.png predict OK (score: 1.91%).
    File anomaly_id_00_00000013.png predict OK (score: 1.56%).
    File anomaly_id_02_00000040.png predict OK (score: 1.53%).
    File anomaly_id_00_00000106.png predict OK (score: 1.38%).
    File normal_id_00_00000048.png predict NOK (score: 1.29%).
    File normal_id_04_00000039.png predict NOK (score: 1.16%).
    File normal_id_00_00000162.png predict NOK (score: 1.14%).
    File normal_id_00_00000774.png predict NOK (score: 1.12%).
    File anomaly_id_02_00000092.png predict OK (score: 1.12%).
    File normal_id_00_00000103.png predict NOK (score: 0.74%).
    File anomaly_id_06_00000073.png predict OK (score: 0.69%).
    File normal_id_06_00000935.png predict NOK (score: 0.58%).
    File normal_id_00_00000024.png predict NOK (score: 0.46%).
    File anomaly_id_04_00000080.png predict OK (score: 0.30%).
    File anomaly_id_06_00000065.png predict OK (score: 0.23%).
    File anomaly_id_00_00000114.png predict OK (score: 0.16%).
    File anomaly_id_04_00000047.png predict OK (score: 0.10%).
    File anomaly_id_00_00000108.png predict OK (score: 0.06%).
"""
""" --- fan: accuracy:  0.92
    File normal_id_06_00000408.png predict OK (score: 100.00%).
    File normal_id_06_00000426.png predict OK (score: 100.00%).
    File normal_id_06_00000307.png predict OK (score: 100.00%).
    File normal_id_00_00000011.png predict OK (score: 99.99%).
    File normal_id_06_00000844.png predict OK (score: 99.99%).
    File normal_id_06_00000679.png predict OK (score: 99.98%).
    File normal_id_00_00000058.png predict OK (score: 99.97%).
    File normal_id_04_00000667.png predict OK (score: 99.94%).
    File normal_id_00_00000020.png predict OK (score: 99.90%).
    File normal_id_02_00000712.png predict OK (score: 99.88%).
    File normal_id_00_00000036.png predict OK (score: 99.79%).
    File normal_id_04_00000083.png predict OK (score: 99.76%).
    File normal_id_00_00000055.png predict OK (score: 99.75%).
    File normal_id_00_00000564.png predict OK (score: 99.68%).
    File normal_id_00_00000472.png predict OK (score: 99.46%).
    File normal_id_02_00000335.png predict OK (score: 99.46%).
    File anomaly_id_00_00000049.png predict NOK (score: 99.22%).
    File normal_id_00_00000043.png predict OK (score: 99.16%).
    File normal_id_04_00000811.png predict OK (score: 99.01%).
    File normal_id_06_00000684.png predict OK (score: 98.88%).
    File normal_id_00_00000016.png predict OK (score: 98.67%).
    File normal_id_06_00000291.png predict OK (score: 98.30%).
    File normal_id_04_00000767.png predict OK (score: 97.51%).
    File normal_id_04_00000175.png predict OK (score: 95.94%).
    File anomaly_id_00_00000008.png predict NOK (score: 94.96%).
    File normal_id_00_00000828.png predict OK (score: 93.86%).
    File normal_id_00_00000002.png predict OK (score: 91.77%).
    File normal_id_06_00000883.png predict OK (score: 88.81%).
    File anomaly_id_04_00000100.png predict NOK (score: 74.33%).
    File normal_id_06_00000227.png predict OK (score: 74.10%).
    File normal_id_00_00000217.png predict OK (score: 70.17%).
    File normal_id_00_00000040.png predict OK (score: 64.03%).
    File anomaly_id_00_00000011.png predict NOK (score: 64.02%).
    File anomaly_id_04_00000102.png predict OK (score: 55.52%).
    File normal_id_00_00000025.png predict NOK (score: 54.67%).
    File anomaly_id_00_00000376.png predict OK (score: 46.62%).
    File anomaly_id_04_00000092.png predict OK (score: 46.43%).
    File anomaly_id_00_00000054.png predict OK (score: 45.09%).
    File anomaly_id_04_00000335.png predict OK (score: 44.43%).
    File anomaly_id_00_00000304.png predict OK (score: 42.36%).
    File anomaly_id_00_00000278.png predict OK (score: 37.95%).
    File anomaly_id_00_00000324.png predict OK (score: 37.93%).
    File anomaly_id_00_00000260.png predict OK (score: 31.11%).
    File anomaly_id_00_00000026.png predict OK (score: 29.05%).
    File anomaly_id_06_00000013.png predict OK (score: 28.96%).
    File anomaly_id_00_00000074.png predict OK (score: 27.03%).
    File anomaly_id_02_00000350.png predict OK (score: 23.41%).
    File anomaly_id_02_00000071.png predict OK (score: 22.78%).
    File anomaly_id_04_00000117.png predict OK (score: 20.35%).
    File anomaly_id_00_00000018.png predict OK (score: 14.67%).
    File anomaly_id_00_00000028.png predict OK (score: 8.65%).
    File anomaly_id_02_00000154.png predict OK (score: 6.86%).
    File anomaly_id_00_00000017.png predict OK (score: 3.66%).
    File anomaly_id_00_00000393.png predict OK (score: 3.31%).
    File anomaly_id_02_00000276.png predict OK (score: 2.15%).
    File anomaly_id_00_00000003.png predict OK (score: 2.03%).
    File anomaly_id_00_00000058.png predict OK (score: 1.15%).
    File anomaly_id_02_00000191.png predict OK (score: 1.00%).
    File anomaly_id_00_00000013.png predict OK (score: 0.51%).
    File anomaly_id_02_00000147.png predict OK (score: 0.21%).
"""
""" --- ToyCar: accuracy:  0.93
    File normal_id_03_00000479.png predict OK (score: 99.90%).
    File normal_id_02_00000238.png predict OK (score: 99.89%).
    File normal_id_01_00000017.png predict OK (score: 99.87%).
    File normal_id_01_00000681.png predict OK (score: 99.85%).
    File normal_id_01_00000003.png predict OK (score: 99.81%).
    File normal_id_01_00000008.png predict OK (score: 99.80%).
    File normal_id_01_00000076.png predict OK (score: 99.75%).
    File normal_id_02_00000051.png predict OK (score: 99.72%).
    File normal_id_01_00000164.png predict OK (score: 99.64%).
    File normal_id_01_00000029.png predict OK (score: 99.58%).
    File normal_id_01_00000025.png predict OK (score: 99.43%).
    File normal_id_01_00000016.png predict OK (score: 99.39%).
    File normal_id_01_00000155.png predict OK (score: 99.01%).
    File normal_id_03_00000353.png predict OK (score: 98.25%).
    File normal_id_01_00000013.png predict OK (score: 98.12%).
    File normal_id_01_00000150.png predict OK (score: 98.06%).
    File anomaly_id_01_00000015.png predict NOK (score: 97.41%).
    File normal_id_04_00000507.png predict OK (score: 97.13%).
    File normal_id_02_00000904.png predict OK (score: 96.13%).
    File normal_id_01_00000018.png predict OK (score: 95.17%).
    File normal_id_01_00000026.png predict OK (score: 94.69%).
    File normal_id_01_00000011.png predict OK (score: 94.67%).
    File normal_id_02_00000676.png predict OK (score: 94.03%).
    File normal_id_02_00000903.png predict OK (score: 93.53%).
    File normal_id_03_00000278.png predict OK (score: 93.08%).
    File normal_id_04_00000255.png predict OK (score: 92.12%).
    File anomaly_id_02_00000029.png predict NOK (score: 89.74%).
    File normal_id_01_00000006.png predict OK (score: 83.35%).
    File normal_id_01_00000160.png predict OK (score: 81.52%).
    File anomaly_id_01_00000095.png predict NOK (score: 78.01%).
    File normal_id_01_00000021.png predict OK (score: 69.46%).
    File normal_id_02_00000348.png predict OK (score: 60.57%).
    File anomaly_id_01_00000013.png predict OK (score: 26.40%).
    File anomaly_id_02_00000147.png predict OK (score: 12.69%).
    File normal_id_01_00000023.png predict NOK (score: 5.13%).
    File anomaly_id_01_00000117.png predict OK (score: 3.36%).
    File anomaly_id_01_00000045.png predict OK (score: 1.80%).
    File anomaly_id_03_00000118.png predict OK (score: 1.33%).
    File anomaly_id_02_00000027.png predict OK (score: 1.22%).
    File anomaly_id_01_00000002.png predict OK (score: 1.16%).
    File anomaly_id_01_00000054.png predict OK (score: 0.67%).
    File anomaly_id_03_00000232.png predict OK (score: 0.53%).
    File anomaly_id_03_00000111.png predict OK (score: 0.44%).
    File anomaly_id_01_00000108.png predict OK (score: 0.44%).
    File anomaly_id_03_00000205.png predict OK (score: 0.39%).
    File anomaly_id_01_00000008.png predict OK (score: 0.29%).
    File anomaly_id_01_00000018.png predict OK (score: 0.17%).
    File anomaly_id_01_00000021.png predict OK (score: 0.15%).
    File anomaly_id_02_00000252.png predict OK (score: 0.06%).
    File anomaly_id_01_00000019.png predict OK (score: 0.04%).
    File anomaly_id_01_00000136.png predict OK (score: 0.03%).
    File anomaly_id_01_00000164.png predict OK (score: 0.02%).
    File anomaly_id_01_00000105.png predict OK (score: 0.02%).
    File anomaly_id_02_00000048.png predict OK (score: 0.02%).
    File anomaly_id_04_00000077.png predict OK (score: 0.02%).
    File anomaly_id_02_00000014.png predict OK (score: 0.02%).
    File anomaly_id_01_00000098.png predict OK (score: 0.01%).
    File anomaly_id_01_00000101.png predict OK (score: 0.00%).
    File anomaly_id_01_00000010.png predict OK (score: 0.00%).
    File anomaly_id_01_00000091.png predict OK (score: 0.00%).
"""
""" --- ToyConveyor: accuracy:  0.50
    File normal_id_01_00000014.png predict OK (score: 100.00%).
    File anomaly_id_01_00000023.png predict NOK (score: 100.00%).
    File normal_id_01_00000911.png predict OK (score: 100.00%).
    File normal_id_01_00000019.png predict OK (score: 100.00%).
    File normal_id_03_00000514.png predict OK (score: 100.00%).
    File anomaly_id_01_00000005.png predict NOK (score: 100.00%).
    File normal_id_01_00000067.png predict OK (score: 100.00%).
    File normal_id_01_00000073.png predict OK (score: 100.00%).
    File normal_id_03_00000487.png predict OK (score: 100.00%).
    File normal_id_01_00000071.png predict OK (score: 100.00%).
    File anomaly_id_01_00000027.png predict NOK (score: 100.00%).
    File normal_id_01_00000079.png predict OK (score: 100.00%).
    File normal_id_03_00000720.png predict OK (score: 100.00%).
    File normal_id_01_00000075.png predict OK (score: 100.00%).
    File normal_id_01_00000062.png predict OK (score: 100.00%).
    File normal_id_01_00000077.png predict OK (score: 100.00%).
    File normal_id_01_00000004.png predict OK (score: 100.00%).
    File normal_id_02_00000227.png predict OK (score: 100.00%).
    File normal_id_01_00000011.png predict OK (score: 100.00%).
    File anomaly_id_01_00000094.png predict NOK (score: 100.00%).
    File normal_id_01_00000002.png predict OK (score: 100.00%).
    File normal_id_01_00000246.png predict OK (score: 100.00%).
    File anomaly_id_03_00000259.png predict NOK (score: 100.00%).
    File normal_id_01_00000032.png predict OK (score: 100.00%).
    File normal_id_01_00000026.png predict OK (score: 100.00%).
    File normal_id_01_00000021.png predict OK (score: 100.00%).
    File normal_id_03_00000377.png predict OK (score: 100.00%).
    File anomaly_id_01_00000101.png predict NOK (score: 100.00%).
    File normal_id_01_00000543.png predict OK (score: 100.00%).
    File normal_id_01_00000397.png predict OK (score: 100.00%).
    File normal_id_01_00000027.png predict OK (score: 100.00%).
    File normal_id_01_00000023.png predict OK (score: 100.00%).
    File anomaly_id_02_00000294.png predict NOK (score: 100.00%).
    File normal_id_01_00000010.png predict OK (score: 100.00%).
    File normal_id_01_00000064.png predict OK (score: 100.00%).
    File anomaly_id_01_00000098.png predict NOK (score: 100.00%).
    File anomaly_id_01_00000065.png predict NOK (score: 100.00%).
    File anomaly_id_01_00000325.png predict NOK (score: 100.00%).
    File normal_id_01_00000078.png predict OK (score: 100.00%).
    File anomaly_id_03_00000276.png predict NOK (score: 100.00%).
    File anomaly_id_02_00000081.png predict NOK (score: 100.00%).
    File anomaly_id_01_00000105.png predict NOK (score: 100.00%).
    File anomaly_id_02_00000342.png predict NOK (score: 100.00%).
    File normal_id_02_00000754.png predict OK (score: 100.00%).
    File anomaly_id_01_00000003.png predict NOK (score: 100.00%).
    File anomaly_id_01_00000092.png predict NOK (score: 100.00%).
    File anomaly_id_03_00000081.png predict NOK (score: 99.99%).
    File anomaly_id_01_00000060.png predict NOK (score: 99.99%).
    File anomaly_id_01_00000020.png predict NOK (score: 99.99%).
    File anomaly_id_01_00000075.png predict NOK (score: 99.98%).
    File anomaly_id_01_00000095.png predict NOK (score: 99.98%).
    File anomaly_id_01_00000057.png predict NOK (score: 99.97%).
    File anomaly_id_01_00000014.png predict NOK (score: 99.97%).
    File anomaly_id_01_00000067.png predict NOK (score: 99.86%).
    File anomaly_id_01_00000059.png predict NOK (score: 99.84%).
    File anomaly_id_01_00000016.png predict NOK (score: 99.84%).
    File anomaly_id_01_00000350.png predict NOK (score: 99.74%).
    File anomaly_id_01_00000064.png predict NOK (score: 99.27%).
    File anomaly_id_01_00000070.png predict NOK (score: 98.68%).
    File anomaly_id_01_00000063.png predict NOK (score: 98.49%).
"""




