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
modelName = '2021-08-03-16-10-24_valve_cnn.h5'

# ---
png_folder = rootFolder + 'data/' + machine + '/png_test/'
# load the model from disk
pathModel = rootFolder + 'dc_classifiers/' + modelName
model = load_model(pathModel)


# browse test files
df_result = pd.DataFrame(columns=['file', 'score']) # df result  with files and scores

indiceFile = 0
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
df_result.head(40)

# doesn't work!
#                           file           score
# 38   normal_id_00_00000028.png    [0.99925923]
# 16   normal_id_00_00000026.png    [0.98695207]
# 15   normal_id_00_00000027.png     [0.9800088]
# 26  anomaly_id_00_00000015.png    [0.97717464]
# 17   normal_id_00_00000032.png     [0.9740279]
# 30  anomaly_id_00_00000002.png     [0.9731842]
# 25  anomaly_id_00_00000011.png    [0.89022774]
# 35  anomaly_id_00_00000018.png     [0.8522206]
# 32  anomaly_id_00_00000017.png     [0.7758606]
# 34  anomaly_id_00_00000019.png    [0.76791215]
# 33  anomaly_id_00_00000003.png     [0.7491205]
# 27  anomaly_id_00_00000001.png    [0.74880254]
# 23  anomaly_id_00_00000004.png       [0.74252]
# 37  anomaly_id_00_00000009.png     [0.7175734]
# 22  anomaly_id_00_00000010.png     [0.7062656]
# 36  anomaly_id_00_00000008.png    [0.69348013]
# 20  anomaly_id_00_00000012.png    [0.68291175]
# 1    normal_id_00_00000022.png     [0.6737879]
# 29  anomaly_id_00_00000014.png    [0.67345834]
# 31  anomaly_id_00_00000016.png     [0.6071769]
# 24  anomaly_id_00_00000005.png    [0.58735144]
# 19  anomaly_id_00_00000013.png     [0.5052569]
# 13   normal_id_00_00000019.png     [0.4960527]
# 21  anomaly_id_00_00000006.png    [0.47579652]
# 0    normal_id_00_00000036.png    [0.32417947]
# 28  anomaly_id_00_00000000.png     [0.3050675]
# 2    normal_id_00_00000023.png    [0.28601164]
# 3    normal_id_00_00000037.png    [0.22367254]
# 39   normal_id_00_00000029.png    [0.21020114]
# 12   normal_id_00_00000025.png    [0.21009189]
# 9    normal_id_00_00000024.png    [0.17366078]
# 18  anomaly_id_00_00000007.png    [0.17295265]
# 7    normal_id_00_00000020.png    [0.15384784]
# 6    normal_id_00_00000034.png    [0.13743669]
# 10   normal_id_00_00000030.png     [0.1017378]
# 4    normal_id_00_00000021.png     [0.0848209]
# 11   normal_id_00_00000031.png   [0.045676976]
# 8    normal_id_00_00000018.png   [0.025991052]
# 5    normal_id_00_00000035.png    [0.00726524]
# 14   normal_id_00_00000033.png  [0.0016297102]