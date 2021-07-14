import numpy as np
import pandas as pd

#import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

from joblib import dump, load
#import pickle


from SoundFile import SoundFile


def convertImage2DataframeDir(imageFolder):

    pngfiles = [f for f in listdir(imageFolder) if isfile(join(imageFolder, f))]

    i=0
    bilan = []

    for f in pngfiles:
        if f[-4:] != '.png':
            # ignore non .png files
            continue

        type = f.split('_')[0]

        if type == 'normal':
            anomaly = 0
        else:
            anomaly = 1

        imagePath = imageFolder + f

        img = plt.imread(imagePath)

        #print(img)

        # les images font 313px de large et 128 de haut
        # exception pour ToyCar : 128*344
        #print(img.shape)

        # retaille l'image en un vecteur ligne de 40064 éléments
        img = img.reshape(img.shape[0]*img.shape[1])

        #print(img.shape)

        img = np.concatenate((img, [anomaly]), axis=0)

        # ajoute le vecteur à la matrice de données
        bilan.append(img)

    return np.array(bilan)


machines = ['fan', 'pump', 'slider', 'ToyCar', 'ToyConveyor', 'valve']
#machines = ['ToyCar', 'ToyConveyor', 'valve']
sets = ['train', 'test']

for machine in machines:

    for s in sets:
        # example : '../data/fan/train_png/'
        imageFolder = '../../data/'+machine+'/'+s+'_png/'

        print("Convert directory ", imageFolder)

        df = convertImage2DataframeDir(imageFolder)
        print(df.shape)

        saveName = '../../data/'+machine+'/df_'+machine+'_'+s+'.joblib'
        dump(df, saveName, compress=True)
        # df_fan_train.joblib :
        # compress = 9 => 188 Mo, enregistrement relativement long
        # compress = False => 1150 Mo
        # compress = True => 223

        # df_fan_test.joblib
        # compress = 9 => 95 Mo
        # compress = False => 586 Mo
        # compress = True => 113
    # break


# Convert directory  ../../data/fan/train_png/			(3675, 40065)
# Convert directory  ../../data/fan/test_png/ 			(1875, 40065)
# Convert directory  ../../data/pump/train_png/ 			(3349, 40065)
# Convert directory  ../../data/pump/test_png/ 			(856, 40065)
# Convert directory  ../../data/slider/train_png/ 		(2804, 40065)
# Convert directory  ../../data/slider/test_png/ 			(1290, 40065)
# Convert directory  ../../data/ToyCar/train_png/			(4000, 44033)
# Convert directory  ../../data/ToyCar/test_png/			(2459, 44033)
# Convert directory  ../../data/ToyConveyor/train_png/	(3000, 40065)
# Convert directory  ../../data/ToyConveyor/test_png/		(3509, 40065)
# Convert directory  ../../data/valve/train_png/			(3291, 40065)
# Convert directory  ../../data/valve/test_png/			(879, 40065)


