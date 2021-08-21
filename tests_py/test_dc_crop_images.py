# -*- coding: utf-8 -*-

"""
    crop images : remove the white margin
    for one machine e.g. 'valve' (list_datasets[0])
    from: https://blog.keras.io/building-autoencoders-in-keras.html
"""

from utils import list_datasets, folders_train_test, rootFolder
import joblib
import sys
import datetime

machine = list_datasets[0]
print('machine:', machine) # use the first folder: e.g. 'valve'

from PIL import Image
 
pathImage = r"/Users/david/DEVS_LOCAL/dev-ia-son/partage-ia-son/data/valve/png_test_v4/anomaly_id_02_00000008.png"
im = Image.open(pathImage)
width, height = im.size
 
# Setting the points for cropped image
left = 54
top = 36
right = 387
bottom = 252
 

# (It will not change original image)
im1 = im.crop((left, top, right, bottom))
 
# im1.show()

im1.save(pathImage)

