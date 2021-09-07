# -*- coding: utf-8 -*-

"""
    Convert wavs from [machine]/wavs to color spectrograms into folders
    # construct dataset of images spectrogram in folders:
        # [machine]/png_v4/normal
        # [machine]/png_v4/anormal
        # [machine]/png_test_v4 is used for imaegs tests never seen by the model (normal & anormal images)

"""

from os import listdir
from os.path import isfile, join
import numpy as np
import sys


nbImagesTotestEachClass = 30 # put this count of images in png_test (for each class)
dictStat = {'normal': 0, 'anomaly': 0} # stats count of files in 2 classes
 
from SoundFile import SoundFile
from utils import list_datasets, rootFolder
import datetime

now = datetime.datetime.now()
print("*************** Start... ******************", now.strftime("%Y-%m-%d %H:%M:%S"))

# from wavs/*.wav, convert to pngs + split pngs in folders png_v3/normal + png_v3/anomaly (train data)
for folder_machine in list_datasets: # ['valve']    
    # todo: empty the 3 png_* folders before start
    # use_folder = rootFolder + 'data/' + folder_machine + '/wavs_mini/' # test
    use_folder = rootFolder + 'data/' + folder_machine + '/wavs/' 
    wavfiles = [f for f in listdir(use_folder) if isfile(join(use_folder, f))]
    if '.DS_Store' in wavfiles:
        wavfiles.remove('.DS_Store')
        
    nbWavs = len(wavfiles)
    countImages =  0    # counter for all images
    countImagesNormal =  0   
    countImagesAnomaly =  0
    
    # count files in each class
    for f in wavfiles:
        arrName = f.split("_") # anomaly_id_00_00000001.wav
        classPrefix = arrName[0] # 'normal' or 'anomaly'
        dictStat['normal'] =  dictStat['normal'] + 1 if classPrefix == 'normal' else dictStat['normal']
        dictStat['anomaly'] =  dictStat['anomaly'] + 1 if classPrefix == 'anomaly' else dictStat['anomaly']

    # print('dictStat: ', dictStat) # 479 anomaly 3291 normal

    # random choose of images for testing at the end: put them in png_test folder_machine
    arrIndicesImagesToTestNormal = np.random.randint(1, dictStat['normal'], nbImagesTotestEachClass) 
    arrIndicesImagesToTestAnormal = np.random.randint(1, dictStat['anomaly'], nbImagesTotestEachClass) 
    
    print('nbWavs: ', nbWavs, ' in ', use_folder)
    print('test normal:', arrIndicesImagesToTestNormal)
    print('test anormal:', arrIndicesImagesToTestAnormal)

    for f in wavfiles:
        arrName = f.split("_") # anomaly_id_00_00000001.wav
        out_folder_png = ''
        classPrefix = arrName[0] # 'normal' or 'anomaly'
        countImages += 1
        if classPrefix == 'normal':
            countImagesNormal += 1
        if classPrefix == 'anomaly':    
            countImagesAnomaly += 1

        out_folder_png = rootFolder + 'data/' + folder_machine + '/png_v4/' + classPrefix + '/' # data/slider/png_v4/normal/
        out_folder_png_test = rootFolder + 'data/' + folder_machine + '/png_test_v4/'
        
        # use some anomaly for the training set:
        if classPrefix == 'normal' and countImagesNormal in arrIndicesImagesToTestNormal or classPrefix == 'anomaly' and countImagesAnomaly in arrIndicesImagesToTestAnormal:
            out_folder_png = out_folder_png_test
            # print('!!! image test: ', classPrefix)
            
        if countImages % 100 == 0:
            print('countImages generated...: ', countImages, ' / ', nbWavs, '(', folder_machine, ')')
        
        s = SoundFile(use_folder + f, out_folder_png)
        s.exportMelSpectrogramColor() # create color file    

now = datetime.datetime.now()
print("*************** End ******************", now.strftime("%Y-%m-%d %H:%M:%S"))


