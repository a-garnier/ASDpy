# -*- coding: utf-8 -*-

"""
    Convert wav to spectrograms into folders
    # construct dataset of images spectrogram in folders:
        # v3/normal
        # v3/anormal
        # png_test is used for some tests at the end (normal & anormally images)

"""

from os import listdir
from os.path import isfile, join
import random
import numpy as np
import sys


countImages =  0    # count all images
nbImagesTotestEachClass = 20 # put this count of images in png_test
dictStat = {'normal': 0, 'anomaly': 0} # count of files in 2 classes
 
from SoundFile import SoundFile
from utils import list_datasets, folders_train_test, rootFolder
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
    
    # random choose of images for testing at the end: put them in png_test folder_machine
    for f in wavfiles:
        arrName = f.split("_") # anomaly_id_00_00000001.wav
        classPrefix = arrName[0] # 'normal' or 'anomaly'
        dictStat['normal'] =  dictStat['normal'] + 1 if classPrefix == 'normal' else dictStat['normal']
        dictStat['anomaly'] =  dictStat['anomaly'] + 1 if classPrefix == 'anomaly' else dictStat['anomaly']


    # sys.exit()    
    # 479 anomaly 3291 normal
    arrIndicesImagesToTestNormal = np.random.randint(1, dictStat['normal'], nbImagesTotestEachClass) 
    arrIndicesImagesToTestAnormal = np.random.randint(1, dictStat['anomaly'], nbImagesTotestEachClass) 
    
    print('nbWavs: ', nbWavs, ' in ', use_folder)
    # print('test:', arrIndicesImagesToTestNormal)
    # print('test:', arrIndicesImagesToTestAnormal)
    # sys.exit()

    for f in wavfiles:
        arrName = f.split("_") # anomaly_id_00_00000001.wav
        out_folder_png = ''
        classPrefix = arrName[0] # 'normal' or 'anomaly'
        
        # out_folder_png = rootFolder + 'data/' + folder_machine + '/png_v3_mini/' + classPrefix + '/' # data/slider/png_v3/normal/
        out_folder_png = rootFolder + 'data/' + folder_machine + '/png_v3/' + classPrefix + '/' # data/slider/png_v3/normal/
        out_folder_png_test = rootFolder + 'data/' + folder_machine + '/png_test/'
        # use some anomaly for the training set:
        if classPrefix == 'normal' and countImages in arrIndicesImagesToTestNormal or classPrefix == 'anomaly' and countImages in arrIndicesImagesToTestAnormal:
            out_folder_png = out_folder_png_test
            # print('!!!to_test:', out_folder_png)
            # sys.exit()
        # print('out_folder_png:', out_folder_png)
        if countImages % 200 == 0:
            print('countImages generated...: ', countImages, ' / ', nbWavs)
        # break
        s = SoundFile(use_folder + f, out_folder_png)
        s.exportMelSpectrogramColor() # create color file
        # s.exportMelSpectrogram() # create black & white file
        # s.showMelSpectrogram()
        countImages = countImages + 1
    # put randomly in png_test folder
    

now = datetime.datetime.now()
print("*************** End ******************", now.strftime("%Y-%m-%d %H:%M:%S"))


