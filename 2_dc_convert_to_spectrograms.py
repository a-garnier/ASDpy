# -*- coding: utf-8 -*-

"""
    Convert wav to spectrograms into folders
    # construct dataset of images spectrogram in folders:
        # png_train/normal
        # png_train/anormal
        # png_validation/normal
        # png_validation/anormal
        # png_test is used for some tests at the end

"""

from os import listdir
from os.path import isfile, join
import random
import numpy as np
import sys


countImages =  0    # count all images
nbImagesTotest = 10 # put these images in png_test (for each train + validation)

from SoundFile import SoundFile
from utils import list_datasets, folders_train_test, rootFolder
import datetime


now = datetime.datetime.now()
print("*************** Start... ******************", now.strftime("%Y-%m-%d %H:%M:%S"))


for folder in list_datasets:
    for ftt in folders_train_test: # train, validation <-- each machine must have these 2 folders which contains the wav files
        # todo: rename 'test' folder to 'validation'
        # todo: empty the 3 png_* folders before start
        use_folder = rootFolder + 'data/' + folder + '/' + ftt + '/'
        wavfiles = [f for f in listdir(use_folder) if isfile(join(use_folder, f))]
        if '.DS_Store' in wavfiles:
            wavfiles.remove('.DS_Store')
            
        nbImages = len(wavfiles)
        
        # random choose of images for testing at the end: put them in png_test folder
        arrIndicesImagesToTest = np.random.randint(1, nbImages, nbImagesTotest)
        print('nbImages: ', nbImages, ' in ', use_folder)
        # print('test:', arrIndicesImagesToTest)
        # sys.exit()
        
        for f in wavfiles:
            arrName = f.split("_") # anomaly_id_00_00000001.wav
            out_folder_png = ''
            classPrefix = arrName[0] # 'normal' or 'anomaly'
            
            out_folder_png = rootFolder + 'data/' + folder + '/png_' + ftt + '/' + classPrefix + '/' # data//slider/train_png/normal/
            # use some anomaly for the training set:
            if classPrefix == 'anomaly':
                randf = random.choice(folders_train_test) # random test or train
                out_folder_png = rootFolder + 'data/' + folder + '/png_' + randf + '/' + classPrefix + '/'
            if countImages in arrIndicesImagesToTest:
                out_folder_png = rootFolder + 'data/' + folder + '/png_test/'
            # print('out_folder_png:', out_folder_png)
            if countImages % 400 == 0:
                print('countImages generated...: ', countImages)
            # break
            s = SoundFile(use_folder + f, out_folder_png)
            s.exportMelSpectrogram()
            countImages = countImages + 1


now = datetime.datetime.now()
print("*************** End ******************", now.strftime("%Y-%m-%d %H:%M:%S"))


# print('nbImages generated: ', nbImages)

# test 1 file ok:
# s1 = SoundFile('./fan/train_mini/normal_id_00_00000001.wav', out_folder_png)
# s1.exportMelSpectrogram()

    
    
# # x-axis has been converted to time using our sample rate. 
# # matplotlib plt.plot(y), would output the same figure, but with sample 
# # number on the x-axis instead of seconds
# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(samples, sr=sample_rate)

# # listen it 
# from IPython.display import Audio
# Audio(AUDIO_FILE)

# # generate spectogram
# # sgram = librosa.stft(samples)
# # librosa.display.specshow(sgram)

# # generate spectogram
# # use the mel-scale instead of raw frequency - v1
# sgram_mag, _ = librosa.magphase(sgram)
# mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate)
# librosa.display.specshow(mel_scale_sgram)



# # use the decibel scale to get the final Mel Spectrogram - v2
# mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
# librosa.display.specshow(mel_sgram, sr=sample_rate, x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')




