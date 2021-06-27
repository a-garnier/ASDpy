# -*- coding: utf-8 -*-

"""
    Convert wav to spectrograms
"""

# construct dataset of images spectrogram if folders:
# train_png/normal
# train_png/anormal
# test_png/normal
# test_png/anormal

# from array import array

from os import listdir
from os.path import isfile, join
import random



from SoundFile import SoundFile
from utils import list_datasets, folders_train_test, rootFolder



# todo : test : normal + anormal prefix

# root_folder = './data/fan/'
# train_folder = root_folder + 'train/'

# limit = 100 # number of file to treat
# indiceFile = 0

for folder in list_datasets:
    for ftt in folders_train_test: #test, train
        use_folder = rootFolder + 'data/' + folder + '/' + ftt + '/'
        wavfiles = [f for f in listdir(use_folder) if isfile(join(use_folder, f))]
        if '.DS_Store' in wavfiles:
            wavfiles.remove('.DS_Store')
        
        for f in wavfiles:
            arrName = f.split("_") # anomaly_id_00_00000001.wav
            out_folder_png = ''
            classPrefix = arrName[0] #normal or anomaly
            
            out_folder_png = rootFolder + 'data/' + folder + '/' + ftt + '_png/' + classPrefix + '/' # data//slider/train_png/normal/
            # use some anomaly for the training set:
            if classPrefix == 'anomaly':
                randf = random.choice(folders_train_test) # random test or train
                out_folder_png = rootFolder + 'data/' + folder + '/' + randf + '_png/' + classPrefix + '/'
            print('out_folder_png:', out_folder_png)
    
            # break
            s = SoundFile(use_folder + f, out_folder_png)
            s.exportMelSpectrogram()
            indiceFile = indiceFile + 1
            # if indiceFile > limit:
            #     break
    
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




