# -*- coding: utf-8 -*-
"""
    Convert wav to spectrograms
"""

# import numpy as np
# from array import array

from os import listdir
from os.path import isfile, join

from SoundFile import SoundFile


train_folder = './fan/train_mini/'
out_folder_png = './fan/train_mini_png/'


wavfiles = [f for f in listdir(train_folder) if isfile(join(train_folder, f))]
wavfiles.remove('.DS_Store')
for f in wavfiles:
    s = SoundFile(train_folder + f, out_folder_png)
    s.exportMelSpectrogram()
    
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




