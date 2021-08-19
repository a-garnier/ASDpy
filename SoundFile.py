#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 12:15:44 2021

SoundFile class
"""

# https://librosa.org/doc/latest/search.html?q=export&check_keywords=yes&area=default#
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import os
from pathlib import Path

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


class SoundFile:
    """ 
    ----------
    SoundFile class contains informations about one sound related to a wav file

    ----------
    Parameters:
        name_file : full path + name of the file ('./fan/train/normal_id_00_00000000.wav')
        out_folder_png: name of output folder in order to save the spectograms as pngs
    """
    # settings
    # hop_length = 512 # number of samples per time-step in spectrogram
    # n_mels = 128 # number of bins in spectrogram. Height of image
    # time_steps = 384 # number of time-steps. Width of image

    def __init__(self, nameFile, out_folder_png):
        # print('init the audio file:', nameFile)
        self.nameFile = nameFile
        self.out_folder_png = out_folder_png
        self.samples, self.sample_rate = librosa.load(nameFile, sr=None)
        # self.samples : (160000,) complex matrix
        # self.sample_rate : 16000
        self.sgram = librosa.stft(self.samples)
        # self.sgram.shape : (1025, 313) matrix
        # (f,t) => f = frequency, t = time
        #print(self.sgram)
        # n_fft=2000 => (1001, 321)
        # n_fft=500 => (251, 1281)
        # n_fft=32 => (17, 200001)
    
    # ok: black and white with skimage
    def exportMelSpectrogram(self): 
        # mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
        sgram_mag, _ = librosa.magphase(self.sgram)
        mels = librosa.feature.melspectrogram(S=sgram_mag, sr=self.sample_rate)
        mels = np.log(mels + 1e-9) # add small number to avoid log(0)
        # min-max scale to fit inside 8-bit range
        img = scale_minmax(mels, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0) # put low frequencies at the bottom in image
        img = 255-img # invert. make black==more energy
        p = Path(self.nameFile)
        namePng = self.out_folder_png + p.stem + '.png' #same name for the png file as wav
        skimage.io.imsave(namePng, img)

        
    #  ok : export a color spectrogram as png 
    def exportMelSpectrogramColor(self):
         # use the decibel scale to get the final Mel Spectrogram - v2
        sgram_mag, _ = librosa.magphase(self.sgram)
        mels = librosa.feature.melspectrogram(S=sgram_mag, sr=self.sample_rate)
        mel_sgram = librosa.amplitude_to_db(mels, ref=np.min)
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        librosa.display.specshow(mel_sgram, sr=self.sample_rate,  ax=ax, x_axis=None, y_axis=None)
        p = Path(self.nameFile)
        namePng = self.out_folder_png + p.stem + '.png' # same name for the png file as wav
        fig.savefig(namePng)
    
    #  ok : display a color spectrogram
    def showMelSpectrogram(self):
         # use the decibel scale to get the final Mel Spectrogram - v2
        sgram_mag, _ = librosa.magphase(self.sgram)
        mels = librosa.feature.melspectrogram(S=sgram_mag, sr=self.sample_rate)
        mel_sgram = librosa.amplitude_to_db(mels, ref=np.min)
        librosa.display.specshow(mel_sgram, sr=self.sample_rate, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB'); # vertical legend

