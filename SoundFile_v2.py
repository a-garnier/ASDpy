#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    SoundFile class V2: from code provided by Jeremy
    
    adapter pour générer de meilleures images ?

"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import os
from pathlib import Path
from PIL import Image


class SoundFile:

    def spectrogram(audio, fe, dt):
        """
        calcul le spectrogramme d'un signal audio par la méthode stft de librosa
        :param audio: np.ndarray: time series data
        :param fe: fréquence d'échantillonage
        :param dt: int>0: résolution temporelle de calcul de la STFT
        :return:np.ndarray: partie réelle de la stft
        """
        return np.abs(librosa.stft(audio,
                                n_fft=int(dt * fe),
                                hop_length=int(dt * fe)
                                )
                    )

    def plot_spectrogram(audio, fe, dt=0.025):
        """
        calcul et affiche le spectrogramme d'un fichier audio
        :param audio:signal audio à transformer 
        :param fe: fréquence d'échantillonnage de l'audio utilisé
        :param dt: int>0: résolution temporelle de la STFT
        :return: None
        """
        im = spectrogram(audio, fe, dt)
        sns.heatmap(np.rot90(im.T), cmap='inferno', vmin=0, vmax=np.max(im) / 3)
        loc, labels = plt.xticks()
        l = np.round((loc - loc.min()) * len(audio) / fe / loc.max(), 2)
        plt.xticks(loc, l)
        loc, labels = plt.yticks()
        l = np.array(loc[::-1] * fe / 2 / loc.max(), dtype=int)
        plt.yticks(loc, l)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
    def logMelSpectrogram(audio, fe, dt):
        """
        :param audio: signal audio à transformer en logMEL
        :param fe: fréquence d'échantillonnage de l'audio
        :param dt: résolution temporelle de la transformation en spectrogramme
        :return: np.array: logMelSpectrogram
        """
        # Spectrogram
        stfts = np.abs(librosa.stft(audio,
                                n_fft=int(dt * fe),
                                hop_length=int(dt * fe),
                                center=True
                                )).T
        num_spectrogram_bins = stfts.shape[-1]
        # MEL filter
        linear_to_mel_weight_matrix = librosa.filters.mel(
        sr=fe,
        n_fft=int(dt * fe) + 1,
        n_mels=num_spectrogram_bins,
        ).T

        # Apply the filter to the spectrogram
        mel_spectrograms = np.tensordot(
        stfts,
        linear_to_mel_weight_matrix,
        1
        )
        return np.rot90(np.log(mel_spectrograms + 1e-6))

    def plot_logMelSpectrogram(audio, fe, dt=0.025):
        """
        affichage du logMelSpectrogram d'un signal audio
        :param audio: signal audio à représenter en temps-fréquence
        :param fe: fréquence d 'échantillonnage du signal analysé
        :param dt: résolution temporelle du spectogramme à calculer
        :return: None
        """
        sns.heatmap(np.rot90(logMelSpectrogram(audio, fe, dt)), cmap='inferno', vmin=-6)
        loc, labels = plt.xticks()
        l = np.round((loc - loc.min()) * len(audio) / fe / loc.max(), 2)
        plt.xticks(loc, l)
        plt.yticks([])
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Mel)")