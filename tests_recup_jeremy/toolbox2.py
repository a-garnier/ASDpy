import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import time
import tensorflow as tf
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import auc

from tqdm.notebook import tqdm, trange

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.models import Model, Sequential, load_model
AUTOTUNE = tf.data.AUTOTUNE

"""
variables et données par défaut nécessaires au fonctionnement
"""
spectro_size = (512,256)
machine_list= []
machine_id_list=[] # doit être généra par le script appelant: il contient les noms des différentes machines etudiées
max_duration = 10
batch_size = 12

def cut_audio(signal, frequency, max_lenght_s = max_duration):
    '''
    coupe un signal à la longueur maximale spécifiée en secondes
    :parameter signal: signal d'entrée à tronquer
    :parameter frequency: frequence du signal d'entrée
    :max_lenght: durée maximum du signal en s
    :return: array
    '''
    return signal[0:min(frequency*max_lenght_s+1,len(signal))]

def load_audio(audio_path, duration = None):
    '''
    charge un fichier audio
    :param audio_path: str: fichier à charger
    :param duration: limite de temps à ne pas dépasser
    :return: y: np.ndarray: audio time series, sr: int >0 fréquence d'échantillonnage
    '''
    return librosa.load(audio_path, duration = duration, sr=None)

def plot_audio(audio_data, sr):
    '''
    affiche un fichier audio
    :param audio_data:np.ndarray: audio_data: time serie data
    :param sr:int: fréquence échantillannage
    :return: None
    '''

    # Intervalle de temps entre deux points
    dt = 1 / sr
    # Variable de temps en seconde.
    t = dt * np.arange(len(audio_data))
    plt.plot(t, audio_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

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

def min_max_scale(data):
    """
    scaling min_max du jeux de données
    recentre les données d'entrée entre 0 et 1
    :param data: tout type itérable
    :return: le même type
    """
    return (data-data.min())/data.max()

def plot_learning(history,title = None ):
    """
    affiche la courbe d'apprentissage d'un modèle après fitting
    :param history: history structure possédant la propriété "accuracy"
    :param title: titre du graphique
    :return: None
    """
    # Labels des axes
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    epochs = len(history.epoch)

    # Courbe de la précision sur l'échantillon d'entrainement
    plt.plot(np.arange(1 , epochs+1, 1),
           history.history['accuracy'],
           label = 'Training Accuracy',
           color = 'blue')

    # Courbe de la précision sur l'échantillon de validation
    plt.plot(np.arange(1 , epochs+1, 1),
           history.history['val_accuracy'],
           label = 'Validation Accuracy',
           color = 'red')

    # Affichage de la légende
    plt.legend()
    if title is not None:
        plt.title(title)

    # Affichage de la figure
    plt.show()

def decode_audio(audio_binary):
    """
    décode le signal binaire encodé en wav en signal time series tensorflow
    :param audio_binary: signal binaire d'un fichier audio wav ouvert avec tensorflow
    :return: tensor (time series audio, sr): fréquence d'échantillonnage
    """
    audio, sr = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1), sr

def get_spectrogram(waveform, sr,max_duration=max_duration):
    """
    calcul le spectrogram des signaux audio, limité en temps à max_duration
    :param waveform: signal audio
    :param sr:int>0: fréquence d'échantillonnage
    :param max_duration: int >0: temps maximum en s
    :return: np.ndarray:spectrogramme
    """
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([sr * max_duration] - tf.shape(waveform), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the
    # same length
    waveform = tf.cast(waveform, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(
        equal_length, frame_length=512, frame_step=256)

    spectrogram = tf.abs(spectrogram)
    return spectrogram

def get_label(file_path):
    """
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    retourne le nom du type de machine du fichier étudié (par exemple ToyCar)
    :param file_path: chemin du fichier étudié
    :return: tf.string: le nom du type de machine
    """
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[1]

def get_machine_id(file_path):
    """
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    retourne un identifiant de la machine correspondant au fichier en paramètre (exemple: ToyCar_00)

    :param file_path: fichier étudié
    :return: tf.string: identifiant de la machine
    """
    parts = tf.strings.split(file_path, os.path.sep)
    machine_id = parts[1] + '_' + tf.strings.split(parts[-1], '_')[2]
    return machine_id

def get_anomaly_label(file_path):
    """
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    retourne la classe d'anomalie du fichier son en fonction du nom du fichier
    :param file_path: chemin vers le fichier son étudié
    :return: int: label -  0 / 1: sans anomalie/avec anomalie
    """
    parts = tf.strings.split(file_path, os.path.sep)
    anomaly = tf.strings.split(parts[-1], '_')[0]
    if anomaly == 'anomaly':
        return 1
    else:
        return 0

def get_waveform_and_label(file_path, max_duration=max_duration): #label
    """
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    ouverture et décodage d'un fichier audio, détermination du label du type de la machine concernée (ex: ToyCar)
    :param file_path: list(string) ou tensor(string)
    :param max_duration: int: durée maximum du signal audio à charger
    :return: tensor(signal audio, type machine, fréquence échantillonnage)
    """
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform, sr = decode_audio(audio_binary)
    # on coupe pour les signaux superieurs à max_duration
    if len(waveform) > sr * max_duration:
        waveform = waveform[:sr * max_duration]
    return waveform, label, sr

def get_spectrogram_and_label_id(audio, label, sr):
    """
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    calcul le spectrogramme, et encode le type de machine du signal audio en fonction de la liste des labels utilisés .
    Le spectrogramme en sortie est redimensionné à la dimension spectro_size de la toolbox

    :param audio: nparray: signal audio à transformer
    :param label: string: type de machine du signal audio
    :param sr: fréquence d'échantillonnage du label audio
    :return: (nparray:spectrogramme, int:label numérique du type de machine)
    """
    global machine_list
    if len(machine_list)==0:
        print("machine_list est vide! ")
        raise

    spectrogram = get_spectrogram(audio, sr)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.resize(spectrogram,spectro_size)
    label_id = tf.argmax(label == machine_list)
    return spectrogram, label_id

def get_waveform_and_machine_id(file_path, max_duration=max_duration): #label
    """
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    ouverture et décodage d'un fichier audio et détermination de l'identifiant de la machine concernée (ex: ToyCar_00)
    :param file_path: list(string) ou tensor(string)
    :param max_duration: int: durée maximum du signal audio à charger
    :return: tensor(signal audio, id machine, fréquence échantillonnage)
    """
    machine_id = get_machine_id(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform, sr = decode_audio(audio_binary)
    # on coupe pour les signaux superieurs à max_duration
    if len(waveform) > sr * max_duration:
        waveform = waveform[:sr * max_duration]
    return waveform, machine_id, sr

def get_spectrogram_and_machine_id(audio, label, sr):
    """
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    calcul le spectrogramme, et encode la machine dont est issu le signal audio en fonction de la liste des labels utilisés

    :param audio: nparray: signal audio à transformer
    :param label: string: identifiant machine du signal audio
    :param sr: fréquence d'échantillonnage du signal audio
    :return: (nparray: spectrogramme, int: label machine_id numérique)
    """
    global machine_id_list
    if len(machine_id_list)==0:
        print("machine_id_list est vide! ")
        raise

    spectrogram = get_spectrogram(audio, sr)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.resize(spectrogram, spectro_size)
    label_id = tf.argmax(label == machine_id_list)
    return spectrogram, label_id

def get_waveform_and_anomaly(file_path, max_duration=max_duration):
    """
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    retourne le signal sonore décodé, la classe d'anomalie du fichier son, et la fréquence d'échantillonnage
    :param file_path: chemin vers le fichier son kaggle
    :return: (array: time series,int:anomalie,int: fréquence d'échantillonnage) (0: sans anomalie, 1: avec anomalie)
    """
    label = get_anomaly_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform, sr = decode_audio(audio_binary)
    # on coupe pour les signaux superieurs à 10s
    if len(waveform) > sr * max_duration:
        waveform = waveform[:sr * max_duration]
    return waveform, label, sr

def get_spectrogram_and_anomaly(audio, label, sr):
    """
        dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
        retourne le spectrogramme du signal sonore décodé, la classe d'anomalie

        :param file_path: chemin vers le fichier son kaggle
        :return: (nparray: spectrogramme,int: classe d'anomalie)
        """
    spectrogram = get_spectrogram(audio, sr)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.resize(spectrogram, spectro_size)
    return spectrogram, label

def get_waveform_and_label_and_anomaly(file_path):
    """
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    retourne le signal sonore décodé, le type de machine, et la classe d'anomalie du fichier son,
    ainsi que la fréquence d'échantillonnage
    :param file_path: chemin vers le fichier son kaggle
    :return: (array: time series,string:identifiant type machine,int:anomalie,int: fréquence d'échantillonnage)
    """
    label = get_label(file_path)
    anomaly = get_anomaly_label(file_path)

    audio_binary = tf.io.read_file(file_path)
    waveform, sr = decode_audio(audio_binary)
    # on coupe pour les signaux superieurs à 10s
    if len(waveform) > sr * 10:
        waveform = waveform[:sr * 10]
    return waveform, label, anomaly, sr

def get_spectrogram_and_label_and_anomaly(audio, label, anomaly, sr):
    """
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    retourne le spectrogramme du signal sonore décodé, le label numérique du type de machine et sa classe d'anomalie

    :param audio: array: time series signal sonore
    :param label: string: type de machine, appartenant au label_set
    :param anomaly: int: 0/1: classe normale/anormale
    :param sr: : int: fréquence d'échantillonnage
    :return: (nparray: spectrogramme , int: label du type de machine, int: anomalie)
    """
    global machine_list
    if len(machine_list)==0:
        print("machine_list est vide! ")
        raise
    spectrogram = get_spectrogram(audio, sr)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.resize(spectrogram, spectro_size)
    label_id = tf.argmax(label == machine_list)
    return spectrogram, label_id, anomaly

def get_waveform_and_machine_id_and_anomaly(file_path):
    """
        dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
        retourne le signal sonore décodé, l'identifiant de machine, et la classe d'anomalie du fichier son,
        ainsi que la fréquence d'échantillonnage
        :param file_path: chemin vers le fichier son kaggle
        :return: (array: time series,string:identifiant machine,int:anomalie,int: fréquence d'échantillonnage)
        """
    machine_id = get_machine_id(file_path)
    anomaly = get_anomaly_label(file_path)

    audio_binary = tf.io.read_file(file_path)
    waveform, sr = decode_audio(audio_binary)
    # on coupe pour les signaux superieurs à 10s
    if len(waveform) > sr * 10:
        waveform = waveform[:sr * 10]
    return waveform, machine_id, anomaly, sr

def get_spectrogram_and_machine_id_and_anomaly(audio, label, anomaly, sr):
    """
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    retourne le spectrogramme, l'identifiant machine encodé, et la classe normale/anormale du signal
    :param audio: array: signal audio décodé
    :param label: string: identifiant machine
    :param anomaly: int: classe normale/anormale
    :param sr: int: fréquence échantillonnage
    :return: (nparray: spectrogramme, int: identifiant machine, int: classe normale/anormale)
    """
    global machine_id_list
    if len(machine_id_list)==0:
        print("machine_id_list est vide! ")
        raise
    spectrogram = get_spectrogram(audio, sr)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.resize(spectrogram, spectro_size)
    label_id = tf.argmax(label == machine_id_list)
    return spectrogram, label_id, anomaly

def get_waveform_and_label_and_machine_id_and_anomaly(file_path):
    """
        dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
        retourne le signal sonore décodé, l'identifiant de machine, et la classe d'anomalie du fichier son,
        ainsi que la fréquence d'échantillonnage
        :param file_path: chemin vers le fichier son kaggle
        :return: (array: time series,string:identifiant machine,int:anomalie,int: fréquence d'échantillonnage)
        """
    machine_id = get_machine_id(file_path) #string
    anomaly = get_anomaly_label(file_path) #int
    label = get_label(file_path) #string
    audio_binary = tf.io.read_file(file_path)
    waveform, sr = decode_audio(audio_binary)
    # on coupe pour les signaux superieurs à 10s
    if len(waveform) > sr * max_duration:
        waveform = waveform[:sr * max_duration]
    return waveform, label,machine_id, anomaly, sr

def get_spectrogram_and_label_and_machine_id_and_anomaly(audio, label_str,machine_id_str, anomaly, sr):
    """
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    retourne le spectrogramme, l'identifiant machine encodé, et la classe normale/anormale du signal
    :param audio: array: signal audio décodé
    :param label: string: identifiant machine
    :param anomaly: int: classe normale/anormale
    :param sr: int: fréquence échantillonnage
    :param label_set: liste de string des identifiants de machine
    :return: (nparray: spectrogramme, int: identifiant machine, int: classe normale/anormale)
    """
    global machine_list
    if len(machine_list)==0:
        print("machine_list est vide! ")
        raise
    global machine_id_list
    if len(machine_id_list)==0:
        print("machine_id_list est vide! ")
        raise
    spectrogram = get_spectrogram(audio, sr)
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.resize(spectrogram, spectro_size)
    label_id = tf.argmax(label_str == machine_list)
    machine_id = tf.argmax(machine_id_str == machine_id_list)
    return spectrogram,label_id,machine_id, anomaly

def generator(machines, X,files,batch_size=batch_size):
    """
    méthode plus utilisée dans le notebook
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    Retourne un batch X_batch de (batch_size*3) lignes et les labels correspondant y_batch.
    Chaque ligne est respectivement un chemin vers un fichier audio et un label de type de machine ou identifiant machine

    2 lignes/3 sont les fichiers d'une première machine, et 1 ligne/3 est un fichier d'une  2eme machine au label différent

    :param machines: string: set de machines disponibles
    :param X: liste complète des chemins vers les fichiers audio
    :param batch_size: nombre de trio de fichiers à constituer
    :return: (string liste de fichiers, liste des labels des fichiers)
    """
    ID_range = range(0, len(X))
    machine_range = range(0, len(machines))
    while (True):
        for i in range(0, len(files), batch_size):
            X_batch = []
            y_batch = []
            for j in range(0, batch_size):
                machineID1, machineID2 = random.sample(machine_range, 2)
                id1_spec_1, id1_spec_2 = random.sample(range(len(X[machineID1])), 2)
                id2_spec_1 = random.sample(range(len(X[machineID2])), 1)[0]
                X_batch.extend([X[machineID1][id1_spec_1], X[machineID1][id1_spec_2], X[machineID2][id2_spec_1]])
                y_batch.extend([machineID1, machineID1, machineID2])

            X_batch = np.array(X_batch)  #
            # ligne nécessaire pour avoir les bonne dimensions en entrée du réseau mais empêche d'afficher par imshow
            X_batch = np.repeat(X_batch[..., np.newaxis], 1, -1)
            y_batch = np.array(y_batch)

            yield X_batch, y_batch  # np.zeros(len(X_batch))

def generator_by_machine(spec_dict, batch_size=batch_size):
    """
    Ce générateur n'est plus utilisé actuellement dans le notebook fina.
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    Retourne un batch X_batch de (batch_size*3) lignes et les labels correspondant y_batch.
    Chaque ligne retournée par le batch est respectivement un chemin vers un fichier audio et un label de type de machine ou identifiant machine
    Le générateur pioche 2 machines d'un type et 1 machine d'un autre type.

    2 lignes/3 sont les fichiers d'une première machine, et 1 ligne/3 est un fichier d'une  2eme machine au label différent

    :param spec_dict: dict: contient les infos de type/machine/spectro-anomaly-label
    :param batch_size: nombre de trio de fichiers à constituer
    :return: (string liste de fichiers, liste des labels des fichiers)
    """
    n_files = 0
    # si dict complet: liste des types de machines, sinon liste des machines dans un type
    for step in spec_dict:
        mach = spec_dict[step]
        n_files += len(mach)

    ID_range = range(0, len(spec_dict))
    machines = list(spec_dict)
    while (True):
        for i in range(0, n_files, batch_size):
            X_batch = []
            y_batch = []
            for j in range(0, batch_size):
                machineID1, machineID2 = random.sample(ID_range, 2)

                id1_spec_1, id1_spec_2 = random.sample(range(len(machines[machineID1])), 2)
                id2_spec_1 = random.sample(range(len(machines[machineID2])), 1)[0]

                X_batch.extend([spec_dict[machines[machineID1]]['spectro'][id1_spec_1],
                                spec_dict[machines[machineID1]]['spectro'][id1_spec_2],
                                spec_dict[machines[machineID2]]['spectro'][id2_spec_1]])

                y_batch.extend([spec_dict[machines[machineID1]]['machine_id'][id1_spec_1],
                                spec_dict[machines[machineID1]]['machine_id'][id1_spec_2],
                                spec_dict[machines[machineID2]]['machine_id'][id2_spec_1]])
            X_batch = np.array(X_batch)  #
            # ligne nécessaire pour avoir les bonne dimensions en entrée du réseau mais empêche d'afficher par imshow
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)

            yield X_batch, y_batch  # np.zeros(len(X_batch))

def generator_by_level(spec_dict,level=1, batch_size=batch_size):
    """
    Générateur de batch utilisé dans le dernier notebook.
    dans le cadre du dataset https://www.kaggle.com/daisukelab/dc2020task2
    Retourne un batch X_batch de (batch_size*3) lignes et les labels correspondant y_batch.
    Chaque ligne est respectivement un chemin vers un fichier audio et un label de type de machine ou identifiant machine

    2 lignes/3 sont les fichiers d'une première machine, et 1 ligne/3 est un fichier d'une  2eme machine différente.

    Selon le level choisi, la distinction entre machine se fait par type de machine (level=1) ou par identifiant machine (level=2).

    :param spec_dict: dict: contient les infos de type/machine/spectro-anomaly-label.
    :param level: level=1: generateur sur type de machine. Generateur = 2: generateur sur machine indivuelle
    :param batch_size: nombre de trio de fichiers à constituer
    :return: (string liste de fichiers, liste des labels des fichiers)
    """
    this_spect = []
    this_machid = []
  
    n_files=0
    mach_file=0
    base_level=0
    n_file_level = 0   

    ID_range = 0 
    machines = []
    machines_range = [] 
    
    for mach_type in spec_dict:
        
        if level==1:
          machines.append(mach_type)
          ID_range+=1
          base_level = n_files
          n_file_level = 0

        for mach in spec_dict[mach_type]:
         
          n_file_mach = len(spec_dict[mach_type][mach]['machine_id'])
          n_files += n_file_mach
          
          this_spect.extend(spec_dict[mach_type][mach]['spectro'])
          this_machid.extend(spec_dict[mach_type][mach]['machine_id'])

          if level==1:
            n_file_level += n_file_mach

          else:
            machines.append(mach)
            ID_range +=1
            n_file_level = n_file_mach
            machines_range.append(range(base_level,base_level + n_file_level))
            base_level = n_files          
        
        if level==1:
          machines_range.append(range(base_level,base_level + n_file_level))

    ID_range = range(0,ID_range) 
    while (True):
        for i in range(0, n_files, batch_size):
            X_batch = []
            y_batch = []
            for j in range(0, batch_size):
                ID1, ID2 = random.sample(ID_range, 2)
                ID1_range = machines_range[ID1]
                ID2_range = machines_range[ID2]
                id1_spec_1, id1_spec_2 = random.sample(ID1_range, 2)
                id2_spec_1 = random.sample(ID2_range,1)[0]
                
                X_batch.extend([this_spect[id1_spec_1],
                                this_spect[id1_spec_2],
                                this_spect[id2_spec_1]])

                y_batch.extend([this_machid[id1_spec_1],
                                this_machid[id1_spec_2],
                                this_machid[id2_spec_1]])
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)

            yield X_batch, y_batch


def createModel():
    """
    Cette méthode crée le modème d'encodage des spectros
    :return: Sequential model
    """

    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=[512,256,1]))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu')) # tests à 128
    model.add(Dropout(0.2))
    model.add(Dense(128)) # test à 64 un peu moins performant

    return model

def normL2(X1, X2):
    """
    calcule la distance euclidienne entre 2 vecteurs

    :param X1: vecteur 1
    :param X2: vecteur 2
    :return: distance euclidienne moyenne entre les points du vecteur
    """
    return tf.reduce_mean(tf.square(X1-X2), axis=-1)

def loss(y_true, y_pred):
    """
    méthode de caclcul de perte cherchant à minimiser la distance entre les 2 premiers vecteurs,
    tout en maximisant leur distance avec le 3eme.

    :param y_true:
    :param y_pred:
    :return:
    """
    p1_id1 = y_pred[::3]
    p1_id2 = y_pred[1::3]
    p2_id1 = y_pred[2::3]
#     return 1/2*tf.reduce_mean(normL2(p1_id1, p1_id2))
#     return 1/4* tf.math.log_sigmoid(tf.sigmoid(normL2(p1_id1, p2_id1)))
#     return -tf.math.log_sigmoid((tf.reduce_mean(-1/2*normL2(p1_id1, p1_id2) + 1/4* normL2(p1_id1, p2_id1) + 1/4* normL2(p1_id2, p2_id1)))) + tf.reduce_mean(1/10*normL2(p1_id1, p1_id2) - 1/4* tf.math.log_sigmoid(normL2(p1_id1, p2_id1)) - 1/4* tf.math.log_sigmoid(normL2(p1_id2, p2_id1)))
    return tf.reduce_mean(1/2*normL2(p1_id1, p1_id2)/(1e-8+1/4* normL2(p1_id1, p2_id1) + 1/4* normL2(p1_id2, p2_id1)))
    #return tf.reduce_mean(1/2*normL2(p1_id1, p1_id2) - 1/4* tf.math.log_sigmoid(normL2(p1_id1, p2_id1)) - 1/4* tf.math.log_sigmoid(normL2(p1_id2, p2_id1)))

def select_inputfile(data_dir, selected_machines):
    """
    méthode de chargement des fichiers, en fonction des type de machines analysées.

    :param data_dir: directory ou sont enregistrés les fichiers
    :param selected_machines: liste de type de machine à étudier
    :return: liste de fichiers train, liste  de fichiers test, liste des type de machine différents, liste de machines différentes,
    dictionnaire des données type/machine/fichier de train et de test
    """
    print("\ndataset sélectionnés:")

    files_train = []
    files_test = []
    for m in selected_machines:
        print('- ',m)

        mach_files_train = tf.io.gfile.glob(str(data_dir) + '/'+m+'/train/*')
        files_train.extend(mach_files_train)
        mach_files_test = tf.io.gfile.glob(str(data_dir) + '/'+m+'/test/*')
        files_test.extend(mach_files_test)

    #on récupère les éléments unique de la liste des machines
    machine_list = []     # liste des label uniques des machines
    machine_id_list = [] # liste des id unique des machines
    # labels = []      # liste des type machine, de même dimension que le nombre de fichiers
    # ids= []          # liste des identifiants machine de même dimension que le nombre de fichiers
    # labels_test = []
    # ids_test = []

    machine_train_dict = {}
    machine_test_dict={}

    print("\nrecherche des labels et id des machines sélectionnées, parmi tous les fichiers d'entraînement:")

    # à optimiser avec tensorflow?
    for file in tqdm(files_train):
        label_t = get_label(file) # on préserve la format tenseur .numpy().decode('utf-8')
        label = label_t.numpy().decode('utf-8')

        if label not in machine_list:
            machine_list.append(label)
            machine_train_dict[label]={}

        id_t = get_machine_id(file)
        id = id_t.numpy().decode('utf-8')

        if id not in machine_id_list:
            machine_id_list.append(id)
            machine_train_dict[label][id]=[]
            print('-',id)

        machine_train_dict[label][id].append(file)

    nb_machines=len(machine_id_list)

    print("\nrecherche des labels et id des machines sélectionnées, parmi tous les fichiers de tests:")

    # à optimiser avec tensorflow?
    for file in tqdm(files_test):
        label_t = get_label(file) # on préserve la format tenseur .numpy().decode('utf-8')
        label = label_t.numpy().decode('utf-8')

        if label not in machine_test_dict:
            machine_test_dict[label]={}

        id_t = get_machine_id(file)
        id = id_t.numpy().decode('utf-8')

        if id not in machine_test_dict[label]:
            machine_test_dict[label][id]=[]
            print('-',id)

        machine_test_dict[label][id].append(file)

    return files_train,files_test,machine_list,machine_id_list,machine_train_dict,machine_test_dict

def make_spectro_ds_from_files(file_list):
    """
    méthode d e création des spectros et des données de label, id_machine, anomalie, à partir d'une liste de fichiers

    :param file_list: liste de fichiers
    :return: dataset tensorflow (spectros, type_machine, id_machine, anomalie
    """
    
    spectro_ds = tf.data.Dataset.from_tensor_slices(file_list)
    spectro_ds = spectro_ds.map(get_waveform_and_label_and_machine_id_and_anomaly, num_parallel_calls=AUTOTUNE)
    spectro_ds = spectro_ds.map(get_spectrogram_and_label_and_machine_id_and_anomaly,
                                num_parallel_calls=AUTOTUNE)
    spectro_ds = spectro_ds.map(lambda x, l, m, a,: {'spectro': x, 'label': l, 'machine_id': m, 'anomaly': a})
    spectro_ds = spectro_ds.batch(100, num_parallel_calls=AUTOTUNE)
    spectro_ds = spectro_ds.cache().prefetch(AUTOTUNE)

    return spectro_ds
                               
def prepare_data_train(mach, machine_file_dict, verbose=1):
    """
    méthode de préparation des données d'apprentissage : à partir d'une liste  de machine à analyser, des disctionnaires contenant les infos des fichiers.

    :param mach: liste de type de machines à analyser
    :param machine_file_dict: dictionnaire des fichiers audio, constitué par la péthode selectInputFile
    :param verbose:
    :return:
    """
    t0 = time.time()
    nb = 0
    spectro_train_dict = {}
    for m in mach:
        print(m)
        spectro_train_dict[m] = {}
        for mach_id in machine_file_dict[m].keys():
            spectro_train_dict[m][mach_id] = {}
            
            spectro_ds = make_spectro_ds_from_files(machine_file_dict[m][mach_id])
          
            for element, spec_dict in enumerate(spectro_ds.as_numpy_iterator()):
                nb += len(spec_dict['label'])
                for key in spec_dict:
                    if key == 'label_str':
                        pass
                    if key == 'machine_id_str':
                        pass

                    if key not in spectro_train_dict[m][mach_id]:
                        spectro_train_dict[m][mach_id][key] = []
                    spectro_train_dict[m][mach_id][key].extend(spec_dict[key])
            if verbose:
                print("  -machine {:15s}: {:4d} spectros générés".format(mach_id,
                                                                        len(spectro_train_dict[m][mach_id]['spectro'])))
    t1 = time.time()
    if verbose:
        print(
            "\ngénération des spectrogrammes de {:d} fichiers en {:.2f}s (temps moyen de {:.3f}ms par fichier)\n".format(
                nb, t1 - t0, (t1 - t0) / nb * 1000))

    return spectro_train_dict
    
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

def train_encoder(data_to_train, level=2, batch_size=12,epochs = 20,model= None, learning_rate=5e-4, plot=True,verbose=1):
    """
    méthode d'entraîenement de l'encodeur de spectros

    :param data_to_train: données préparées par la méthode prepare_data_train
    :param level: niveau d'encodage: 1! type de machine. 2: identifiant machine
    :param batch_size: 12, sinon plante
    :param epochs: nb d'épochs à faire pour l'apprentissage
    :param model: si model==None, l'apprentissage est initialisé. Si modèle est passé, l'apprentissage sera complété sans perdre les poids calculés auparavant.
    :param learning_rate: comme son nom l'indique
    :param plot: affichage des courbes d'apprentissage
    :param verbose: affichage de l'avancement et différents renseignements (verbose = 1 ou 2 selon le niveau de détails
    :return: le modèle après apprentissage
    """
    n_epochs = epochs
    lr = learning_rate

    if model==None:
        model = createModel()
        model.compile(loss=loss, optimizer=Adam(lr))#5e-3

    if verbose:
      print("jeu de données d'apprentissage:",list(data_to_train.keys()))
      if level==1:
        print("==============================================================\n"
              "  apprentissage pour encodage différenciant le type de machine\n"
              "===============================================================")
      else:
        print("==============================================================\n"
              "  apprentissage pour encodage différenciant chaque machine\n"
              "===============================================================")
        for mach_type in data_to_train:
          print(list(data_to_train[mach_type].keys()))
        print("===============================================================\n")
      
    if verbose >1:
        model.summary()

    n=0
    for mach in data_to_train:
        for mach_id in data_to_train[mach].keys():
            n+=len(data_to_train[mach][mach_id]['spectro'])

    gen = generator_by_level(data_to_train, level, batch_size)
    
    t0 = time.time()
    fit_verbose = verbose-1
    history = model.fit(gen, epochs=n_epochs,steps_per_epoch=int(n/10/len(data_to_train[mach])),verbose=fit_verbose)
    t1 = time.time()
    if verbose:
        print("machine {}: apprentissage effectué en {:.3f}mn".format(mach,(t1-t0)/60))
        print("____________________________________________________________________")

    if plot:
        plt.figure()
        plt.plot(history.history['loss'])
        plt.title("apprentissage machine {} avec learning rate= {:.2E}".format(mach,lr));

    return model

def loadModel(path):
    """
    méthode de chargement d'un modèle préentraîné, enregistré sur disque
    :param path: chemin d'accès au fichier  de poids
    :return: modèle préentraîné
    """
    model = createModel()
    model.load_weights(path)
    return model

def createData(files, verbose=1):
    '''
    Méthode qui crée les données de spectros, label machine, identifiant machine, et anomalie machine à partir d'une liste de fichier
    :param files: liste  des fichiers à traiter
    :param verbose: affichage des infos d'avancement
    :return: (spectro, type_machine,id_machine,anomalie
    '''

    t0 = time.time()
    if verbose:
        print("création du pipeline de preprocessing des données ...")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(tb.get_waveform_and_label_and_machine_id_and_anomaly, num_parallel_calls=tb.AUTOTUNE)
    dataset = dataset.map(tb.get_spectrogram_and_label_and_machine_id_and_anomaly, num_parallel_calls=tb.AUTOTUNE).batch(len(files)) #.batch(1000)
    dataset = dataset.shuffle(len(files))

    if verbose:
        print("création des données...")
    X, label,machine,anomaly = next(iter(dataset)) #bien faire un batch de la longueur voulue

    t1 = time.time()
    if verbose:
        print("génération des spectrogrammes de {:d} fichiers en {:.2f}s (temps moyen de {:.3f}ms par fichier)".format(len(files),t1-t0,(t1-t0)/(len(files))*1000))

    return X,label,machine,anomaly

def createEncodedData(files, encoder, verbose=1):
    '''
    Méthode qui crée les données de spectros, label machine, identifiant machine, et anomalie machine à partir d'une liste de fichier
    Elle retourne les données de spectro encodées selon le modèle pasé en paramètre
    :param files: liste  des fichiers à traiter
    :param encoder: modèle d'encodeur transformant les spectrogramme
    :param verbose: affichage des infos d'avancement
    :return: (spectro, type_machine,id_machine,anomalie
    '''

    t0 = time.time()
    if verbose:
        print("création du pipeline de preprocessing des données ...")
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(get_waveform_and_label_and_machine_id_and_anomaly, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(get_spectrogram_and_label_and_machine_id_and_anomaly, num_parallel_calls=AUTOTUNE).batch(len(files)) #.batch(1000)
    dataset = dataset.shuffle(len(files))

    if verbose:
        print("création des données...")
    X, mach_type,mach_id, anomaly = next(iter(dataset)) #bien faire un batch de la longueur voulue

    if verbose:
        print("encodage")
    X = encoder.predict(X)

    t1 = time.time()
    if verbose:
        print("génération des spectrogrammes de {:d} fichiers en {:.2f}s (temps moyen de {:.3f}ms par fichier)".format(len(files),t1-t0,(t1-t0)/(len(files))*1000))

    return X,mach_type,mach_id,anomaly

def trainResult(row):
    """
    retourne la proportion d'erreur d'une ligne d'un dataframe contenant une colonne train_error et train_size
    :param row: ligne de dataframe contenant une colonne train_error et train_size
    :return: proportion d'erreur d'apprentissage
    """

    return row.train_error / row.train_size

def specificite(row):
    """
    retorune la proportion de vrais négatifs d'une dataframe contenant une colonne "normal_detect" et une colonne "normal_size"
    :param row:
    :return:
    """
    return row.normal_detect / row.normal_size

def sensibilite(row):
    """

    :param row:
    :return:
    """
    return row.anormal_detect / row.anormal_size

def aggResult(df, plot=True, title="résultats de détection"):
    """
    méthode d'&ggregation des résultats par type de machine, et affichage histogramme eventuel

    :param df:
    :param plot:
    :param title:
    :return:
    """
    # on aggrege par type de machine
    aggregate_result = df.groupby(['type'])[
        ['train_size', 'train_error', 'normal_size', 'normal_detect', 'anormal_size', 'anormal_detect']].sum()
    result = pd.DataFrame()
    # on calcule les perfs
    result['specificite'] = aggregate_result.apply(specificite, axis=1) * 100
    result['sensibilite'] = aggregate_result.apply(sensibilite, axis=1) * 100

    if plot:
        plotHist(result, result.index, title, figsize=(20, 5))
        plotHist(df, df.machine, title, figsize=(20, 5), orientationX=45)

    return result

def plotHist(result, category, title, figsize=(6, 6), orientationX=0):
    
    sns.color_palette("Set2")

    categorical = category
    # colors      = ['blue', 'orange']
    numerical = [result.sensibilite.values,
                 result.specificite.values]
    number_groups = 2
    bin_width = 1.0 / (number_groups + 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(number_groups):
        ax.bar(x=np.arange(len(categorical)) + i * bin_width,
               height=numerical[i],
               width=bin_width,
               # color=colors[i],
               align='center')
    ax.set_xticks(np.arange(len(categorical)) + number_groups / (4 * (number_groups + 1)))
    ax.set_ylim([0, 110])
    plt.xticks(rotation=orientationX)
    # number_groups/(2*(number_groups+1)): décalage du xticklabel
    ax.set_xticklabels(categorical)
    ax.legend(['Sensibilité', 'Spécificité'], facecolor='w')
    ax.set_title(title)
    plt.show()
    # plt.figure(figsize=(20,5))
    # ax2 = sns.barplot(x='machine',y="%VP",data=df);
    # plt.xticks(rotation=45);
    # ax2.set_title(title)

def run_test_LOF(X_train, X_test, y_train, y_test, anomaly_test, contamination, params_algo, pca_reduction=False,
                 components=2, seuil1=70, seuil2=90, verbose=2):
    """
    méthode effectuant, pour chaque type de machine de la liste machine_id_list issue de la méthode selectInputFile, la boucle de traitements
    - calcul d'ACP (si acp=True) selon le nombre de composantes choisies,
    - apprentissage du modèle Local Outlier Factor sur une boucle de 2 paramètres:
            -  nombre de voisins à tester
            - taux de contamination du set

    :param X_train: spectros encodés, set d'apprentissage, ne comportant pas d'anomalies
    :param X_test: spectros encodés, set de test comportant des anomalies
    :param y_train: identifiants machine correspondant aux données d'entrées de X_train
    :param y_test: identifiants machine correspondant aux données d'entrées de X_test
    :param anomaly_test: vecteurs de 0 et 1 correspondant à la présence d'anomalie dans les données X_test (target de détection)
    :param contamination: liste de taux de contamination à tester permettant le calcul ultérieur d'une courbe ROC
    :param params_algo: nombre de voisins du LOF
    :param pca_reduction:si True, calcule au préalable l'ACP sur les données par machine
    :param components: nb de composantes de l'ACP si ACP = TRue
    :param seuil1: non utilisé
    :param seuil2:non utilisé
    :param verbose: affichage de l'avancement des traitements
    :return: vecteur 2D de spécificite, vecteur 2D de sensibilite (1 dimension machine, 1 dimension contamination),
    dataframe  contenant les lignes du meilleur choix de voisns du LOF, et dataframe contenant tous les résultats de toutes les itérations
    """
    results = pd.DataFrame(
        columns=['algo', 'machine', 'train_size', 'train_error', 'normal_size', 'normal_detect', 'anormal_size',
                 'anormal_detect', 'specificite', 'sensibilite', 'neighbors', 'outlier_fraction','idx_outlier', 'component', 'learn_duration',
                 'predict_duration'])
    resultatFinal = pd.DataFrame(
        columns=['algo', 'machine', 'train_size', 'train_error', 'normal_size', 'normal_detect', 'anormal_size',
                 'anormal_detect', 'specificite', 'sensibilite', 'neighbors', 'outlier_fraction','idx_outlier', 'component', 'learn_duration',
                 'predict_duration'])
    #specificite = VN
    #sensibilite = VP

    # neighbors= [80,90,100,110,120,130,150,]
    name = 'LOF'
    sensibilite = []
    specificite = []
    for n, m in enumerate(tqdm(machine_id_list)):
        
        sensibilite.append([])
        specificite.append([])

        idx_test = (y_test == n)
        idx_train = (y_train == n)
        anomalie_test = anomaly_test[idx_test]

        img_train = X_train[idx_train, :]
        img_test = X_test[idx_test, :]

        if pca_reduction:
            pca = PCA(n_components=components)
            img_train = pca.fit_transform(img_train)
            img_test = pca.transform(img_test)

        for nei, neighbor in enumerate(params_algo):

            specificite[n].append([])
            sensibilite[n].append([])

            for c,conta in enumerate(contamination):
           
                clf = LocalOutlierFactor(
                    novelty=True,
                    n_neighbors=neighbor,
                    contamination=conta)

                # entraînement
                t0 = time.time()
                clf.fit(img_train)
                t1 = time.time()
                # prédictions
                y_pred_train = clf.predict(img_train)
                y_pred_test = clf.predict(img_test)
                t2 = time.time()

                # score
                n_error_train_FP = y_pred_train[y_pred_train == -1].size
                n_VN = (y_pred_test[(y_pred_test == 1) & (anomalie_test == 0)]).size
                # n_error_test_FN = (y_pred_test[(y_pred_test == 1) & (anomalie_test==1)]).size
                n_VP = (y_pred_test[(y_pred_test == -1) & (anomalie_test == 1)]).size
                resultat = {'algo': name,
                            'machine': m,
                            'train_size': len(y_pred_train),
                            'train_error': n_error_train_FP,
                            'normal_size': len(y_pred_test[anomalie_test == 0]),
                            'normal_detect': n_VN,
                            'anormal_size': len(y_pred_test[anomalie_test == 1]),
                            'anormal_detect': n_VP,
                            'specificite': n_VN / len(y_pred_test[anomalie_test == 0]),
                            'sensibilite': n_VP / len(y_pred_test[anomalie_test == 1]),
                            'neighbors': neighbor,
                            'learn_duration': t1 - t0,
                            'predict_duration': t2 - t1,
                            'outlier_fraction': conta,
                            'idx_outlier':c,
                            'component': components
                            }

                results = results.append(resultat, ignore_index=True)
                specificite[n][nei].append(n_VN / len(y_pred_test[anomalie_test == 0]))
                sensibilite[n][nei].append(n_VP / len(y_pred_test[anomalie_test == 1]))

            select = results[results.machine == m]
            select = select.sort_values(by=['specificite'],ascending=False)
            idx = select['sensibilite'].idxmax()
            resultatFinal = resultatFinal.append(results.iloc[idx])

    
            if verbose >= 2:
              print("machine {:20s}: meilleure détection pour neighbors={:4d} \tsensibilité:{:4.2f} \tspecificite: {:4.2f}"
                    .format(m,
                            results.iloc[idx]['neighbors'],
                            results.iloc[idx]['sensibilite'],
                            results.iloc[idx]['specificite']))

    resultatFinal['type'] = resultatFinal.machine.apply(lambda x: x.split('_')[0])
    results['type'] = results.machine.apply(lambda x: x.split('_')[0])

    return specificite,sensibilite,resultatFinal,results


def analyse_test(specificite, sensibilite):
    """
    méthode de calcul du score AUC  entre le vecteur specificité vs sensibilité

    :param specificite: vecteur 2D de spécificités, issus de la méthode run_test_lof (même taille que la liste du tableau de contamination testé)
    :param sensibilite: vecteur 2D de sensibilité correspondant
    :return: vecteurs d'indices du meilleur score et score correspondant, de la même taille que le nombre de machines concernées.
    """
    # plot=True
    # if plot:
    #   plt.figure(figsize=(15,8))

    idx = []
    score = []

    for mach in range(0, len(specificite)):
        # mach=10
        auc_scoring = []
        for nei in range(0, len(specificite[mach])):
            # if plot:
            #   plt.plot(np.ones(len(specificite[mach][nei][:]))-sensibilite[mach][nei][:],specificite[mach][nei][:],label=tb.machine_id_list[mach]+" n={}".format(neighbors[nei]))
            #   plt.xlabel('1-specificité')
            #   plt.ylabel('sensibilité')
            #   plt.legend()
            auc_scoring.append(
                auc(np.ones(len(specificite[mach][nei][:])) - specificite[mach][nei][:], sensibilite[mach][nei][:]))

        idx.append(np.argmax(auc_scoring))
        score.append(auc_scoring[idx[-1]])

    # if plot:
    #   plt.title("machine {}: auc max:{:.2f} pour n={}".format(tb.machine_id_list[mach],auc_scoring[idx[-1]],neighbors[idx[-1]]))
    # print("machine {}: auc max:{:.2f} pour n={}".format(tb.machine_id_list[mach],auc_scoring[idx[-1]],neighbors[idx[-1]]))

    return (idx, score)
