# -*- coding: utf-8 -*-

"""
    global variables & fonctions
"""
import pandas as pd
from os.path import isfile, join
import os
import seaborn as sns
import matplotlib.pyplot as plt

list_datasets = ['valve', 'slider', 'pump', 'ToyCar', 'fan', 'ToyConveyor']  # folders machine  +  'valve', 'slider'

# list_datasets = ['slider'] # for testing on only 1 machine
    
def countInFolder():
    """
        get folders names and count all files in each folder
    """
    list_folders= []
    list_stats= []

    for machine_folder in list_datasets:
        train_folder = '../../_data_origin/' + machine_folder + '/train/'
        # print('f=', train_folder)
        wavFilesTrain = [f for f in os.listdir(train_folder) if isfile(join(train_folder, f))]
        if '.DS_Store' in wavFilesTrain: wavFilesTrain.remove('.DS_Store')
        list_folders.append(train_folder)
        list_stats.append(len(wavFilesTrain))
    
    df_train = pd.DataFrame(data = {'folder': list_folders, 'nb': list_stats})
    return(df_train)
    

def data_allFilesByType(typeFolder):
    """
        get data about all files in folders: filename, folder, file size, file type
    """
    # typeFolder : 'train' or 'test'
    list_files = []
    list_folders = []
    list_size = []
    list_type = []

    # get a df with all files and categories
    for folder in list_datasets:
        rel_folder =  folder + '/' + typeFolder
        full_folder = '../../_data_origin/' + rel_folder + '/'
        # full_folder = rootFolder  + 'data/' + '/' + rel_folder + '/'
        # print('full_folder=', full_folder)
        wavFilesTrain = [f for f in os.listdir(full_folder) if isfile(join(full_folder, f))]
        if '.DS_Store' in wavFilesTrain: wavFilesTrain.remove('.DS_Store')

        for f in wavFilesTrain:
            # f : filename (ex : normal_id_06_00000092.wav)
            list_files.append(f)

            # rel_folder: folder (ex : valve/test/)
            list_folders.append(rel_folder)

            # append file size in bytes
            list_size.append(os.path.getsize(full_folder + '/' + f))

            # append file type (normal/anomalous)
            list_type.append(f.split('_')[0])

    return(list_folders, list_files, list_size, list_type)


def data_allFiles():
    """
        for each filde in folders "train" & "test", get folder name, file name, file size
    """
    list_folder1, list_files1, list_size1, list_type1 = data_allFilesByType('train')
    list_folder2, list_files2, list_size2, list_type2 = data_allFilesByType('test')

    list_folder1.extend(list_folder2)
    list_files1.extend(list_files2)    
    list_size1.extend(list_size2)
    list_type1.extend(list_type2)

    return ({'folder': list_folder1, 'file': list_files1, 'size': list_size1, 'type': list_type1})




