#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 15:15:55 2021

preparation of data sets:
    generate charts about amount of data
    --- todo: ---
    generate metadata file
    generate spectograms in folders
"""

import pandas as pd
from os.path import isfile, join
import os
import seaborn as sns

rootFolder = '/Users/david/DEVS_LOCAL/dev-ia-son-data/'
list_datasets_train = ['fan', 'ToyCar', 'pump']  # folders in rootFolder
# list_datasets_train = ['fan'] # test
list_datasets_validation = ['slider', 'valve'] # folders in rootFolder

def countInFolder():
    # count all files in folders
    list_folders= []
    list_stats= []

    for folder in list_datasets_train:
        train_folder = './' + folder + '/train/'
        print('f=', train_folder)
        wavFilesTrain = [f for f in os.listdir(train_folder) if isfile(join(train_folder, f))]
        if '.DS_Store' in wavFilesTrain: wavFilesTrain.remove('.DS_Store')
        list_folders.append(train_folder)
        list_stats.append(len(wavFilesTrain))
    

    df_train = pd.DataFrame(data = {'folder': list_folders, 'nb': list_stats})
    df_train.head()


def data_allFilesByType(typeFolder):
    # typeFolder : 'train' or 'test'
    list_files = []
    list_folders = []
    list_size = []
    # get a df with all files and categories
    for folder in list_datasets_train:
        rel_folder =  folder + '/' + typeFolder + '/'
        full_folder = rootFolder + '/' + rel_folder
        print('full_folder=', full_folder)
        wavFilesTrain = [f for f in os.listdir(full_folder) if isfile(join(full_folder, f))]
        if '.DS_Store' in wavFilesTrain: wavFilesTrain.remove('.DS_Store')
        for f in wavFilesTrain:
            list_files.append(f)
            list_folders.append(rel_folder)
            list_size.append(os.path.getsize(full_folder + '/' + f))
    return(list_folders, list_files, list_size)


def data_allFiles():
    # for each filde in folders "train" & "test", get folder name, file name, file size
    list_folder1, list_files1, list_size1 = data_allFilesByType('train')
    list_folder2, list_files2, list_size2 = data_allFilesByType('test')
    list_folder1.extend(list_folder2)
    list_files1.extend(list_files2)    
    list_size1.extend(list_size2)    
    return ({'folder': list_folder1, 'file': list_files1, 'size': list_size1})

# countInFolder()

df = pd.DataFrame(data = data_allFiles()) # init df: list of files in folders
sns.countplot(y="folder", data=df); # number of files in each folder
# sns.countplot(x="size", data=df_train); # compare sizes files
df.boxplot(column= 'size', by='folder', figsize= (7,7)); # compare sizes files

