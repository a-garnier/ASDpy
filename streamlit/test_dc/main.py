"""
Created on Thu May 27 18:49:06 2021

@author: david
launch, use: streamlit run main.py <---------
help: https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py

"""

import streamlit as st
import matplotlib as plt
import seaborn as sns
import pandas as pd
import numpy as np

from modelisation import display_stats_cnn1


options= ['Présentation', 'Exploration des datas', 'Démo cnn', 'Démo auto-encoder', 'Démo vecteurs']
choix = st.sidebar.radio('Aller à la section :', options = options)
list_machines = ['ToyCar', 'ToyConveyor', 'fan', 'pump', 'slider', 'valve']

    
# presentation of the project
if choix ==options[0]:
    st.title('ASDpy')
    st.header('(Anomalous Sound Detection)')
    st.markdown("""
                L’objectif est de concevoir un ou plusieurs modèles de machine learning permettant de détecter, à partir de sons émis, les machines défaillantes dans un parc de machines. 
    Les machines seront donc classifiées sous état “normal” ou “anormal”.
    Dataset : https://www.kaggle.com/daisukelab/dc2020task2
                """)
    
    st.markdown("""
            Les fichiers sont classés par machine. Il y a 6 machines :
        Valve : 1,22 Go pour 3 771 fichiers
        Slider : 1,2 Go pour 3 695 fichiers
        Pump : 1,23 Go pour 3 806 fichiers
        ToyCar : 1,78 Go pour 5 060 fichiers
        Fan : 1,67 Go pour 5 151 fichiers
        ToyConveyor : 1,33 Go pour 4 111 fichiers
            """)


# exploration data
if choix ==options[1]:
    spectro_normal = '../../data_v5/all_png_test_v5/normal_id_00_00000001_slider.png'
    spectro_anormal = '../../data_v5/all_png_test_v5/anomaly_id_00_00000000_pump.png'
    st.header(choix)
    st.subheader('Nombre de fichiers')
    st.subheader('Convertion en spectrogrammes')
    st.write('exemple de spectrogramme normal : ' + spectro_normal)
    st.image(spectro_normal)
    st.write('exemple de spectrogramme anormal : ' + spectro_anormal)
    st.image(spectro_anormal)

# démo cnn 1
if choix ==options[2]:
    st.header(choix)
    choix_machine = st.selectbox('Modèle', list_machines)
    st.subheader(choix_machine)
    name_csv_logs = '../../_final/cnn1/cnn_results.csv' 
    df_logs = pd.read_csv(name_csv_logs)
    df_logs['pred_f'] =  df_logs['file'].apply(lambda x: x.split('_')[0]) 
    df_logs['pred_r'] =  np.random.rand(df_logs.shape[0])
    # df_filtre['pred_n'] = df_filtre['score'].apply(lambda sc: sc > cutoff ) 
    # df_filtre['pred_n'] =  df_filtre['correctPrediction'].apply(lambda pred: 1 if pred == 'OK' else 0.95 ) 
    df_logs = display_stats_cnn1(df_logs, choix_machine)

# démo auto encoder
if choix ==options[3]:
    st.header(choix)
    st.write('auto encoder')
    
# démo vecteurs
if choix ==options[4]:
    st.header(choix)
    st.write('vecteurs')
    
    
