"""
Created on Thu May 27 18:49:06 2021

@author: david
launch, use: streamlit run main.py <---------
help: https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py

"""

import streamlit as st
import matplotlib as plt
import pandas as pd
import numpy as np
import os
from PIL import Image 
from modelisation import display_stats_cnn1
from keras.preprocessing import image
from keras.models import load_model

cutoff = 0.9
options= ['Présentation', 'Exploration des datas', 'Démo cnn', 'Résultats cnn', 'Conclusion']
choix = st.sidebar.radio('Aller à la section :', options = options)
st.sidebar.write('Promotion : **DS - mars 21**')
st.sidebar.write('Participants :')
st.sidebar.write('Frédéric Aymé [frederic.ayme@gmail.com](frederic.ayme@gmail.com)')
st.sidebar.write('David Campion [cmpdvd@gmail.com](cmpdvd@gmail.com) ')
st.sidebar.write('Antoine Garnier [garnier.antoine66@gmail.com](garnier.antoine66@gmail.com) ')

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 

list_machines = ['slider', 'fan', 'ToyCar', 'ToyConveyor',  'pump',  'valve']
    
# presentation projet
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
    spectro_normal = '../data_v5/all_png_test_v5/normal_id_00_00000001_slider.png'
    spectro_anormal = '../data_v5/all_png_test_v5/anomaly_id_00_00000000_pump.png'
    st.header(choix)
    st.subheader('Nombre de fichiers')
    st.subheader('Convertion en spectrogrammes')
    st.write('exemple de spectrogramme normal : ' + spectro_normal)
    st.image(spectro_normal)
    st.write('exemple de spectrogramme anormal : ' + spectro_anormal)
    st.image(spectro_anormal)

# démo 1 fichier cnn 1
if choix ==options[2]:
    st.header(choix)
    st.write('Prédiction sur 1 spectrogramme')
    # Charger une image spectrogramme
    image_file = st.file_uploader("",type=['png'])
    if image_file is not None:
        file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
        # st.write(file_details)
        img = load_image(image_file)
        st.image(img)
        if st.button("Prediction du spectrogramme"):
            machine_folder = 'slider' # en dur : slider
            image_size = (333, 216)
            png_tests_folder = '../_data_png_cnn1/' + machine_folder + '/png_test/' # en dur 
            model_folder = '../_classifiers_cnn1/cnn_' + machine_folder + '.h5'
            st.write('nameFilePngTotest:', png_tests_folder + image_file.name)
            model = load_model(model_folder)
            test_image = image.load_img(png_tests_folder + image_file.name, target_size = image_size) 
            img_array = image.img_to_array(test_image)
            img_array = np.expand_dims(test_image, axis = 0)
            predictions = model.predict(img_array)
            score = predictions[0][0]
            st.write('score:', score)
            result_txt = '> Son normal' if score > cutoff else '> Son anormal'
        
            arrName = image_file.name.split("_") # anomaly_id_00_00000001.wav
            classPrefix = arrName[0] # anomaly
            isNormalPredict = 1 if score > cutoff else 0
            isNormalReal = 1 if classPrefix == "normal" else 0
        
            # isNormalPredict = 'Correct' if score > cutoff else 'Incorrect'
            correctPrediction = 'Correct' if isNormalReal == isNormalPredict else 'Incorrect'
        
            st.subheader(result_txt + ' : ' + correctPrediction)


# démo résultats cnn 1
if choix == options[3]:
    st.header(choix)
    choix_machine = st.selectbox('Modèle', list_machines)
    st.subheader(choix_machine)
    name_csv_logs = '../_final/cnn1/cnn_results.csv' 
    df_logs = pd.read_csv(name_csv_logs)
    df_logs['pred_f'] =  df_logs['file'].apply(lambda x: x.split('_')[0]) 
    df_logs['pred_r'] =  np.random.rand(df_logs.shape[0])
    # df_filtre['pred_n'] = df_filtre['score'].apply(lambda sc: sc > cutoff ) 
    # df_filtre['pred_n'] =  df_filtre['correctPrediction'].apply(lambda pred: 1 if pred == 'OK' else 0.95 ) 
    df_logs = display_stats_cnn1(df_logs, choix_machine)

    
# conclusion
if choix ==options[4]:
    st.header(choix)
    st.write('conclusion')
    
    
