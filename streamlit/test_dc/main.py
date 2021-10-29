"""
Created on Thu May 27 18:49:06 2021

@author: david
help: https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py

"""

import streamlit as st
# import matplotlib as plt
# import seaborn as sns
import pandas as pd
import numpy as np

from modelisation import generate_data

m = generate_data()

# st.text(m)

# st.sidebar.text('Fixed width text')
# >>> a = st.sidebar.radio('R:',[1,2])
# img =plt.imread('titanic.jpg')
# st.image(img)

# #chart
# sns.countplot(df['colonne'])
# fig = plt.gcf()

# st.pyplot(fig)


options= ['Présentation', 'Exploration des datas', 'Utilisation cnn', 'Utilisation auto-encoder', 'Convertir fichiers audio en vecteurs']
choix = st.sidebar.radio('Aller à la section :', options = options)

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

# page exploration
spectro_normal = '../../data_v5/all_png_test_v5/normal_id_00_00000001_slider.png'
spectro_anormal = '../../data_v5/all_png_test_v5/anomaly_id_00_00000000_pump.png'

if choix ==options[1]:
    st.header(choix)
    st.subheader('Nombre de fichiers')
    st.subheader('Convertion en spectrogrammes')
    st.write('exemple de spectrogramme normal : ' + spectro_normal)
    st.image(spectro_normal)
    st.write('exemple de spectrogramme anormal : ' + spectro_anormal)
    st.image(spectro_anormal)


if choix ==options[2]:
    st.header(choix)
    st.write('xxx')

if choix ==options[3]:
    st.header(choix)
    st.write('xxx')
    
    
    
    
