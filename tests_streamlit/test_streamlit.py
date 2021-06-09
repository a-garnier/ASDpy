#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 18:49:06 2021

@author: david
"""

import streamlit as st
# import matplotlib as plt
# import seaborn as sns

from modelisation import generate_data 

m = generate_data()

st.text(m)


st.title('title texte')
st.text('ceci est du dataset titanic')
st.markdown("""
            plusieurs lignes 
            ce projet est ... https://
            """)
            
            
# img =plt.imread('titanic.jpg')
# st.image(img)

# #chart
# sns.countplot(df['colonne'])
# fig = plt.gcf()

# st.pyplot(fig)


options= ['mmmm', 'oooooo']
choix = st.radio('choisir modele', options = options)

# df =

# st.write(df)



st.write(choix)

if choix ==options[0]:
    st.write('kkkkk')
    
if choix ==options[1]:
    st.write('xxx')
    
    
optionsMenu = ['mmmm', 'oooooo']
# choixside = st.sidebar.radio('choisir modele', options = optionsMenu)








