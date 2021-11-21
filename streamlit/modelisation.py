
# import utils
# import sys
# import datetime
import streamlit as st
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import seaborn as sns
import matplotlib.pyplot as plt


cutoff = 0.9 # if score > cutoff, the machine predicted as "normal" otherwise as "anomaly"

# df_logs: full dataframe of logs
# machine_folder: 'slider'
def display_stats_cnn1(df_logs, machine_folder):
    df_filtre = df_logs[df_logs['machine'] == machine_folder]
    # df_filtre = df_filtre.sort_values(by = ['score'], ascending = False)

    # display stats
    nbCorrect = df_filtre[df_filtre['correctPrediction'] == 'OK'].shape[0]
    nbTotal = df_filtre.shape[0]
    st.text('accuracy: ' + str(round(nbCorrect / nbTotal, 3)) + ' on files count: ' + str(nbTotal))

    # display chart 1: ok 
    g = sns.relplot(data=df_filtre, 
                x="score", y="pred_r", hue="pred_f", 
                height=6, aspect=2, s=190)
    g._legend.set_bbox_to_anchor((.5, .9))
    g.set(xlim=(-0.05,1.05), 
            ylim=(0,1.05),
            xticks=np.arange(0, 1.1, 0.1), 
            yticks=np.arange(0, 1.1, 0.1))
    plt.plot([cutoff, cutoff], [1, 0], 'r--', linewidth=1) # vertical line cutoff
    st.pyplot(g) 
    # display chart 2
    g = sns.relplot(data=df_filtre, 
                x="score", y="pred_r", hue="correctPrediction", palette=["b", "r"],
                height=6, aspect=2, s=190)
    g._legend.set_bbox_to_anchor((.5, .9))
    g.set(xlim=(-0.05,1.05), 
        ylim=(0,1.05),
        xticks=np.arange(0, 1.1, 0.1), 
        yticks=np.arange(0, 1.1, 0.1))
    plt.plot([0.9, 0.9], [1, 0], 'r--', linewidth=1)
    st.pyplot(g) 
    # fig, ax = plt.subplots(figsize=(5, 2))
    # g = sns.histplot(df_filtre, x="score", hue='correctPrediction', bins=10)
    # g.set(xlim=(-0.05,1.05), 
    #   xticks=np.arange(0, 1.1, 0.1), 
    # )
    # st.pyplot(fig)
    
    # display dataframe 
    # st.dataframe(df_filtre)
    
    return 1
        

