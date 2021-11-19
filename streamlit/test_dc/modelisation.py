
# import utils
# import sys
# import datetime
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image
from keras.models import load_model


image_size = (333, 216)
cutoff = 0.9 # if score > cutoff, the machine predicted as "normal" otherwise as "anomaly"
# indiceFile = 0
# name_csv_logs = 'cnn_results.csv'
# df_logs_h = pd.DataFrame(columns=['machine', 'file', 'correctPrediction', 'score'])
# df_logs_h.to_csv(name_csv_logs, header = True, index = False)

def display_stats_cnn1(machine_folder):
    name_csv_logs = '../../_final/cnn1/cnn_results.csv' 
    df_logs = pd.read_csv(name_csv_logs)
    df_filtre = df_logs[df_logs['machine'] == machine_folder]
    return df_filtre
    
def predict_one_machine(machine_folder): # trop lent, utiliser display_stats_cnn1 pour charts des résultats 
    png_tests_folder = '../../_data_png_cnn1/' + machine_folder + '/png_test/'
    model_folder = '../../_classifiers_cnn1/cnn_' + machine_folder + '.h5'
    # renit the result dataframe (store with files tested and scores of the model)
    df_result = pd.DataFrame(columns=['file', 'score'])
     # load the model from disk
    model = load_model(model_folder)
    wavfiles = [f for f in listdir(png_tests_folder) if isfile(join(png_tests_folder, f))]
    for nameFilePngTotest in wavfiles:
        if nameFilePngTotest[-4:] != '.png': # ignore non .png files
            continue
        # print('test nameFilePngTotest: ', nameFilePngTotest)
        test_image = image.load_img(png_tests_folder + nameFilePngTotest, target_size = image_size) 
        img_array = image.img_to_array(test_image)
        img_array = np.expand_dims(test_image, axis = 0)
        predictions = model.predict(img_array)
        score = predictions[0]
        df_result = df_result.append({'file': nameFilePngTotest, 'score': score}, ignore_index=True)

    df_result = df_result.sort_values(by = ['score'], ascending = False)
    countPrediction = 0
    countCorrectPrediction = 0
    for index, row in df_result.iterrows():
        countPrediction += 1
        arrName = row['file'].split("_") # anomaly_id_00_00000001.wav
        classPrefix = arrName[0] # 'normal' or 'anomaly'
        isNormalPredict = 1 if row['score'] > cutoff else 0
        isNormalReal = 1 if classPrefix == "normal" else 0
        if isNormalReal == isNormalPredict:
            countCorrectPrediction += 1
        # if countPrediction % 20 == 0: # display result predictions (1 of 20)
        # correctPrediction = 'OK' if isNormalReal == isNormalPredict else 'NOK'
        # print("file %s predict %s (score: %.2f%%)." % (row['file'], correctPrediction, 100 * row['score']))
        # df_logs = pd.read_csv(name_csv_logs)
        # new_row = {'machine': machine_folder, 'file': row['file'], 'correctPrediction': correctPrediction, 'score': row['score'][0]}
        # df_logs = df_logs.append(new_row, ignore_index=True)
        # df_logs.to_csv(name_csv_logs, header = True, index = False)
        
    # display general result for the machine (correct predictions / total of préeditions)
    # print("****** %s: accuracy:  %.2f (%i fichiers)" % (machine_folder, countCorrectPrediction / len(df_result), countPrediction))
    recap = {'machine': machine_folder, 'accuracy': countCorrectPrediction / len(df_result), 'countPrediction': countPrediction}
    return df_result, recap

    
# def generate_data():
#     predict_one_machine('fan')
