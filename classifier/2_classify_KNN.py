# Author : Antoine
# First try of classification with KNN
# use only fan data, use normal and anomalous data for training
import pandas as pd
import numpy as np

from joblib import dump, load

from sklearn import neighbors
from sklearn.model_selection import train_test_split


working_directory = '../../data/fan/'

if 0:
    print("CREATE TRAINING DATAFRAME...")
    # part 1 : create dataframe from fragments
    # need to be run on time
    data_part_1 = load(working_directory+'df_fan_train.joblib')
    data_part_2 = load(working_directory+'df_fan_test.joblib')

    # data_part_1 : <class 'numpy.ndarray'> (3675, 40065)
    # data_part_2 : <class 'numpy.ndarray'> (1875, 40065)

    # concaténation des deux ndarray sur l'axe vertical
    data = np.concatenate([data_part_1, data_part_2], axis=0)

    # data : <class 'numpy.ndarray'> (5550, 40065)

    # réduit la dimension de l'array pour les tests
    np.random.shuffle(data)
    data = data[0:500,:]

    df = pd.DataFrame(data)

    print("SAVE DATAS...")
    dump(data, working_directory+'data.joblib', compress=True)
    dump(df, working_directory+'df.joblib', compress=True)

df = load(working_directory+'df.joblib')
print("TRAINING DATAFRAME : ")
print(type(df))
print(df.shape)
# (500, 40065)
# 40065, besoin impératif de réduire la dimension...

# séparation des données de la cible
data = df.iloc[:,0:-1]
target = df[df.iloc[:,-1:]]


print("SPLIT DATA...")
# séparation des données d'entrainement (70%) et de test (30%)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.30, random_state=66)


# classification
# initialisation du classifieur KNN
knn = neighbors.KNeighborsClassifier(n_neighbors=7, metric='minkowski')

# ajustement du classifieur
print("FIT KNN (minkowski)...")
knn.fit(X_train, y_train)

# application du classifieur aux données
y_pred = knn.predict(X_test)

# génération de la matrice de confusion
print(pd.crosstab(y_test, y_pred, rownames=['Realité'], colnames=['Prédiction']))

# initialisation du classifieur KNN
knn_m = neighbors.KNeighborsClassifier(n_neighbors=5, metric='manhattan')

# ajustement du classifieur
print("FIT KNN (manhattan)...")
knn_m.fit(X_train, y_train)


#Score du modèle utilisant la distance de minkowski
score_minkowski = knn.score(X_test, y_test)

#Score du modèle utilisant la distance de manhattan
score_manhattan = knn_m.score(X_test, y_test)

print("Score Minkowski : ", round(score_minkowski, 3))
print("Score Manhattan : ", round(score_manhattan, 3))
#Score Minkowski :  0.981
#Score Manhattan :  0.986

