# ASDpy
## project de reconnaissance de sons de machines anormaux

Utiliser python 3.8.5 64bits.

## dossiers :
```_final : scripts python
_final : code à livrer (explorations de différentes solutions)
_data_origin : datas récupérées de kaggle sous la forme :
    -- fan/
        -- test
        -- train
    -- pump/
        -- test
        -- train
    -- slider/
        -- test
        -- train
    -- Toycar/
        -- test
        -- train
    -- ToyConveyor/
        -- test
        -- train
    -- valve/
        -- test
        -- train

_data_png : (généré par 2_convert_to_spectrograms.ipynb) contient les spectrogrammes
_classifiers : (généré par 2_convert_to_spectrograms.ipynb) contient les modèles
cnn1 : scripts pour la solution cnn v1
cnn2 : scripts pour la solution cnn v2
```

##  fichiers :
```
utils.py : fonctions diverses et variables globales
SoundFile.py : classe pour le traitement d'un fichier wav

cnn1/1_data_exploration.ipynb : data vizualisation du jeu de données complet 
cnn1/2_convert_to_spectrograms.ipynb : créé les dossiers et génère pour chaque machine les images des spectrogrammes dans _data_png
cnn1/3_cnn_fit.ipynb : entraine un modèle cnn sur les images de chaque machine dans _data_png + enregistre ce modèle pour chaque machine dans _classifiers
cnn1/4_cnn_use.ipybn : utilise le modèle relatif à chaque machine sur les données de test _data_png/[machine] et affiche les résultats
```







    


