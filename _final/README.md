# ASDpy (Anomalous Sound Detection)
## Projet de reconnaissance de sons de machines anormaux

Utiliser python 3.8.5 64bits.

## Dossiers :
```
_final : scripts python (explorations de différentes solutions)
_final/cnn1 : scripts pour la solution cnn v1
_final/cnn2 : scripts pour la solution cnn v2

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

_data_png_cnn1 : (généré par _final/cnn1/2_cnn1_convert_to_spectrograms.ipynb) contient les spectrogrammes pour le cnn v1
_data_png_cnn2 : (généré par _final/cnn2/2_cnn2_convert_to_spectrograms.ipynb) contient les spectrogrammes pour le cnn v2
_classifiers_cnn1 : contient les modèles pour cnn1
_classifier_cnn2: contient le modèle pour cnn2
```

##  Fichiers :
```
utils.py : fonctions diverses et variables globales
SoundFile.py : classe pour le traitement d'un fichier wav

1_data_exploration.ipynb : data vizualisation du jeu de données complet 

cnn1/2_cnn1_convert_to_spectrograms.ipynb : créé les dossiers et génère pour chaque machine les images des spectrogrammes dans _data_png_cnn1
cnn1/3_cnn1_fit.ipynb : entraine un modèle cnn sur les images de chaque machine dans _data_png_cnn1 + enregistre ce modèle pour chaque machine dans _classifiers
cnn1/4_cnn1_use.ipybn : utilise le modèle relatif à chaque machine sur les données de test _data_png/[machine] et affiche les résultats

cnn2/2_cnn2_convert_to_spectrograms.ipynb : créé les dossiers et génère pour chaque machine les images des spectrogrammes dans _data_png_cnn2
cnn2/3_cnn2_fit.ipynb : entraine un modèle cnn sur les images des machine dans _data_png_cnn2 + enregistre un modèle cnn global dans _classifiers
cnn2/4_cnn2_use.ipynb : utilise le modèle cnn global sur les données de test de _data_png_cnn2 et affiche les résultats
```







    


