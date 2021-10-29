Repo pour le project son


utiliser python 3.8.5 64bits

**** dossiers :
_final : scripts python
_data_origin : datas récupérées de kaggle
_data_png : (généré) contient les spectrogrammes
_classifiers : (généré) contient les modèles


**** fichiers :
utils.py : fonctions diverses et variables globales
SoundFile.py : classe pour le traitement d'un fichier wav

1_data_exploration.ipynb : data vizualisation du jeu de données complet 
2_convert_to_spectrograms.ipynb : créé les dossiers et génère pour chaque machine les images des spectrogrammes dans _data_png
3_cnn_fit.ipynb : entraine un modèle cnn sur les images de chaque machine dans _data_png + enregistre ce modèle pour chaque machine dans _classifiers
4_cnn_use.ipybn : utilise le modèle relatif à chaque machine sur les données de test _data_png/[machine] et affiche les résultats






todo : 
4_dc_use_cnn3.py : load model + affiche charts de prédiction
rapport doc
streamlit

    


