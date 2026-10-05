# Prédiction du revenu - Adult Census

Projet de machine learning sur le jeu de données **Adult Census** : prédire si une personne gagne plus de 50K par an (`>50K`) ou non (`<=50K`).

## Contenu du projet

- **Exploration et nettoyage** : valeurs manquantes (mode), doublons, outliers (percentile 99 / IQR), encodage OneHot, normalisation `StandardScaler`.
- **Données** : le jeu d'entraînement (`adult_train.csv`) et le jeu de test (`adult_test.csv`) sont séparés d'avance ; le test n'est utilisé que pour l'évaluation finale.
- **Clustering** : K-Means (choix de k avec la méthode du coude et le score silhouette) et interprétation des clusters.
- **Classification** : Decision Tree, Random Forest et KNN, optimisés par `GridSearchCV`, comparés à un modèle de référence (Dummy) et à une régression logistique.
- **Évaluation** : accuracy, precision, recall, F1 de la classe `>50K`, courbes ROC / AUC et Précision-Rappel, importance des variables.
- **Interface** : application Streamlit pour saisir un profil et obtenir la prédiction des trois modèles (avec explication par les plus proches voisins pour KNN).

## Structure du dépôt

```
├── Untitled2_corrige.ipynb   # notebook complet
├── app.py                    # interface Streamlit
├── adult_train.csv           # données d'entraînement
├── adult_test.csv            # données de test
├── requirements.txt
└── README.md
```

`modele_adult.pkl` (modèles entraînés) n'est pas versionné : il est généré par la Cellule 22 du notebook.

## Lancer le projet

```bash
pip install -r requirements.txt
jupyter notebook          # ouvrir le notebook et faire Run All
streamlit run app.py      # lancer l'interface (après la Cellule 22)
```

Tous les fichiers doivent être dans le même dossier (chemins relatifs).

## Limites

- Jeu de données historique et déséquilibré (≈76 % de `<=50K`) : l'accuracy seule est trompeuse, d'où le suivi du recall et du F1 de `>50K`.
- Les prédictions sont des estimations statistiques et ne constituent pas une décision.
