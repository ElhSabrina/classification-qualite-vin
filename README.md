# Classification de la qualité du vin

Projet 2 du cours Machine Learning 1, EFREI 2026-2027, réalisé par Sabrina EL HASSANI, Florielle LODJO et Christina PAYONG.

Le but est de prédire si un vin est bon, c'est à dire noté au moins 7 sur 10 par des dégustateurs, à partir de 11 mesures de laboratoire et de sa couleur.

**[Parcourir les notebooks avec le menu de navigation](https://elhsabrina.github.io/classification-qualite-vin/)**

## Sommaire

1. [Les notebooks](#les-notebooks)
2. [Les questions de chaque partie](#les-questions-de-chaque-partie)
3. [L'application web](#lapplication-web)
4. [Lancer le projet](#lancer-le-projet)
5. [Source des données](#source-des-données)

## Les notebooks

Le notebook [projet_qualite_vin.ipynb](projet_qualite_vin.ipynb) est le livrable, il réunit les trois parties avec toutes leurs sorties. Les parties existent aussi séparément et se lancent dans l'ordre depuis la racine du dépôt.

| Partie | Notebook | Version .py |
|---|---|---|
| 1 | [1_preparation.ipynb](1_preparation.ipynb) | [1_preparation.py](version_py/1_preparation.py) |
| 2 | [2_modelisation.ipynb](2_modelisation.ipynb) | [2_modelisation.py](version_py/2_modelisation.py) |
| 3 | [3_evaluation.ipynb](3_evaluation.ipynb) | [3_evaluation.py](version_py/3_evaluation.py) |
| Complet | [projet_qualite_vin.ipynb](projet_qualite_vin.ipynb) | [projet_qualite_vin.py](version_py/projet_qualite_vin.py) |

## Les questions de chaque partie

Chaque lien ouvre la section dans le site de navigation.

### Partie 1. Peut-on faire confiance aux données ?

[1.1 Que contiennent les données ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/1_preparation.html#s1-1)  
[1.2 Manque-t-il des valeurs ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/1_preparation.html#s1-2)  
[1.3 Qu'est-ce qu'un bon vin, et pourquoi l'accuracy ne suffit-elle pas ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/1_preparation.html#s1-3)  
[1.4 Quels vins garde-t-on pour le test final ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/1_preparation.html#s1-4)  
[1.5 Qu'est-ce qui distingue un bon vin ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/1_preparation.html#s1-5)  
[1.6 Des mesures disent-elles la même chose ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/1_preparation.html#s1-6)  
[1.7 Comment vérifier nos choix sans toucher au test ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/1_preparation.html#s1-7)  
[1.8 Faut-il supprimer les doublons ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/1_preparation.html#s1-8)  
[1.9 Faut-il retirer les valeurs extrêmes ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/1_preparation.html#s1-9)  
[1.10 Faut-il mettre les mesures à la même échelle ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/1_preparation.html#s1-10)  
[1.11 Faut-il retirer des mesures ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/1_preparation.html#s1-11)

### Partie 2. Comment construire et régler les cinq modèles ?

[2.1 Quels sont les cinq modèles, et lesquels sont linéaires ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/2_modelisation.html#s2-1)  
[2.2 Que donnent-ils sans réglage ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/2_modelisation.html#s2-2)  
[2.3 Comment choisir les hyperparamètres ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/2_modelisation.html#s2-3)  
[2.4 Pourquoi régler avec l'AUC plutôt qu'avec le F1 ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/2_modelisation.html#s2-4)  
[2.5 La recherche aléatoire fait-elle aussi bien ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/2_modelisation.html#s2-5)  
[2.6 Comment éviter un modèle trop rigide ou trop collé aux données ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/2_modelisation.html#s2-6)  
[2.7 Qu'a appris chaque modèle ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/2_modelisation.html#s2-7)  
[2.8 Combiner les modèles aide-t-il ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/2_modelisation.html#s2-8)

### Partie 3. Quel modèle choisir, et peut-on s'y fier ?

[3.1 Quel modèle est le meilleur, et sur quoi se base-t-on ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/3_evaluation.html#s3-1)  
[3.2 Que donne le modèle, et à partir de quelle probabilité dit-on « bon » ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/3_evaluation.html#s3-2)  
[3.3 Que vaut-il sur des vins qu'il n'a jamais vus ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/3_evaluation.html#s3-3)  
[3.4 Est-il aussi bon sur les rouges que sur les blancs ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/3_evaluation.html#s3-4)  
[3.5 Où se trompe-t-il, et pourquoi ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/3_evaluation.html#s3-5)  
[3.6 Quel seuil pour quel usage ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/3_evaluation.html#s3-6)  
[3.7 Quelles sont les limites ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/3_evaluation.html#s3-7)  
[3.8 Que retenir de tout le projet ?](https://elhsabrina.github.io/classification-qualite-vin/notebooks/3_evaluation.html#s3-8)

## L'application web

On saisit les 11 mesures d'un vin, sa couleur et un usage. L'application affiche la probabilité que le vin soit bon, la décision selon le seuil de l'usage et les notes des huit vins les plus ressemblants.

| Usage | Seuil |
|---|---|
| Présélection avant dégustation | 0,10 |
| Équilibre | 0,26 |
| Label prestige | 0,72 |

<p align="center"><img src="figures/application_web.png" alt="Capture de l'application" width="360"></p>

## Lancer le projet

Avec Docker, qui réentraîne le modèle et lance l'application

```bash
docker build -t qualite-vin .
docker run -p 5000:5000 qualite-vin
```

Sans Docker

```bash
pip install -r requirements.txt
python -m src.entrainement
python app/app.py
```

L'application s'ouvre sur http://localhost:5000 et les 12 tests unitaires se lancent avec `pytest`.

## Source des données

P. Cortez, A. Cerdeira, F. Almeida, T. Matos et J. Reis. Modeling wine preferences by data mining from physicochemical properties. Decision Support Systems, 47(4), 547-553, 2009.
