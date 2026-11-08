"""Préparation des données, les mêmes étapes que dans le notebook de la partie 1."""
import pandas as pd
from sklearn.model_selection import train_test_split

GRAINE = 42
NUMERIQUES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
              "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
VARIABLES = NUMERIQUES + ["est_rouge"]


def charger_donnees(dossier="data"):
    """Lit les deux fichiers et les réunit avec une colonne type (rouge ou blanc)."""
    rouge = pd.read_csv(f"{dossier}/winequality-red.csv", sep=";")
    blanc = pd.read_csv(f"{dossier}/winequality-white.csv", sep=";")
    rouge["type"] = "rouge"
    blanc["type"] = "blanc"
    return pd.concat([rouge, blanc], ignore_index=True)


def preparer(vins_brut):
    """Supprime les doublons, encode la couleur et crée la cible bon (note d'au moins 7)."""
    vins = vins_brut.drop_duplicates().reset_index(drop=True)
    vins["est_rouge"] = (vins["type"] == "rouge").astype(int)
    vins["bon"] = (vins["quality"] >= 7).astype(int)
    return vins


def decouper(vins, part_test=0.2):
    """Sépare entraînement et test en respectant la part de bons vins et la part de rouges."""
    groupe = vins["type"] + "_" + vins["bon"].astype(str)
    return train_test_split(vins, test_size=part_test, stratify=groupe, random_state=GRAINE)
