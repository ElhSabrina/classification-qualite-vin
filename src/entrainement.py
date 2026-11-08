"""Entraîne le modèle final et l'enregistre pour l'application web.

Lancement depuis la racine du projet :  python -m src.entrainement
"""
import os

import joblib
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.preparation import GRAINE, VARIABLES, charger_donnees, decouper, preparer


def construire_modele():
    """Le modèle retenu dans les notebooks : standardisation puis SVM à noyau gaussien."""
    return Pipeline([
        ("echelle", StandardScaler()),
        ("modele", SVC(kernel="rbf", C=1, gamma=0.1, class_weight="balanced",
                       probability=True, random_state=GRAINE)),
    ])


def meilleur_seuil(y_vrai, probas):
    """Seuil de probabilité qui donne le meilleur F1-score."""
    precisions, rappels, seuils = precision_recall_curve(y_vrai, probas)
    f1 = 2 * precisions * rappels / (precisions + rappels + 1e-12)
    return float(seuils[f1[:-1].argmax()])


def entrainer(dossier="data", sortie="app/modele_final.joblib"):
    train, _ = decouper(preparer(charger_donnees(dossier)))
    X_train, y_train = train[VARIABLES], train["bon"]

    modele = construire_modele()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=GRAINE)
    probas_hors_pli = cross_val_predict(modele, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    seuil = meilleur_seuil(y_train, probas_hors_pli)

    # Seuils des deux autres usages, fixés eux aussi sur les prédictions hors pli
    precisions, rappels, seuils_possibles = precision_recall_curve(y_train, probas_hors_pli)
    seuils_usages = {"preselection": float(seuils_possibles[rappels[:-1] >= 0.90].max()),
                     "prestige": float(seuils_possibles[precisions[:-1] >= 0.80].min())}

    modele.fit(X_train, y_train)
    os.makedirs(os.path.dirname(sortie), exist_ok=True)
    joblib.dump({"modele": modele, "nom": "SVM", "seuil": seuil, "variables": VARIABLES,
                 "seuils_usages": seuils_usages, "vins_connus": train[VARIABLES + ["quality"]]}, sortie)
    return modele, seuil


if __name__ == "__main__":
    _, seuil = entrainer()
    print(f"Modèle enregistré dans app/modele_final.joblib avec le seuil {seuil:.2f}")
