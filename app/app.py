"""Application web : le carnet de cave.

Lancement depuis la racine du projet :  python app/app.py   puis ouvrir http://localhost:5000
"""
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

DOSSIER = os.path.dirname(os.path.abspath(__file__))
paquet = joblib.load(os.path.join(DOSSIER, "modele_final.joblib"))
modele, nom_modele, seuil_equilibre = paquet["modele"], paquet["nom"], paquet["seuil"]
variables, vins_connus = paquet["variables"], paquet["vins_connus"]

# Pour justifier une réponse, on retrouve les vins connus qui ressemblent le plus au vin saisi
echelle = StandardScaler().fit(vins_connus[variables])
recherche_voisins = NearestNeighbors(n_neighbors=8).fit(echelle.transform(vins_connus[variables]))

# Le même modèle sert trois besoins, seul le seuil change
seuils_usages = paquet["seuils_usages"]
BESOINS = {
    "preselection": ("Présélection avant dégustation", round(seuils_usages["preselection"], 2)),
    "equilibre": ("Équilibre entre précision et rappel", round(seuil_equilibre, 2)),
    "prestige": ("Label prestige", round(seuils_usages["prestige"], 2)),
}

CHAMPS = [
    ("fixed acidity", "Acidité fixe", 7.0, 0.1), ("volatile acidity", "Acidité volatile", 0.30, 0.01),
    ("citric acid", "Acide citrique", 0.31, 0.01), ("residual sugar", "Sucre résiduel", 2.7, 0.1),
    ("chlorides", "Chlorures", 0.047, 0.001), ("free sulfur dioxide", "Soufre libre", 28, 1),
    ("total sulfur dioxide", "Soufre total", 116, 1), ("density", "Densité", 0.9947, 0.0001),
    ("pH", "pH", 3.21, 0.01), ("sulphates", "Sulfates", 0.51, 0.01), ("alcohol", "Alcool", 10.4, 0.1),
]

app = Flask(__name__)


def predire(mesures, besoin="equilibre"):
    """Renvoie la probabilité, le verdict et les notes des vins les plus ressemblants."""
    vin = pd.DataFrame([mesures])[variables]
    probabilite = float(modele.predict_proba(vin)[0, 1])
    libelle, seuil = BESOINS[besoin]
    _, positions = recherche_voisins.kneighbors(echelle.transform(vin))
    return {
        "probabilite": round(probabilite, 3),
        "seuil": seuil,
        "besoin": libelle,
        "bon": bool(probabilite >= seuil),
        "notes_des_vins_ressemblants": [int(n) for n in vins_connus["quality"].values[positions[0]]],
    }


def lire_mesures(source):
    """Transforme les champs reçus en nombres. Lève une erreur si un champ manque."""
    mesures = {nom: float(source[nom]) for nom, _, _, _ in CHAMPS}
    mesures["est_rouge"] = int(source["est_rouge"])
    return mesures


@app.route("/", methods=["GET", "POST"])
def accueil():
    resultat, erreur, saisie = None, None, {}
    if request.method == "POST":
        saisie = request.form
        try:
            resultat = predire(lire_mesures(request.form), request.form.get("besoin", "equilibre"))
        except (KeyError, ValueError):
            erreur = "Il manque une mesure ou une valeur n'est pas un nombre."
    return render_template("index.html", champs=CHAMPS, besoins=BESOINS, resultat=resultat,
                           erreur=erreur, saisie=saisie, nom_modele=nom_modele)


@app.route("/api/predire", methods=["POST"])
def api_predire():
    donnees = request.get_json(silent=True) or {}
    try:
        return jsonify(predire(lire_mesures(donnees), donnees.get("besoin", "equilibre")))
    except (KeyError, ValueError):
        return jsonify({"erreur": "Il faut les 11 mesures et est_rouge (0 ou 1), en nombres."}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
