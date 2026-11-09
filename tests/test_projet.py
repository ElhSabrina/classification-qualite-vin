"""Tests unitaires. Lancement depuis la racine du projet :  pytest"""
import os
import sys

import pytest
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entrainement import construire_modele, entrainer, meilleur_seuil
from src.preparation import VARIABLES, charger_donnees, decouper, preparer

VIN_EXEMPLE = {"fixed acidity": 6.4, "volatile acidity": 0.15, "citric acid": 0.36, "residual sugar": 1.8,
               "chlorides": 0.034, "free sulfur dioxide": 43, "total sulfur dioxide": 150, "density": 0.9922,
               "pH": 3.42, "sulphates": 0.69, "alcohol": 11.0, "est_rouge": 0}


@pytest.fixture(scope="module")
def vins():
    return preparer(charger_donnees("data"))


@pytest.fixture(scope="module")
def decoupage(vins):
    return decouper(vins)


# Préparation des données

def test_chargement_complet():
    vins_brut = charger_donnees("data")
    assert len(vins_brut) == 6497
    assert vins_brut.isna().sum().sum() == 0


def test_plus_aucun_doublon(vins):
    assert vins.duplicated().sum() == 0
    assert len(vins) == 5320


def test_cible_et_encodage(vins):
    assert (vins["bon"] == (vins["quality"] >= 7)).all()
    assert (vins["est_rouge"] == (vins["type"] == "rouge")).all()


def test_decoupage_stratifie_sur_la_cible_et_la_couleur(decoupage):
    train, test = decoupage
    assert abs(train["bon"].mean() - test["bon"].mean()) < 0.01
    assert abs(train["est_rouge"].mean() - test["est_rouge"].mean()) < 0.01
    rouges_train, rouges_test = train[train["est_rouge"] == 1], test[test["est_rouge"] == 1]
    assert abs(rouges_train["bon"].mean() - rouges_test["bon"].mean()) < 0.01


def test_aucune_fuite_entre_entrainement_et_test(decoupage):
    """Aucun vin du test ne doit avoir un jumeau exact dans l'entraînement."""
    train, test = decoupage
    vus = set(train[VARIABLES].apply(tuple, axis=1))
    assert not test[VARIABLES].apply(tuple, axis=1).isin(vus).any()


# Modèle

def test_seuil_sur_un_cas_simple():
    assert meilleur_seuil([0, 0, 1, 1], [0.1, 0.2, 0.6, 0.9]) == pytest.approx(0.6)


def test_modele_meilleur_que_le_hasard(decoupage):
    train, _ = decoupage
    auc = cross_val_score(construire_modele(), train[VARIABLES], train["bon"], cv=3, scoring="roc_auc").mean()
    assert auc > 0.80


def test_entrainement_complet(tmp_path):
    modele, seuil = entrainer("data", str(tmp_path / "modele.joblib"))
    assert 0 < seuil < 0.5                      # la classe est rare, le seuil est donc sous 0,5
    assert (tmp_path / "modele.joblib").exists()


# Application web

@pytest.fixture(scope="module")
def client():
    if not os.path.exists("app/modele_final.joblib"):
        entrainer()
    from app.app import app
    return app.test_client()


def test_page_accueil(client):
    assert client.get("/").status_code == 200


def test_api_prediction(client):
    resultat = client.post("/api/predire", json=VIN_EXEMPLE).get_json()
    assert 0 <= resultat["probabilite"] <= 1
    assert len(resultat["notes_des_vins_ressemblants"]) == 8


def test_le_besoin_change_le_seuil_pas_la_probabilite(client):
    large = client.post("/api/predire", json={**VIN_EXEMPLE, "besoin": "preselection"}).get_json()
    strict = client.post("/api/predire", json={**VIN_EXEMPLE, "besoin": "prestige"}).get_json()
    assert large["probabilite"] == strict["probabilite"]
    assert large["seuil"] < strict["seuil"]


def test_api_refuse_une_saisie_incomplete(client):
    assert client.post("/api/predire", json={"alcohol": 11}).status_code == 400
