# Version script du notebook 2_modelisation.ipynb, à lancer depuis la racine du dépôt.
# Les blocs « # %% » se lisent comme des cellules dans VS Code ou Spyder.

# %% [markdown]
# # Partie 2. Construire, régler et comprendre les cinq modèles
#
# Projet 2 de Machine Learning 1, EFREI 2026-2027.
#
# Cette partie reprend le jeu d'entraînement préparé en partie 1. Elle entraîne les cinq modèles du sujet, les règle, puis cherche à comprendre ce que chacun a appris. Le jeu de test n'est jamais ouvert ici. Tous les scores de cette partie viennent de la validation croisée expliquée en 1.7.

# %%
import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.model_selection import (StratifiedKFold, cross_validate, cross_val_score, cross_val_predict,
                                     GridSearchCV, RandomizedSearchCV, validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import precision_recall_curve
from scipy.stats import loguniform

# Style commun des graphiques du groupe
ENCRE, PRUNE, MIEL, BRUME = "#1F2A44", "#7A3B69", "#E3A72F", "#C5CAD3"
plt.rcParams.update({
    "font.family": "serif", "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#EAEAEA", "axes.axisbelow": True,
    "axes.titlelocation": "left", "axes.titleweight": "bold", "axes.titlesize": 12,
    "legend.frameon": False,
})
GRAINE = 42
pd.set_option("display.precision", 3)

# %%
train = pd.read_csv("donnees_preparees/train.csv")
variables = [c for c in train.columns if c not in ["quality", "type", "bon"]]
X_train, y_train = train[variables], train["bon"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=GRAINE)
print(len(X_train), "vins d'entraînement,", len(variables), "variables,", f"{y_train.mean():.1%} de bons vins")

# %% [markdown]
# ## 2.1 Quels sont les cinq modèles, et lesquels sont linéaires ?
#
# Nous comparons une régression logistique, qui est le seul modèle linéaire, à quatre modèles qui ne le sont pas. Un modèle linéaire sépare les bons vins des mauvais par une frontière droite. Les autres peuvent tracer une frontière plus souple.
#
# | Modèle | Comment il décide | Linéaire | Ce qu'il suppose |
# |---|---|---|---|
# | Régression logistique | Il additionne les mesures, chacune multipliée par un poids, et transforme le total en probabilité | Oui | Que l'effet de chaque mesure s'ajoute à celui des autres |
# | k plus proches voisins | Il regarde les k vins les plus ressemblants et compte combien sont bons | Non | Que deux vins proches ont la même qualité |
# | Arbre de décision | Il pose une suite de questions du type « alcool inférieur à 10,6 ? » | Non | Rien de particulier |
# | Naive Bayes | Il calcule la probabilité d'être bon d'après chaque mesure prise séparément, puis multiplie ces probabilités | Non | Que les mesures sont indépendantes et suivent une loi normale |
# | SVM | Il cherche la frontière qui passe le plus loin possible des deux classes, et un noyau lui permet de la courber | Selon le noyau | Que la distance entre deux vins a un sens |

# %% [markdown]
# ## 2.2 Que donnent-ils sans réglage ?
#
# Sans réglage, la régression logistique et le SVM sont en tête, et l'arbre de décision est de loin le moins bon. Pour chaque modèle, nous relevons son score sur les vins qui ont servi à l'entraîner et son score sur les vins de validation.

# %%
modeles = {
    "Régression logistique": Pipeline([("echelle", StandardScaler()), ("modele", LogisticRegression(max_iter=2000))]),
    "k plus proches voisins": Pipeline([("echelle", StandardScaler()), ("modele", KNeighborsClassifier())]),
    "Arbre de décision": Pipeline([("modele", DecisionTreeClassifier(random_state=GRAINE))]),
    "Naive Bayes": Pipeline([("modele", GaussianNB())]),
    "SVM": Pipeline([("echelle", StandardScaler()), ("modele", SVC())]),
}

lignes = []
for nom, modele in modeles.items():
    notes = cross_validate(modele, X_train, y_train, cv=cv, scoring=["roc_auc", "f1"], return_train_score=True)
    lignes.append({"modèle": nom,
                   "AUC entraînement": notes["train_roc_auc"].mean(),
                   "AUC validation": notes["test_roc_auc"].mean(),
                   "F1 validation": notes["test_f1"].mean()})
depart = pd.DataFrame(lignes).set_index("modèle")
depart["écart d'AUC"] = depart["AUC entraînement"] - depart["AUC validation"]
depart.round(3)

# %% [markdown]
# L'arbre obtient 1.000 sur les vins d'entraînement et seulement 0.649 en validation. Il a retenu les vins un par un au lieu d'apprendre une règle générale. La régression logistique et Naive Bayes ont au contraire presque le même score des deux côtés. La section 2.6 revient sur ces deux comportements.
#
# Le F1-score de ce tableau utilise la décision par défaut de chaque modèle. Il ne classe pas les modèles dans le même ordre que l'AUC. Naive Bayes a par exemple le meilleur F1-score et l'une des moins bonnes AUC. La section 2.4 explique pourquoi.
#
# ## 2.3 Comment choisir les hyperparamètres ?
#
# Nous essayons toutes les combinaisons d'une grille de valeurs et nous gardons celle qui obtient le meilleur score en validation croisée à 5 plis. Cette méthode s'appelle la recherche par grille. Un hyperparamètre est un réglage que le modèle n'apprend pas seul et qu'il faut lui fixer.
#
# | Modèle | Réglage | Ce qu'il change |
# |---|---|---|
# | Régression logistique | `C` | Un petit C oblige les poids à rester petits, ce qui simplifie le modèle |
# | Régression logistique et SVM | `class_weight` | Avec `balanced`, une erreur sur un bon vin coûte environ quatre fois plus qu'une erreur sur un mauvais |
# | k plus proches voisins | `n_neighbors` | Avec peu de voisins le modèle suit chaque vin de près, avec beaucoup il fait une moyenne |
# | k plus proches voisins | `weights` et `p` | Vote égal ou vote plus fort des voisins les plus proches, et façon de mesurer la distance |
# | Arbre de décision | `max_depth`, `min_samples_leaf`, `criterion` | La taille de l'arbre, et le critère de choix des questions (Gini ou entropie) |
# | Naive Bayes | nombre de mesures | On ajoute une par une la mesure qui aide le plus, comme annoncé en 1.11 |
# | SVM | `kernel`, `C`, `gamma` | Frontière droite ou courbe. Un grand C tolère peu d'erreurs. Un grand gamma donne une frontière très sinueuse |

# %%
grilles = {
    "Régression logistique": {"modele__C": [0.01, 0.1, 1, 10, 100],
                              "modele__class_weight": [None, "balanced"]},
    "k plus proches voisins": {"modele__n_neighbors": [1, 3, 5, 9, 15, 25, 35, 51, 75],
                               "modele__weights": ["uniform", "distance"],
                               "modele__p": [1, 2]},
    "Arbre de décision": {"modele__max_depth": [2, 3, 4, 6, 8, 10, None],
                          "modele__min_samples_leaf": [1, 5, 10, 20, 50, 80],
                          "modele__criterion": ["gini", "entropy"]},
    "Naive Bayes": {"selection__n_features_to_select": [1, 2, 3, 4, 5, 6, 8, 10]},
    "SVM": [{"modele__kernel": ["rbf"], "modele__C": [0.3, 1, 3, 10], "modele__gamma": [0.003, 0.01, 0.03, 0.1, 0.3],
             "modele__class_weight": [None, "balanced"]},
            {"modele__kernel": ["linear"], "modele__C": [0.1, 1], "modele__class_weight": [None, "balanced"]}],
}

# Pour Naive Bayes, le réglage utile est le choix des variables (partie 1)
a_regler = dict(modeles)
a_regler["Naive Bayes"] = Pipeline([("selection", SequentialFeatureSelector(GaussianNB(), scoring="roc_auc", cv=5)),
                                    ("modele", GaussianNB())])

recherches = {}
lignes = []
for nom, modele in a_regler.items():
    debut = time.time()
    recherche = GridSearchCV(modele, grilles[nom], cv=cv, scoring={"auc": "roc_auc", "f1": "f1"},
                             refit="auc", n_jobs=-1, return_train_score=True)
    recherche.fit(X_train, y_train)
    recherches[nom] = recherche
    i = recherche.best_index_
    lignes.append({"modèle": nom,
                   "essais": len(recherche.cv_results_["params"]),
                   "durée (s)": round(time.time() - debut),
                   "AUC entraînement": recherche.cv_results_["mean_train_auc"][i],
                   "AUC validation": recherche.best_score_,
                   "écart type": recherche.cv_results_["std_test_auc"][i],
                   "F1 validation par défaut": recherche.cv_results_["mean_test_f1"][i],
                   "réglages retenus": {k.split("__")[1]: v for k, v in recherche.best_params_.items()}})
optimisation = pd.DataFrame(lignes).set_index("modèle")
optimisation["gain d'AUC"] = optimisation["AUC validation"] - depart["AUC validation"]
pd.set_option("display.max_colwidth", 90)
optimisation.round(3)

# %% [markdown]
# Les deux modèles qui retenaient les vins un par un sont ceux qui gagnent le plus. L'arbre gagne 0.159 d'AUC et les voisins 0.051. Les voisins passent de 5 à 25 voisins, et l'arbre demande au moins 50 vins dans chaque feuille. Dans les deux cas, le modèle réglé s'appuie sur plus de vins pour chaque décision.
#
# La régression logistique ne gagne rien. Elle n'a que 12 poids à apprendre pour 4256 vins, elle ne risque donc pas de retenir les vins un par un.
#
# Le SVM gagne surtout grâce à `class_weight="balanced"`. Sans cette option, il cherche à faire le moins d'erreurs possible au total. Comme les mauvais vins sont quatre fois plus nombreux, il place sa frontière pour bien classer les mauvais et néglige les bons. Avec l'option, les deux classes pèsent autant et la frontière se replace entre elles. Pour la régression logistique, la même option ne change presque pas l'AUC, car elle décale surtout les probabilités sans beaucoup modifier l'ordre des vins.
#
# Deux chiffres de ce tableau demandent une précision. Les voisins ont une AUC de 1 sur les vins d'entraînement. Avec un vote plus fort pour les voisins les plus proches, chaque vin d'entraînement est son propre voisin à distance nulle, ce score ne veut donc rien dire. La colonne du F1-score par défaut est expliquée en 2.4.
#
# ## 2.4 Pourquoi régler avec l'AUC plutôt qu'avec le F1 ?
#
# Parce que le F1-score par défaut récompense les modèles qui retiennent trop les vins d'entraînement. Pour le montrer, nous regardons le réglage que le F1-score aurait choisi pour chaque modèle. Nous calculons aussi le meilleur F1-score que chaque réglage peut atteindre quand la décision est bien placée, ce que la partie 3 explique en 3.2.

# %%
def meilleur_f1_possible(modele):
    """F1-score obtenu en choisissant le meilleur seuil, sur des prédictions hors pli."""
    methode = "predict_proba" if hasattr(modele[-1], "predict_proba") else "decision_function"
    notes = cross_val_predict(modele, X_train, y_train, cv=cv, method=methode)
    notes = notes[:, 1] if notes.ndim == 2 else notes
    precisions, rappels, _ = precision_recall_curve(y_train, notes)
    return (2 * precisions * rappels / (precisions + rappels + 1e-12)).max()

lignes = []
for nom, recherche in recherches.items():
    resultats = pd.DataFrame(recherche.cv_results_)
    j = resultats["mean_test_f1"].idxmax()                       # le réglage qu'aurait choisi le F1-score
    modele_f1 = clone(a_regler[nom]).set_params(**resultats.loc[j, "params"])
    lignes.append({"modèle": nom,
                   "réglage choisi par le F1": {k.split("__")[1]: v for k, v in resultats.loc[j, "params"].items()},
                   "son F1 au seuil 0,5": resultats.loc[j, "mean_test_f1"],
                   "son AUC": resultats.loc[j, "mean_test_auc"],
                   "son F1 au meilleur seuil": meilleur_f1_possible(modele_f1),
                   "F1 au meilleur seuil du réglage choisi par l'AUC": meilleur_f1_possible(recherche.best_estimator_)})
comparaison_criteres = pd.DataFrame(lignes).set_index("modèle")
comparaison_criteres.round(3)

# %% [markdown]
# Pour les voisins, le F1-score aurait choisi 3 voisins au lieu de 25. Ce réglage a une AUC de 0.785 seulement. Et même avec la décision bien placée, son F1-score plafonne à 0.527, contre 0.568 pour le réglage choisi par l'AUC.
#
# L'explication est la suivante. Les bons vins sont rares, donc un modèle prudent annonce rarement un bon vin et son F1-score par défaut est faible. Un modèle qui a retenu les vins un par un répond de façon plus tranchée, il annonce plus de bons vins et son F1-score paraît meilleur. Le F1-score par défaut mélange donc deux choses, la qualité du classement et la façon de trancher.
#
# Nous séparons ces deux questions. L'AUC sert à régler et à choisir le modèle. Le F1-score sert ensuite à régler la décision, en partie 3.
#
# ## 2.5 La recherche aléatoire fait-elle aussi bien ?
#
# Oui, elle arrive au même score que la grille avec beaucoup moins d'essais. Au lieu d'essayer toutes les combinaisons, elle en tire quelques-unes au hasard. Nous la comparons à la grille sur le SVM, qui est le modèle le plus long à régler.

# %%
debut = time.time()
recherche_aleatoire = RandomizedSearchCV(
    modeles["SVM"].set_params(modele__class_weight="balanced"),
    {"modele__C": loguniform(0.3, 10), "modele__gamma": loguniform(0.003, 0.3)},
    n_iter=10, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=GRAINE).fit(X_train, y_train)
duree_aleatoire = time.time() - debut

print(f"Recherche par grille   {optimisation.loc['SVM', 'essais']} essais, {optimisation.loc['SVM', 'durée (s)']} s, AUC {optimisation.loc['SVM', 'AUC validation']:.3f}")
print(f"Recherche aléatoire    10 essais, {duree_aleatoire:.0f} s, AUC {recherche_aleatoire.best_score_:.3f}")
print("Réglages trouvés au hasard", {k.split("__")[1]: round(v, 3) for k, v in recherche_aleatoire.best_params_.items()})

# %% [markdown]
# Avec 10 essais au lieu de 44, la recherche aléatoire obtient 0.845 contre 0.844 pour la grille. La grille passe du temps sur des zones entières qui ne donnent rien, comme tous les essais avec un grand gamma. La recherche aléatoire ne garantit pas de trouver le meilleur réglage, mais elle essaie des valeurs intermédiaires que la grille ne contient pas.
#
# Nous gardons les réglages de la grille, qui reste rapide sur ce projet. Avec plus de données ou plus de réglages, la recherche aléatoire serait le bon point de départ.
#
# ## 2.6 Comment éviter un modèle trop rigide ou trop collé aux données ?
#
# Il faut régler la souplesse du modèle pour obtenir le meilleur score de validation, et non le meilleur score d'entraînement. Un modèle trop rigide ne peut pas saisir ce qui distingue les bons vins. On dit qu'il a du biais. Un modèle trop souple retient les particularités des vins d'entraînement, qui ne se retrouvent pas sur de nouveaux vins. On dit qu'il a de la variance.
#
# Pour l'arbre, les voisins et le SVM, nous faisons varier un seul réglage et nous traçons les deux scores. Le cercle marque le meilleur score de validation.

# %%
def courbe_de_complexite(ax, modele, parametre, valeurs, titre, echelle_log=False):
    notes_train, notes_val = validation_curve(modele, X_train, y_train, param_name=parametre,
                                              param_range=valeurs, cv=cv, scoring="roc_auc", n_jobs=-1)
    ax.plot(valeurs, notes_train.mean(axis=1), marker="o", color=BRUME, label="vins d'entraînement")
    ax.plot(valeurs, notes_val.mean(axis=1), marker="o", color=PRUNE, label="vins de validation")
    meilleur = notes_val.mean(axis=1).argmax()
    ax.scatter(valeurs[meilleur], notes_val.mean(axis=1)[meilleur], s=220, facecolors="none", edgecolors=ENCRE, linewidths=2, zorder=5)
    if echelle_log:
        ax.set_xscale("log")
    ax.set_ylim(0.6, 1.02); ax.set_title(titre); ax.set_ylabel("AUC")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
courbe_de_complexite(axes[0], modeles["Arbre de décision"], "modele__max_depth", [1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
                     "Arbre, profondeur maximale\nplus complexe vers la droite")
courbe_de_complexite(axes[1], modeles["k plus proches voisins"], "modele__n_neighbors", [1, 3, 5, 9, 15, 25, 35, 51, 75, 101],
                     "Voisins, nombre k\nplus complexe vers la gauche")
courbe_de_complexite(axes[2], modeles["SVM"], "modele__gamma", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
                     "SVM, paramètre gamma\nplus complexe vers la droite", echelle_log=True)
axes[2].legend(loc="upper left")
plt.tight_layout(); plt.show()

# %% [markdown]
# Les trois courbes ont la même forme. Du côté souple, le score d'entraînement monte vers 1 pendant que le score de validation baisse. Du côté rigide, les deux scores sont proches et bas. Le meilleur réglage est entre les deux.
#
# Pour l'arbre, chaque niveau ajouté au-delà de 4 améliore l'entraînement et dégrade la validation. La grille retient pourtant une profondeur de 8. Ce n'est pas contradictoire, car elle impose en même temps au moins 50 vins par feuille, ce qui empêche l'arbre de créer des règles trop fines.
#
# Le tableau suivant fait le bilan pour les cinq modèles réglés.

# %%
bilan = optimisation[["AUC entraînement", "AUC validation"]].copy()
bilan["écart"] = bilan["AUC entraînement"] - bilan["AUC validation"]
bilan["écart avant réglage"] = depart["écart d'AUC"]
bilan.sort_values("AUC validation", ascending=False).round(3)

# %% [markdown]
# La régression logistique et Naive Bayes ont le même score en entraînement et en validation. Ils sont stables mais plafonnent, leur limite est le biais. L'arbre était le modèle le plus collé aux données, avec un écart de 0.351 avant réglage. Le réglage ramène cet écart à 0.056, mais son score reste en retrait. Le SVM a la meilleure validation avec un écart modéré de 0.041. L'écart des voisins ne se lit pas, pour la raison donnée en 2.3.
#
# ## 2.7 Qu'a appris chaque modèle ?
#
# Chaque modèle confirme que l'alcool est la mesure la plus utile, et leurs différences montrent que la qualité dépend de la façon dont les mesures se combinent.
#
# **La régression logistique**

# %%
# Les cinq modèles avec leurs meilleurs réglages, entraînés sur tous les vins d'entraînement
optimises = {nom: recherche.best_estimator_ for nom, recherche in recherches.items()}
optimises["SVM"].set_params(modele__probability=True, modele__random_state=GRAINE)   # pour obtenir des probabilités
for modele in optimises.values():
    modele.fit(X_train, y_train)

# %%
logistique = optimises["Régression logistique"].named_steps["modele"]
poids = pd.Series(logistique.coef_[0], index=variables).sort_values()

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.hlines(range(len(poids)), 0, poids, color=BRUME, linewidth=2)
ax.scatter(poids, range(len(poids)), color=[PRUNE if v > 0 else ENCRE for v in poids], s=80, zorder=3)
ax.axvline(0, color=ENCRE, linewidth=1)
ax.set_yticks(range(len(poids))); ax.set_yticklabels(poids.index)
ax.set_xlabel("poids de la mesure standardisée (vers la droite, augmente les chances d'être bon)")
ax.set_title("Poids de chaque mesure dans la régression logistique")
plt.show()
print(f"Corrélation directe entre le sucre résiduel et la cible {X_train['residual sugar'].corr(y_train):+.2f}")

# %% [markdown]
# Le plus gros poids positif est celui du sucre résiduel. On pourrait en conclure que le sucre est le premier facteur de qualité. Ce serait une erreur, car le lien direct entre le sucre et la cible est légèrement négatif.
#
# Un poids se lit en supposant que toutes les autres mesures restent égales. Or la densité dépend presque entièrement de l'alcool et du sucre, comme vu en 1.6. À densité égale, un vin plus sucré est forcément plus alcoolisé, puisque l'alcool compense le poids du sucre. Le poids du sucre et celui de la densité sont donc deux façons indirectes de dire que le vin a plus d'alcool. Quand des mesures se répètent, leurs poids ne s'interprètent pas un par un.
#
# En 1.5, nous avions vu que certaines mesures n'ont pas le même effet chez les rouges et chez les blancs. Nous pensions que cela expliquait le retard de la régression logistique sur les meilleurs modèles. Nous testons cette idée, ainsi qu'une seconde.

# %%
# Que manque-t-il à la régression logistique ? Deux explications possibles sont testées.
modele_logistique = optimises["Régression logistique"]

# Explication 1, il lui faudrait un poids différent pour les rouges (interactions avec la couleur)
X_couleur = X_train.copy()
for variable in variables[:-1]:
    X_couleur[variable + " x rouge"] = X_train[variable] * X_train["est_rouge"]

# Explication 2, il lui faudrait pouvoir combiner les mesures entre elles (carrés et produits)
avec_produits = Pipeline([("echelle", StandardScaler()), ("produits", PolynomialFeatures(2, include_bias=False)),
                          ("modele", LogisticRegression(max_iter=5000, C=0.1))])

essais = pd.Series({
    "Régression logistique de départ": cross_val_score(modele_logistique, X_train, y_train, cv=cv, scoring="roc_auc").mean(),
    "avec un poids différent pour les rouges": cross_val_score(modele_logistique, X_couleur, y_train, cv=cv, scoring="roc_auc").mean(),
    "avec les carrés et les produits de mesures": cross_val_score(avec_produits, X_train, y_train, cv=cv, scoring="roc_auc").mean(),
}, name="AUC validation")
essais.round(3)

# %% [markdown]
# Notre première idée était fausse. Donner à la régression logistique un poids différent pour les rouges ne change presque rien (0.826 contre 0.825). En revanche, lui donner les produits de mesures la fait monter à 0.839, presque au niveau des meilleurs modèles. Ce qui lui manque, ce sont donc les combinaisons entre mesures. L'effet d'une mesure sur la qualité dépend des autres mesures du même vin, ce qui correspond à l'idée d'un vin équilibré.
#
# **L'arbre de décision**

# %%
arbre = optimises["Arbre de décision"].named_steps["modele"]
plt.figure(figsize=(15, 5))
plot_tree(arbre, max_depth=2, feature_names=variables, class_names=["mauvais", "bon"],
          filled=True, impurity=False, proportion=True, fontsize=9)
plt.title("Les premières questions de l'arbre réglé")
plt.show()
print(f"Arbre de profondeur {arbre.get_depth()} avec {arbre.get_n_leaves()} feuilles")

# %% [markdown]
# La première question de l'arbre porte sur l'alcool, et l'alcool revient aux niveaux suivants avec l'acidité volatile. L'arbre retrouve donc les deux mesures repérées en 1.5. Sa faiblesse vient de sa façon de travailler. La qualité augmente peu à peu avec l'alcool, et l'arbre doit imiter cette pente avec une série de seuils, comme un escalier.
#
# **Naive Bayes**

# %%
# Naive Bayes souffre-t-il vraiment des mesures qui se répètent ? On les décorrèle avant de les lui donner.
bayes_acp = Pipeline([("echelle", StandardScaler()), ("acp", PCA()), ("modele", GaussianNB())])

essais_bayes = pd.Series({
    "Naive Bayes avec les 12 mesures": cross_val_score(GaussianNB(), X_train, y_train, cv=cv, scoring="roc_auc").mean(),
    "Naive Bayes après décorrélation des mesures": cross_val_score(bayes_acp, X_train, y_train, cv=cv, scoring="roc_auc").mean(),
    "Naive Bayes avec les mesures sélectionnées": optimisation.loc["Naive Bayes", "AUC validation"],
}, name="AUC validation")

selection_bayes = optimises["Naive Bayes"].named_steps["selection"]
print("Mesures gardées par la sélection", list(X_train.columns[selection_bayes.get_support()]))
essais_bayes.round(3)

# %% [markdown]
# En 1.11, Naive Bayes se dégradait quand on lui donnait des mesures qui se répètent. Nous le vérifions autrement. L'analyse en composantes principales remplace les 12 mesures par 12 combinaisons qui ne sont plus corrélées entre elles. Avec ces combinaisons, Naive Bayes passe de 0.771 à 0.795. C'est bien la répétition entre mesures qui le gênait. La sélection de mesures retenue en 2.3 arrive au même résultat (0.801) en restant plus simple à expliquer.
#
# **Le SVM et les k plus proches voisins**

# %%
svm = optimises["SVM"].named_steps["modele"]
part_supports = svm.n_support_.sum() / len(X_train)
print(f"Le SVM utilise {svm.n_support_.sum()} vecteurs supports sur {len(X_train)} vins, soit {part_supports:.0%}")

# %% [markdown]
# 54% des vins d'entraînement sont des vecteurs supports, c'est-à-dire des vins proches de la frontière ou du mauvais côté. Dans un cas facile, quelques vins suffiraient à fixer la frontière. Ici il en faut plus de la moitié, ce qui montre que les bons et les mauvais vins sont très mélangés.
#
# Le SVM à noyau gaussien et les k plus proches voisins raisonnent tous les deux par ressemblance. Les voisins comparent directement un vin aux vins les plus proches de lui. Le SVM gaussien décide aussi d'après la distance du vin à ses vecteurs supports, et les plus proches comptent le plus. Dans les deux cas, toutes les mesures du vin sont prises en compte ensemble. Ce sont justement les deux meilleurs modèles.
#
# | Modèle | Pourquoi ce résultat est cohérent avec le vin |
# |---|---|
# | k plus proches voisins | Deux vins de même composition ont de bonnes chances d'avoir le même goût, donc une note proche |
# | SVM | Sa frontière courbe tient compte des combinaisons entre mesures, et il tolère des vins mal classés près de la frontière |
# | Régression logistique | Elle saisit bien l'effet de l'alcool mais additionne les mesures sans les combiner |
# | Arbre de décision | Il découpe en tranches une qualité qui évolue progressivement |
# | Naive Bayes | Les mesures d'un vin sont liées par la chimie, son hypothèse d'indépendance est donc fausse |
#
# ## 2.8 Combiner les modèles aide-t-il ?
#
# Très peu, car nos modèles hésitent sur les mêmes vins. Combiner des modèles n'est utile que s'ils se trompent sur des vins différents. Nous commençons donc par regarder si leurs probabilités se ressemblent.

# %%
# Les modèles hésitent-ils sur les mêmes vins ? On compare leurs probabilités, calculées en validation croisée.
probas_cv = pd.DataFrame({nom: cross_val_predict(modele, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
                          for nom, modele in optimises.items()})
accord = probas_cv.corr()
accord.round(2)

# %% [markdown]
# Les probabilités des cinq modèles sont très liées, avec des corrélations entre 0.76 et 0.88. Nous testons ensuite les trois méthodes d'ensemble du cours.
#
# | Méthode | Principe |
# |---|---|
# | Forêt aléatoire | 300 arbres entraînés chacun sur un tirage différent des vins, puis un vote |
# | AdaBoost | De très petits arbres entraînés à la suite, chacun insistant sur les erreurs du précédent |
# | Stacking | Une régression logistique apprend à combiner les réponses de nos cinq modèles |

# %%
ensembles = {
    "Forêt aléatoire (bagging)": RandomForestClassifier(n_estimators=300, random_state=GRAINE, n_jobs=-1),
    "AdaBoost (boosting)": AdaBoostClassifier(n_estimators=200, random_state=GRAINE),
    "Stacking des 5 modèles": StackingClassifier(estimators=list(optimises.items()),
                                                 final_estimator=LogisticRegression(max_iter=1000), cv=5, n_jobs=-1),
}
lignes = []
for nom, modele in ensembles.items():
    notes = cross_validate(modele, X_train, y_train, cv=cv, scoring="roc_auc", return_train_score=True)
    lignes.append({"modèle": nom, "AUC entraînement": notes["train_score"].mean(),
                   "AUC validation": notes["test_score"].mean(), "écart type": notes["test_score"].std()})
resultats_ensembles = pd.DataFrame(lignes).set_index("modèle")
resultats_ensembles.round(3)

# %% [markdown]
# La forêt aléatoire fait passer l'arbre de 0.808 à 0.850. Le défaut de l'arbre était la variance, et faire voter 300 arbres différents la réduit.
#
# AdaBoost est en dessous. Il insiste sur les vins mal classés, et nous verrons en partie 3 que ces vins sont souvent ceux dont la note est la plus discutable.
#
# La forêt et le stacking ne dépassent le SVM (0.844) que de 0.010 au plus. C'est moins que l'écart type entre plis, qui est d'environ 0.013. Aucune méthode ne dépasse nettement 0,85. Nous restons donc sur les cinq modèles du sujet pour la partie 3.

# %%
os.makedirs("modeles", exist_ok=True)
joblib.dump({"modeles": optimises, "optimisation": optimisation}, "modeles/modeles_optimises.joblib")
print("Cinq modèles réglés enregistrés pour la partie 3")
