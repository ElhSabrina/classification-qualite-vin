# Version script du notebook 3_evaluation.ipynb, à lancer depuis la racine du dépôt.
# Les blocs « # %% » se lisent comme des cellules dans VS Code ou Spyder.

# %% [markdown]
# # Partie 3. Choisir le modèle et juger sa fiabilité
#
# Projet 2 de Machine Learning 1, EFREI 2026-2027.
#
# Cette partie recharge les cinq modèles réglés en partie 2. Elle choisit le meilleur et explique pourquoi il gagne, fixe sa règle de décision, puis ouvre le jeu de test une seule fois pour mesurer ce qu'il vaut, comprendre ses erreurs et en tirer les limites.

# %%
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import clone
from sklearn.model_selection import (StratifiedKFold, RepeatedStratifiedKFold, cross_val_score,
                                     cross_val_predict, learning_curve)
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             precision_recall_curve, roc_curve, confusion_matrix)

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
test = pd.read_csv("donnees_preparees/test.csv")
variables = [c for c in train.columns if c not in ["quality", "type", "bon"]]
X_train, y_train = train[variables], train["bon"]
X_test, y_test = test[variables], test["bon"]

paquet = joblib.load("modeles/modeles_optimises.joblib")
optimises, optimisation = paquet["modeles"], paquet["optimisation"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=GRAINE)
print(len(X_train), "vins d'entraînement,", len(X_test), "vins de test, cinq modèles réglés chargés")

# %% [markdown]
# ## 3.1 Quel modèle est le meilleur, et sur quoi se base-t-on ?
#
# Pour comparer les modèles nous utilisons l'AUC. Elle vérifie que le modèle donne en général une probabilité plus haute aux bons vins qu'aux mauvais. Elle vaut 0,5 pour un modèle qui répondrait au hasard et 1 pour un modèle qui donnerait toujours une probabilité plus haute à un bon vin qu'à un mauvais. Elle ne dépend pas du seuil à partir duquel on dit bon, que nous ne fixerons qu'en 3.2. On peut donc comparer la qualité des modèles avant de régler leur décision.
#
# Les écarts entre modèles sont faibles et un seul découpage pourrait avantager un modèle par chance. Un découpage est une façon de partager les vins d'entraînement entre ceux qui servent à apprendre et ceux qui servent à noter le modèle. La validation croisée de la section 1.7 en fait 5. Nous la répétons 5 fois en mélangeant les vins différemment à chaque fois, ce qui donne 25 notes par modèle. Les cinq modèles sont notés sur exactement les mêmes découpages, et nous comptons sur combien de découpages le premier fait mieux que chacun des autres.

# %%
# 25 découpages, soit une validation croisée à 5 plis répétée 5 fois avec des tirages différents
cv25 = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=GRAINE)

a_comparer = dict(optimises)
a_comparer["SVM"] = clone(optimises["SVM"]).set_params(modele__probability=False)   # même modèle, calcul plus rapide

auc_25 = pd.DataFrame({nom: cross_val_score(modele, X_train, y_train, cv=cv25, scoring="roc_auc", n_jobs=-1)
                       for nom, modele in a_comparer.items()})

classement = pd.DataFrame({"AUC moyenne": auc_25.mean(), "écart type": auc_25.std()}).sort_values("AUC moyenne", ascending=False)
choisi = classement.index[0]
classement[f"découpages où le {choisi} fait mieux (sur 25)"] = [(auc_25[choisi] > auc_25[nom]).sum() for nom in classement.index]

# Le noyau linéaire, déjà testé en partie 2, est remis sur les mêmes 25 découpages pour comparer
svm_lineaire = clone(a_comparer["SVM"]).set_params(modele__kernel="linear")
auc_lineaire = cross_val_score(svm_lineaire, X_train, y_train, cv=cv25, scoring="roc_auc", n_jobs=-1)

# Une boîte par modèle avec ses 25 AUC, le modèle retenu en violet et en haut
ordre = list(classement.index[::-1])
fig, ax = plt.subplots(figsize=(8, 3.8))
boites = ax.boxplot([auc_25[nom] for nom in ordre], orientation="horizontal", widths=0.55, patch_artist=True,
                    medianprops=dict(color=ENCRE, linewidth=2),
                    flierprops=dict(marker="o", markersize=4, markerfacecolor=BRUME, markeredgecolor=BRUME))
for boite, nom in zip(boites["boxes"], ordre):
    boite.set(facecolor=PRUNE if nom == choisi else BRUME, edgecolor=ENCRE, alpha=0.85)
ax.set_yticks(range(1, len(ordre) + 1)); ax.set_yticklabels(ordre)
ax.set_xlabel("AUC obtenue sur chacun des 25 découpages")
ax.set_title("AUC des cinq modèles sur 25 découpages")
plt.show()

print("Modèle retenu avant d'ouvrir le jeu de test,", choisi)
print(f"Le même SVM avec un noyau linéaire obtient une AUC moyenne de {auc_lineaire.mean():.3f}")
classement.round(3)

# %% [markdown]
# Chaque boîte résume les 25 AUC d'un modèle. Le trait au milieu est la médiane, c'est à dire la valeur du milieu quand on range les 25 AUC. La boîte contient la moitié des découpages, ceux du milieu, et les moustaches vont jusqu'aux valeurs les plus éloignées. Plus une boîte est à droite, meilleur est le modèle, et plus elle est étroite, plus ses résultats sont réguliers d'un découpage à l'autre.
#
# La boîte du SVM est la plus à droite, avec une AUC moyenne de 0,843. Il fait mieux que la régression logistique, l'arbre et Naive Bayes sur les 25 découpages, et leurs boîtes sont nettement à gauche de la sienne. Celle des k plus proches voisins, à 0,840 de moyenne, chevauche largement celle du SVM et elle est même un peu plus étroite. Sur un seul découpage, on ne pourrait donc pas départager ces deux modèles. C'est pour cela que nous comptons les découpages gagnés. Le SVM fait mieux que les voisins sur 17 découpages sur 25, l'écart est petit mais il va presque toujours dans le même sens. Nous retenons donc le SVM, en gardant en tête que les voisins seraient un choix presque aussi bon.
#
# C'est aussi le modèle qui offrait le meilleur compromis entre biais et variance en 2.6. Avec un noyau linéaire, déjà écarté par la recherche par grille en 2.3, le même SVM retombe ici à 0,824, au niveau de la régression logistique. Cela confirme que c'est sa frontière courbe qui fait la différence.
#
# Ce choix est fait avant d'ouvrir le jeu de test.

# %% [markdown]
# ### Pourquoi ces deux modèles sont-ils les meilleurs pour juger un vin ?
#
# Le classement s'explique par la façon dont un vin est jugé. Un dégustateur ne note pas l'alcool d'un côté et l'acidité de l'autre, il juge un goût d'ensemble. Un vin plaît quand ses mesures s'équilibrent entre elles, par exemple un degré d'alcool élevé avec peu d'acidité volatile. Deux vins dont la composition est très proche ont donc un goût proche et reçoivent en général une note proche.
#
# La section 2.7 a montré que les k plus proches voisins et le SVM à noyau gaussien raisonnent tous les deux par ressemblance entre vins, en regardant toutes les mesures ensemble. C'est exactement ce qu'il faut pour tenir compte de cet équilibre, et c'est pourquoi ils arrivent en tête. Les trois autres modèles reposent sur des hypothèses qui correspondent moins bien au vin.
#
# ## 3.2 Que donne le modèle, et à partir de quelle probabilité dit-on « bon » ?
#
# Le modèle ne répond pas directement bon ou pas bon. Pour chaque vin il donne une probabilité d'être bon, par exemple 0,62, et c'est un seuil qui transforme cette probabilité en décision. Si la probabilité dépasse le seuil, on dit bon. Le seuil habituel est 0,5.
#
# Comme seulement 19 % des vins sont bons, le modèle donne rarement plus de 0,5 et il rate alors la plupart des bons vins. Nous cherchons donc le seuil qui donne le meilleur F1-score. Changer le seuil ne change pas le modèle, seulement la décision. Il est choisi sur les vins d'entraînement grâce à la validation croisée et jamais sur le test.

# %%
def meilleur_seuil(y_vrai, probas):
    """Seuil de probabilité qui donne le meilleur F1-score."""
    precisions, rappels, seuils_possibles = precision_recall_curve(y_vrai, probas)
    f1 = 2 * precisions * rappels / (precisions + rappels + 1e-12)
    return seuils_possibles[f1[:-1].argmax()]

# Probabilités calculées en validation croisée, chaque vin est noté par un modèle qui ne l'a pas vu
probas_cv = {nom: cross_val_predict(modele, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
             for nom, modele in optimises.items()}
seuils = {nom: meilleur_seuil(y_train, p) for nom, p in probas_cv.items()}

lignes = []
for nom, p in probas_cv.items():
    lignes.append({"modèle": nom,
                   "part des probabilités au-dessus de 0,5": (p >= 0.5).mean(),
                   "rappel au seuil 0,5": recall_score(y_train, p >= 0.5),
                   "F1 au seuil 0,5": f1_score(y_train, p >= 0.5),
                   "seuil retenu": seuils[nom],
                   "rappel au seuil retenu": recall_score(y_train, p >= seuils[nom]),
                   "F1 au seuil retenu": f1_score(y_train, p >= seuils[nom])})
tableau_seuils = pd.DataFrame(lignes).set_index("modèle")
tableau_seuils.round(3)

# %% [markdown]
# Le SVM ne donne une probabilité supérieure à 0,5 qu'à 12 % des vins. Avec ce seuil il ne retrouve que 36 % des bons vins. Avec un seuil de 0,26 il en retrouve 73 % et son F1-score passe de 0,444 à 0,566. Le même constat vaut pour les cinq modèles, qui ont tous un seuil retenu entre 0,22 et 0,34.
#
# Une remarque sur le SVM. Son F1-score au seuil 0,5 vaut ici 0,444 alors que le tableau de la section 2.3 donnait 0,551. Ce n'est pas une erreur. En partie 2 la recherche par grille utilisait la réponse directe du SVM, c'est à dire le côté de la frontière où tombe le vin. Ici nous utilisons sa probabilité comparée à 0,5, et ces deux façons de décider ne donnent pas la même réponse pour tous les vins. La différence disparaît une fois le seuil réglé.

# %% [markdown]
# Le graphique suivant montre d'où vient ce problème. Il compare les probabilités données par le SVM aux vrais bons vins et aux vrais mauvais vins, toujours calculées en validation croisée sur les vins d'entraînement.

# %%
# Probabilités données par le SVM aux vrais bons vins et aux vrais mauvais vins (validation croisée)
p_bons = probas_cv[choisi][y_train.values == 1]
p_mauvais = probas_cv[choisi][y_train.values == 0]
intervalles = np.linspace(0, 0.85, 35)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(p_mauvais, bins=intervalles, color=BRUME, label="vins réellement mauvais")
ax.hist(p_bons, bins=intervalles, color=PRUNE, alpha=0.85, label="vins réellement bons")
ax.axvline(0.5, color="gray", linestyle=":"); ax.text(0.51, 720, "seuil habituel 0,5", color="gray")
ax.axvline(seuils[choisi], color=PRUNE, linestyle="--"); ax.text(seuils[choisi] + 0.01, 600, "seuil retenu", color=PRUNE)
ax.set_xlabel("probabilité d'être bon donnée par le SVM"); ax.set_ylabel("nombre de vins")
ax.set_title("Probabilités données aux vrais bons et aux vrais mauvais vins")
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.6))
plt.show()

print(f"Probabilité médiane des mauvais vins {np.median(p_mauvais):.2f}, des bons vins {np.median(p_bons):.2f}")
print(f"Bons vins au-dessus de 0,5 {(p_bons >= 0.5).mean():.0%}, au-dessus du seuil retenu {(p_bons >= seuils[choisi]).mean():.0%}, sous 0,1 {(p_bons < 0.1).mean():.0%}")
print(f"Au seuil retenu, {(p_bons >= seuils[choisi]).sum()} bons vins retrouvés et {(p_mauvais >= seuils[choisi]).sum()} mauvais vins annoncés bons ({(p_mauvais >= seuils[choisi]).mean():.0%} des mauvais)")

# %% [markdown]
# Les mauvais vins sont presque tous regroupés près de 0, avec une probabilité médiane de 0,06. Les bons vins reçoivent des probabilités nettement plus hautes, avec une médiane de 0,43, mais elles sont très étalées et rarement au dessus de 0,5. Seuls 36 % des bons vins dépassent ce seuil, ce qui explique le rappel faible du seuil habituel.
#
# Les bons vins ne se ressemblent pas tous. Certains, au dessus de 0,5, ont un profil chimique typique d'un bon vin. Beaucoup se trouvent entre 0,2 et 0,5, dans la zone où ils se mélangent aux mauvais vins. Environ 10 % ont même une probabilité inférieure à 0,1, car leur chimie ressemble à celle d'un vin moyen. Les mauvais vins se ressemblent et sont faciles à reconnaître, alors que les bons vins sont plus variés.
#
# Le seuil retenu garde 73 % des bons vins mais laisse aussi passer 20 % des mauvais. Comme les mauvais vins sont quatre fois plus nombreux, cela fait presque autant de fausses alertes que de bons vins retrouvés, 691 contre 591. C'est pourquoi la précision reste moyenne. La courbe suivante montre comment ce compromis évolue quand on déplace le seuil.

# %%
p = probas_cv[choisi]
precisions, rappels, seuils_possibles = precision_recall_curve(y_train, p)
f1 = 2 * precisions * rappels / (precisions + rappels + 1e-12)

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(seuils_possibles, precisions[:-1], color=MIEL); ax.text(0.80, 0.93, "précision", color=MIEL)
ax.plot(seuils_possibles, rappels[:-1], color=ENCRE); ax.text(0.80, 0.10, "rappel", color=ENCRE)
ax.plot(seuils_possibles, f1[:-1], color=PRUNE, linewidth=2.5); ax.text(0.80, 0.30, "F1-score", color=PRUNE)
ax.axvline(0.5, color="gray", linestyle=":"); ax.text(0.51, 1.02, "seuil habituel", color="gray")
ax.axvline(seuils[choisi], color=PRUNE, linestyle="--"); ax.text(seuils[choisi] - 0.01, 1.02, "seuil retenu", color=PRUNE, ha="right")
ax.set_xlabel("seuil de probabilité à partir duquel on dit bon"); ax.set_xlim(0, 1)
ax.set_title(f"Précision, rappel et F1-score du {choisi} selon le seuil", pad=18)
plt.show()

# %%
# Ce que donnent quelques seuils, avec les probabilités de validation croisée sur les vins d'entraînement
# Les seuils 0,10 et 0,72 sont ceux des usages présélection et prestige, calculés de la même façon qu'en 3.6
precisions, rappels, seuils_possibles = precision_recall_curve(y_train, probas_cv[choisi])
seuil_large = seuils_possibles[rappels[:-1] >= 0.90].max()
seuil_strict = seuils_possibles[precisions[:-1] >= 0.80].min()

lignes = []
for seuil_essaye in [seuil_large, seuils[choisi], 0.50, seuil_strict]:
    decision = probas_cv[choisi] >= seuil_essaye
    lignes.append({"seuil": round(float(seuil_essaye), 2), "vins annoncés bons": int(decision.sum()),
                   "précision": precision_score(y_train, decision), "rappel": recall_score(y_train, decision),
                   "F1-score": f1_score(y_train, decision)})
pd.DataFrame(lignes).set_index("seuil").round(2)

# %% [markdown]
# Quand le seuil baisse, le rappel monte et la précision descend. On retrouve plus de bons vins mais on se trompe plus souvent quand on annonce un bon vin. Le tableau donne les valeurs pour quatre seuils.
#
# Au seuil habituel de 0,5, le modèle n'annonce bons que 498 vins. Sa précision est correcte (58 %) mais il ne retrouve que 36 % des bons vins. Au seuil retenu de 0,26, il annonce 1 282 vins, retrouve 73 % des bons vins et garde une précision de 46 %. C'est le meilleur F1-score (0,57), donc le meilleur équilibre entre les deux erreurs. Au seuil de 0,10, il retrouve 90 % des bons vins, mais seulement 35 % des vins annoncés bons le sont vraiment. Au seuil de 0,72, 81 % des vins annoncés bons le sont vraiment, mais il n'en annonce que 21 et ne retrouve que 2 % des bons vins.
#
# Aucun seuil n'est donc bon dans l'absolu, chacun correspond à un besoin. Le seuil de 0,26 convient quand les deux erreurs sont aussi graves, celui de 0,10 sert à ne rater presque aucun bon vin et celui de 0,72 à ne garder que des vins très sûrs.
#
# Sur la courbe, le F1-score est assez plat entre 0,2 et 0,4, le choix exact du seuil compte donc peu dans cette zone. À droite de 0,7 la précision devient très irrégulière car très peu de vins ont une probabilité aussi haute.
#
# ## 3.3 Que vaut-il sur des vins qu'il n'a jamais vus ?
#
# Tous les choix sont faits, nous ouvrons donc le jeu de test une seule fois. Chaque modèle est évalué avec le seuil fixé en 3.2. Nous ajoutons un modèle qui répond toujours mauvais pour avoir un point de comparaison.
#
# Comme le test ne contient qu'un millier de vins, l'AUC mesurée dépend un peu des vins qui s'y trouvent. Pour savoir à quel point, nous utilisons le bootstrap. On tire au hasard 1 064 vins parmi ceux du test en acceptant de tirer plusieurs fois le même, puis on recalcule l'AUC. En répétant ce tirage 1 000 fois, on obtient un intervalle qui contient l'AUC 95 fois sur 100.

# %%
def mesures(y_vrai, probas, seuil):
    prediction = probas >= seuil
    return {"accuracy": accuracy_score(y_vrai, prediction),
            "précision": precision_score(y_vrai, prediction, zero_division=0),
            "rappel": recall_score(y_vrai, prediction),
            "F1-score": f1_score(y_vrai, prediction),
            "AUC": roc_auc_score(y_vrai, probas)}

def intervalle_auc(y_vrai, probas, tirages=1000):
    """Intervalle à 95 % de l'AUC par bootstrap. On retire au hasard autant de vins du test, avec remise, 1 000 fois."""
    hasard = np.random.default_rng(GRAINE)
    y_vrai = np.asarray(y_vrai)
    aucs = []
    for _ in range(tirages):
        tirage = hasard.integers(0, len(y_vrai), len(y_vrai))
        aucs.append(roc_auc_score(y_vrai[tirage], probas[tirage]))
    bas, haut = np.percentile(aucs, [2.5, 97.5])
    return f"{bas:.3f} à {haut:.3f}"

probas_test = {nom: modele.predict_proba(X_test)[:, 1] for nom, modele in optimises.items()}

lignes = [{"modèle": nom, "seuil": seuils[nom], **mesures(y_test, p, seuils[nom]),
           "intervalle de l'AUC à 95 %": intervalle_auc(y_test, p)} for nom, p in probas_test.items()]
naif = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
lignes.append({"modèle": "Toujours répondre mauvais", "seuil": np.nan, **mesures(y_test, naif.predict_proba(X_test)[:, 1], 0.5),
               "intervalle de l'AUC à 95 %": ""})
resultats = pd.DataFrame(lignes).set_index("modèle").sort_values("AUC", ascending=False)
resultats.round(3)

# %%
# Comparaison des cinq modèles sur le jeu de test, mesure par mesure
mesures_a_comparer = ["accuracy", "précision", "rappel", "F1-score", "AUC"]
noms_modeles = [nom for nom in resultats.index if nom in optimises]
couleurs = {"SVM": PRUNE, "k plus proches voisins": ENCRE, "Régression logistique": MIEL,
            "Arbre de décision": BRUME, "Naive Bayes": "#8FA6C1"}
largeur = 0.16
fig, ax = plt.subplots(figsize=(9, 4))
for k, nom in enumerate(noms_modeles):
    positions = np.arange(len(mesures_a_comparer)) + (k - 2) * largeur
    ax.bar(positions, resultats.loc[nom, mesures_a_comparer], width=largeur, color=couleurs[nom], label=nom)
ax.set_xticks(range(len(mesures_a_comparer))); ax.set_xticklabels(mesures_a_comparer)
ax.set_ylim(0, 1); ax.set_ylabel("score sur le jeu de test")
ax.set_title("Les cinq modèles sur le jeu de test, avec le seuil fixé en 3.2")
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# %% [markdown]
# La dernière ligne confirme ce qui était annoncé en 1.3. Répondre toujours mauvais donne 81 % d'accuracy, plus que tous nos modèles, sans trouver un seul bon vin. C'est pour cela que nous ne jugeons pas les modèles sur l'accuracy.
#
# Sur 100 bons vins, le SVM en retrouve 79. Sur 100 vins qu'il annonce bons, 46 le sont vraiment. En tirant des vins au hasard, on n'en aurait que 19 de bons sur 100. Le modèle fait donc 2,4 fois mieux que le hasard.
#
# L'AUC du SVM se situe entre 0,823 et 0,875. Les intervalles des trois premiers modèles se recouvrent presque entièrement, on ne peut donc pas dire sur le test que l'un est meilleur que les autres. Cela confirme la section 3.1 où le SVM et les voisins étaient déjà très proches. L'arbre et Naive Bayes restent en retrait comme en validation. Enfin l'AUC du SVM sur le test (0,850) est très proche de celle obtenue en validation croisée (0,843). Le score annoncé avant d'ouvrir le test était donc fiable.
#
# Ces mesures complètent l'AUC. L'AUC dit si le modèle range bien les vins, alors que la précision, le rappel et le F1-score jugent ses décisions une fois le seuil fixé, c'est à dire ce qu'obtiendra réellement celui qui utilise le modèle. Le rappel dit combien de bons vins sont retrouvés et la précision si l'on peut faire confiance à un vin annoncé bon. Avec si peu de bons vins, ce sont ces deux mesures qui comptent le plus.
#
# Le graphique reprend ces mesures pour les cinq modèles. Le SVM, les voisins et la régression logistique sont presque à égalité sur chaque mesure. Ce qui les distingue surtout est le partage entre précision et rappel. Les voisins ont la meilleure précision (0,48) mais le rappel le plus faible des trois (0,70), alors que le SVM a le meilleur rappel (0,79). L'arbre et Naive Bayes sont en dessous en F1-score et en AUC. L'accuracy change peu d'un modèle à l'autre et reste sous les 81 % du modèle qui répond toujours mauvais, elle ne sert donc pas à départager les modèles.

# %%
fig, ax = plt.subplots(figsize=(7, 5))
for nom, p in probas_test.items():
    faux_positifs, vrais_positifs, _ = roc_curve(y_test, p)
    retenu = nom == choisi
    ax.plot(faux_positifs, vrais_positifs, color=PRUNE if retenu else BRUME, linewidth=2.6 if retenu else 1.4,
            zorder=3 if retenu else 2, label=f"{nom} ({roc_auc_score(y_test, p):.3f})")
ax.plot([0, 1], [0, 1], color=ENCRE, linestyle=":")
ax.set_xlabel("part des mauvais vins annoncés bons"); ax.set_ylabel("rappel")
ax.set_title("Courbes ROC sur le jeu de test (AUC entre parenthèses)"); ax.legend(loc="lower right")
plt.show()

# %% [markdown]
# Une courbe ROC montre ce que l'on gagne et ce que l'on paie quand on baisse le seuil. En abscisse on lit la part des mauvais vins annoncés bons à tort, et en ordonnée la part des bons vins retrouvés. La diagonale en pointillés correspond à un modèle qui répondrait au hasard, et un modèle parfait passerait par le coin en haut à gauche.
#
# Toutes les courbes sont nettement au dessus de la diagonale. Pour le SVM, si l'on accepte d'annoncer bons 10 % des mauvais vins, on retrouve 47 % des bons vins. Si l'on accepte 20 %, on en retrouve 76 %. Au delà, la courbe s'aplatit et chaque bon vin supplémentaire coûte beaucoup de fausses alertes.
#
# Les courbes du SVM, des voisins et de la régression logistique se croisent plusieurs fois. Aucune n'est au dessus des autres partout, ce qui confirme qu'ils sont équivalents sur le test. Celles de l'arbre et de Naive Bayes sont un peu en dessous sur la plus grande partie du graphique.

# %%
fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
for ax, nom in zip(axes, optimises):
    matrice = confusion_matrix(y_test, probas_test[nom] >= seuils[nom])
    ax.imshow(matrice, cmap="Purples"); ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrice[i, j], ha="center", va="center", fontsize=12,
                    color="white" if matrice[i, j] > matrice.max() / 2 else ENCRE)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["prédit mauvais", "prédit bon"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["réel mauvais", "réel bon"])
    ax.set_title(nom)
plt.tight_layout(); plt.show()

# %% [markdown]
# Chaque matrice se lit ainsi. En haut à gauche on trouve les mauvais vins bien reconnus, en bas à droite les bons vins retrouvés, en haut à droite les fausses alertes et en bas à gauche les bons vins ratés.
#
# Les cinq matrices ont la même forme. Tous les modèles font plus de fausses alertes que de bons vins ratés, par exemple 186 contre 43 pour le SVM. C'est la conséquence des seuils bas choisis en 3.2. Nous acceptons d'annoncer trop de bons vins pour en rater moins.
#
# La comparaison entre le SVM et les voisins illustre bien le compromis entre précision et rappel. Les voisins font moins de fausses alertes (156 contre 186) mais ratent plus de bons vins (60 contre 43). Pour une AUC égale, le SVM retrouve donc plus de bons vins, tandis que les voisins se trompent un peu moins souvent quand ils annoncent un bon vin.
#
# ## 3.4 Est-il aussi bon sur les rouges que sur les blancs ?
#
# Comme la séparation de la section 1.4 respectait la part de rouges, nous pouvons évaluer le modèle séparément sur chaque couleur.

# %%
lignes = []
for couleur in ["rouge", "blanc"]:
    masque = (test["type"] == couleur).values
    lignes.append({"couleur": couleur, "vins": masque.sum(), "bons vins": y_test[masque].sum(),
                   **mesures(y_test[masque], probas_test[choisi][masque], seuils[choisi])})
par_couleur = pd.DataFrame(lignes).set_index("couleur")
par_couleur.round(3)

# %% [markdown]
# L'AUC est de 0,882 sur les rouges et de 0,836 sur les blancs, le modèle semble donc un peu meilleur sur les rouges. Il faut rester prudent car le test ne contient que 37 bons vins rouges. Il s'agit d'une tendance et pas d'une certitude.
#
# ## 3.5 Où se trompe-t-il, et pourquoi ?
#
# Le modèle n'a jamais vu la note sur 10, seulement la cible bon ou pas bon. Nous regardons pourtant ses réponses note par note pour voir où se trouvent ses erreurs.

# %%
analyse = pd.DataFrame({"note": test["quality"].values, "probabilité": probas_test[choisi],
                        "prédit bon": probas_test[choisi] >= seuils[choisi]})
par_note = analyse.groupby("note").agg(vins=("note", "size"), probabilite_mediane=("probabilité", "median"),
                                       part_predite_bonne=("prédit bon", "mean"))
par_note["part d'erreurs"] = np.where(par_note.index >= 7, 1 - par_note["part_predite_bonne"], par_note["part_predite_bonne"])

# Pour chaque note réelle, part des vins annoncés bons (violet) et annoncés mauvais (gris)
notes_vues = list(par_note.index)
annonces_bons = par_note["part_predite_bonne"] * 100
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(notes_vues, annonces_bons, width=0.65, color=PRUNE, label="annoncés bons")
ax.bar(notes_vues, 100 - annonces_bons, bottom=annonces_bons, width=0.65, color=BRUME, label="annoncés mauvais")
for n in notes_vues:
    ax.text(n, 102, f"{annonces_bons[n]:.0f} %", ha="center", color=PRUNE)
ax.axvline(6.5, color=ENCRE, linestyle="--")
ax.text(6.45, 112, "vins réellement mauvais", ha="right", color=ENCRE)
ax.text(6.55, 112, "vins réellement bons", ha="left", color=ENCRE)
ax.set_ylim(0, 120); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_xticks(notes_vues); ax.set_xticklabels([f"{n}\n{c} vins" for n, c in zip(notes_vues, par_note["vins"])])
ax.set_xlabel("note réelle donnée par les dégustateurs, jamais montrée au modèle")
ax.set_ylabel("part des vins (%)")
ax.set_title("Ce que le SVM annonce selon la note réelle du vin", pad=12)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

faux_positifs = analyse[(analyse["note"] < 7) & analyse["prédit bon"]]
faux_negatifs = analyse[(analyse["note"] >= 7) & ~analyse["prédit bon"]]
print(f"{len(faux_positifs)} faux positifs, dont {(faux_positifs['note'] == 6).mean():.0%} sont des vins notés 6")
print(f"{len(faux_negatifs)} faux négatifs, dont {(faux_negatifs['note'] == 7).mean():.0%} sont des vins notés 7")
par_note.round(3)

# %% [markdown]
# Chaque barre regroupe les vins d'une même note réelle. La partie violette est la part des vins que le modèle annonce bons et la partie grise celle qu'il annonce mauvais. À gauche de la ligne pointillée se trouvent les vins réellement mauvais, pour eux le violet est une erreur. À droite se trouvent les vins réellement bons, pour eux c'est le gris qui est une erreur. Les notes 3 et 9 ne comptent que 4 et 2 vins, on ne peut donc rien en conclure.
#
# Le modèle ne connaît pas la note, pourtant il annonce bons de plus en plus de vins quand elle monte. C'est le cas de 3 % des vins notés 4, 8 % des 5, 34 % des 6, 76 % des 7 et 90 % des 8. Le tableau sous le graphique montre que la probabilité médiane suit la même progression, avec 0,04 pour un 5, 0,12 pour un 6, 0,43 pour un 7 et 0,47 pour un 8. Le modèle a donc bien appris ce qui fait monter la qualité.
#
# Les erreurs se concentrent sur les deux notes qui entourent la frontière. Un tiers des vins notés 6 est annoncé bon à tort, et près d'un quart des vins notés 7 est raté. C'est pourquoi 84 % des fausses alertes sont des vins notés 6 et 93 % des bons vins ratés sont des vins notés 7. Les vins notés 4, 5 ou 8, plus éloignés de la frontière, sont presque toujours bien classés.
#
# Pour comprendre pourquoi cette frontière est difficile, nous cherchons des vins qui se ressemblent beaucoup et qui n'ont pas la même étiquette.

# %%
# Pour chaque vin d'entraînement, on cherche le vin le plus ressemblant (distance calculée sur les mesures standardisées)
X_std = StandardScaler().fit_transform(X_train)
distances, positions = NearestNeighbors(n_neighbors=2).fit(X_std).kneighbors(X_std)
paires = pd.DataFrame({"vin": range(len(train)), "voisin": positions[:, 1], "distance": distances[:, 1]})
paires["étiquettes différentes"] = train["bon"].values[paires["vin"]] != train["bon"].values[paires["voisin"]]

# Les deux vins les plus ressemblants qui n'ont pas la même étiquette
plus_proche = paires[paires["étiquettes différentes"]].sort_values("distance").iloc[0]
jumeaux = train.iloc[[int(plus_proche["vin"]), int(plus_proche["voisin"])]][["type"] + variables[:-1] + ["quality"]]
jumeaux.index = ["vin A", "vin B"]

# Point de comparaison, deux vins pris au hasard ont des étiquettes différentes dans 2 x p x (1 - p) des cas
part_bons = y_train.mean()
presque_identiques = paires[paires["distance"] < 0.5]
print(f"Deux vins pris au hasard ont des étiquettes différentes dans {2 * part_bons * (1 - part_bons):.0%} des cas.")
print(f"Un vin et son voisin le plus ressemblant ont des étiquettes différentes dans {paires['étiquettes différentes'].mean():.0%} des cas.")
print(f"{len(presque_identiques)} vins ont un voisin presque identique (distance inférieure à 0,5), "
      f"et l'étiquette diffère pour {presque_identiques['étiquettes différentes'].sum()} d'entre eux, soit {presque_identiques['étiquettes différentes'].mean():.0%}.")
jumeaux.T

# %% [markdown]
# Les deux vins du tableau ont des mesures presque identiques, pourtant l'un a reçu 6 et l'autre 7. Aucun modèle ne peut les séparer à partir de la chimie.
#
# Pour savoir si ce cas est fréquent, il faut un point de comparaison. Deux vins pris au hasard ont des étiquettes différentes dans 31 % des cas. Un vin et son voisin le plus ressemblant n'en ont des différentes que dans 19 % des cas, et ce chiffre tombe à 7 % pour les vins qui ont un voisin presque identique. Plus deux vins se ressemblent, plus ils ont la même étiquette. La composition explique donc bien une grande partie de la qualité, ce qui confirme l'idée de ressemblance de la section 3.1. Mais elle ne l'explique pas entièrement.
#
# La note est donnée par des dégustateurs, et entre un 6 et un 7 l'avis d'une seule personne peut suffire à changer l'étiquette. Une partie des erreurs vient donc de la note elle-même, et aucun modèle ne pourra la corriger.
#
# ## 3.6 Quel seuil pour quel usage ?
#
# Le bon seuil dépend de l'erreur qui coûte le plus cher à celui qui utilise le modèle. Monter le seuil limite les faux positifs, c'est à dire les mauvais vins annoncés bons, mais fait rater plus de bons vins. Baisser le seuil limite au contraire les faux négatifs, les bons vins ratés, mais donne plus de fausses alertes. Par exemple, un caviste qui prépare une dégustation baissera le seuil pour ne laisser passer aucun bon vin, quitte à goûter quelques vins moyens en plus. À l'inverse, un concours qui décerne un label montera le seuil pour ne jamais récompenser un vin moyen, quitte à oublier quelques bons vins. Le seuil de 3.2 convient si rater un bon vin et annoncer à tort un bon vin sont aussi graves l'un que l'autre. Pour deux autres usages, nous partons d'un objectif et nous calculons le seuil qui l'atteint sur les vins d'entraînement. Nous vérifions ensuite sur le test.

# %%
# Les seuils des deux autres usages sont calculés sur les vins d'entraînement, à partir d'un objectif
precisions, rappels, seuils_possibles = precision_recall_curve(y_train, probas_cv[choisi])
seuil_large = seuils_possibles[rappels[:-1] >= 0.90].max()        # le seuil le plus haut qui retrouve encore 90 % des bons vins
seuil_strict = seuils_possibles[precisions[:-1] >= 0.80].min()    # le seuil le plus bas qui atteint 80 % de précision

usages = {"Présélection (retrouver 90 % des bons vins)": seuil_large,
          "Équilibre (meilleur F1-score)": seuils[choisi],
          "Prestige (80 % de précision)": seuil_strict}
lignes = []
for usage, seuil in usages.items():
    sur_train, sur_test = probas_cv[choisi] >= seuil, probas_test[choisi] >= seuil
    lignes.append({"usage": usage, "seuil": round(seuil, 2),
                   "précision entraînement": precision_score(y_train, sur_train),
                   "rappel entraînement": recall_score(y_train, sur_train),
                   "vins retenus sur le test": sur_test.sum(),
                   "précision test": precision_score(y_test, sur_test, zero_division=0),
                   "rappel test": recall_score(y_test, sur_test),
                   "F1 test": f1_score(y_test, sur_test)})
tableau_usages = pd.DataFrame(lignes).set_index("usage")
large, strict = tableau_usages.iloc[0], tableau_usages.iloc[-1]
print(f"Sur les vins d'entraînement, seuls {(probas_cv[choisi] >= seuil_strict).sum()} vins dépassent le seuil prestige de {seuil_strict:.2f}.")
tableau_usages.round(2)

# %% [markdown]
# Pour une présélection avant dégustation, l'erreur grave est de rater un bon vin. L'objectif est de retrouver au moins 90 % des bons vins, ce qui donne un seuil de 0,10. Sur le test, le modèle retrouve 91 % des bons vins en ne retenant que 523 vins sur 1 064. Les dégustateurs n'ont donc plus à goûter qu'environ la moitié des vins tout en gardant presque tous les bons.
#
# Pour un label prestige, l'erreur grave est au contraire d'annoncer bon un vin qui ne l'est pas. L'objectif est une précision d'au moins 80 %, ce qui donne un seuil de 0,72. Sur le test, seuls 6 vins dépassent ce seuil et 4 sont bons, soit 67 % de précision. L'objectif n'est pas tenu, et c'était prévisible. Même sur les vins d'entraînement, seuls 21 vins dépassaient 0,72, le seuil a donc été fixé sur très peu de cas. Le modèle donne d'ailleurs très rarement des probabilités aussi hautes, seuls 5 % des vins du test dépassent 0,6 et aucun n'atteint 0,8. Notre modèle est donc utile pour faire un premier tri, mais pas pour garantir qu'un vin est excellent.
#
# ## 3.7 Quelles sont les limites ?
#
# La principale limite est un plafond d'AUC autour de 0,85. Aucun des cinq modèles ne le dépasse nettement, même après réglage. La section 3.5 en donne une raison, puisqu'une partie des erreurs vient de la note. Nous regardons si plus de vins permettraient d'aller plus loin.

# %%
tailles, notes_train, notes_val = learning_curve(a_comparer[choisi], X_train, y_train, cv=cv, scoring="roc_auc",
                                                 train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(tailles, notes_train.mean(axis=1), marker="o", color=BRUME, label="vins d'entraînement")
ax.plot(tailles, notes_val.mean(axis=1), marker="o", color=PRUNE, label="vins de validation")
ax.set_ylim(0.75, 1.0); ax.set_xlabel("nombre de vins utilisés pour apprendre"); ax.set_ylabel("AUC")
ax.set_title(f"Courbe d'apprentissage du {choisi}"); ax.legend()
plt.show()
gain_fin = notes_val.mean(axis=1)[-1] - notes_val.mean(axis=1)[-3]
print(f"Gain d'AUC de validation entre {tailles[-3]} et {tailles[-1]} vins {gain_fin:+.3f}")

# %% [markdown]
# La courbe d'apprentissage montre le score du modèle selon le nombre de vins utilisés pour apprendre. Le score de validation monte vite au début puis ne progresse presque plus. Entre 2 178 et 3 404 vins, il ne gagne que 0,002 d'AUC. Collecter davantage de vins du même type n'apporterait donc presque rien à ce modèle.
#
# Les deux courbes se rapprochent mais restent séparées d'environ 0,04, avec 0,885 en entraînement contre 0,844 en validation. Le SVM colle donc encore un peu aux vins d'entraînement. Cet écart reste modéré et c'est le prix de la meilleure validation obtenue en 2.6.
#
# D'autres limites viennent des données elles-mêmes. Tous les vins sont des vinhos verdes portugais, rien ne garantit donc que le modèle fonctionne sur des vins d'une autre région. Les 11 mesures ne décrivent pas non plus tout ce qui fait un vin. Le fichier ne contient ni le cépage, ni la marque, ni le prix, alors que ces informations jouent sûrement sur la note. Le fichier contient aussi peu de bons vins, seulement 19 %, et presque aucun vin noté 3, 8 ou 9. Le modèle dispose donc de peu d'exemples des vins les plus intéressants. La courbe d'apprentissage montre que plus de vins n'aiderait presque pas, mais elle garde toujours la même part de bons vins. Ajouter surtout des bons vins pourrait aider un peu, même si l'option `class_weight="balanced"` compense déjà en partie ce déséquilibre.
#
# Plusieurs pistes permettraient d'aller plus loin. On pourrait prédire la note elle-même puis appliquer la limite de 7, pour que le modèle sache qu'un 6 est plus proche d'un 7 que d'un 3. On pourrait faire noter chaque vin par plus de dégustateurs, ce qui rendrait les étiquettes plus sûres. Enfin il faudrait ajouter des informations qui manquent aujourd'hui, comme le cépage.
#
# ## 3.8 Que retenir de tout le projet ?
#
# À partir de 11 mesures de laboratoire et de la couleur, notre modèle repère les bons vins 2,4 fois mieux qu'un tirage au hasard. Il peut aider à trier les vins avant une dégustation, mais il ne peut pas remplacer l'avis des dégustateurs.
#
# | Modèle | AUC test | F1 test | Sa force | Sa faiblesse | La raison liée au vin |
# |---|---|---|---|---|---|
# | SVM | 0,850 | 0,581 | Meilleure AUC sur 25 découpages | Difficile à expliquer mesure par mesure | Sa frontière courbe tient compte de l'équilibre entre les mesures |
# | k plus proches voisins | 0,850 | 0,568 | Presque aussi bon, et il peut montrer les vins ressemblants | Inutilisable sans mise à l'échelle | Deux vins de composition proche ont un goût et une note proches |
# | Régression logistique | 0,846 | 0,572 | Stable et lisible | Additionne les mesures sans les combiner | Elle saisit l'alcool mais pas l'équilibre du vin |
# | Arbre de décision | 0,811 | 0,519 | Des règles que l'on peut lire | Retient les vins un par un s'il n'est pas limité | Il découpe en tranches une qualité qui évolue progressivement |
# | Naive Bayes | 0,810 | 0,527 | Très rapide et stable | Se dégrade quand les mesures se répètent | Les mesures d'un vin sont liées par la chimie |
#
# | Choix | Raison | Chiffre |
# |---|---|---|
# | Supprimer les doublons | Ils laissaient le modèle reconnaître des vins déjà vus | 18 % des lignes, le F1-score des voisins passait de 0,47 à 0,67 avec eux |
# | Séparer en respectant la cible et la couleur | Les bons vins rouges sont rares | 184 dans tout le fichier |
# | Garder les valeurs extrêmes | Elles signalent les vins ratés | 12 % de bons vins parmi eux, contre 21 % |
# | Standardiser dans un pipeline | Les voisins et le SVM calculent des distances | Les voisins passent de 0,705 à 0,827 |
# | Garder les 12 mesures, sauf pour Naive Bayes | Lui seul se dégrade quand les mesures se répètent | Il passe de 0,798 à 0,771 avec les 12 |
# | Régler avec l'AUC | Le F1-score par défaut favorise les modèles trop collés aux données | Il aurait choisi 3 voisins |
# | Retenir le SVM gaussien | Meilleure AUC moyenne sur 25 découpages | 0,843, devant les voisins sur 17 découpages et le noyau linéaire à 0,824 |
# | Baisser le seuil | Les bons vins sont rares, le modèle dépasse rarement 0,5 | 0,26 au lieu de 0,5, rappel de 79 % sur le test |
#
# Le modèle final est enregistré pour l'application web, avec les trois seuils calculés dans cette partie.

# %%
os.makedirs("app", exist_ok=True)
joblib.dump({"modele": optimises[choisi], "nom": choisi, "seuil": float(seuils[choisi]), "variables": variables,
             "seuils_usages": {"preselection": float(seuil_large), "prestige": float(seuil_strict)},
             "vins_connus": train[variables + ["quality"]]}, "app/modele_final.joblib")
print("Modèle final enregistré,", choisi, "avec les seuils", round(float(seuil_large), 2), round(float(seuils[choisi]), 2), round(float(seuil_strict), 2))
