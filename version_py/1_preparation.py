# Version script du notebook 1_preparation.ipynb, à lancer depuis la racine du dépôt.
# Les blocs « # %% » se lisent comme des cellules dans VS Code ou Spyder.

# %% [markdown]
# # Partie 1. Préparer des données fiables
#
# Projet 2 de Machine Learning 1, EFREI 2026-2027.
#
# Nous voulons prédire si un vin est bon à partir de 11 mesures de laboratoire et de sa couleur. Cette première partie nettoie les données, les explore et vérifie chaque choix de préparation par une expérience. Elle enregistre à la fin un jeu d'entraînement et un jeu de test que les parties 2 et 3 réutilisent.

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

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

# %% [markdown]
# ## 1.1 Que contiennent les données ?
#
# Les données décrivent 6 497 vins portugais par 11 mesures de laboratoire, une couleur et une note de dégustation sur 10. Elles arrivent en deux fichiers, un pour les rouges et un pour les blancs. Nous les réunissons en gardant la couleur dans une colonne `type`.

# %%
rouge = pd.read_csv("data/winequality-red.csv", sep=";")
blanc = pd.read_csv("data/winequality-white.csv", sep=";")
rouge["type"] = "rouge"
blanc["type"] = "blanc"

vins_brut = pd.concat([rouge, blanc], ignore_index=True)
print("Vins rouges", len(rouge), "| vins blancs", len(blanc), "| total", len(vins_brut))
vins_brut.describe().T[["mean", "std", "min", "50%", "max"]]

# %% [markdown]
# Les mesures n'ont pas du tout la même échelle. Le dioxyde de soufre total monte jusqu'à 440 alors que la densité varie de quelques centièmes. Certains maximums sont aussi très loin de la médiane, par exemple 65,8 grammes de sucre pour une médiane de 3. Ces deux remarques sont reprises en 1.9 et en 1.10.
#
# ## 1.2 Manque-t-il des valeurs ?
#
# Non, le tableau ne contient aucune valeur manquante, il n'y a donc rien à compléter. En revanche, beaucoup de lignes sont présentes en plusieurs exemplaires.

# %%
print("Valeurs manquantes dans tout le tableau", vins_brut.isna().sum().sum())
print("Lignes identiques à une autre ligne", vins_brut.duplicated().sum(),
      f"soit {vins_brut.duplicated().mean():.1%} du tableau")

# %% [markdown]
# 1177 lignes sont la copie exacte d'une autre ligne. Nous les retirons dès maintenant pour ne garder qu'un exemplaire de chaque vin. La section 1.8 explique ce choix et le prouve par une expérience.
#
# ## 1.3 Qu'est-ce qu'un bon vin, et pourquoi l'accuracy ne suffit-elle pas ?
#
# Un bon vin est un vin noté au moins 7 sur 10, et comme ces vins sont rares, compter simplement les bonnes réponses ne suffit pas à juger un modèle. Nous créons la cible `bon`, qui vaut 1 pour ces vins et 0 pour les autres. Nous transformons aussi la couleur en une colonne `est_rouge` qui vaut 1 ou 0. Une seule colonne suffit puisqu'il n'y a que deux couleurs.

# %%
vins = vins_brut.drop_duplicates().reset_index(drop=True)
vins["est_rouge"] = (vins["type"] == "rouge").astype(int)
vins["bon"] = (vins["quality"] >= 7).astype(int)

numeriques = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
              "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
variables = numeriques + ["est_rouge"]

print("Vins distincts", len(vins), "| bons vins", vins["bon"].sum(), f"({vins['bon'].mean():.1%})")
vins.groupby("type")["bon"].agg(nombre="size", part_de_bons="mean").round(3)

# %%
notes = vins["quality"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar(notes.index, notes.values, color=[PRUNE if n >= 7 else BRUME for n in notes.index], width=0.7)
for note, nombre in notes.items():
    ax.text(note, nombre + 25, str(nombre), ha="center")
ax.axvline(6.5, color=ENCRE, linestyle="--")
ax.text(6.55, notes.max() * 0.9, "frontière du sujet\nbon à partir de 7", color=ENCRE)
ax.set_xlabel("note donnée par les dégustateurs"); ax.set_ylabel("nombre de vins")
ax.set_title("Nombre de vins par note (bons vins en violet)")
plt.show()

# %% [markdown]
# Il reste 5320 vins distincts et 19% d'entre eux sont bons. Il y a donc environ un bon vin pour quatre mauvais. La frontière passe entre les notes 6 et 7, qui sont les deux notes les plus fréquentes.
#
# L'accuracy est la part de bonnes réponses sur tous les vins. Ici elle est trompeuse. Un modèle qui répondrait toujours mauvais aurait 81% d'accuracy sans trouver un seul bon vin. Nous utilisons donc quatre autres mesures dans tout le projet.
#
# | Mesure | Ce qu'elle dit |
# |---|---|
# | Rappel | Sur 100 bons vins, combien le modèle en retrouve |
# | Précision | Sur 100 vins que le modèle annonce bons, combien le sont vraiment |
# | F1-score | Une seule note qui n'est élevée que si le rappel et la précision le sont tous les deux |
# | AUC | La chance que le modèle donne une meilleure note à un bon vin qu'à un mauvais, quand on en tire un de chaque au hasard. Elle vaut 0,5 pour un modèle qui répond au hasard et 1 pour un classement parfait |
#
# ## 1.4 Quels vins garde-t-on pour le test final ?
#
# Nous mettons de côté 20 % des vins et nous n'y touchons plus jusqu'à la partie 3. Ce jeu de test sert à mesurer le modèle final sur des vins qu'il n'a jamais vus. Si nous l'utilisions pour explorer ou pour régler les modèles, ce score final serait trop optimiste.
#
# Le tirage est stratifié, ce qui veut dire qu'il respecte certaines proportions des deux côtés. Nous lui imposons de garder la même part de bons vins et la même part de rouges. Les bons vins rouges sont rares, il n'y en a que 184 dans tout le fichier. Avec un tirage libre, le test pourrait en recevoir trop ou trop peu par malchance.

# %%
# Quatre groupes à respecter des deux côtés, blanc mauvais, blanc bon, rouge mauvais, rouge bon
groupe = vins["type"] + "_" + vins["bon"].astype(str)

train, test = train_test_split(vins, test_size=0.2, stratify=groupe, random_state=GRAINE)
X_train, y_train = train[variables], train["bon"]

controle = pd.DataFrame({
    "nombre de vins": [len(train), len(test)],
    "part de bons vins": [train["bon"].mean(), test["bon"].mean()],
    "part de rouges": [train["est_rouge"].mean(), test["est_rouge"].mean()],
    "part de bons parmi les rouges": [train[train["est_rouge"] == 1]["bon"].mean(), test[test["est_rouge"] == 1]["bon"].mean()],
}, index=["entraînement", "test"])
controle.round(3)

# %% [markdown]
# Les deux jeux ont bien la même part de bons vins, la même part de rouges et la même part de bons vins parmi les rouges. Toute la suite de cette partie n'utilise que le jeu d'entraînement.
#
# ## 1.5 Qu'est-ce qui distingue un bon vin ?
#
# C'est surtout le degré d'alcool, et dans une moindre mesure une faible acidité volatile. Nous calculons la corrélation de chaque mesure avec la cible, une fois chez les rouges et une fois chez les blancs.

# %%
# Corrélation de chaque variable avec la cible, calculée séparément pour les rouges et pour les blancs
rouges, blancs = train[train["type"] == "rouge"], train[train["type"] == "blanc"]
corr_couleur = pd.DataFrame({"rouges": rouges[numeriques].corrwith(rouges["bon"]),
                             "blancs": blancs[numeriques].corrwith(blancs["bon"])})
corr_couleur = corr_couleur.loc[corr_couleur.abs().max(axis=1).sort_values().index]

fig, ax = plt.subplots(figsize=(8, 5))
positions = np.arange(len(corr_couleur))
ax.barh(positions + 0.2, corr_couleur["rouges"], height=0.4, color=PRUNE, label="vins rouges")
ax.barh(positions - 0.2, corr_couleur["blancs"], height=0.4, color=MIEL, label="vins blancs")
ax.axvline(0, color=ENCRE, linewidth=1)
ax.set_yticks(positions); ax.set_yticklabels(corr_couleur.index)
ax.set_xlabel("corrélation avec la cible (vers la droite, la variable va avec les bons vins)")
ax.set_title("Corrélation de chaque variable avec la cible, par couleur de vin")
ax.legend(loc="lower right")
plt.show()
corr_couleur.round(2)

# %% [markdown]
# Une barre vers la droite signifie que la mesure est plus élevée chez les bons vins. Une barre vers la gauche signifie l'inverse. Plus la barre est longue, plus le lien est fort.
#
# L'alcool a les deux barres les plus longues, et elles sont presque égales (+0.39 pour les rouges et +0.42 pour les blancs). Pour d'autres mesures, les deux couleurs ne se ressemblent pas. L'acidité volatile, qui donne un goût de vinaigre, pénalise les rouges (-0.26) et presque pas les blancs (-0.05). Nous nous sommes demandé si cette différence entre couleurs gênait certains modèles. La partie 2 teste cette idée en 2.7.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, variable in zip(axes, ["alcohol", "chlorides"]):
    for classe, couleur, nom in [(0, BRUME, "mauvais"), (1, PRUNE, "bons")]:
        ax.hist(train[train["bon"] == classe][variable], bins=30, density=True, color=couleur, alpha=0.75, label=nom)
    ax.set_xlabel(variable); ax.set_yticks([])
axes[0].set_title("Distribution de l'alcool")
axes[1].set_title("Distribution des chlorures")
axes[0].legend()
plt.tight_layout(); plt.show()

print("Asymétrie de chaque variable (0 pour une loi normale)")
print(train[numeriques].skew().round(1).sort_values(ascending=False).to_string())

# %% [markdown]
# Même pour l'alcool, les bons et les mauvais vins se recouvrent largement. Aucune mesure ne suffit seule, et aucun modèle n'atteindra un score parfait. On voit aussi que les chlorures ont une distribution très étalée vers la droite, loin de la forme en cloche d'une loi normale.
#
# ## 1.6 Des mesures disent-elles la même chose ?
#
# Oui, plusieurs mesures se répètent, en particulier la densité qui dépend presque entièrement des autres.

# %%
colonnes = numeriques + ["est_rouge", "bon"]
correlations = train[colonnes].corr()
triangle = correlations.where(np.tril(np.ones(correlations.shape, dtype=bool)))

fig, ax = plt.subplots(figsize=(9.5, 7.5))
image = ax.imshow(triangle, cmap="PuOr_r", vmin=-1, vmax=1)
for i in range(len(colonnes)):
    for j in range(i + 1):
        valeur = correlations.iloc[i, j]
        ax.text(j, i, f"{valeur:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(valeur) > 0.6 else ENCRE)
ax.set_xticks(range(len(colonnes))); ax.set_xticklabels(colonnes, rotation=45, ha="right")
ax.set_yticks(range(len(colonnes))); ax.set_yticklabels(colonnes)
ax.grid(False); fig.colorbar(image, shrink=0.7)
ax.set_title("Corrélations sur le jeu d'entraînement (la dernière ligne est la cible)")
plt.show()

autres = [v for v in variables if v != "density"]
r2_densite = LinearRegression().fit(train[autres], train["density"]).score(train[autres], train["density"])
print(f"Les 11 autres variables permettent de recalculer la densité à {r2_densite:.0%}.")

# %% [markdown]
# La densité est fortement liée à l'alcool (-0.66), et les deux mesures de dioxyde de soufre vont ensemble (+0.72). La densité se recalcule même à 95% à partir des autres mesures. C'est physique, l'alcool est plus léger que l'eau et le sucre plus lourd.
#
# Cette répétition aura des conséquences pour deux modèles de la partie 2. Naive Bayes suppose que les mesures sont indépendantes, et les poids de la régression logistique deviennent difficiles à lire quand deux mesures portent la même information.
#
# ## 1.7 Comment vérifier nos choix sans toucher au test ?
#
# Nous utilisons la validation croisée, qui permet de noter un modèle sur des vins qu'il n'a pas vus en n'utilisant que le jeu d'entraînement. Les vins d'entraînement sont coupés en 5 groupes. Le modèle apprend sur 4 groupes et il est noté sur le cinquième. On recommence 5 fois en changeant le groupe mis de côté, puis on fait la moyenne des 5 notes.
#
# Cette méthode sert dans tout le projet. Elle sert ici à tester les choix de préparation, puis en partie 2 à régler les modèles et en partie 3 à choisir le modèle final et son seuil.

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=GRAINE)

def notes_cv(modele, X, y, decoupage=cv, groupes=None):
    """AUC et F1-score moyens en validation croisée à 5 plis."""
    notes = cross_validate(modele, X, y, cv=decoupage, groups=groupes, scoring=["roc_auc", "f1"])
    return notes["test_roc_auc"].mean(), notes["test_f1"].mean()

temoins = {
    "Régression logistique": Pipeline([("echelle", StandardScaler()), ("modele", LogisticRegression(max_iter=1000))]),
    "k plus proches voisins": Pipeline([("echelle", StandardScaler()), ("modele", KNeighborsClassifier(15, weights="distance"))]),
    "Arbre de décision": DecisionTreeClassifier(random_state=GRAINE),
    "Forêt aléatoire": RandomForestClassifier(200, random_state=GRAINE, n_jobs=-1),
}

# Quatre modèles simples pour les expériences sur la mise à l'échelle et les valeurs extrêmes
candidats = {
    "Régression logistique": LogisticRegression(max_iter=5000),
    "k plus proches voisins": KNeighborsClassifier(15),
    "Arbre de décision": DecisionTreeClassifier(max_depth=4, random_state=GRAINE),
    "Naive Bayes": GaussianNB(),
}

# %% [markdown]
# ## 1.8 Faut-il supprimer les doublons ?
#
# Oui, car ils permettent au modèle de reconnaître des vins déjà vus, ce qui gonfle les scores sans que le modèle soit meilleur. Pour le montrer, nous comparons trois situations sur les données complètes. Dans la première, les doublons sont gardés. Dans la deuxième, ils sont gardés mais les copies d'un même vin restent toujours dans le même groupe de la validation croisée. Dans la troisième, ils sont supprimés.

# %%
X_brut = vins_brut.assign(est_rouge=(vins_brut["type"] == "rouge").astype(int))[variables]
y_brut = (vins_brut["quality"] >= 7).astype(int)
numero_du_vin = X_brut.groupby(variables).ngroup()      # deux lignes identiques portent le même numéro
cv_groupe = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=GRAINE)

lignes = []
for nom, modele in temoins.items():
    auc_a, f1_a = notes_cv(modele, X_brut, y_brut)
    auc_b, f1_b = notes_cv(modele, X_brut, y_brut, cv_groupe, numero_du_vin)
    auc_c, f1_c = notes_cv(modele, X_train, y_train)
    lignes.append({"modèle": nom, "AUC doublons mélangés": auc_a, "AUC doublons jamais séparés": auc_b,
                   "AUC doublons supprimés": auc_c, "F1 doublons mélangés": f1_a, "F1 doublons supprimés": f1_c})
exp_doublons = pd.DataFrame(lignes).set_index("modèle")

fig, ax = plt.subplots(figsize=(8, 3.4))
positions = np.arange(len(exp_doublons))
ax.hlines(positions, exp_doublons["AUC doublons supprimés"], exp_doublons["AUC doublons mélangés"], color=BRUME, linewidth=3)
ax.scatter(exp_doublons["AUC doublons mélangés"], positions, color=MIEL, s=90, label="doublons gardés", zorder=3)
ax.scatter(exp_doublons["AUC doublons jamais séparés"], positions, color=ENCRE, s=40, marker="D", label="doublons gardés, copies du même côté", zorder=4)
ax.scatter(exp_doublons["AUC doublons supprimés"], positions, color=PRUNE, s=90, label="doublons supprimés", zorder=3)
ax.set_yticks(positions); ax.set_yticklabels(exp_doublons.index); ax.set_xlabel("AUC en validation croisée")
ax.set_title("AUC de quatre modèles avec et sans les doublons")
ax.legend(loc="lower right")
plt.show()
exp_doublons.round(3)

# %% [markdown]
# Avec les doublons, la forêt aléatoire obtient une AUC de 0.923. Sans eux, 0.850. Le F1-score des voisins passe de 0.67 à 0.47.
#
# La deuxième situation donne la preuve. Les données sont exactement les mêmes que dans la première, on interdit seulement qu'un vin soit à la fois dans les groupes d'apprentissage et dans le groupe noté. Le score de la forêt retombe alors à 0.853. Tout le gain venait donc de copies que le modèle avait déjà vues. On appelle cela une fuite de données.
#
# La régression logistique ne change presque pas. Elle trace une seule frontière et ne peut pas retenir un vin en particulier, contrairement aux voisins, à l'arbre et à la forêt.
#
# Onze mesures identiques jusqu'à la dernière décimale avec la même note, cela ressemble plus à un échantillon saisi plusieurs fois qu'à des vins différents. Nos scores seront donc plus bas que ceux d'une analyse qui garde les doublons, mais ils disent ce que le modèle sait faire sur un vin vraiment nouveau.
#
# ## 1.9 Faut-il retirer les valeurs extrêmes ?
#
# Non, car elles signalent souvent un vin raté et aident donc à reconnaître les mauvais vins. Nous repérons d'abord les valeurs extrêmes avec la règle de la boîte à moustaches.

# %%
def est_aberrante(serie):
    """Règle de la boîte à moustaches, au-delà de 1,5 fois l'écart interquartile."""
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    return (serie < q1 - 1.5 * (q3 - q1)) | (serie > q3 + 1.5 * (q3 - q1))

vin_atypique = train[numeriques].apply(est_aberrante).any(axis=1)
print(f"Vins avec au moins une valeur aberrante   {vin_atypique.sum()} soit {vin_atypique.mean():.0%}")
print(f"Part de bons vins parmi ces vins atypiques {train[vin_atypique]['bon'].mean():.0%}")
print(f"Part de bons vins parmi les autres         {train[~vin_atypique]['bon'].mean():.0%}")

# %% [markdown]
# Cette règle signale 21% des vins. Seulement 12% d'entre eux sont bons, contre 21% des autres. Une valeur extrême n'est donc pas une erreur de mesure. Elle correspond souvent à un défaut, comme trop d'acide acétique ou trop de sel.
#
# Nous vérifions ensuite ce que donnerait leur suppression.

# %%
lignes = []
for nom, modele in candidats.items():
    modele = Pipeline([("echelle", StandardScaler()), ("modele", modele)])
    lignes.append({"modèle": nom,
                   "tous les vins": notes_cv(modele, X_train, y_train)[0],
                   "sans les vins atypiques": notes_cv(modele, X_train[~vin_atypique], y_train[~vin_atypique])[0]})
exp_aberrantes = pd.DataFrame(lignes).set_index("modèle")
print(f"Supprimer les vins atypiques ferait perdre {vin_atypique.mean():.0%} des vins d'entraînement.")
exp_aberrantes.round(3)

# %% [markdown]
# Les quatre modèles sont moins bons quand on retire ces vins, et on perd un cinquième des données. Nous les gardons.
#
# ## 1.10 Faut-il mettre les mesures à la même échelle ?
#
# Oui pour les modèles qui calculent des distances, et cela ne change rien pour les autres.

# %%
lignes = []
for nom, modele in candidats.items():
    lignes.append({"modèle": nom,
                   "sans mise à l'échelle": notes_cv(modele, X_train, y_train)[0],
                   "standardisation": notes_cv(Pipeline([("echelle", StandardScaler()), ("modele", modele)]), X_train, y_train)[0],
                   "normalisation min-max": notes_cv(Pipeline([("echelle", MinMaxScaler()), ("modele", modele)]), X_train, y_train)[0]})
exp_echelle = pd.DataFrame(lignes).set_index("modèle")
exp_echelle.round(3)

# %% [markdown]
# Le modèle des voisins passe de 0.705 à 0.827. Il cherche les vins les plus proches, et sans mise à l'échelle la distance est dominée par le dioxyde de soufre total, dont les valeurs sont les plus grandes. Le SVM de la partie 2 repose aussi sur des distances.
#
# L'arbre de décision et Naive Bayes ne bougent pas. L'arbre compare chaque mesure à un seuil, et changer l'unité déplace le seuil sans changer la réponse. Naive Bayes calcule une moyenne et un écart type par mesure, il s'adapte donc tout seul.
#
# La standardisation et la normalisation min-max donnent presque le même résultat. Nous retenons la standardisation, qui est moins sensible aux valeurs extrêmes que nous avons gardées. Avec le min-max, le vin à 65,8 grammes de sucre écraserait tous les autres vers zéro.
#
# La standardisation est placée dans un pipeline avec le modèle. De cette façon, pendant la validation croisée, la moyenne et l'écart type sont calculés sur les 4 groupes d'apprentissage seulement. Le groupe noté n'influence pas la préparation.
#
# ## 1.11 Faut-il retirer des mesures ?
#
# Non, sauf pour Naive Bayes. Nous appliquons les trois familles de méthodes du cours.
#
# | Famille | Principe | Méthode utilisée |
# |---|---|---|
# | Filtre | Donner une note à chaque mesure, sans entraîner de modèle | Corrélation de Pearson, information mutuelle |
# | Enveloppante | Entraîner un modèle, retirer la mesure la moins utile, recommencer | Élimination récursive, notée RFE |
# | Intégrée | Laisser le modèle faire le tri pendant son entraînement | Régression logistique avec pénalité Lasso |
#
# Le test du khi-deux, aussi présenté en cours, compare des catégories ou des comptages. Nos mesures sont continues, nous utilisons donc l'information mutuelle comme second filtre.

# %%
X_std = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=variables, index=X_train.index)

pearson = X_train.corrwith(y_train).abs()                                                    # filtre
info_mutuelle = pd.Series(mutual_info_classif(X_train, y_train, random_state=GRAINE), index=variables)   # filtre
rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=1).fit(X_std, y_train)     # enveloppante
lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.01).fit(X_std, y_train)     # intégrée

selection = pd.DataFrame({
    "rang Pearson": pearson.rank(ascending=False).astype(int),
    "rang information mutuelle": info_mutuelle.rank(ascending=False).astype(int),
    "rang RFE": rfe.ranking_,
    "poids Lasso": lasso.coef_[0].round(3),
}, index=variables).sort_values("rang RFE")
selection

# %% [markdown]
# Les quatre méthodes placent l'alcool en premier. Elles ne sont pas d'accord sur la densité et les chlorures. Les deux filtres les classent deuxième et troisième, alors que le Lasso leur donne un poids nul. Un filtre note chaque mesure séparément, il voit donc que la densité est liée à la qualité. Le Lasso regarde toutes les mesures ensemble et il écarte la densité parce qu'elle répète l'alcool.
#
# Nous regardons maintenant si retirer des mesures améliore les modèles, en les ajoutant une par une de la plus utile à la moins utile.

# %%
ordre = list(selection.index)
a_tester = {"Régression logistique": temoins["Régression logistique"],
            "k plus proches voisins": temoins["k plus proches voisins"],
            "Naive Bayes": GaussianNB()}
courbes = pd.DataFrame({nom: [notes_cv(modele, X_train[ordre[:k]], y_train)[0] for k in range(1, 13)]
                        for nom, modele in a_tester.items()}, index=range(1, 13))

fig, ax = plt.subplots(figsize=(8, 3.8))
for nom, couleur in zip(courbes.columns, [ENCRE, PRUNE, MIEL]):
    ax.plot(courbes.index, courbes[nom], marker="o", color=couleur)
    ax.text(12.2, courbes[nom].iloc[-1], nom, color=couleur, va="center")
ax.set_xlim(0.5, 16); ax.set_xticks(range(1, 13))
ax.set_xlabel("nombre de variables gardées, de la plus utile à la moins utile"); ax.set_ylabel("AUC")
ax.set_title("AUC selon le nombre de variables gardées")
plt.show()
courbes.round(3).T

# %% [markdown]
# La régression logistique et les voisins progressent quand on ajoute des mesures, puis se stabilisent vers 10. Les dernières n'apportent rien mais ne gênent pas.
#
# Naive Bayes fait l'inverse. Il est à son meilleur avec 2 mesures (0.798) et descend à 0.771 avec les 12. Ce modèle traite chaque mesure comme une preuve indépendante. Quand deux mesures se répètent, il compte la même preuve deux fois.
#
# Nous gardons donc les 12 mesures pour tous les modèles sauf Naive Bayes, dont le nombre de mesures sera réglé en partie 2.

# %%
import os
os.makedirs("donnees_preparees", exist_ok=True)
train.to_csv("donnees_preparees/train.csv", index=False)
test.to_csv("donnees_preparees/test.csv", index=False)
print("Fichiers enregistrés pour les parties 2 et 3,", len(train), "vins d'entraînement et", len(test), "vins de test")

# %% [markdown]
# Les vins d'entraînement et de test sont enregistrés pour les deux parties suivantes.
