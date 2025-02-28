import pandas as pd
import numpy as np
import pickle

# Charger le fichier Excel
data_cat = pd.read_excel("AMA08001.xlsx")
data_dog = pd.read_excel("VYE15004.xlsx")
# Afficher les premières lignes
print(data_cat.head())

## Option 3 dog

# Identifier le gagnant et le perdant
data_dog["Winner"] = data_dog.apply(lambda row: row["ALIMENT_B_LIBELLE"] if row["RATIO_B"] > row["RATIO_A"] else row["ALIMENT_A_LIBELLE"], axis=1)
data_dog["Loser"] = data_dog.apply(lambda row: row["ALIMENT_A_LIBELLE"] if row["RATIO_B"] > row["RATIO_A"] else row["ALIMENT_B_LIBELLE"], axis=1)

# Obtenir la liste unique des croquettes
croquettes = list(set(data_dog["Winner"]).union(set(data_dog["Loser"])))
print(croquettes)
with open("croquettes_dog.npy", "wb") as f:
    pickle.dump(croquettes, f)

# Créer une matrice vide avec les croquettes en index et colonnes
comparison_matrix_dog_opt3 = pd.DataFrame(0, index=croquettes, columns=croquettes)

# Remplir la matrice avec le nombre de victoires
for _, row in data_dog.iterrows():
    comparison_matrix_dog_opt3.loc[row["Winner"], row["Loser"]] += 1  # Ajoute 1 lorsque l'un est préféré à l'autre

# Convertir la matrice pandas en numpy array
comparison_matrix_dog_opt3_np = comparison_matrix_dog_opt3.to_numpy()

# Afficher la matrice numpy
print(comparison_matrix_dog_opt3_np)

## Option 3 cat

data_cat["Winner"] = data_cat.apply(lambda row: row["ALIMENT_B_LIBELLE"] if row["RATIO_B"] > row["RATIO_A"] else row["ALIMENT_A_LIBELLE"], axis=1)
data_cat["Loser"] = data_cat.apply(lambda row: row["ALIMENT_A_LIBELLE"] if row["RATIO_B"] > row["RATIO_A"] else row["ALIMENT_B_LIBELLE"], axis=1)

# Obtenir la liste unique des croquettes
croquettes = list(set(data_cat["Winner"]).union(set(data_cat["Loser"])))
print(croquettes)

# Sauvegarder les matrices numpy dans un fichier pickle
with open("croquettes_cat.npy", "wb") as f:
    pickle.dump(croquettes, f)

# Créer une matrice vide avec les croquettes en index et colonnes
comparison_matrix_cat_opt3 = pd.DataFrame(0, index=croquettes, columns=croquettes)

# Remplir la matrice avec le nombre de victoires
for _, row in data_cat.iterrows():
    comparison_matrix_cat_opt3.loc[row["Winner"], row["Loser"]] += 1  # Ajoute 1 lorsque l'un est préféré à l'autre

# Convertir la matrice pandas en numpy array
comparison_matrix_cat_opt3_np = comparison_matrix_cat_opt3.to_numpy()

# Afficher la matrice numpy
print(comparison_matrix_cat_opt3_np)

# Sauvegarder les matrices numpy dans un fichier pickle
with open("comparison_matrix_dog_opt3.npy", "wb") as f:
    pickle.dump(comparison_matrix_dog_opt3_np, f)

with open("comparison_matrix_cat_opt3.npy", "wb") as f:
    pickle.dump(comparison_matrix_cat_opt3_np, f)


