# Sur dog data
import pandas as pd
import pickle
data_dog = pd.read_excel("VYE15004.xlsx")
dog_pref = data_dog[["ALIMENT_A_LIBELLE", "ALIMENT_B_LIBELLE", "CONSO_A", "CONSO_B", "RATIO_A", "RATIO_B"]]


# Fonction pour trier les colonnes et inverser les valeurs associées
def versus_ordre_aphabetique(row):
    # Vérifie si inversion nécessaire
    if row['ALIMENT_A_LIBELLE'] > row['ALIMENT_B_LIBELLE']:
        row['ALIMENT_A_LIBELLE'], row['ALIMENT_B_LIBELLE'] = row['ALIMENT_B_LIBELLE'], row['ALIMENT_A_LIBELLE']  # Échange des aliments
        row['CONSO_A'], row['CONSO_B'] = row['CONSO_B'], row['CONSO_A']  # Échange des consomations
        row['RATIO_A'], row['RATIO_B'] = row['RATIO_B'], row['RATIO_A']  # Échange des ratios
    return row


dog_pref = dog_pref.apply(versus_ordre_aphabetique, axis=1)


# Colonnes de regroupement
groupby_cols = ['ALIMENT_A_LIBELLE', 'ALIMENT_B_LIBELLE']

# Dictionnaire des fonctions d'agrégation
agg_funcs = {
    'CONSO_A': 'sum',
    'CONSO_B': 'sum',
    'RATIO_A': 'mean',
    'RATIO_B': 'mean'
}

# GroupBy avec agrégation
dog_pref = dog_pref.groupby(groupby_cols).agg(agg_funcs).reset_index()

dog_pref["CONSO_AB"] = dog_pref["CONSO_A"] + dog_pref["CONSO_B"]
dog_pref["CONSO_Mij"] = dog_pref["RATIO_A"] * dog_pref["CONSO_AB"] / 100
dog_pref["CONSO_Mji"] = dog_pref["RATIO_B"] * dog_pref["CONSO_AB"] / 100

# dog_pref_red = dog_pref[["ALIMENT_A_LIBELLE", "ALIMENT_B_LIBELLE", "CONSO_Mij", "CONSO_Mji"]]


# Création de la matrice avec pivot_table
dog_mat = pd.pivot_table(dog_pref, values='CONSO_Mij', index='ALIMENT_A_LIBELLE', columns='ALIMENT_B_LIBELLE', fill_value=0)

# Ajout des valeurs symétriques
for _, row in dog_pref.iterrows():
    dog_mat.at[row['ALIMENT_B_LIBELLE'], row['ALIMENT_A_LIBELLE']] = row['CONSO_Mji']  # Valeur pour (j, i)

# Remplacer les NaN éventuels par 0
dog_mat = dog_mat.fillna(0)

# Réorganiser les colonnes : déplacer la dernière colonne en première
dog_mat = dog_mat[[dog_mat.columns[-1]] + list(dog_mat.columns[:-1])]

# Affichage du résultat
dog_mat_np = dog_mat.to_numpy()


# Pour les chats
data_cat = pd.read_excel("AMA08001.xlsx")

cat_pref = data_cat[["ALIMENT_A_LIBELLE", "ALIMENT_B_LIBELLE", "CONSO_A", "CONSO_B", "RATIO_A", "RATIO_B"]]
cat_pref = cat_pref.apply(versus_ordre_aphabetique, axis=1)

cat_pref = cat_pref.groupby(groupby_cols).agg(agg_funcs).reset_index()

cat_pref["CONSO_AB"] = cat_pref["CONSO_A"] + cat_pref["CONSO_B"]
cat_pref["CONSO_Mij"] = cat_pref["RATIO_A"] * cat_pref["CONSO_AB"] / 100
cat_pref["CONSO_Mji"] = cat_pref["RATIO_B"] * cat_pref["CONSO_AB"] / 100

# Création de la matrice avec pivot_table
cat_mat = pd.pivot_table(cat_pref, values='CONSO_Mij', index='ALIMENT_A_LIBELLE', columns='ALIMENT_B_LIBELLE', fill_value=0)

# Ajout des valeurs symétriques
for _, row in cat_pref.iterrows():
    cat_mat.at[row['ALIMENT_B_LIBELLE'], row['ALIMENT_A_LIBELLE']] = row['CONSO_Mji']  # Valeur pour (j, i)

# Remplacer les NaN éventuels par 0
cat_mat = cat_mat.fillna(0)

# Réorganiser les colonnes : déplacer la dernière colonne en première
cat_mat = cat_mat[[cat_mat.columns[-1]] + list(cat_mat.columns[:-1])]

# Convertir la matrice pandas en numpy array
cat_mat_np = cat_mat.to_numpy()

print(cat_mat_np)

# Sauvegarder les matrices numpy dans un fichier pickle
with open("dog_mat.npy", "wb") as f:
    pickle.dump(dog_mat_np, f)

with open("cat_mat.npy", "wb") as f:
    pickle.dump(cat_mat_np, f)
