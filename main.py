import numpy as np
import First_part_project.Bradley_Terry_Model_2D.calcul_affichage as calcul_affichage
import First_part_project.Bradley_Terry_model_1D.calcul_affichage as calcul_1D
import Second_part_project.Bradley_Terry_model_2D.calcul_affichage as calcul_graphique

# Exemple de matrice 
N = np.array([
  [0, 39, 64, 40, 61, 76, 46],
  [61, 0, 65, 59, 55, 85, 60],
  [36, 35, 0, 31, 25, 41, 35],
  [60, 41, 69, 0, 41, 80, 28],
  [39, 45, 75, 59, 0, 71, 37],
  [24, 15, 59, 20, 29, 0, 18],
  [54, 40, 65, 72, 63, 82, 0]])

# Donner le label correspondant à la matrice
labels = ['1', '2', '3', "4", "5", '6', '7']

N_cat_1 = np.array([
    [ 0, 11, 17,  9,  2, 20, 18,  5],
    [18,  0, 11, 20, 10, 12, 18,  6],
    [19, 26,  0, 16, 14, 23, 18, 11],
    [15, 16, 20,  0, 15, 18, 15,  8],
    [33, 28, 27, 21,  0, 24, 20, 11],
    [16, 21, 17, 17, 16,  0,  8, 12],
    [23, 19, 13, 17, 17, 27,  0, 24],
    [31, 31, 24, 33, 23, 23, 16,  0]
])

N_cat_2 = np.array([
    [ 0, 13, 13, 15,  8, 18, 14, 26, 16, 10, 14, 19, 13, 13,  9, 11, 15],
    [23,  0, 16, 19, 23, 27, 21, 53, 20, 21, 19, 21, 20, 23, 23, 20, 23],
    [23, 18,  0, 27, 14, 24, 21, 30, 18, 20, 28, 25, 25, 18, 25, 22, 25],
    [22, 16, 11,  0, 29, 23, 12, 24, 20, 21, 16, 16, 21, 15, 17, 44, 18],
    [26, 14, 22, 43,  0, 29, 26, 23, 25, 26, 25, 26, 20, 19, 25, 24, 19],
    [16, 10, 11, 15,  7,  0, 14, 25, 17, 15, 16, 12, 10, 18, 15, 18, 17],
    [20, 14, 11, 24, 10, 22,  0, 25, 21, 21, 24, 11, 13,  6, 18, 21, 17],
    [11, 16,  6,  9, 13, 10, 13,  0, 17, 10,  9, 14, 20, 15, 14, 15, 16],
    [19, 15, 18, 15,  8, 18, 17, 19,  0, 15, 12, 14, 13, 15, 14, 11, 14],
    [22, 14,  9, 14,  9, 21, 15, 17, 21,  0, 21, 14, 13, 11, 13, 15, 16],
    [21, 17,  8, 20, 11, 22, 13, 25, 24, 15,  0, 10,  9,  8, 15, 21, 21],
    [19, 11, 13, 19, 12, 25, 19, 21, 20, 22, 26,  0, 11, 18, 15, 25, 21],
    [23, 14, 12, 13, 16, 26, 24, 14, 23, 25, 26, 24,  0, 17, 16, 15, 16],
    [22, 13, 17, 22, 17, 16, 18, 20, 23, 23, 18, 18, 21,  0, 20, 28, 21],
    [27, 15,  9, 19,  8, 20, 17, 19, 22, 19, 23, 21, 18, 18,  0, 25, 20],
    [24, 16, 14, 23, 12, 16, 14, 22, 23, 19, 15, 10, 11,  8, 13,  0, 24],
    [19, 10,  7, 17, 16, 18, 17, 20, 21, 20, 14, 15, 20, 15, 16, 42,  0]
])
labels_cat_1 = ['SPF1', 'SPF4', 'BENCH3', 'SPF3', 'SPF2', 'BENCH4', 'BENCH2', 'BENCH1']
labels_cat_2 = ['LWFP', 'LDFA', 'HDEA', 'HWEA', 'LDEA', 'HWEP', 'HWFA', 'HWFP', 'LWEP', 'HDEP', 'HDFP', 'LWEA', 'LWFA', 'HDFA', 'M', 'LDFP', 'LDEP']

##########
# PARTIE 1
##########

# calcul_1D.graphique_1D(N,labels)

# calcul_affichage.graphique_2D(N_cat_2,labels_cat_2,reverse_v1=False,reverse_v2=True)

# calcul_affichage.ellipses(N_cat_1,labels_cat_1,reverse_v1=False,reverse_v2=True)

# print(calcul_affichage.deviances(N,reverse_v1=False,reverse_v2=True))

##########
# PARTIE 2
##########

# calcul_graphique.graphique_IPM(N_cat_2,method = "trust-constr",reverse_v1=False,reverse_v2=True,labels=labels_cat_2)


D0 = -57.55180687718192
D1 = -1308.2720255202005
# Définir les paramètres
G2 = D0 - D1  # Différence de déviance
df = 17-5     # Différence de degrés de liberté

from scipy.stats import chi2
# Calculer la p-valeur
p_value = 1 - chi2.cdf(G2, df)
print("p-value :", p_value)