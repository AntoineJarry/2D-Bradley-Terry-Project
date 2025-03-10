import numpy as np
import First_part_project.Bradley_Terry_Model_2D.calcul_affichage as calcul_affichage
import First_part_project.Bradley_Terry_model_1D.calcul_affichage as calcul_1D
import Second_part_project.Bradley_Terry_model_2D.calcul_affichage as calcul_graphique

#####################
# Exemples de matrice
##################### 
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


Teams = np.array(["BRO", "DK", "DRX", "FOX", "GEN", "HLE", "KDF", "KT", "NS", "T1"])

LCK = np.array([
  [0, 10, 15, 10, 5, 12, 22, 10, 16, 8],
  [29, 0, 32, 34, 22, 32, 27, 28, 32, 22],
  [28, 8, 0, 19, 3, 8, 25, 14, 21, 7],
  [26, 8, 31, 0, 8, 19, 19, 16, 21, 11],
  [30, 36, 31, 32, 0, 45, 30, 26, 32, 40],
  [25, 16, 33, 23, 11, 0, 30, 17, 32, 30],
  [21, 9, 18, 20, 7, 14, 0, 19, 24, 12],
  [27, 21, 30, 29, 18, 32, 18, 0, 26, 23],
  [22, 7, 21, 20, 1, 14, 15, 16, 0, 13],
  [29, 42, 34, 29, 30, 30, 31, 41, 26, 0]])

#######
# Matrice de comparaison option 2
#######

croquettes_dog = ['SPF2', 'SPF4', 'BENCH4', 'SPF1', 'SPF3', 'BENCH1', 'BENCH2', 'BENCH3']
mat_comp_dog_opt2 = np.array([
    [ 0, 28, 24, 33, 21, 11, 20, 27],
    [10,  0, 12, 18, 20,  6, 18, 11],
    [16, 21,  0, 16, 17, 12,  8, 17],
    [ 2, 11, 20,  0,  9,  5, 18, 17],
    [15, 16, 18, 15,  0,  8, 15, 20],
    [23, 31, 23, 31, 33,  0, 16, 24],
    [17, 19, 27, 23, 17, 24,  0, 13],
    [14, 26, 23, 19, 16, 11, 18,  0]
])

croquettes_cat = ['LWFA', 'LWEA', 'HWEP', 'HDFA', 'HDEA', 'LWFP', 'HWFA', 'HDFP', 'LDFA', 'LDEA', 'LWEP', 'LDEP', 'HDEP', 'LDFP', 'HWEA', 'M', 'HWFP']
mat_comp_cat_opt2 = np.array([
    [ 0, 24, 26, 17, 12, 23, 24, 26, 14, 16, 23, 16, 25, 15, 13, 16, 14],
    [11,  0, 25, 18, 13, 19, 19, 26, 11, 12, 20, 21, 22, 25, 19, 15, 21],
    [10, 12,  0, 18, 11, 16, 14, 16, 10,  7, 17, 17, 15, 18, 15, 15, 25],
    [21, 18, 16,  0, 17, 22, 18, 18, 13, 17, 23, 21, 23, 28, 22, 20, 20],
    [25, 25, 24, 18,  0, 23, 21, 28, 18, 14, 18, 25, 20, 22, 27, 25, 30],
    [13, 19, 18, 13, 13,  0, 14, 14, 13,  8, 16, 15, 10, 11, 15,  9, 26],
    [13, 11, 22,  6, 11, 20,  0, 24, 14, 10, 21, 17, 21, 21, 24, 18, 25],
    [ 9, 10, 22,  8,  8, 21, 13,  0, 17, 11, 24, 21, 15, 21, 20, 15, 25],
    [20, 21, 27, 23, 16, 23, 21, 19,  0, 23, 20, 23, 21, 20, 19, 23, 53],
    [20, 26, 29, 19, 22, 26, 26, 25, 14,  0, 25, 19, 26, 24, 43, 25, 23],
    [13, 14, 18, 15, 18, 19, 17, 12, 15,  8,  0, 14, 15, 11, 15, 14, 19],
    [20, 15, 18, 15,  7, 19, 17, 14, 10, 16, 21,  0, 20, 42, 17, 16, 20],
    [13, 14, 21, 11,  9, 22, 15, 21, 14,  9, 21, 16,  0, 15, 14, 13, 17],
    [11, 10, 16,  8, 14, 24, 14, 15, 16, 12, 23, 24, 19,  0, 23, 13, 22],
    [21, 16, 23, 15, 11, 22, 12, 16, 16, 29, 20, 18, 21, 44,  0, 17, 24],
    [18, 21, 20, 18,  9, 27, 17, 23, 15,  8, 22, 20, 19, 25, 19,  0, 19],
    [20, 14, 10, 15,  6, 11, 13,  9, 16, 13, 17, 16, 10, 15,  9, 14,  0]
])

##########
# PARTIE 1
##########

# calcul_1D.graphique_1D(mat_comp_dog_opt2,croquettes_dog)

# calcul_affichage.graphique_2D(LCK,Teams,reverse_v1=True,reverse_v2=False)

# calcul_affichage.ellipses(N_cat_1,labels_cat_1,reverse_v1=False,reverse_v2=True)

# print(calcul_affichage.deviances(N_cat_2,reverse_v1=False,reverse_v2=True))

##########
# PARTIE 2
##########

# calcul_graphique.graphique_IPM(mat_comp_dog_opt2,method = "trust-constr",reverse_v1=False,reverse_v2=True,labels=croquettes_dog)

calcul_graphique.deviance_NR_IPM(LCK,method="trust-constr",reverse_v1=False,reverse_v2=True)
