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

#######
# Matrice de comparaison option 3
#######

croquettes_dog = ['SPF2', 'SPF4', 'BENCH4', 'SPF1', 'SPF3', 'BENCH1', 'BENCH2', 'BENCH3']
mat_comp_dog_opt3 = np.array([
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
mat_comp_cat_opt3 = np.array([
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

calcul_1D.graphique_1D(mat_comp_dog_opt3,croquettes_dog)

#calcul_affichage.graphique_2D(mat_comp_dog_opt3,croquettes_dog,reverse_v1=False,reverse_v2=True)

# calcul_affichage.ellipses(N_cat_1,labels_cat_1,reverse_v1=False,reverse_v2=True)

# print(calcul_affichage.deviances(N_cat_2,reverse_v1=False,reverse_v2=True))

##########
# PARTIE 2
##########

# calcul_graphique.graphique_IPM(mat_comp_dog_opt3,method = "trust-constr",reverse_v1=False,reverse_v2=True,labels=croquettes_dog)

# calcul_graphique.deviance_NR_IPM(N,method="trust-constr",reverse_v1=False,reverse_v2=True)
