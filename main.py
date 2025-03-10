import numpy as np
import matplotlib.pyplot as plt
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
# Matrice de comparaison option 1
#######
croquettes_dog_opt1 = ['BENCH1', 'BENCH2', 'BENCH3', 'BENCH4', 'SPF1', 'SPF2', 'SPF3', 'SPF4']
mat_comp_dog_opt1 = np.array([
    [   0.  , 2627.31, 4646.45, 5213.66, 7135.71, 4673.5 , 6741.33, 6943.78],
    [4556.69,    0.  , 2231.8 , 5303.63, 5335.36, 2759.81, 2521.65, 3158.33],
    [2329.55, 3188.2 ,   0.   , 3446.43, 5088.27, 3077.23, 2930.57, 6433.18],
    [2643.34, 1665.37, 2943.57,    0.  , 3703.12, 3543.14, 2497.41, 3073.71],
    [1629.29, 3852.64, 4183.73, 4309.88,    0.  ,  683.85, 1246.74, 1674.88],
    [3224.5 , 3399.19, 6100.77, 4437.86, 6657.15,    0.  , 4833.42, 5339.36],
    [1835.67, 2367.35, 4582.43, 3158.59, 1925.26, 3510.58,    0.  , 3319.4 ],
    [1569.22, 3490.67, 3401.82, 1851.29, 2916.12, 2398.64, 4384.6 ,     0.  ],
    ])

croquettes_cat_opt1 = ['HDEA', 'HDEP', 'HDFA', 'HDFP', 'HWEA', 'HWEP', 'HWFA', 'HWFP', 'LDEA', 'LDEP', 'LDFA', 'LDFP', 'LWEA', 'LWEP', 'LWFA', 'LWFP','M']

mat_comp_cat_opt1 = np.array([
    [   0.  , 1120.07, 1024.73, 1579.53, 1693.24, 1387.1 , 1175.44, 1394.53,  830.44,  1282.91, 1008.31, 1342.53, 1349.23, 1044.58, 1514.23, 1272.52, 1276.79],
    [ 689.93,    0.  ,  717.74,  891.97,  947.02, 1149.35,  849.83, 1003.66,  551.88,   963.72,  803.86,  953.99,  965.32, 1166.06,  776.2 , 1034.44,  771.65],
    [1138.27, 1138.26,    0.  ,  982.99, 1301.5 ,  979.04,  931.91, 1178.44,  981.66,  1336.66,  999.03, 1549.22, 1173.07, 1464.09, 1236.28, 1144.31, 1170.82],
    [ 560.47,  845.03,  615.01,    0.  , 1094.81, 1053.96,  926.67, 1351.51,  720.59,  1152.5 ,  787.04, 1145.09,  736.32, 1311.16,  576.52, 1246.09,  827.5 ],
    [ 977.77, 1210.98,  949.5 , 1074.19,    0.  , 1329.  ,  945.19, 1335.11, 2044.13,  1162.78,  861.21, 2532.55,  997.74,  972.43, 1061.55, 1268.81,  906.88],
    [ 708.9 ,  896.66,  935.96, 1129.04,  961.  ,    0.  ,  880.85, 1384.51,  445.68,  1052.58,  677.42,  966.14,  842.93,  862.8 ,  591.86, 1013.68,  858.67],
    [ 872.56, 1134.17,  479.09, 1254.33, 1223.81, 1201.15,    0.  , 1132.13,  752.81,  1066.11,  879.26,  974.44,  657.73, 1248.05,  885.89, 1055.76, 1021.6 ],
    [ 478.47,  658.34,  852.56,  685.49,  697.89,  745.49,  873.87,    0.  ,  797.9 ,   821.62, 1271.83,  947.15,  750.49, 1038.17, 1050.49,  767.83,  803.29],
    [1135.56, 1609.12,  944.34, 1462.41, 2360.87, 1489.32, 1288.19, 1340.1 ,    0.  ,   963.46, 1025.1 , 1473.17, 1355.99, 1268.56, 1139.01, 1402.38, 1303.11],
    [ 584.09, 1074.28,  969.34,  814.5 , 1108.22, 1189.42,  935.89, 1041.38,  997.54,     0.  ,  649.6 , 2060.81,  823.66, 1089.61,  962.7 ,  928.63,  782.25],
    [ 943.7 , 1062.14, 1267.97, 1086.96, 1403.79, 1382.58, 1288.74, 2650.18,  1160.9,  1230.4 ,    0.  , 1024.99, 1062.54, 1045.06, 1072.96, 1282.88, 1347.53],
    [ 934.47, 1028.01,  753.78,  917.91, 1804.45,  882.86,  732.56, 1117.85,  794.84,  1471.19,  670.01,    0.  ,  682.93, 1301.66,  626.06, 1150.71, 1013.01],
    [ 829.77, 1132.68, 1240.93, 1269.68, 1092.26, 1533.07,  959.27, 1269.51,  878.01,  1443.34,  600.46, 1206.07,    0.  , 1009.46,  758.26, 1303.09,  918.57],
    [ 796.42,  922.94,  879.91,  721.84,  734.57, 1088.2 , 1132.95, 1180.83,  564.44,   952.39,  569.94,  757.34,  655.54,    0.  , 1018.85, 1036.13,  890.14],
    [ 867.77, 1321.8 , 1096.72, 1478.48,  877.45, 1379.14, 1410.11,  959.51,  810.99,   905.3 ,  955.04,  965.94, 1258.74, 1076.15,    0.  , 1305.42, 897.  ],
    [ 759.48,  625.56,  611.69,  844.91,  984.19, 1018.32,  918.24, 1352.17,  584.62,   778.37,  839.12,  864.29, 1235.91,  827.87,  880.58,    0.  ,  809.25],
    [ 808.21, 1055.36, 1037.19, 1218.5 , 1253.12, 1026.33, 1147.4 ,  974.71,  685.89,  1000.75,  861.48, 1246.99, 1118.43, 1249.86, 1025.  , 1570.75,    0.  ]
])



#######
# Matrice de comparaison option 3
#######

croquettes_dog_opt3 = ['SPF2', 'SPF4', 'BENCH4', 'SPF1', 'SPF3', 'BENCH1', 'BENCH2', 'BENCH3']
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

croquettes_cat_opt3 = ['LWFA', 'LWEA', 'HWEP', 'HDFA', 'HDEA', 'LWFP', 'HWFA', 'HDFP', 'LDFA', 'LDEA', 'LWEP', 'LDEP', 'HDEP', 'LDFP', 'HWEA', 'M', 'HWFP']
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

# calcul_1D.graphique_1D(mat_comp_dog_opt3, croquettes_dog)

# calcul_affichage.graphique_2D(mat_comp_dog_opt3,croquettes_dog,reverse_v1=False,reverse_v2=True)

# calcul_affichage.ellipses(N_cat_1,labels_cat_1,reverse_v1=False,reverse_v2=True)

print(calcul_affichage.deviances(LCK,reverse_v1=False,reverse_v2=True))

##########
# PARTIE 2
##########

# calcul_graphique.graphique_IPM(mat_comp_dog_opt3, method="trust-constr", reverse_v1=False, reverse_v2=True, labels=croquettes_dog_opt3)

#calcul_graphique.deviance_NR_IPM(mat_comp_dog_opt3,method="trust-constr",reverse_v1=False,reverse_v2=True)


##########
# PARTIE X
##########

# ceci est un test pour le rapport, seulement pour les chiens


# calcul_1D.graphique_1D(mat_comp_dog_opt3, croquettes_dog_opt3, False)
# plt.savefig("graphes/1D_opt3.png", dpi=300, bbox_inches='tight')
# calcul_1D.graphique_1D(mat_comp_dog_opt1, croquettes_dog_opt1, False)
# plt.savefig("graphes/1D_opt1.png", dpi=300, bbox_inches='tight')

# calcul_graphique.graphique_IPM(mat_comp_dog_opt3, method="trust-constr", reverse_v1=False, reverse_v2=True, labels=croquettes_dog_opt3, affichage=False)
# plt.savefig("graphes/2D_opt3.png", dpi=300, bbox_inches='tight')

# calcul_graphique.graphique_IPM(mat_comp_dog_opt1, method="trust-constr", reverse_v1=False, reverse_v2=True, labels=croquettes_dog_opt1, affichage=False)
# plt.savefig("graphes/2D_opt1.png", dpi=300, bbox_inches='tight')