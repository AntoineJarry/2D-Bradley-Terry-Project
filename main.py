import numpy as np
import First_part_project.Bradley_Terry_Model_2D.calcul_affichage as calcul_affichage
import First_part_project.Bradley_Terry_model_1D.calcul_affichage as calcul_1D

N = np.array([
  [0, 39, 64, 40, 61, 76, 46],
  [61, 0, 65, 59, 55, 85, 60],
  [36, 35, 0, 31, 25, 41, 35],
  [60, 41, 69, 0, 41, 80, 28],
  [39, 45, 75, 59, 0, 71, 37],
  [24, 15, 59, 20, 29, 0, 18],
  [54, 40, 65, 72, 63, 82, 0]])

#calcul_1D.graphique_1D(N)

#calcul_affichage.graphique_2D(N)

labels = ['COMPLÈTEMENT', 'SOUS', 'ORTILLION', "L'ALCOOL", "L'EMPRISE DE", 'BAPTISTE', 'EST']
#calcul_affichage.ellipses(N,labels)

#print(calcul_affichage.deviances(N))