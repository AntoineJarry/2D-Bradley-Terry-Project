# import matplotlib.pyplot as plt
import Data.dog_matrix as dog
import Data.cat_matrix as cat

# import 1D
import First_part_project.Bradley_Terry_model_1D.calcul_affichage as calcul_1D
import Second_part_project.Bradley_Terry_model_1D.calcul_affichage as calcul_1D_IPM


# import 2D
import First_part_project.Bradley_Terry_Model_2D.calcul_affichage as calcul_affichage
import Second_part_project.Bradley_Terry_model_2D.calcul_affichage as calcul_graphique


##########
# PARTIE 1
##########

# calcul_1D.graphique_1D(dog.mat_comp_dog_opt3, dog.croquettes_dog_opt3)

# calcul_affichage.graphique_2D(dog.mat_comp_dog_opt3, dog.croquettes_dog_opt3, reverse_v1=False, reverse_v2=True)

# calcul_affichage.ellipses(N_cat_1,labels_cat_1,reverse_v1=False,reverse_v2=True)

# print(calcul_affichage.deviances(mat_comp_dog_opt1,reverse_v1=False,reverse_v2=True))

##########
# PARTIE 2
##########

# calcul_graphique.graphique_IPM(mat_comp_dog_opt3, method="trust-constr", reverse_v1=False, reverse_v2=True, labels=croquettes_dog_opt3)

# Attention, le test de déviance ne permet que de comparer des modèles emboîtés, ce n'est pas le cas entre IPM et NR
# --> utiliser directement la vraisemblance.
# calcul_graphique.vraisemblance_NR_IPM(mat_comp_dog_opt3,method="trust-constr",reverse_v1=False,reverse_v2=True)


##########
# PARTIE X
##########

# ceci est un test pour le rapport, seulement pour les chiens


### 1D ###

# calcul_1D.graphique_1D(mat_comp_dog_opt3, croquettes_dog_opt3, False)
# plt.savefig("graphes/1D_opt3.png", dpi=300, bbox_inches='tight')
# calcul_1D.graphique_1D(mat_comp_dog_opt1, croquettes_dog_opt1, False)
# plt.savefig("graphes/1D_opt1.png", dpi=300, bbox_inches='tight')

# calcul_1D_IPM.graphique_1D_IPM(dog.mat_comp_dog_opt3, dog.croquettes_dog_opt3)


### 2D ###

# calcul_graphique.graphique_IPM(dog.mat_comp_dog_opt3, method="trust-constr", reverse_v1=False, reverse_v2=True, labels=dog.croquettes_dog_opt3, affichage=True)
# plt.savefig("graphes/2D_opt3.png", dpi=300, bbox_inches='tight')

# calcul_graphique.graphique_IPM(dog.mat_comp_dog_opt1, method="trust-constr", reverse_v1=False, reverse_v2=True, labels=dog.croquettes_dog_opt1, affichage=True)
# plt.savefig("graphes/2D_opt1.png", dpi=300, bbox_inches='tight')


# Chats
calcul_1D_IPM.graphique_1D_IPM(cat.mat_comp_cat_opt1, cat.croquettes_cat_opt1)
calcul_graphique.graphique_IPM(cat.mat_comp_cat_opt1, method="trust-constr", reverse_v1=False, reverse_v2=True, labels=cat.croquettes_cat_opt1, affichage=True)
