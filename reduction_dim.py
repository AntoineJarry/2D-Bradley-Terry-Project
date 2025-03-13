import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, Isomap, TSNE
import umap
import pickle
from sklearn.decomposition import PCA
import First_part_project.Bradley_Terry_Model_2D.NR_algorithm.starting_point as starting_point
import Second_part_project.Bradley_Terry_model_2D.Reduction_model as Reduction_model

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

dist_matrix = starting_point.matrice_Q(mat_comp_dog_opt3) # Our distance matrix is defined as in the first part of the project
## Maybe later we can try to used another distance because is probably not the best here ?
Reduction_model.check_distance_matrix(dist_matrix) ## Important to test this and look if we have any problem

# 1. UMAP
reducer = umap.UMAP(n_components=2, metric='precomputed',min_dist=0.6,n_neighbors = 6) # Hyper paramètre a tester
X_umap = reducer.fit_transform(dist_matrix)

# 2. t-SNE
tsne = TSNE(n_components=2, metric="precomputed",perplexity=6,init='random',method = 'exact') # Hyper paramètre à ajuster aussi
X_tsne = tsne.fit_transform(dist_matrix)

# 3. PCA not useful here because PCA need features here. So we will keep four method that we will explain when presenting

#5. Metric MDS 
Mmds = MDS(n_components=2, dissimilarity="precomputed",metric=True,n_init = 50,n_jobs = -1,normalized_stress=False) # For compare to our starting point which is a MDS 
X_Mmds = Mmds.fit_transform(dist_matrix)


#6. classical MDS
mds = starting_point.starting_point(mat_comp_dog_opt3,reverse_v1=False,reverse_v2=False)


#7. Non metric MDS 

Nmds = MDS(n_components=2, dissimilarity="precomputed",metric=False,n_init = 50,n_jobs = -1,normalized_stress=True) # For compare to our starting point which is a MDS 
X_Nmds = Nmds.fit_transform(dist_matrix)

teams = croquettes_dog_opt3

# Plotting the results with labels
fig, axs = plt.subplots(3, 2, figsize=(10, 10))


# UMAP plot
axs[0, 0].scatter(X_umap[:, 0], X_umap[:, 1])
for i, label in enumerate(teams):
    axs[0, 0].text(X_umap[i, 0], X_umap[i, 1], label, fontsize=9)
axs[0, 0].set_title("UMAP - 2D")

# t-SNE plot
axs[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1])
for i, label in enumerate(teams):
    axs[0, 1].text(X_tsne[i, 0], X_tsne[i, 1], label, fontsize=9)
axs[0, 1].set_title("t-SNE - 2D")



# MDS plot
mds = np.hstack((mds[0:len(mat_comp_dog_opt3)], mds[len(mat_comp_dog_opt3):]))
axs[1, 1].scatter(mds[:, 0], mds[:, 1])
for i, label in enumerate(teams):
    axs[1, 1].text(mds[i, 0], mds[i, 1], label, fontsize=9)
axs[1, 1].set_title("MDS - 2D")

# Metric MDS plot

axs[2, 0].scatter(X_Mmds[:, 0], X_Mmds[:, 1])
for i, label in enumerate(teams):
    axs[2, 0].text(X_Mmds[i, 0], X_Mmds[i, 1], label, fontsize=9)
axs[2, 0].set_title(" Metric MDS - 2D")

# Non metric MDS plot

axs[2, 1].scatter(-X_Nmds[:, 0], X_Nmds[:, 1])
for i, label in enumerate(teams):
    axs[2, 1].text(-X_Nmds[i, 0], X_Nmds[i, 1], label, fontsize=9)
axs[2, 1].set_title(" Non-Metric MDS - 2D")

plt.tight_layout()
plt.show()