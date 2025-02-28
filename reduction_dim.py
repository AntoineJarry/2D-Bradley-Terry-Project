import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, Isomap, TSNE
import umap
from sklearn.decomposition import PCA
import First_part_project.Bradley_Terry_Model_2D.NR_algorithm.starting_point as starting_point
import Second_part_project.Bradley_Terry_model_2D.Reduction_model as Reduction_model

N = np.array([
  [0, 39, 64, 40, 61, 76, 46],
  [61, 0, 65, 59, 55, 85, 60],
  [36, 35, 0, 31, 25, 41, 35],
  [60, 41, 69, 0, 41, 80, 28],
  [39, 45, 75, 59, 0, 71, 37],
  [24, 15, 59, 20, 29, 0, 18],
  [54, 40, 65, 72, 63, 82, 0]])

Score_cumules = np.array([
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

dist_matrix = starting_point.matrice_Q(Score_cumules) # Our distance matrix is defined as in the first part of the project
## Maybe later we can try to used another distance because is probably not the best here ?
Reduction_model.check_distance_matrix(dist_matrix) ## Important to test this and look if we have any problem
"""
# 1. UMAP
reducer = umap.UMAP(n_components=2, metric='precomputed',min_dist=0.5,n_neighbors = 3) # Hyper paramètre a tester
X_umap = reducer.fit_transform(dist_matrix)

# 2. t-SNE
tsne = TSNE(n_components=2, metric="precomputed",perplexity=5,init='random') # Hyper paramètre à ajuster aussi
X_tsne = tsne.fit_transform(dist_matrix)

# 3. PCA not useful here because PCA need features here. So we will keep four method that we will explain when presenting
"""
# 4. Isomap
isomap = Isomap(n_components=2,n_neighbors=2) # Adjusted parameter 
X_isomap = isomap.fit_transform(dist_matrix)

"""
#6. classical MDS
mds = starting_point.starting_point(Score_cumules,reverse_v1=False,reverse_v2=False)

#7. Non metric MDS 

Nmds = MDS(n_components=2, dissimilarity="precomputed",metric=False,n_init = 50,n_jobs = -1,normalized_stress=True) # For compare to our starting point which is a MDS 
X_Nmds = Nmds.fit_transform(dist_matrix)
"""
teams = np.array(["BRO", "DK", "DRX", "FOX", "GEN", "HLE", "KDF", "KT", "NS", "T1"])

# Plotting the results with labels
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

"""
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
"""

# Isomap plot
axs[1, 0].scatter(X_isomap[:, 0], X_isomap[:, 1])
for i, label in enumerate(teams):
    axs[1, 0].text(X_isomap[i, 0], X_isomap[i, 1], label, fontsize=9)
axs[1, 0].set_title("Isomap - 2D")
"""
# MDS plot
mds = np.hstack((mds[0:len(Score_cumules)], mds[len(Score_cumules):]))
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

axs[2, 1].scatter(X_Nmds[:, 0], X_Nmds[:, 1])
for i, label in enumerate(teams):
    axs[2, 1].text(X_Nmds[i, 0], X_Nmds[i, 1], label, fontsize=9)
axs[2, 1].set_title(" Non-Metric MDS - 2D")
"""

plt.tight_layout()
plt.show()