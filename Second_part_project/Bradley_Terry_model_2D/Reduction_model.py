import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, Isomap, TSNE
import umap
from sklearn.decomposition import PCA
from First_part_project.Bradley_Terry_Model_2D.fonctions import matrice_Q

def check_distance_matrix(D, tol=1e-6):
    """
    Vérifie si la matrice D est une matrice de distance valide pour MDS.
    
    Paramètres:
    - D : np.array (matrice NxN) supposée être une matrice de distance
    - tol : Tolérance numérique pour vérifier les égalités
    
    Retourne:
    - True si la matrice est une distance valide, False sinon.
    """
    N = D.shape[0]
    
    # 1. Vérifier si la matrice est carrée
    if D.shape[0] != D.shape[1]:
        print("❌ Erreur : La matrice n'est pas carrée.")
        return False
    
    # 2. Vérifier la symétrie : D[i, j] == D[j, i]
    if not np.allclose(D, D.T, atol=tol):
        print("❌ Erreur : La matrice n'est pas symétrique.")
        return False
    
    # 3. Vérifier la diagonale : D[i, i] == 0
    if not np.allclose(np.diag(D), 0, atol=tol):
        print("❌ Erreur : La diagonale de la matrice n'est pas nulle.")
        return False
    
    # 4. Vérifier que toutes les distances sont positives ou nulles
    if np.any(D < -tol):  # Tolérance pour éviter erreurs numériques
        print("❌ Erreur : Certaines distances sont négatives.")
        return False
    
    # 5. Vérifier l’inégalité triangulaire : D[i, j] ≤ D[i, k] + D[k, j]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                if D[i, j] > D[i, k] + D[k, j] + tol:
                    print(f"❌ Erreur : L'inégalité triangulaire est violée pour (i={i}, j={j}, k={k}).")
                    return False
    
    print("✅ La matrice est une matrice de distance valide pour MDS.")
    return True

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

dist_matrix = matrice_Q(Score_cumules) # Our distance matrix is defined as in the first part of the project
## Maybe later we can try to used another distance because is probably not the best here ? 

check_distance_matrix(dist_matrix) ## Important to test this and look if we have any problem

# 1. UMAP
reducer = umap.UMAP(n_components=2, metric='precomputed',min_dist=0.5,n_neighbors = 3) # Hyper paramètre a tester
X_umap = reducer.fit_transform(dist_matrix)

# 2. t-SNE
tsne = TSNE(n_components=2, metric="precomputed",perplexity=5,init='random') # Hyper paramètre à ajuster aussi
X_tsne = tsne.fit_transform(dist_matrix)

# 3. PCA not useful here because PCA need features here. So we will keep four method that we will explain when presenting

# 4. Isomap
isomap = Isomap(n_components=2,n_neighbors=2) # Adjusted parameter 
X_isomap = isomap.fit_transform(dist_matrix)

# 5. MDS
mds = MDS(n_components=2, dissimilarity="precomputed") # For compare to our starting point which is a MDS 
X_mds = mds.fit_transform(dist_matrix)

teams = np.array(["BRO", "DK", "DRX", "FOX", "GEN", "HLE", "KDF", "KT", "NS", "T1"])

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

# PCA plot
axs[1, 0].scatter(X_pca[:, 0], X_pca[:, 1])
for i, label in enumerate(teams):
    axs[1, 0].text(X_pca[i, 0], X_pca[i, 1], label, fontsize=9)
axs[1, 0].set_title("PCA - 2D")

# Isomap plot
axs[1, 1].scatter(X_isomap[:, 0], X_isomap[:, 1])
for i, label in enumerate(teams):
    axs[1, 1].text(X_isomap[i, 0], X_isomap[i, 1], label, fontsize=9)
axs[1, 1].set_title("Isomap - 2D")

# MDS plot

axs[2, 1].scatter(X_mds[:, 0], X_mds[:, 1])
for i, label in enumerate(teams):
    axs[2, 1].text(X_mds[i, 0], X_mds[i, 1], label, fontsize=9)
axs[2, 1].set_title("MDS - 2D")


plt.tight_layout()
plt.show()
