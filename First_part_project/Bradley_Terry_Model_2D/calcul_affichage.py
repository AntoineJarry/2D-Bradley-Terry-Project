from math import log
import numpy as np
import numpy.linalg as lng
import matplotlib.pyplot as plt
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from math import factorial, comb

import First_part_project.Bradley_Terry_Model_2D.NR_algorithm.starting_point as starting_point
import First_part_project.Bradley_Terry_Model_2D.NR_algorithm.NR_algo as NR_algo
import First_part_project.Bradley_Terry_Model_2D.fonctions as fonctions
import First_part_project.Bradley_Terry_model_1D.Algorithm.functions as functions

def calcul_lambda(N,reverse_v1,reverse_v2):
    # lambda_0 = np.random.rand(14, 1)  # Vecteur colonne avec des valeurs aléatoires entre 0 et 1 ne converge pas
    lambda_0 = starting_point.starting_point(N,reverse_v1,reverse_v2) # vecteur colonne début censé être cool
    #print(lambda_0) ## selon la page 248 lambda général comme ça (vecteur colonne)
    a_0 = np.zeros((3,1))  ## selon la page 250 définie comme ça (vecteur colonne) + p252 dit choix arbitraire pas de soucis sur la convergence (à vérifier)
    param_estim , mat_cov_var = NR_algo.newton_raphson(N,lambda_0, a_0)
    return param_estim, mat_cov_var
    #print(param_estim)

def graphique_2D(N,labels,reverse_v1,reverse_v2):

    param_estim, mat_cov_var = calcul_lambda(N,reverse_v1,reverse_v2)
    n = len(N)
    # Sample data (coordinates of items in two dimensions)
    # Replace these with actual model values
    lambda_1 = param_estim[0:n,]  # X-coordinates
    lambda_2 = param_estim[n:2*n,]  # Y-coordinates

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(lambda_1, lambda_2, color='blue')

    # Annotating points with labels
    for i, label in enumerate(labels):
        plt.text(lambda_1[i] + 0.02, lambda_2[i] + 0.02, label, fontsize=12)

    # Add titles and labels
    plt.title('2D Bradley-Terry Model Representation')
    plt.xlabel('$\lambda_1$ (Dimension 1)')
    plt.ylabel('$\lambda_2$ (Dimension 2)')

    # Add grid and display plot
    plt.grid(True)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.show()

def ellipses(N,labels,reverse_v1,reverse_v2):

    param_estim, mat_cov_var = calcul_lambda(N,reverse_v1,reverse_v2)
    n = len(N)
    lambda_1 = param_estim[0:n, 0]  # Coordonnées X
    lambda_2 = param_estim[n:2*n, 0]  # Coordonnées Y

    # Créer les paires d'indices (1, 8), (2, 9), ..., (7, 14)
    idx_pairs = [[i, i + n] for i in range(n)]

    # Seuil chi2 pour un intervalle de confiance de 95% et 2 degrés de liberté
    chi2_val = chi2.ppf(0.95, df=2)

    # Créer le graphique pour les points et ellipses
    fig, ax = plt.subplots(figsize=(8, 6))

    # Affichage des points
    ax.scatter(lambda_1, lambda_2, color='blue')

    # Annoter les points avec les labels
    for i, label in enumerate(labels):
        ax.text(lambda_1[i] + 0.02, lambda_2[i] + 0.02, label, fontsize=12)

    # Itérer sur chaque paire d'indices pour ajouter les ellipses
    for i, idx in enumerate(idx_pairs):
        cov_2x2 = mat_cov_var[np.ix_(idx, idx)]

        # Calculer les valeurs propres et vecteurs propres de la matrice de covariance
        eigenvalues, eigenvectors = np.linalg.eigh(cov_2x2)

        # Calculer la longueur des axes de l'ellipse
        axis_lengths = np.sqrt(chi2_val * eigenvalues)
        
        if reverse_v1==True and reverse_v2==True :
            angle = np.degrees(np.arctan2(-eigenvectors[1, 0], -eigenvectors[0, 0]))
        elif reverse_v1 == True and reverse_v2 == False : 
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], -eigenvectors[0, 0])) 
        elif reverse_v1 == False and reverse_v2 == True : 
            angle = np.degrees(np.arctan2(-eigenvectors[1, 0], eigenvectors[0, 0]))
        else :
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))# Calculer l'angle de rotation de l'ellipse
        

        # Extraire les estimations correspondantes (moyennes)
        mean_2d = np.vstack((lambda_1[idx[0]], lambda_2[idx[0]]))

        # Ajouter l'ellipse au graphique
        ellipse = Ellipse(
            xy=mean_2d,
            width=2 * axis_lengths[0],  # Largeur = 2 * écart-type sur l'axe principal
            height=2 * axis_lengths[1],  # Hauteur = 2 * écart-type sur l'axe secondaire
            angle=angle,
            edgecolor='black',  # Couleur des bords
            facecolor='none',  # Pas de couleur de remplissage
            linewidth=1.5,
            label=f'Paire {i+1}'
        )
        ax.add_patch(ellipse)


    # Ajuster l'affichage
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Ligne horizontale noire
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Ligne verticale noire

    # Définir les limites des axes
    ax.set_xlim(-1.5, 1.5)  # Limite de l'axe des X de -1.5 à 1.5
    ax.set_ylim(-1, 1)      # Limite de l'axe des Y de -1 à 1

    # Labels et titre
    ax.set_xlabel('$\lambda_1$', fontsize=12, color='black')
    ax.set_ylabel('$\lambda_2$', fontsize=12, color='black')
    ax.set_title('2D Bradley-Terry Model avec Ellipses de Confiance', fontsize=14, color='black')

    # Grille et autres éléments de style
    ax.grid(True, color='black')  # Grille noire
    plt.show()

def calcul_pi_ij(N,lambd, i, j):
    """
    Calcule pi_ij = sigma_ij * sqrt[(lambda_i,1 − lambda_j,1)^2 + (lambda_i,2 − lambda_j,2)^2]

    Arguments :
    - lambd : vecteur np.array de taille (2*n,), contient les coordonnées lambda sous la forme [lambda_1,1, lambda_1,2, ..., lambda_n,1, lambda_n,2].
    - i, j : index du vecteur lambda.

    Retourne :
    - pi_ij : Un entier.
    """
    n = int(len(lambd) / 2)
    lamb = np.hstack((lambd[0:n,], lambd[n:2*n,]))
    pi_ij = fonctions.inv_logit(fonctions.logit_pi_ij(N,lamb,i,j))
    return pi_ij

def log_vraisemblance_max(N):
    log_Vraisemblance = 0
    for i in range(N.shape[0]):
        for j in range(i+1, N.shape[1]):
          if j != i:
              nij = N[i, j]
              mij = N[i, j] + N[j, i]
              pi_ij = nij/mij
              log_Vraisemblance += np.log(factorial(mij)/(factorial(mij-nij)*factorial(nij))) + nij * np.log(pi_ij) + (mij - nij) * np.log(1 - pi_ij)
    return log_Vraisemblance

def log_Vraisemblance_mod_1(N, lambd, a): # lambd : lambda 2D
    """
    Calcule la log-vraisemblance à partir des estimations des paramètres.
    """
    #param_estim, mat_cov_var = calcul_lambda(N)
    #n = len(N)
    #lambd = param_estim[0:2*n, 0]  # Lambda
    #a = param_estim[2*n:2*n+3, 0]  # Coef de Lagrange a
    log_Vraisemblance = 0
    for i in range(N.shape[0]):
        for j in range(i+1, N.shape[1]):
          if j != i:
              nij = N[i, j]
              mij = N[i, j] + N[j, i]
              pi_ij = calcul_pi_ij(N,lambd, i, j)
              log_Vraisemblance += np.log(factorial(mij)/(factorial(mij-nij)*factorial(nij))) + nij * np.log(pi_ij) + (mij - nij) * np.log(1 - pi_ij)
    log_Vraisemblance += np.sum(a * fonctions.phi(lambd))
    return log_Vraisemblance

def log_vraisemblance_M0(N):
    log_L = 0
    for i in range(N.shape[0]):
        for j in range(i+1, N.shape[1]):
            nij = N[i, j]
            mij = N[i, j] + N[j, i]
            # Calcul de la log-vraisemblance pour pi_ij = 0.5
            log_L += np.log(factorial(mij)/(factorial(mij-nij)*factorial(nij))) + mij * np.log(0.5)
    return log_L

def log_vraisemblance_M1(N, lambd): #lambd : lambda 1D
    log_L = 0
    for i in range(N.shape[0]):
        for j in range(i+1, N.shape[1]):
            nij = N[i, j]
            mij = N[i, j] + N[j, i]
            # Calcul de pi_ij pour M1
            diff_lambda = abs(lambd[i] - lambd[j])
            sigma_ij = fonctions.estim_sigma(N, i, j)
            logit_pi_ij = sigma_ij * diff_lambda
            pi_ij = fonctions.inv_logit(logit_pi_ij)

            # Log-vraisemblance
            log_L += np.log(factorial(mij)/(factorial(mij-nij)*factorial(nij))) + nij * np.log(pi_ij) + (mij - nij) * np.log(1 - pi_ij)
    return log_L

def deviances(N):
    param_estim, mat_cov_var = calcul_lambda(N)
    n = int(len(N))
    lambda_ = functions.bradley_terry_iterative(N)
    lambd = param_estim[0:2*n, 0]  # Lambda 2D
    a = param_estim[2*n:2*n+3, 0]  # Coef de Lagrange a

    D0 = -2*(log_vraisemblance_M0(N) - log_vraisemblance_max(N))
    D1 = -2*(log_vraisemblance_M0(N) - log_vraisemblance_M1(N, lambda_))
    D2 = -2*(log_vraisemblance_M1(N, lambda_) - log_Vraisemblance_mod_1(N, lambd, a))
    D_residual = -2*(log_Vraisemblance_mod_1(N, lambd, a) - log_vraisemblance_max(N))

    return f"Déviance du modèle nul : {D0}\n Déviance du modèle en 1D : {D1}\n Déviance du modèle en 2D : {D2}\n Déviance résiduelle : {D_residual}"