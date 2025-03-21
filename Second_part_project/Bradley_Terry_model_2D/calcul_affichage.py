import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import chi2
from matplotlib.patches import Ellipse

# Récupérer le chemin absolu du dossier racine du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Ajouter ce chemin à sys.path
sys.path.append(project_root)

import First_part_project.Bradley_Terry_Model_2D.NR_algorithm.starting_point as starting_point
import First_part_project.Bradley_Terry_Model_2D.calcul_affichage as calcul_affichage
import First_part_project.Bradley_Terry_Model_2D.fonctions as fonctions
import First_part_project.Bradley_Terry_model_1D.Algorithm.functions as functions

import Second_part_project.Bradley_Terry_model_2D.IPM_algorithm as IPM_algorithm



def graphique_IPM(N, method, reverse_v1, reverse_v2, labels, affichage=True):
    lambda_0 = starting_point.starting_point(N, reverse_v1, reverse_v2)
    lambda_0.reshape(2*len(N),1)
    a_0 = np.zeros((3, 1))
    res = IPM_algorithm.IPM_algorithm(N, a0=a_0, lam0=lambda_0, method=method)

    # Extraire les paramètres optimaux depuis `res`
    n = len(N)  # Taille de la matrice N
    optimized_params = res.x  # Prendre les valeurs optimisées
    optimal_lambda = optimized_params[:-3].reshape(n, 2)  # Reshape en (n,2)
    optimal_a = optimized_params[-3:]
    num_params = len(res.x)
    print("Optimal lambda (10 x 2):\n", optimal_lambda)
    print("Optimal a (3 values):\n", optimal_a)
    print("Maximum log-likelihood:", -res.fun)
    print("Nombre de paramètres du modèle M3 :", num_params)

    # Annotating points with labels
    for i, label in enumerate(labels):
        plt.text(-optimal_lambda[:, 1][i] + 0.02, -optimal_lambda[:, 0][i] + 0.02, label, fontsize=12)
    # Tracer optimal_lambda[:, 0] contre optimal_lambda[:, 1]
    plt.scatter(-optimal_lambda[:, 1], -optimal_lambda[:, 0], color='b', label=labels)
    plt.xlabel("Lambda 1")
    plt.ylabel("Lambda 2")
    plt.grid(True)

    if affichage:
        plt.show()

def graphique_IPM_multistart(N, method, labels, num_starts=10, affichage=True):
    log_V, res = IPM_algorithm.IPM_multistart(N,method,num_starts)
    # Extraire les paramètres optimaux depuis `res`
    n = len(N)  # Taille de la matrice N
    optimized_params = res.x  # Prendre les valeurs optimisées
    optimal_lambda = optimized_params[:-3].reshape(n, 2)  # Reshape en (n,2)
    optimal_a = optimized_params[-3:]
    num_params = len(res.x)
    print("Optimal lambda (10 x 2):\n", optimal_lambda)
    print("Optimal a (3 values):\n", optimal_a)
    print("Maximum log-likelihood:", -res.fun)
    print("Nombre de paramètres du modèle M3 :", num_params)

    # Annotating points with labels
    for i, label in enumerate(labels):
        plt.text(optimal_lambda[:, 1][i] + 0.02, -optimal_lambda[:, 0][i] + 0.02, label, fontsize=12)
    # Tracer optimal_lambda[:, 0] contre optimal_lambda[:, 1]
    plt.scatter(optimal_lambda[:, 1], -optimal_lambda[:, 0], color='b', label=labels)
    plt.xlabel("Lambda 1")
    plt.ylabel("Lambda 2")
    plt.title("Optimized Lambda Values")
    plt.grid(True)

    if affichage:
        plt.show()


def vraisemblance_NR_IPM(N, method, reverse_v1, reverse_v2):
    # Calcul de la vraisemblance de IPM
    lambda_0 = starting_point.starting_point(N, reverse_v1, reverse_v2)
    a_0 = np.zeros((3, 1))
    n = int(len(N))
    result = IPM_algorithm.IPM_algorithm(N, a0=a_0, lam0=lambda_0, method=method)
    logv_ipm = -result.fun

    # Calcul de la vraisemblance de Newton-Raphson
    param_estim, mat_cov_var = calcul_affichage.calcul_lambda(N, reverse_v1, reverse_v2)
    lambd = param_estim[0:2*n, 0]  # Lambda 2D
    a = param_estim[2*n:2*n+3, 0]  # Coef de Lagrange a
    logv_NR = calcul_affichage.log_Vraisemblance_mod_1(N, lambd, a)

    print("Log-vraisemblance Maximum IPM:", logv_ipm)
    print("Log-vraisemblance Newton-Raphson:", logv_NR)

def deviances(N,method,reverse_v1,reverse_v2):
    # Récupération des paramètres Lambda
    n = len(N)
    lambda_ = functions.bradley_terry_iterative(N)

    # Calcul des log-V des modèles
    log_v_M0 = calcul_affichage.log_vraisemblance_M0(N)
    log_v_max = calcul_affichage.log_vraisemblance_max(N)
    log_v_M1 = calcul_affichage.log_vraisemblance_M1(N, lambda_)

    lambda_0 = starting_point.starting_point(N, reverse_v1, reverse_v2)
    lambda_0.reshape(2*len(N),1)
    a_0 = np.zeros((3, 1))
    res = IPM_algorithm.IPM_algorithm(N, a0=a_0, lam0=lambda_0, method=method)

    log_v_mod_1 = -res.fun

    # Calcul des déviances et du nombre de paramètres de chaque modèle
    D0 = 2*np.absolute(log_v_M0 - log_v_max)
    D1 = 2*np.absolute(log_v_M0 - log_v_M1)
    D2 = 2*np.absolute(log_v_M1 - log_v_mod_1)
    D_residual = 2*np.absolute(log_v_mod_1 - log_v_max)
    n_param_0 = n*(n-1)/2
    n_param_1 = n-1
    n_param_2 = n-2
    n_param_max = n_param_0 - (n_param_1+n_param_2)

    # Calcul des p-valeurs
    G2_1 = D0 - D1  # Différence de déviance
    df_1 = n_param_0 - n_param_1     # Différence de degrés de liberté
    G2_2 = D1 - D2  # Différence de déviance
    df_2 = n_param_1 - n_param_2     # Différence de degrés de liberté (vaut 1)
    # Calculer la p-valeur
    p_valeur_1 = 1 - chi2.cdf(G2_1, df_1)
    p_valeur_2 = 1 - chi2.cdf(G2_2, df_2)

    return f"Modèle nul : Log-V = {log_v_M0}. Déviance = {D0}. Nombre de paramètres = {n_param_0}\nModèle en 1D : Log-V = {log_v_M1}. Déviance = {D1}. Nombre de paramètres = {n_param_1}. p-valeur = {p_valeur_1} \nModèle en 2D : Log-V = {log_v_mod_1}. Déviance = {D2}. Nombre de paramètres = {n_param_2}. p-valeur = {p_valeur_2}\nModel résiduel : Log-V = {log_v_max}. Déviance = {D_residual}. Nombre de paramètres = {n_param_max}"


def ellipses_IPM(N, method, reverse_v1, reverse_v2, labels, affichage=True):
    lambda_0 = starting_point.starting_point(N, reverse_v1, reverse_v2)
    a_0 = np.zeros((3, 1))
    res = IPM_algorithm.IPM_algorithm(N, a0=a_0, lam0=lambda_0, method=method)

    # Extraire les paramètres optimaux depuis `res`
    n = len(N)  # Taille de la matrice N
    optimized_params = res.x  # Prendre les valeurs optimisées
    optimal_lambda = optimized_params[:-3].reshape(n,2)[:, ::-1]  # Swap columns
    optimal_lambda = optimal_lambda.T.flatten()[:, np.newaxis]  # Ensures column vector
    #print("optimal_lambda :",optimal_lambda)
    #print("optimized_params :",optimized_params)

    # Calcul mat_cov_var Option 1 :
    matrice = np.block([[np.zeros((n,n)),np.eye(n)],
                       [np.eye(n),np.zeros((n,n))]]) # matrice taille 2n carré
    mat_cov_var = fonctions.extract_submatrix(np.linalg.inv(np.block([[-fonctions.second_derivative_L_star(N,optimal_lambda[0:2*n])-optimal_lambda[-1,]*matrice, fonctions.d_phi(optimal_lambda[0:2*n])],
                                                             [np.transpose(fonctions.d_phi(optimal_lambda[0:2*n])), np.zeros((3,3))]])), n) # moins en haut à gauche propt silvey

    # optimal_lambda = optimized_params[:-3].reshape(2*n, 1)
    lambda_1 = -optimal_lambda[0:n, 0]  # Coordonnées X
    lambda_2 = -optimal_lambda[n:2*n, 0]  # Coordonnées Y

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

        if reverse_v1 == True and reverse_v2 == True:
            angle = np.degrees(np.arctan2(-eigenvectors[1, 0], -eigenvectors[0, 0]))
        elif reverse_v1 == True and reverse_v2 == False:
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], -eigenvectors[0, 0]))
        elif reverse_v1 == False and reverse_v2 == True:
            angle = np.degrees(np.arctan2(-eigenvectors[1, 0], eigenvectors[0, 0]))
        else:
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
    
    ax.set_xticks(ax.get_xticks()[::-1])
    # Grille et autres éléments de style
    ax.grid(True, color='black')  # Grille noire
    plt.show()


