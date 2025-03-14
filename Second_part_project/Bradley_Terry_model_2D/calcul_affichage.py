import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import chi2

# Récupérer le chemin absolu du dossier racine du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Ajouter ce chemin à sys.path
sys.path.append(project_root)

import First_part_project.Bradley_Terry_Model_2D.NR_algorithm.starting_point as starting_point
import First_part_project.Bradley_Terry_Model_2D.calcul_affichage as calcul_affichage
import Second_part_project.Bradley_Terry_model_2D.IPM_algorithm as IPM_algorithm


def graphique_IPM(N, method, reverse_v1, reverse_v2, labels, affichage=True):
    lambda_0 = starting_point.starting_point(N, reverse_v1, reverse_v2)
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
