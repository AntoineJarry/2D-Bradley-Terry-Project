from math import log
import numpy as np
import numpy.linalg as lng
import matplotlib.pyplot as plt

import Implementation_py.TwoD_model.NR_algorihm.starting_point as starting_point
import Implementation_py.TwoD_model.NR_algorihm.NR_algo as NR_algo

def calcul_lambda(N):
    # lambda_0 = np.random.rand(14, 1)  # Vecteur colonne avec des valeurs aléatoires entre 0 et 1 ne converge pas
    lambda_0 = starting_point.starting_point(N) # vecteur colonne début censé être cool
    #print(lambda_0) ## selon la page 248 lambda général comme ça (vecteur colonne)
    a_0 = np.zeros((3,1))  ## selon la page 250 définie comme ça (vecteur colonne) + p252 dit choix arbitraire pas de soucis sur la convergence (à vérifier)
    param_estim , mat_cov_var = NR_algo.newton_raphson(N,lambda_0, a_0)
    return param_estim
    #print(param_estim)

def graphique_2D(N):
    param_estim = calcul_lambda(N)
    # Sample data (coordinates of items in two dimensions)
    # Replace these with actual model values
    labels = ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5','Item 6','Item 7']
    lambda_1 = param_estim[0:7,]  # X-coordinates
    lambda_2 = param_estim[7:14,]  # Y-coordinates

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