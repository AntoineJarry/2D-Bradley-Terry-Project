import matplotlib.pyplot as plt
import numpy as np
import Second_part_project.Bradley_Terry_model_1D.IPM_algorithm as functions


def graphique_1D_IPM(N, labels, affichage=True):
    n = len(labels)
    lam0 = np.random.uniform(-5, 5, size=(n))
    result = functions.IPM_1D(N, lam0, method='trust-constr')

    lambda_x = result.x
    lambda_y = np.zeros(len(N))
    lambda_y_lab = np.zeros(len(N))

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(lambda_x, lambda_y, color='blue', alpha=0.4)

    # Annotating points with labels
    for i, label in enumerate(labels):
        plt.text(lambda_x[i] + 0.01, lambda_y_lab[i] + 0.007, label, fontsize=12)

    # Add titles and labels
    plt.title('1D Bradley-Terry Model Representation with IPM')
    plt.xlabel('$\lambda$ (Dimension 1)')

    # Add grid and display plot
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    if affichage:
        plt.show()
