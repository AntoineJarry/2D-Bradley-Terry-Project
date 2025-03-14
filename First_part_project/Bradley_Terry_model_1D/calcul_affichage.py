import matplotlib.pyplot as plt
import numpy as np
import First_part_project.Bradley_Terry_model_1D.Algorithm.functions as functions


def graphique_1D(N, labels, affichage=True):
    # Sample data (coordinates of items in two dimensions)
    # Replace these with actual model values
    lambda_ = functions.bradley_terry_iterative(N)
    lambda_y = np.zeros(len(N))

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(lambda_, lambda_y, color='blue', alpha=0.4)

    lambda_y_lab = np.zeros(len(N))
    # Annotating points with labels
    for i, label in enumerate(labels):
        plt.text(lambda_[i] + 0.01, lambda_y_lab[i] + 0.007, label, fontsize=12)

    # Add titles and labels
    plt.title('1D Bradley-Terry Model Representation')
    plt.xlabel('$\lambda$ (Dimension 1)')

    # Add grid and display plot
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    if affichage:
        plt.show()
