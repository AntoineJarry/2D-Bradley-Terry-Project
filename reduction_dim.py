import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.manifold import MDS, TSNE
from scipy.optimize import minimize
import umap
import First_part_project.Bradley_Terry_Model_2D.NR_algorithm.starting_point as starting_point
import Second_part_project.Bradley_Terry_model_2D.Reduction_model as Reduction_model
import First_part_project.Bradley_Terry_Model_2D.NR_algorithm.NR_algo as NR_algo
import Second_part_project.Bradley_Terry_model_2D.IPM_algorithm as IPM_algorithm

np.random.seed(42)  # Global seed for reproducibility

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

dist_matrix = starting_point.matrice_Q(mat_comp_dog_opt3)
Reduction_model.check_distance_matrix(dist_matrix)

methods = []
time_results = {}
sum_results = {}

# UMAP with different n_neighbors
for n_neighbors in [2, 4, 6]:
    start_time = time.time()
    reducer = umap.UMAP(n_components=2, metric='precomputed', init='random', min_dist=0.5,
                        n_neighbors=n_neighbors, n_epochs=5000, random_state=42)
    X_umap = reducer.fit_transform(dist_matrix)
    elapsed_time = time.time() - start_time
    methods.append((f"UMAP (n_neighbors={n_neighbors})", X_umap))
    time_results[f"UMAP-{n_neighbors}"] = elapsed_time
    sum_results[f"UMAP-{n_neighbors}"] = (
        sum(X_umap[:, 0]), sum(X_umap[:, 1]), sum(X_umap[:, 0] * X_umap[:, 1])
    )

# t-SNE with different perplexities
for perplexity in [2, 4, 6]:
    start_time = time.time()
    tsne = TSNE(n_components=2, metric="precomputed", perplexity=perplexity, init='random',
                method='exact', max_iter=5000, random_state=42)
    X_tsne = tsne.fit_transform(dist_matrix)
    elapsed_time = time.time() - start_time
    methods.append((f"t-SNE (perplexity={perplexity})", X_tsne))
    time_results[f"tSNE-{perplexity}"] = elapsed_time
    sum_results[f"tSNE-{perplexity}"] = (
        sum(X_tsne[:, 0]), sum(X_tsne[:, 1]), sum(X_tsne[:, 0] * X_tsne[:, 1])
    )

# Classical MDS
start_time = time.time()
mds = starting_point.starting_point(mat_comp_dog_opt3, reverse_v1=False, reverse_v2=False)
mds = np.hstack((mds[:len(mat_comp_dog_opt3)], mds[len(mat_comp_dog_opt3):]))
elapsed_time = time.time() - start_time
methods.append(("Classical MDS", mds))
time_results["MDS"] = elapsed_time
sum_results["MDS"] = (
    sum(mds[:, 0]), sum(mds[:, 1]), sum(mds[:, 0] * mds[:, 1])
)

# Non-metric MDS
start_time = time.time()
Nmds = MDS(n_components=2, dissimilarity="precomputed", metric=False, n_init=50, n_jobs=1,
           verbose=1, normalized_stress=True, random_state=42)
X_Nmds = Nmds.fit_transform(dist_matrix)
elapsed_time = time.time() - start_time
methods.append(("Non-metric MDS", X_Nmds))
time_results["NonMetric-MDS"] = elapsed_time
sum_results["NonMetric-MDS"] = (
    sum(X_Nmds[:, 0]), sum(X_Nmds[:, 1]), sum(X_Nmds[:, 0] * X_Nmds[:, 1])
)


# Création et sauvegarde d'une image par modèle
for title, data in methods:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(data[:, 0], data[:, 1])
    for j, label in enumerate(croquettes_dog_opt3):
        ax.text(data[j, 0], data[j, 1], label, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.tight_layout()
    filename = f"{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Ferme la figure pour libérer la mémoire

# Affichage des temps de calcul et des sommes
print("Computation times:")
for method, elapsed_time in time_results.items():
    print(f"{method}: {elapsed_time:.4f} seconds")

print("\nSum results:")
for method, sums in sum_results.items():
    print(f"{method}: sum_x={sums[0]:.4f}, sum_y={sums[1]:.4f}, sum_xy={sums[2]:.4f}")

"""""
"NR METHODS"

for title, data in methods:
    # Use vstack to create a 1D column vector (lambda_0) from x and y coordinates
    lambda_0 = np.vstack((data[:, 0], data[:, 1])).flatten()
    lambda_0 = lambda_0.reshape(-1,1)  # Stack x and y values to form lambda_0
    print(lambda_0)
    a_0 = np.zeros((3, 1))  # Initialize a_0 (adjust according to your algorithm requirements)

    # Apply Newton-Raphson algorithm with the flattened data
    param_estim_mds, mat_cov_var_mds = NR_algo.newton_raphson(mat_comp_dog_opt3, lambda_0, a_0)
    
    # Print or store the results as needed
    print(f"Method: {title}, Estimated Params: {param_estim_mds}, Covariance Matrix: {mat_cov_var_mds}")

"""""
time_results = {}  # Dictionary to store computation times
iteration_results = {}  # Dictionary to store number of iterations
param_results = {}  # Dictionary to store estimated parameters
successful_methods = {}  # This dictionary will track if a method was successful

# Loop through methods and use different initial points
for title, data in methods:
    try:
        # Use vstack to create a 1D column vector (lambda_0) from x and y coordinates
        lam_0 = np.vstack((data[:, 0], data[:, 1])).flatten()  # Flatten to 1D vector
        print(f"Initial lambda_0 for {title}: {lam_0}")
        
        a0 = np.zeros((3, 1))  # Initialize a_0 (adjust according to your algorithm requirements)
        initial_guess = np.concatenate([lam_0, a0.flatten()])  # Concatenate x, y, and a0
        
        # Define constraints (adjust based on your requirements)
        constraints = {'type': 'eq', 'fun': IPM_algorithm.eq_constraint, 'args': (mat_comp_dog_opt3,)}
        
        # Track optimization time
        start_time = time.time()

        # Run the optimization
        result = minimize(fun=IPM_algorithm.objective, 
                          x0=initial_guess, 
                          args=(mat_comp_dog_opt3,), 
                          method='trust-constr', 
                          constraints=constraints)
        
        # If the optimization runs successfully, store results
        elapsed_time = time.time() - start_time
        time_results[title] = elapsed_time  # Store time taken
        iteration_results[title] = result.nit  # Number of iterations
        param_results[title] = result.x  # Optimized parameters
        
        # Mark the method as successful
        successful_methods[title] = True
        
        # Print the results for each method
        print(f"Method: {title}")
        print(f"Optimized Params: {result.x}")
        print(f"Computation Time: {elapsed_time:.4f} seconds")
        print(f"Iterations: {result.nit}")
        print(f"Log-vraisemblance : {result.fun}")
        print('-' * 50)
    
    except Exception as e:
        # Handle errors gracefully by printing an error message
        print(f"Method {title} failed with error: {e}")
        print("Skipping to the next method...\n")
        
        # Mark the method as failed
        successful_methods[title] = False
        continue  # Skip the current iteration and continue with the next method

# Print summary of results only for successful methods
print("\nSummary of results:")
for title in successful_methods:
    if successful_methods[title]:  # Only print for successful methods
        print(f"{title} - Time: {time_results[title]:.4f}s, Iterations: {iteration_results[title]}, Params: {param_results[title]}")
