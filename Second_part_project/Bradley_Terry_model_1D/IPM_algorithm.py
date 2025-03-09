import numpy as np
from scipy import minimize
# Objective function: Log-likelihood + penalty terms


def objective(lam, N):
    likelihood = 0
    n = len(N)
    for i in range(n):
        for j in range(n):
            likelihood += N[i, j] * (lam[i] - np.log(np.exp(lam[i])+np.exp(lam[j])))
    return -(likelihood)

# Equality constraint: phi(lambda) = 0


def eq_constraint(lam):
    phi = np.sum(lam)
    return phi


def IPM_1D(N, initial_guess, method): 

    constraints = {'type': 'eq', 'fun': eq_constraint}
    result = minimize(objective, initial_guess, args=(N,), method=method, constraints=constraints)
    # Results
    if result.success:
        optimized_lambda = result.x
        print("Optimal lambda :\n", optimized_lambda)
        print("Maximum log-likelihood:", -result.fun)
    else:
        print("Optimization failed:", result.message)

    return result
