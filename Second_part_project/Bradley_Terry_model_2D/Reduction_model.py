import numpy as np

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
