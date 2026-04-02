import numpy as np
import networkx as nx
import random

# Reproducibility
SEED = 0
np.random.seed(SEED)
random.seed(SEED)

def selfconsistent_minimum(E, k, MSM, gamma, cap_k=1, threshold=1e-10, max_repetitions=1000, verbose=False):
    
    # Normalize the capacities in the initial guess:        
    k *= cap_k / (np.sum(k**gamma)**(1/gamma))

    # Iterate until convergence:
    last_change = 1e5
    iterations = 0
    while last_change > threshold:

        # Compute the PTDF matrix via the Graph Laplacian:
        L = E @ np.diag(k) @ E.T                 
        PTDF = np.diag(k) @ E.T @ np.linalg.pinv(L)        

        # Compute the second moments of the flows
        F_sm = PTDF @ MSM @ PTDF.T
        F_squared = np.diag(F_sm)

        # Update the edge capacities a la Corson:
        k_old = k.copy()
        denom = (np.sum(F_squared**(gamma/(1+gamma))))**(1/gamma)
        k = F_squared**(1/(1+gamma))  / denom * cap_k

        # Compute the dissipation:
        D = 0
        for e in range(len(k)):
            if k[e] > 0:
                D += F_squared[e] / k[e]

        # change of edge capacities in this step. Used to check convergence.
        last_change = np.sum((k - k_old)**2)

        # Output to console in verbose mode:
        if verbose:
            print(f'Iteration {iterations}, Last Change {last_change:.4E}')

        # Put out a warning if the max number of iterations has been exceeded:
        iterations += 1
        if iterations > max_repetitions:
            print('Maximum number of repetitions reached!')
            break
        
    return k, D

def random_minimum(E, G, MSM, gamma):

    # Compute the local minima for different initial guesses
    D_min = 1e10
    num_edges = np.shape(E)[1]
    k_min = np.ones(num_edges)
    num_rand = 10
    num_tree = 50

    # All edges present in initial guess
    for counter in range(num_rand):

        # Initial guess:
        k_guess = np.ones(num_edges)
        if counter > 0:
            k_guess = k_guess * np.random.rand(num_edges)

        # Compute local min and keep if appropriate:
        k_localmin, D_localmin = selfconsistent_minimum(E, k_guess, MSM, gamma, verbose=False)
        if D_localmin < D_min:
            D_min = D_localmin
            k_min = k_localmin

    # One edge removed in initial guess
    for e in range(num_edges):
        for counter in range(num_rand):

            # Initial guess:

            k_guess = np.ones(num_edges)
            k_guess[e] = 0
            if counter > 0:
                k_guess = k_guess * np.random.rand(num_edges)

            # Compute local min and keep if appropriate:
            k_localmin, D_localmin = selfconsistent_minimum(E, k_guess, MSM, gamma, verbose=False)
            if D_localmin < D_min:
                D_min = D_localmin
                k_min = k_localmin

    # Spanning trees:
    for counter in range(num_tree):

        # Find a random spanning tree:
        T = nx.random_spanning_tree(G, weight=None, seed = SEED)  

        # set the capacities to be non-zero only on the spanning tree:
        k_guess = np.zeros(num_edges)
        for e in range(num_edges):
            f = np.argmax(E[:, [e]])
            t = np.argmin(E[:, [e]])
            if T.has_edge(f, t):
                k_guess[e] = 1.0 + 0.1 * np.random.rand()

        # Compute local min and keep if appropriate:
        k_localmin, D_localmin = selfconsistent_minimum(E, k_guess, MSM, gamma, verbose=False)
        if D_localmin < D_min:
            D_min = D_localmin
            k_min = k_localmin

    # Use a special spanning tree in the case of the cube:
    if np.shape(E)[1] == 12:
        k_guess = np.array([4.31e-4, 0, 0, 2.46e-4, 3.11e-4, 2.46e-4, 0 , 0, 2.46e-4, 4.31e-4, 2.46e-4, 0])
        k_localmin, D_localmin = selfconsistent_minimum(E, k_guess, MSM, gamma, verbose=False)
        if D_localmin < D_min:
            D_min = D_localmin
            k_min = k_localmin

        k_guess = np.array([4.02466443e-02, 3.68715851e-02, 0.0, 0.0, 2.56245112e-02, 3.68698432e-02, 4.02494210e-02, 3.68698188e-02, 2.56224456e-02, 4.02491591e-02, 2.56211581e-02, 4.27565815e-09])
        k_localmin, D_localmin = selfconsistent_minimum(E, k_guess, MSM, gamma, verbose=False)
        if D_localmin < D_min:
            D_min = D_localmin
            k_min = k_localmin

        k_guess = np.array([0.0, 3.62166214e-02, 3.32021744e-02, 2.29177442e-02, 3.62194970e-02, 2.29201247e-02, 2.29163145e-02, 0.0, 3.32002297e-02, 3.32001449e-02, 1.38556524e-09, 3.62198857e-02])
        k_localmin, D_localmin = selfconsistent_minimum(E, k_guess, MSM, gamma, verbose=False)
        if D_localmin < D_min:
            D_min = D_localmin
            k_min = k_localmin

        k_guess = np.array([0.00059444, 0.0, 0.0, 0.00034142 ,0.00042977, 0.00034142, 0.0, 0.0, 0.00034142, 0.00059444, 0.00034142, 0.0])
        k_localmin, D_localmin = selfconsistent_minimum(E, k_guess, MSM, gamma, verbose=False)
        if D_localmin < D_min:
            D_min = D_localmin
            k_min = k_localmin

        k_guess = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0  ])
        k_localmin, D_localmin = selfconsistent_minimum(E, k_guess, MSM, gamma, verbose=False)
        if D_localmin < D_min:
            D_min = D_localmin
            k_min = k_localmin

        k_guess = np.array([2.319e-02, 2.523e-02, 0.0, 2.319e-02, 2.319e-02, 0.0, 2.523e-02, 1.566e-02, 2.523e-02, 1.566e-02, 0.0, 1.565e-02])
        k_localmin, D_localmin = selfconsistent_minimum(E, k_guess, MSM, gamma, verbose=False)
        if D_localmin < D_min:
            D_min = D_localmin
            k_min = k_localmin

        k_guess = np.array([0.03142871, 0.03426775, 0.0, 0.03142862, 0.03142871, 0.0, 0.0342679, 0.02161574, 0.03426775, 0.02161574, 0.0, 0.02161556])
        k_localmin, D_localmin = selfconsistent_minimum(E, k_guess, MSM, gamma, verbose=False)
        if D_localmin < D_min:
            D_min = D_localmin
            k_min = k_localmin


    return k_min, D_min