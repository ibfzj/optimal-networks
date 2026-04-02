import numpy as np
import networkx as nx
from .corson_algorithm import selfconsistent_minimum, random_minimum
import random

# Reproducibility
SEED = 0
np.random.seed(SEED)
random.seed(SEED)

# === Figure 8 ===
def simple_multiplex_network():
    """Build the 8-node two-layer cube network used in Figure 8."""
    # General rules:
    # Nodes must be numbered 0,1,2,3,4,...
    # Generators have G.nodes[u]['s'] = +1
    # Consumers  have G.nodes[u]['s'] = -1
    
    # Initialize Graph and a disct  ionary of position for plotting
    G = nx.Graph()
    pos = {}

    # Generate the nodes with alternating generators and consumers:
    G.add_node(0 , s=+1)
    pos[0] = [0,0,0]
    G.add_node(1 , s=-1)
    pos[1] = [0,1,0]
    G.add_node(2 , s=+1)
    pos[2] = [1,1,0]
    G.add_node(3 , s=-1)
    pos[3] = [1,0,0]
    G.add_node(4 , s=-1)
    pos[4] = [0,0,1]
    G.add_node(5 , s=+1)
    pos[5] = [0,1,1]
    G.add_node(6 , s=-1)
    pos[6] = [1,1,1]
    G.add_node(7 , s=+1)
    pos[7] = [1,0,1]

    # Add the edges:
    elist = [(0,1),(1,2),(2,3),(3,0),  (4,5),(5,6),(6,7),(7,4),  (0,4),(1,5),(2,6),(3,7) ]
    G.add_edges_from(elist)

    return G, pos

def compute_second_moment_matrix(G, beta, mu):
    """
    Build the source covariance matrix for the given network and fluctuation parameters,
    with entries M[i,j] = <S_i S_j>.
    """
    nodes = list(G.nodes)
    n = len(nodes)
    node_to_idx = {u: k for k, u in enumerate(nodes)}

    # Split nodes into generators and consumers
    idx_gen = [node_to_idx[u] for u in nodes if G.nodes[u]['s'] > 0]
    idx_con = [node_to_idx[u] for u in nodes if G.nodes[u]['s'] <= 0]

    Ng = len(idx_gen)
    Nc = len(idx_con)

    M = np.zeros((n, n), dtype=float)

    # Consumer-consumer: <S_i S_j> = mu^2
    for i in idx_con:
        for j in idx_con:
            M[i, j] = mu**2

    # Generator-consumer and consumer-generator:
    # <S_i S_j> = -(Nc/Ng) * mu^2
    gc_val = -(Nc / Ng) * mu**2
    for i in idx_gen:
        for j in idx_con:
            M[i, j] = gc_val
            M[j, i] = gc_val

    # Generator-generator
    offdiag_gg = (Nc**2 / Ng**2) * mu**2 - beta**2
    diag_gg = (Nc**2 / Ng**2) * mu**2 + (Ng - 1) * beta**2

    for i in idx_gen:
        for j in idx_gen:
            M[i, j] = offdiag_gg
        M[i, i] = diag_gg

    return M

def extract_unique_guesses(k_guess_dict, atol=1e-2):
    """Extract numerically distinct capacity vectors from a dictionary of guesses."""
    unique_guesses = []

    for guess in k_guess_dict.values():
        if guess is None:
            continue
        if not any(np.allclose(guess, known, atol=atol) for known in unique_guesses):
            unique_guesses.append(np.array(guess, copy=True))

    return unique_guesses


def compute_coarse_phase_diagram(G, gamma_values, beta_values, mu, nonzero_tol=1e-8, verbose=True):
    """Compute the coarse Figure 8 phase diagram using random initial searches."""

    # Incidence matrix of the network (fixed across the scan)
    E = nx.incidence_matrix(G, oriented=True)

    # Output arrays: number of active edges and stored solutions
    num_nonzero_edges = np.zeros((len(gamma_values), len(beta_values)))
    k_guesses = {}

    # Progress tracking
    total_points = len(beta_values) * len(gamma_values)
    point_counter = 0

    # Loop over parameters
    for j, beta in enumerate(beta_values):
        # MSM depends only on beta
        MSM = compute_second_moment_matrix(G, beta, mu)
        for i, gamma in enumerate(gamma_values):
            point_counter += 1
            if verbose:
                print(f"[{point_counter}/{total_points}] ({i}, {j})", end="\r", flush=True)

            # Solve using a random initial condition
            k_min, _ = random_minimum(E, G, MSM, gamma)

            # Store solution and count active edges
            k_guesses[(i, j)] = np.array(k_min, copy=True)
            num_nonzero_edges[i, j] = np.sum(k_min > nonzero_tol)

    return {"gamma_values": np.array(gamma_values), "beta_values": np.array(beta_values), "num_nonzero_edges": num_nonzero_edges,
            "k_guesses": k_guesses}


def compute_fine_phase_diagram(G, gamma_values, beta_values, mu, unique_guesses, nonzero_tol=1e-8, verbose=True):
    """Compute the refined Figure 8 phase diagram from a bank of previously identified seeds."""

    # Dimensions and fixed matrices
    num_g = len(gamma_values)
    num_b = len(beta_values)
    E = nx.incidence_matrix(G, oriented=True)
    num_edges = E.shape[1]

    # Output arrays
    k_final_map = np.zeros((num_g, num_b, num_edges))
    D_final_map = np.full((num_g, num_b), np.inf)
    N_edges_map = np.zeros((num_g, num_b))

    # Progress tracking
    total_points = num_b * num_g
    point_counter = 0

    # Loop over parameter space
    for j, beta in enumerate(beta_values):
        MSM = compute_second_moment_matrix(G, beta, mu)

        for i, gamma in enumerate(gamma_values):
            point_counter += 1
            if verbose:
                print(f"[{point_counter}/{total_points}] ({i}, {j})", end="\r", flush=True)

            # Track best solution among all initial guesses
            best_k = None
            best_D = np.inf
            best_N = 0

            # Try all seeds from the unique guess bank
            for k_guess in unique_guesses:
                try:
                    k_candidate, D = selfconsistent_minimum(E, k_guess.copy(), MSM, gamma=gamma)
                except np.linalg.LinAlgError:
                    continue
                except Exception:
                    continue

                # Skip non-converged or invalid solutions
                if not np.all(np.isfinite(k_candidate)) or not np.isfinite(D):
                    continue

                # Count active edges for this candidate
                N_edges = np.sum(k_candidate > nonzero_tol)

                # Keep the lowest-dissipation solution
                if D < best_D:
                    best_D = D
                    best_k = np.array(k_candidate, copy=True)
                    best_N = N_edges

            # If no valid solution found, leave defaults
            if best_k is None:
                continue

            # Store best solution for this parameter point
            k_final_map[i, j] = best_k
            D_final_map[i, j] = best_D
            N_edges_map[i, j] = best_N

    if verbose:
        print()

    return {"gamma_values": np.array(gamma_values), "beta_values": np.array(beta_values), "num_nonzero_edges": N_edges_map,
            "k_values": k_final_map, "D_values": D_final_map}


def save_phase_diagram_npz(filename, results):
    """Save Figure 8 phase-diagram results to an .npz file."""
    save_dict = {"gamma_values": results["gamma_values"], "beta_values": results["beta_values"],
                 "num_nonzero_edges": results["num_nonzero_edges"]}

    if "k_guesses" in results:
        save_dict["k_guesses"] = np.array([results["k_guesses"]], dtype=object)

    if "k_values" in results:
        save_dict["k_values"] = results["k_values"]

    if "D_values" in results:
        save_dict["D_values"] = results["D_values"]

    np.savez(filename, **save_dict)

def create_graph(G, beta, gamma, mu):
    """Solve one parameter point and return a copy of the network with optimized edge weights."""
    E = nx.incidence_matrix(G, oriented=True)
    MSM = compute_second_moment_matrix(G, beta, mu)

    k_min, _ = random_minimum(E=E, G=G, MSM=MSM, gamma=gamma)

    Gres = G.copy()
    threshold = 1e-8

    for e in range(len(k_min)):
        f = np.argmax(E[:, [e]])
        t = np.argmin(E[:, [e]])
        if k_min[e] < threshold:
            Gres.remove_edge(f, t)
        else:
            Gres.edges[f, t]["weight"] = k_min[e]

    return Gres

