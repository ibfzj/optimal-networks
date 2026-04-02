import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def compute_second_moment_matrix(mu, beta, N):
    total_nodes = 2 * N 
    matrix = np.zeros((total_nodes, total_nodes)) 

    for i in range(total_nodes):
        for j in range(total_nodes):
            if i == j:  # Diagonal elements
                if i % 2 == 0:  # i even
                    matrix[i, j] = mu**2 + (N-1) * beta**2
                else:  # i odd
                    matrix[i, j] = mu**2
            else:  # Off-diagonal elements
                if i % 2 == 0 and j % 2 == 0:  # Both i and j are even
                    matrix[i, j] = mu**2 - beta**2
                elif i % 2 == 1 and j % 2 == 1:  # Both i and j are odd
                    matrix[i, j] = mu**2
                else:  # One index is even, the other is odd
                    matrix[i, j] = -mu**2

    return matrix

def build_ring_incidence(N):
    """
    Oriented incidence E for a 2N-cycle with edges e=(e, e+1 mod 2N),
    oriented from e -> e+1.
    Returns E (2N x 2N).
    """
    n = 2 * N
    E = np.zeros((n, n))
    for e in range(n):
        i = e
        j = (e + 1) % n
        # edge e starts at i, ends at j
        E[i, e] = 1.0
        E[j, e] = -1.0
    return E


def build_ring_graph(N):
    """
    Build the 2N-cycle graph with node attribute s:
    odd nodes -> generators (+1)
    even nodes -> consumers (-1)
    """
    G = nx.Graph()

    for n in range(2 * N):
        if n % 2 == 1:
            G.add_node(n, s=+1)
        else:
            G.add_node(n, s=-1)

    for e in range(2 * N):
        G.add_edge(e, (e + 1) % (2 * N))

    return G

def expected_dissipation_ring(N, k_vec, Sigma, tol=1e-10):
    """
    Computes <D> = <S^T L^+ S> = Tr(L^+ <S S^T>) with L = E K E^T.
    """
    k = np.asarray(k_vec, dtype=float).reshape(-1)

    if np.any(k < 0):
        raise ValueError("capacities must be >= 0")

    Sigma = np.asarray(Sigma, dtype=float)

    E = build_ring_incidence(N)
    K = np.diag(k)
    L = E @ K @ E.T
    Lp = np.linalg.pinv(L)
    D = np.trace(Lp @ Sigma) # <D> = Tr(L^+ <S S^T>)

    return float(D)

def k_symmetric(N, gamma, kappa=1.0):
    m = 2 * N
    k0 = kappa * (m ** (-1.0 / gamma))
    return np.full(m, k0, dtype=float)

def D_symmetric(N, gamma, mu, beta, kappa=1.0):
    Sigma = compute_second_moment_matrix(mu, beta, N)
    k = k_symmetric(N, gamma, kappa)
    return expected_dissipation_ring(N, k, Sigma), k

def compute_asymmetric_case(N, gamma, beta, mu, kappa=1.0, broken_edge=-1):
    """
    Symmetry-broken ring: cut one edge -> path.
    Use KCL on the path: F_e = sum_{i=0}^e S_i (with orientation 0->1->...->m-1)
    Then apply Corson update: k_e ~ <F_e^2>^{1/(1+gamma)}, normalized to sum k^gamma = kappa^gamma
    Returns:
      D (expected dissipation)
      K (m x 1 capacities, with K[broken_edge]=0)
    """
    m = 2 * N
    Sigma = compute_second_moment_matrix(mu, beta, N)

    # Choose which edge is removed; default is the "closing" edge (m-1 -> 0).
    # For the KCL prefix formula below to hold without extra bookkeeping,
    # we assume the remaining path edges are (0-1,1-2,...,m-2 - (m-1)).
    # That corresponds to broken_edge being the closing edge.
    if broken_edge not in (-1, m - 1):
        raise ValueError("This closed-form path formula assumes the broken edge is the closing edge (m-1).")

    # Edge e on the path separates nodes {0..e} from {e+1..m-1}.
    # F_e = sum_{i=0}^e S_i  =>  <F_e^2> = sum_{i,j=0}^e <S_i S_j>
    F2_path = np.array([np.sum(Sigma[:e+1, :e+1]) for e in range(m - 1)], dtype=float)

    # Numerical safety: due to rounding, tiny negatives can occur
    if F2_path.min() < -1e-10:
        raise ValueError(f"<F^2> became significantly negative: {F2_path.min()}")
    F2_path = np.maximum(F2_path, 0.0)

    # Corson optimal capacities on the path (only for the m-1 remaining edges)
    # k_e = [F2_e^(1/(1+g))] / [ (sum F2^(g/(1+g)))^(1/g) ] * kappa
    numer = F2_path ** (1.0 / (1.0 + gamma))
    denom = (np.sum(F2_path ** (gamma / (1.0 + gamma)))) ** (1.0 / gamma)
    k_path = (numer / denom) * kappa  # sum k_path^gamma = kappa^gamma

    # Expected dissipation on the path: sum_e <F_e^2>/k_e
    D_path = np.sum(np.divide(F2_path, k_path, out=np.zeros_like(F2_path), where=(k_path > 0)))

    # Embed back into ring edge list with one broken edge set to zero
    K = np.zeros(m, dtype=float)
    K[:m-1] = k_path
    K[m-1] = 0.0
    return float(D_path), K.reshape(-1, 1)


def kb_from_constraint_2branch(ka, gamma, N, kappa=1.0):
    """
    For a two-value branch on a 2N-edge ring:
        N * ka^gamma + N * kb^gamma = kappa^gamma
    """
    rhs = (kappa**gamma) / N - ka**gamma
    if rhs < 0:
        return np.nan
    return rhs**(1.0 / gamma)

def D_alternating(N, gamma, mu, beta, ka, kappa=1.0):
    """
    Weakly symmetry-broken branch [ka, kb, ka, kb, ...].
    General-N formula.
    """
    kb = kb_from_constraint_2branch(ka, gamma, N, kappa)
    if not np.isfinite(kb) or kb <= 0 or ka <= 0:
        return np.inf, np.array([])

    D = (mu**2 * N / (ka + kb) + beta**2 * N * (N**2 - 1) / 12.0 * (ka + kb) / (ka * kb))

    k = np.array([ka if i % 2 == 0 else kb for i in range(2 * N)], dtype=float)
    return float(D), k

def D_block_N2(gamma, mu, beta, ka, kappa=1.0):
    """
    Block branch for N=2: [ka, ka, kb, kb].
    Constraint: 2 ka^gamma + 2 kb^gamma = kappa^gamma
    """
    N = 2
    kb = kb_from_constraint_2branch(ka, gamma, N, kappa)
    if not np.isfinite(kb) or kb <= 0 or ka <= 0:
        return np.inf, np.array([])

    D = mu**2 * (ka + kb) / (2.0 * ka * kb) + 2.0 * beta**2 / (ka + kb)
    k = np.array([ka, ka, kb, kb], dtype=float)
    return float(D), k

def find_best_alternating(N, gamma, mu, beta, kappa=1.0, n_grid=2000):
    ka_max = (kappa**gamma / N)**(1.0 / gamma)
    # prevent division by zero
    ka_vals = np.linspace(1e-8, ka_max - 1e-8, n_grid)

    best_D = np.inf
    best_k = None

    for ka in ka_vals:
        D, k = D_alternating(N, gamma, mu, beta, ka, kappa)
        if D < best_D:
            best_D = D
            best_k = k

    return best_D, best_k

def find_best_block_N2(gamma, mu, beta, kappa=1.0, n_grid=2000):
    ka_max = (kappa**gamma / 2.0)**(1.0 / gamma)
    ka_vals = np.linspace(1e-8, ka_max - 1e-8, n_grid)

    best_D = np.inf
    best_k = None

    for ka in ka_vals:
        D, k = D_block_N2(gamma, mu, beta, ka, kappa)
        if D < best_D:
            best_D = D
            best_k = k

    return best_D, best_k

def get_capacities_all_branches(N, mu, beta_values, gamma_values, kappa=1.0):
    capacity_values = np.zeros((len(beta_values), len(gamma_values)))
    branch_labels = [[None for _ in gamma_values] for _ in beta_values]

    for i, beta in enumerate(beta_values):
        Sigma = compute_second_moment_matrix(mu, beta, N)

        for j, gamma in enumerate(gamma_values):
            candidates = []

            # symmetric
            k_sym = k_symmetric(N, gamma, kappa=kappa)
            D_sym = expected_dissipation_ring(N, k_sym, Sigma)
            candidates.append(("symmetric", D_sym, k_sym))

            # strongly symmetry-broken
            D_brk, k_brk = compute_asymmetric_case(N, gamma, beta, mu, kappa=kappa, broken_edge=-1)
            candidates.append(("strongly_broken", D_brk, k_brk.flatten()))

            # weakly symmetry-broken / alternating
            D_alt, k_alt = find_best_alternating(N, gamma, mu, beta, kappa=kappa)
            candidates.append(("weakly_broken", D_alt, k_alt))

            # block branch, only for N=2
            if N == 2:
                D_blk, k_blk = find_best_block_N2(gamma, mu, beta, kappa=kappa)
                candidates.append(("block", D_blk, k_blk))

            # choose best
            label, D_best, k_best = min(candidates, key=lambda x: x[1])

            capacity_values[i, j] = np.min(k_best)
            branch_labels[i][j] = label

    return capacity_values, branch_labels

def get_capacities(N, mu, beta_values, gamma_values):
    capacity_values = np.zeros((len(beta_values), len(gamma_values)))

    for i, beta in enumerate(beta_values):
        Sigma = compute_second_moment_matrix(mu, beta, N)  # reuse per beta 
        for j, gamma in enumerate(gamma_values):
            k_sym = k_symmetric(N, gamma, kappa=1.0)
            D_sym = expected_dissipation_ring(N, k_sym, Sigma)

            D_brk, k_brk = compute_asymmetric_case(N, gamma, beta, mu, kappa=1.0, broken_edge=-1)

            capacity_values[i, j] = np.min(k_sym) if (D_sym < D_brk) else np.min(k_brk)
            
    return capacity_values



