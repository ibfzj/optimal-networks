import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.signal import argrelextrema

from .rings_calc import build_ring_graph, compute_ring_second_moment_matrix, D_of_a_N2, global_optimum_N2_piecewise, kvec_N3_from_ab, expected_dissipation_ring, meaningful_minima_N3, global_candidates_from_minima, build_weighted_ring_graph_from_kvec
from .corson_algorithm import random_minimum

class ScalarFormatterOneDecimal(ScalarFormatter):
    """Scalar formatter that displays one decimal place in scientific notation."""
    def _set_format(self):
        self.format = '%.1f'

# === Figure 2 ===

def minima_indices(y, include_endpoints=True):
    """Return indices of local minima, optionally including endpoint minima."""
    y = np.asarray(y)
    idx = list(argrelextrema(y, np.less)[0])  # interior strict minima
    if include_endpoints and len(y) >= 2:
        if np.isfinite(y[0]) and np.isfinite(y[1]) and y[0] <= y[1]:
            idx.append(0)
        if np.isfinite(y[-1]) and np.isfinite(y[-2]) and y[-1] <= y[-2]:
            idx.append(len(y) - 1)
    return sorted(set(idx))

def mark_minima(ax, x, y, rtol=1e-10):
    """Mark local and global minima of a 1D curve on an existing axis."""
    x = np.asarray(x)
    y = np.asarray(y)

    idx = minima_indices(y, include_endpoints=True)
    idx = [i for i in idx if np.isfinite(y[i])]
    if not idx:
        return

    y_min = np.min(y[idx])
    tol = rtol * max(1.0, abs(y_min))

    global_idx = [i for i in idx if abs(y[i] - y_min) <= tol]
    local_only = [i for i in idx if i not in global_idx]

    # local (non-global): hollow circles
    if local_only:
        ax.scatter(x[local_only], y[local_only], marker="o", s=220, facecolors="none", edgecolors="k", linewidths=1, zorder=20)

    # global: x + hollow circle around it
    ax.scatter(x[global_idx], y[global_idx], marker="x", s=140, c="black", linewidths=1, zorder=23)
    ax.scatter(x[global_idx], y[global_idx], marker="o", s=220, facecolors="yellow", edgecolors="black", linewidths=1, zorder=22)
    

def plot_dissipation_N2(gammas, mu=0.0, beta=0.1, kappa=1.0, ax=None, legend_elements=None):
    """Plot the N=2 dissipation as a function of the reduced variable a for selected gamma values."""
    colors = ["#003366", "#E69F00"]
    if ax is None:
        fig, ax = plt.subplots()

    for i, gamma in enumerate(gammas):
        a_max = kappa * (2.0 ** (-1.0 / gamma))
        a_vals = np.linspace(1e-16, a_max, 800)

        D_vals = np.array([D_of_a_N2(a, gamma, mu, beta, kappa=kappa) for a in a_vals])

        ax.plot(a_vals, D_vals, label=fr"$\gamma={gamma}$", lw=2, color=colors[i], zorder=1)

        mark_minima(ax, a_vals, D_vals)

    ax.set_xlabel(r"$k_1$")
    ax.set_ylabel(r"$\bar D$")

    if legend_elements is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + legend_elements, labels + ['Local min', 'Global min'], loc = 'center')
    else:
        ax.legend(loc = 'upper right')

    return ax

def plot_fig2b_N2(gamma_vals, kappa=1.0, ax = None):
    """Plot the analytic N=2 optimal capacities as a function of gamma."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))
    k1 = []
    k3 = []
    for g in gamma_vals:
        a, b = global_optimum_N2_piecewise(g, kappa=kappa)
        k1.append(a)  # k1=k2=a
        k3.append(b)  # k3=k4=b

    k1 = np.array(k1)
    k3 = np.array(k3)

    g = np.asarray(gamma_vals)
    mask_left  = g < 1.0
    mask_right = g > 1.0

    # left branch (gamma < 1)
    ax.plot(g[mask_left],  k1[mask_left],  color="#E69F00", linewidth=4, label=r"$k_1^*=k_2^*$")
    ax.plot(g[mask_left],  k3[mask_left],  color="#003366", linewidth=2, label=r"$k_3^*=k_4^*$")

    # right branch (gamma > 1)
    ax.plot(g[mask_right], k3[mask_right], color="#E69F00", linewidth=4)
    ax.plot(g[mask_right], k1[mask_right], color="#003366", linewidth=2)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$k_e^*$")
    ax.legend(loc="lower right")

# === Figure 3 ===

def plot_Figure3(gamma, mu, beta, kappa=1.0, grid=260, ax=None):
    """Plot the reduced N=3 dissipation and mark its relevant minima."""
    if ax is None:
        fig, ax = plt.subplots()

    # Create (a, b) parameter grid
    a_max = ((kappa**gamma)/2.0)**(1.0/gamma)
    a_vals = np.linspace(0.0, a_max, grid)
    b_vals = a_vals

    # Evaluate the dissipation on the grid
    N = 3
    Sigma = compute_ring_second_moment_matrix(mu, beta, N)
    D = np.full((grid, grid), np.nan, float)

    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            k = kvec_N3_from_ab(a, b, gamma, kappa=kappa)
            if k is None:
                continue
            D[i, j] = expected_dissipation_ring(N, k, Sigma)
            
    # Plot heatmap of dissipation
    im = ax.imshow(np.log(D).T, origin="lower", aspect="auto", extent=[a_vals[0], a_vals[-1], b_vals[0], b_vals[-1]])
    plt.colorbar(im, ax=ax, label=r"$ln(\bar D)$")

    # Compute minima (interior + boundaries)
    interior_min, boundary_mins = meaningful_minima_N3(gamma, mu, beta, kappa=kappa, n_grid=grid, eps_frac=0.01)

    candidates = []
    if interior_min is not None:
        candidates.append(("sym", *interior_min))
    for key, (a_m, b_m, D_m) in boundary_mins.items():
        candidates.append((key, a_m, b_m, D_m))

    best = min(candidates, key=lambda t: t[3])
    _, a_g, b_g, _ = best

    # Markers
    # Boundary minima: yellow filled circles with black edge
    if gamma < 1: # we analytically showed this
        for (a_b, b_b, D_b) in boundary_mins.values():
            ax.scatter([a_b], [b_b], marker="o", s=260, facecolors="yellow", edgecolors="black", linewidths=2.0, zorder=9)

    # Interior minimum (if exists): hollow circle
    if interior_min is not None:
        a_i, b_i, D_i = interior_min
        ax.scatter([a_i], [b_i], marker="o", s=260, facecolors="yellow", edgecolors="black", linewidths=2.0, zorder=10)

    globals_ = global_candidates_from_minima(candidates, use_log=True, atol_log=5e-3)

    for tag, a_g, b_g, D_g in globals_:
        ax.scatter([a_g], [b_g], marker="x", s=140, c="black", linewidths=3.0, zorder=11)

    ax.scatter([a_g], [b_g], marker="x", s=140, c="black", linewidths=3.0, zorder=11)

    ax.set_xlabel(r"$k_1=k_2$")
    ax.set_ylabel(r"$k_3=k_4$")
    ax.text(0.12, 0.95, rf"$\gamma={gamma}$", transform=ax.transAxes, ha="left", va="top")
    return ax

# Insets for Fig 3
def add_ring_inset(ax, kvec, bounds, color="white", edge_scale=8, node_size=300):
    """Add a weighted ring inset at a specified position in axis coordinates:
    bounds = [x0, y0, w, h] in ax.transAxes coordinates.
    """
    iax = ax.inset_axes(bounds, transform=ax.transAxes)
    G = build_weighted_ring_graph_from_kvec(kvec)
    draw_ring_inset(G, iax, color=color, node_size=node_size, edge_scale=edge_scale)
    return iax

def add_insets_to_Figure3(ax, gamma, kappa=1.0):
    """
    Places representative insets in panels of Figure 3.
    Bounds are [x0, y0, w, h] in axes coordinates.
    """
    def k_sym_N3(gamma, kappa=1.0):
        x = kappa * 6.0 ** (-1.0 / gamma)
        return x, x, x

    def k_brk_N3(gamma, kappa=1.0):
        x = kappa * 4.0 ** (-1.0 / gamma)
        return 0.0, x, x

    def kvec_hex_from_abc(a, b, c):
        return np.array([a, a, b, b, c, c], dtype=float)
    
    gc = np.log(3/2) / np.log(2)
    gb = 1.0

    # Representative capacity patterns
    a_sym, b_sym, c_sym = k_sym_N3(gamma, kappa)
    a_brk, b_brk, c_brk = k_brk_N3(gamma, kappa)

    k_sym = kvec_hex_from_abc(a_sym, b_sym, c_sym)
    k_brk = kvec_hex_from_abc(a_brk, b_brk, c_brk)

    if gamma < gc:
        # gamma=0.55: global broken, local symmetric
        add_ring_inset(ax, k_brk, bounds=[0.62, 0.5, 0.23, 0.23], color="black", edge_scale=5, node_size=180)
    elif gamma < gb:
        # gamma=0.60: global symmetric, local broken
        add_ring_inset(ax, k_sym, bounds=[0.60, 0.5, 0.23, 0.23], color="black", edge_scale=5, node_size=180)
    else:
        # gamma=1.05: only symmetric local/global minimum
        add_ring_inset(ax, k_sym, bounds=[0.58, 0.5, 0.24, 0.24], color="black", edge_scale=5, node_size=180)

# === Figure 4 ===

def plot_Figure4(ax=None, kappa=1.0, gamma_min=0.32, gamma_max=1.3, n=500):
    """Plot the analytic N=3 branch diagram showing global and local minima versus gamma."""    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))

    def k_sym_N3(gamma, kappa=1.0):
        return kappa * 6.0 ** (-1.0 / gamma)

    def k_brk_N3(gamma, kappa=1.0):
        return kappa * 4.0 ** (-1.0 / gamma)

    gc = np.log(3/2) / np.log(2)
    gb = 1.0

    g_left = np.linspace(gamma_min, gc, n)
    g_mid  = np.linspace(gc, gb, n)

    # Branch values
    ksym_left = k_sym_N3(g_left, kappa)
    kbrk_left = k_brk_N3(g_left, kappa)
    kbrk_mid  = k_brk_N3(g_mid, kappa)
    color_brk = "#003366"
    color_sym = "#E69F00"

    # Symmetry-broken branch: k1*=k3*=x, k2*=0
    # solid for global (g < gc), dashed for local (gc < g < 1)
    ax.plot(g_left, kbrk_left, color=color_brk, lw=2.5)
    ax.plot(g_left, np.zeros_like(g_left), color=color_brk, lw=2.5)

    ax.plot(g_mid, kbrk_mid, color=color_brk, lw=2.5, ls=(4, (4, 3)))
    ax.plot(g_mid, np.zeros_like(g_mid), color=color_brk, lw=2.5, ls=(4, (4, 3)))

    # Symmetric branch: k1*=k2*=k3*=y
    # dashed for local (g < gc), solid for global (g > gc)
    ax.plot(g_left, ksym_left, color=color_sym, lw=2.5, ls="--")
    ax.plot(np.linspace(gc, gamma_max, n), k_sym_N3(np.linspace(gc, gamma_max, n), kappa), color=color_sym, lw=2.5)

    # Labels and style
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$k_e^\ast$")

    ax.plot([], [], color=color_brk, lw=2.0, label="symmetry broken")
    ax.plot([], [], color=color_sym, lw=2.0, label="symmetric")

    ax.legend(frameon=True, loc="upper left", fontsize=12)

    ax.set_xlim(gamma_min, gamma_max)
    ax.set_ylim(-0.02, 1.05 * max(k_brk_N3(gamma_min, kappa), k_sym_N3(gamma_max, kappa)))
    ax.axvline(1.0, linestyle="dashdot", color = 'k', lw = 1)
    ax.axvline(np.log(3/2)/np.log(2), linestyle=":", color = 'k', lw = 1)

    return ax

# === Figure 5 ===

def save_phase_data(N, capacity_values, branch_labels, beta_values, gamma_values):
    df_cap = pd.DataFrame(capacity_values, index=beta_values, columns=gamma_values)
    df_cap.to_csv(f"capacity_values_N{N}.csv")
    df_branch = pd.DataFrame(branch_labels, index=beta_values, columns=gamma_values)
    df_branch.to_csv(f"branch_labels_N{N}.csv")

def plot_capacity_values(capacity_values, N, beta_values, gamma_values, label, savefig = False):
    """Plot the ring phase diagram using the minimum edge capacity as the displayed observable."""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(capacity_values, aspect='auto', origin='lower', 
            extent=[gamma_values[0], gamma_values[-1], beta_values[0], beta_values[-1]], cmap='viridis')
    
    cbar = fig.colorbar(im, ax=ax, label=r'$min_e(K_e)$')
    cbar.ax.tick_params(labelsize=35)  
    cbar.set_label(r'$\min _e(k^*_e)$', fontsize=35) 

    formatter = ScalarFormatterOneDecimal(useMathText=True)
    formatter.set_powerlimits((0, 0))   # always scientific notation
    cbar.formatter = formatter
    cbar.update_ticks()
    fig.canvas.draw()

    cbar.ax.tick_params(labelsize=35)
    offset = cbar.ax.yaxis.get_offset_text()
    offset.set_size(35)
    offset.set_x(3)

    ax.set_xlabel(r'$\gamma$', fontsize=35)
    ax.set_ylabel(r'$\beta$', fontsize=35)
    ax.tick_params(axis='both', labelsize=35) 

    ax.annotate(f'{label}', xy=(0.0, 0.93), xycoords='figure fraction', fontsize=35, color='black', ha='left', va='top')
    ax.annotate(rf'$N={N}$', xy=(0.6, 0.85), xycoords='figure fraction', fontsize=35, color='black', ha='left', va='top')
    if savefig == True:
        fig.savefig(f"capacity_phase_N{N}.pdf", bbox_inches="tight")

    return fig, ax


def draw_ring_inset(G, ax, color="white", node_size=500, edge_scale=8):
    """Draw a weighted 4-, 6-, or 8-node ring inset on an existing axis."""

    num_nodes = G.number_of_nodes()

    if num_nodes == 4:
        pos = {
            0: (0, 0),
            1: (-0.15, 0),
            2: (-0.15, -0.15),
            3: (0, -0.15),
        }
        xlim = (-0.2, 0.2)
        ylim = (-0.2, 0.2)

    elif num_nodes == 6:
        pos = {
            0: (0, 0.2),
            1: (-0.173, 0.1),
            2: (-0.173, -0.1),
            3: (0, -0.2),
            4: (0.173, -0.1),
            5: (0.173, 0.1),
        }
        xlim = (-0.25, 0.25)
        ylim = (-0.25, 0.25)

    elif num_nodes == 8:
        pos = {
            0: (0, 0.2),
            1: (-0.141, 0.141),
            2: (-0.2, 0),
            3: (-0.141, -0.141),
            4: (0, -0.2),
            5: (0.141, -0.141),
            6: (0.2, 0),
            7: (0.141, 0.141),
        }
        xlim = (-0.25, 0.25)
        ylim = (-0.25, 0.25)

    else:
        raise ValueError("draw_ring_inset only supports 4, 6, or 8 nodes.")

    if G.number_of_edges() == 0:
        ew_max = 1.0
    else:
        ew_max = max(G.edges[u, v].get("weight", 1.0) for u, v in G.edges())
        if ew_max <= 0:
            ew_max = 1.0

    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ew = G.edges[u, v].get("weight", 1.0)

        if ew <= 1e-9:
            continue

        lw = np.floor(1 + edge_scale * ew / ew_max)
        ax.plot([x1, x2], [y1, y2], linewidth=lw, color=color, zorder=0)

    for u in G.nodes():
        x, y = pos[u]
        marker = "s" if u % 2 == 0 else "o"
        ax.scatter(x, y, marker=marker, s=node_size, c=color, zorder=10)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")

def draw_weighted_network_on_ax(G, ax, pos, color="black", edge_scale=8, node_size=120):
    """Draw a weighted network with prescribed node positions on an existing axis."""
    ew_max = max([G.edges[u, v]["weight"] for u, v in G.edges()])
    if ew_max <= 0:
        ew_max = 1.0

    for u, v in G.edges():
        ew = G.edges[u, v]["weight"]

        # hide truly vanishing edges
        if ew <= 1e-14:
            continue

        x1, y1 = pos[u]
        x2, y2 = pos[v]

        lw = np.floor(1 + edge_scale * ew / ew_max) * 0.5
        ax.plot([x1, x2], [y1, y2], linewidth=float(lw), color=color, zorder=0)

    for u in G.nodes():
        x, y = pos[u]
        marker = "s" if u % 2 == 0 else "o"
        ax.scatter(x, y, marker=marker, s=node_size, c=color, zorder=10)

    ax.set_xlim(-0.2, 0.2)
    ax.set_ylim(-0.2, 0.2)
    ax.set_aspect("equal")
    ax.axis("off")

def build_optimized_ring_graph(N, beta, gamma, mu, threshold=1e-9):
    """Build the numerically optimized ring graph for use in phase-diagram insets."""

    G = build_ring_graph(N)

    MSM = compute_ring_second_moment_matrix(mu, beta, N)

    num_edges = G.number_of_edges()
    E = nx.incidence_matrix(G, oriented=True)

    # Solve the full optimization problem numerically
    k_min, D_min = random_minimum(E, G, MSM, gamma)

    # Store only active edges and attach optimized capacities as weights
    Gres = G.copy()
    for e in range(num_edges):
        f = np.argmax(E[:, [e]])
        t = np.argmin(E[:, [e]])

        if k_min[e] < threshold:
            Gres.remove_edge(f, t)
        else:
            Gres.edges[f, t]["weight"] = k_min[e]

    return Gres, k_min, D_min

def add_network_inset(ax, G, beta, gamma, size=0.25, color="white"):
    """Add a ring-network inset centered at the data point (gamma, beta)."""    
    # Convert data -> axis coordinates
    x_ax, y_ax = ax.transAxes.inverted().transform(ax.transData.transform((gamma, beta)))

    # Center inset on that point
    rect = [x_ax - size/2, y_ax - size/2, size, size]

    axins = ax.inset_axes(rect)

    draw_ring_inset(G, axins, color=color)

    return axins


def plot_phase_with_insets(N, mu, gamma_values, beta_values, capacity_values, inset_params, label, save_final_fig=False):
    """Plot the ring phase diagram and add representative optimized-network insets:
    inset_params = list of tuples:
        [(gamma1, beta1, color1),
         (gamma2, beta2, color2),
         (gamma3, beta3, color3)]
    """

    # Draw the main phase diagram
    fig, ax = plot_capacity_values(capacity_values, N, beta_values, gamma_values, label, savefig=False)

    # Create and place one optimized inset network for each requested parameter point
    for gamma, beta, color in inset_params:
        Gres, _, _ = build_optimized_ring_graph(N=N, beta=beta, gamma=gamma, mu=mu)
        add_network_inset(ax, Gres, gamma=gamma, beta=beta, color=color)

    if save_final_fig == True:
        fig.savefig(f"capacity_phase_N{N}.pdf", bbox_inches="tight")

    return fig, ax


# === Figure 5 d, e

def plot_curves_fixed_betas(fixed_beta, gamma_values, minK, branch_labels, label):
    """Plot minimum capacities versus gamma for selected beta values, marking branch changes."""
    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(fixed_beta)))

    all_jump_positions = []

    for c, beta in enumerate(fixed_beta):
        y = np.asarray(minK[c, :], dtype=float)

        # Extract 1D labels for this fixed beta
        labels_1d = np.array(branch_labels[c], dtype=object)

        # Smoothing against flickers
        labels_smooth = labels_1d.copy()
        for i in range(1, len(labels_smooth) - 1):
            if labels_smooth[i-1] == labels_smooth[i+1] != labels_smooth[i]:
                labels_smooth[i] = labels_smooth[i-1]

        # Detect true branch changes
        change_idx = np.where(labels_smooth[:-1] != labels_smooth[1:])[0]
        jump_positions = 0.5 * (gamma_values[change_idx] + gamma_values[change_idx + 1])
        jump_positions = jump_positions[:2]    

        # Break the colored curve at jumps
        y_plot = y.copy()
        for j in change_idx:
            y_plot[j + 1] = np.nan

        ax.plot(gamma_values, y_plot, color=colors[c], lw=2, label=rf'$\beta = {beta}$')

        # Draw dashed vertical lines at actual branch transitions
        for xj in jump_positions:
            ax.axvline(x=xj, color="black", lw=1.8, ls="--", zorder=0)

    ax.set_xlabel(r'$\gamma$', fontsize=35)
    ax.set_ylabel(r'$\min_e(k^*_e)$', fontsize=35)
    ax.tick_params(axis='both', labelsize=35)
    ax.legend(loc='best', fontsize=30)

    plt.annotate(f'{label}', xy=(0.0, 0.93), xycoords='figure fraction', fontsize=35, color='black', ha='left', va='top')

    return fig, ax, all_jump_positions


def plot_curves_fixed_gammas(beta_values, fixed_gamma, minK, branch_labels, label):
    """Plot minimum capacities versus beta for selected gamma values, marking branch changes."""
    fig, ax = plt.subplots(figsize=(24, 10))
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, len(fixed_gamma)))

    all_jump_positions = []

    for c, gamma in enumerate(fixed_gamma):
        y = np.asarray(minK[:, c], dtype=float)

        # Extract 1D labels for this fixed gamma
        labels_1d = np.array([row[c] for row in branch_labels], dtype=object)

        # Smoothing against flickers
        labels_smooth = labels_1d.copy()
        for i in range(1, len(labels_smooth) - 1):
            if labels_smooth[i-1] == labels_smooth[i+1] != labels_smooth[i]:
                labels_smooth[i] = labels_smooth[i-1]

        # Detect true branch changes
        change_idx = np.where(labels_smooth[:-1] != labels_smooth[1:])[0]
        jump_positions = 0.5 * (beta_values[change_idx] + beta_values[change_idx + 1])
        jump_positions = np.array([jump_positions[0], jump_positions[1], jump_positions[-2], jump_positions[-1]])

        # Break the curve at jumps
        y_plot = y.copy()
        for j in change_idx:
            y_plot[j + 1] = np.nan

        ax.plot(beta_values, y_plot, color=colors[c], lw=2, label=rf'$\gamma = {gamma}$')

        # Draw one dashed vertical line per actual branch transition
        for xj in jump_positions:
            ax.axvline(x=xj, color="black", lw=2, ls="--", zorder=0)

    ax.set_xlabel(r'$\beta$', fontsize=35)
    ax.set_ylabel(r'$\min_e(k^*_e)$', fontsize=35)
    ax.tick_params(axis='both', labelsize=35)
    ax.legend(loc='lower center', fontsize=30)

    plt.annotate(f'{label}', xy=(0.0, 0.93), xycoords='figure fraction', fontsize=35, color='black', ha='left', va='top')

    return fig, ax

def add_network_inset_xy(ax, G, x, y, size=0.25, color="white"):
    """Add a ring-network inset centered at generic data coordinates (x, y)."""
    # convert data -> axis coordinates
    x_ax, y_ax = ax.transAxes.inverted().transform(ax.transData.transform((x, y)))

    rect = [x_ax - size/2, y_ax - size/2, size, size]
    axins = ax.inset_axes(rect)

    draw_ring_inset(G, axins, color=color)

    return axins