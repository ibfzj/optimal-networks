import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
import warnings

from core.multilayer_calc import create_graph

# === Figure 8 ===

def plot_phase_diagram(gamma_values, beta_values, num_nonzero_edges, cmap="viridis", figtext=None, savepath=None):
    """Plot the Figure 8 phase diagram showing the number of active edges across parameter space."""

    # Identify discrete values from the data 
    discrete_values = np.unique(num_nonzero_edges[np.isfinite(num_nonzero_edges)])

    # Build boundaries so each integer value is centered in its color bin
    vmin = discrete_values.min() - 0.5
    vmax = discrete_values.max() + 0.5
    boundaries = np.arange(vmin, vmax + 1)

    norm = BoundaryNorm(boundaries, ncolors=plt.cm.get_cmap(cmap).N, clip=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Transpose so axes align with (gamma horizontal, beta vertical)
    data = num_nonzero_edges.T

    im = ax.imshow(data, origin="lower", aspect="auto", extent=[gamma_values[0], gamma_values[-1], beta_values[0], beta_values[-1]],
                   cmap=cmap, norm=norm)

    fontsize=30
    tick_fontsize=30

    # Use only values that actually appear as colorbar ticks
    cbar_ticks = discrete_values

    cbar = fig.colorbar(im, ax=ax, ticks=cbar_ticks)
    cbar.set_label(r"$\mathcal{N}^*$", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    ax.set_xlabel(r"$\gamma$", fontsize=fontsize)
    ax.set_ylabel(r"$\beta$", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # Optional panel label (e.g. "(b)")
    if figtext is not None:
        x, y, text, fs = figtext
        fig.text(x, y, text, fontsize=fs, ha="left", va="top")

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    plt.show()

    return fig, ax

def draw_network_3d(ax, G, pos):
    """Draw a weighted multiplex network in 3D with node types and edge thickness encoding capacities."""
    ew_max = max(G.edges[u, v].get("weight", 0.0) for u, v in G.edges())
    if ew_max <= 0:
        ew_max = 1.0

    # Nodes
    for u in G.nodes():
        x, y, z = pos[u]
        if G.nodes[u]["s"] > 0:
            ax.scatter(x, y, z, marker="s", s=40, c="#1f77b4", depthshade=False)
        else:
            ax.scatter(x, y, z, marker="o", s=40, c="#ff7f0e", depthshade=False)

    # Edges
    for u, v in G.edges():
        x1, y1, z1 = pos[u]
        x2, y2, z2 = pos[v]
        ew = G.edges[u, v].get("weight", 0.0)
        lw = 1 + 6 * ew / ew_max
        ax.plot([x1, x2], [y1, y2], [z1, z2],
                linewidth=lw,
                color=str(0.5 - 0.5 * ew / ew_max))

    # Layout
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.set_facecolor((1, 1, 1, 0))
    ax.view_init(elev=22, azim=-60)


def plot_k_vs_gamma_with_insets(gamma_values, k_values, num_nonzero_edges, beta, mu, G, pos, savepath=None,
                                phase_labels = None, inset_positions = None, inset_gammas = None):
    """Plot optimized edge capacities versus gamma with inset network visualizations at representative points."""

    # Detect jumps
    jumps = np.where(np.diff(num_nonzero_edges) != 0)[0]
    jump_gamma_values = gamma_values[jumps + 1]

    # Sort k values
    sorted_k_values = [sorted(k) for k in k_values]
    num_edges = len(sorted_k_values[0])

    # Edge colors
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, num_edges))

    fig, ax = plt.subplots(figsize=(16, 6))

    # Edge capacities
    for e in range(num_edges):
        edge_k_values = [k[e] for k in sorted_k_values]
        ax.plot(gamma_values, edge_k_values, alpha=0.6, label=f"e={e}", color=colors[e], lw=2)

    # Vertical jump lines
    for x in jump_gamma_values:
        ax.axvline(x=x, color='k', linestyle='--', linewidth=2)

    # Labels
    ax.set_xlabel(r"$\gamma$", fontsize=30)
    ax.set_ylabel(r"$k_e^*$", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)

    fig.text(0.0, 1.0, "(a)", fontsize=32, ha='left', va='top')

    '''
    ax.text(0.17, 0.075, r"$\mathcal{N}^* = 7$", fontsize=30)
    ax.text(0.34, 0.075, r"$\mathcal{N}^* = 8$", fontsize=30)
    ax.text(0.58, 0.075, r"$\mathcal{N}^* = 9$", fontsize=30)
    ax.text(0.69, 0.075, r"$\mathcal{N}^* = 12$", fontsize=30)
    ''' 
    # Phase labels
    if phase_labels is not None:
        for x, y, text, kwargs in phase_labels:
            ax.text(x, y, text, **kwargs)

    # Inset networks
    if inset_positions is not None and inset_gammas is not None:
        for rect, gamma in zip(inset_positions, inset_gammas):
            ax_in = fig.add_axes(rect, projection='3d')
            Gres = create_graph(G, beta, gamma, mu)
            draw_network_3d(ax_in, Gres, pos)
    # Insets
    '''
    inset_positions = [
        [0.1, 0.58, 0.12, 0.28],
        [0.28, 0.58, 0.12, 0.28],
        [0.5, 0.58, 0.12, 0.28],
        [0.64, 0.58, 0.12, 0.28],
    ]

    representative_gammas = [0.2, 0.4, 0.6, 0.8]

    for rect, gamma in zip(inset_positions, representative_gammas):
        ax_in = fig.add_axes(rect, projection='3d')
        Gres = create_graph(G, beta, gamma, mu)
        draw_network_3d(ax_in, Gres, pos)
    '''
    ax.legend(fontsize=20, ncol=2, loc='lower right', bbox_to_anchor=(0.995, 0.005), borderaxespad=0)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout"
        )
        fig.tight_layout()

        if savepath is not None:
            plt.savefig(savepath, bbox_inches='tight')

    plt.show()

    return fig, ax
