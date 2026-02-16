import json
from matplotlib.lines import Line2D
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import numpy as np
import networkx as nx

def draw_graph_from_json(
    nodes_path: str | Path,
    edges_path: str | Path,
    save_fname: str | Path,
    node_size: float = 30,      # base size for non-OD nodes
    od_scale: float = 1.2,      # OD nodes will be node_size * od_scale
    edge_width: float = 0.5,
    figsize: tuple[float, float] = (3.4, 3.0),
    show_legend: bool = False,
) -> None:
    """
    Draw a graph from nodes/edges JSON.
    - Node sizes are parameterised by `node_size` (user input).
    - Nodes with is_od==True are drawn as red 6-point stars.
    """

    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)

    with nodes_path.open("r") as f:
        nodes = json.load(f)
    with edges_path.open("r") as f:
        edges = json.load(f)

    G = nx.Graph()
    pos = {}
    is_od = {}

    # Nodes
    for nid, attrs in nodes.items():
        G.add_node(nid)
        pos[nid] = (attrs["x"], attrs["y"])
        is_od[nid] = bool(attrs.get("is_od", False))

    # Edges (treat as undirected; your file uses directed:false)
    for _, e in edges.items():
        G.add_edge(e["from"], e["to"])

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width, alpha=1.0)

    # Draw non-OD nodes
    regular = [n for n in G.nodes if not is_od[n]]
    nx.draw_networkx_nodes(
        G, pos, nodelist=regular, ax=ax,
        node_size=node_size,
        node_color="#467EA4",
        linewidths=0.0,
        alpha=1.0,
    )

    # Draw OD nodes as red markers
    od_nodes = [n for n in G.nodes if is_od[n]]
    nx.draw_networkx_nodes(
        G, pos, nodelist=od_nodes, ax=ax,
        node_size=node_size * od_scale,
        node_color="red",
        node_shape="x",  
        linewidths=3.0,
        alpha=1.0,
    )

    ax.set_aspect("equal")
    ax.set_axis_off()

    # --- Legend (node types) ---
    if show_legend:
        legend_elements = [
            Line2D(
                [0], [0],
                marker="x",
                linestyle="None",
                label="OD node",
                color="red",
                markersize=8,
                markeredgewidth=3.0,
            ),
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True,
            handletextpad=0.2,
            borderpad=0.4,
        )

    # --- Manually set axis limits from node positions (to minimise whitespace) ---
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

    pad = 0.02  # small fractional padding
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    dx = xmax - xmin
    dy = ymax - ymin

    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)

    fig.tight_layout()
    fig.savefig( save_fname, dpi=300, bbox_inches="tight", pad_inches=0.0 )


def draw_brc_monitor_data(
    data_paths: list,
    labels: list,
) -> None:
    """
    Draws four separate figures on BRC monitor data:
    - number of reference states vs. unknown probability (with legend)
    - number of reference states vs. number of disjoint events
    - number of reference states vs. memory usage
    - number of reference states vs. cumulative days
    """

    if len(data_paths) != len(labels):
        raise ValueError("data_paths and labels must have the same length.")

    sec_to_hr = 1.0 / 3600.0

    n_rules, unk_prob, n_brs, mem, c_time = {}, {}, {}, {}, {}

    for path, label in zip(data_paths, labels):
        with open(path, "r") as f:
            data_ = json.load(f)

        n_rules[label] = data_["no_ra"]
        unk_prob[label] = data_["pr_bu"]
        n_brs[label] = data_["no_br"]
        mem[label] = data_["rss_gb"]
        c_time[label] = np.cumsum(data_["time"]) * sec_to_hr

    def _plot_all(ydata_dict, ylabel, linestyle, save_fname, show_legend=False, ylog=False):

        if len(ydata_dict.keys()) != len(labels):
            raise ValueError("ydata_dict and labels must have the same length.")

        # Figure parameters
        font_name = "Times New Roman"
        plt.rcParams["font.family"] = font_name
        fsz = 11
        # ---

        fig = plt.figure(figsize=(3.4, 3.0), constrained_layout=True)
        ax = fig.add_axes([0.12, 0.08, 0.83, 0.83])

        for i, label in enumerate(labels):
            x = n_rules[label]
            y = ydata_dict[label]
            n = min(len(x), len(y))  # safety
            ax.plot(
                x[:n], y[:n],
                linestyle=linestyle[i], linewidth=1.5,
                label=label
            )

        ax.set_xlabel("Number of reference states", fontsize=fsz)
        ax.set_ylabel(ylabel, fontsize=fsz)
        ax.tick_params(axis="both", which="major", labelsize=fsz-1)
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
        if ylog:
            ax.set_yscale("log")

        if show_legend:
            ax.legend(loc="best", frameon=True, fontsize=fsz-1.0, labelspacing=0.25, handletextpad=0.4, handlelength=1.1, borderpad=0.2,)

        # --- add memory limit line if plotting memory ---
        if ylabel.lower().startswith("rss"):
            y_limit = 20

            # horizontal line
            ax.axhline(
                y=y_limit,
                color="black",
                linestyle="-",
                linewidth=1.2,
            )

            # label near the right edge, slightly above the line
            ax.text(
                ax.get_xlim()[1]-10,           # xmax (data coordinates)
                y_limit * 1.01,              # a bit above the line 
                "Memory limit",
                color="black",
                fontsize=fsz,
                ha="right",
                va="bottom",
            )
            ax.set_ylim(0, 22.5)  # ensure limit line is visible


        plt.savefig(save_fname, dpi=300, bbox_inches="tight", pad_inches=0.02)

    # --- Four separate figures ---
    HERE = Path(__file__).resolve().parent
    _plot_all(
        ydata_dict=unk_prob,
        ylabel="Unclassified probability",
        linestyle=['-', '--'],
        save_fname=HERE / "figs/brc_refs_vs_log_unc_prob.png",
        ylog=True,
        show_legend=False,
    )

    _plot_all(
        ydata_dict=n_brs,
        ylabel="Number of disjoint events",
        linestyle=['-', '--'],
        save_fname=HERE / "figs/brc_refs_vs_log_n_brs.png",
        ylog=True,
        show_legend=True,
    )

    _plot_all(
        ydata_dict=mem,
        ylabel="RSS (GB)",
        linestyle=['-', '--'],
        save_fname=HERE / "figs/brc_refs_vs_mem.png",
    )

    _plot_all(
        ydata_dict=c_time,
        ylabel="Cumulative time (hours)",
        linestyle=['-', '--'],
        save_fname=HERE / "figs/brc_refs_vs_c_time.png",
    )

def draw_chen12_data() -> None:
    """
    Draw the computation statistics reported in Chen and Lin (2012)
    for their search for minimal paths.
    """

    HERE = Path(__file__).resolve().parent

    # Data from Chen & Lin (2012)
    n_edges = [17, 22, 24, 27, 31, 38, 40, 45, 49, 52, 58, 59, 60]
    n_mps = [
        38, 125, 184, 414, 976, 5382, 8512,
        29738, 79384, 163496, 752061, 896476, 1262816
    ]

    # --- Figure / font parameters (journal-safe) ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    # --- Plot ---
    ax.plot(
        n_edges,
        n_mps,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    # --- Labels ---
    ax.set_xlabel("Number of component variables")
    ax.set_ylabel("Number of reference states")

    ax.set_yscale("log")  # MP counts grow exponentially

    # --- Grid (subtle, print-safe) ---
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(HERE / "figs/chen12_mps.png", dpi=300, bbox_inches="tight", pad_inches=0.02)



if __name__ == "__main__":

    HERE = Path(__file__).resolve().parent

    # --- Draw random graphs from JSON ---
    for rg, leg in zip(
        ['rg1', 'rg2', 'rg3', 'rg4'],
        [True, False, True, False],  
    ):
        draw_graph_from_json(
            nodes_path=HERE.parent / f"results/{rg}/v1/data/nodes.json",
            edges_path=HERE.parent / f"results/{rg}/v1/data/edges.json",
            save_fname=HERE / f"figs/{rg}_graph.png",
            show_legend=leg,
        )

    # --- BRC monitor data ---
    draw_brc_monitor_data(
        data_paths=[
            HERE.parent / "results/rg1/v1/brc/monitor_conn.json",
            # HERE.parent / "results/rg2/v1/brc/monitor_conn.json",
            HERE.parents[2] / "brc_data/monitor_autosave.json",
        ],
        labels=[
            "59 nodes; 262 edges; 1 OD conn.",
            "119 nodes; 295 edges; 1 OD conn.",
        ]
    )

    # --- Chen & Lin (2012) data ---
    draw_chen12_data()