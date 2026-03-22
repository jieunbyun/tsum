"""
igraph-accelerated system functions for tsum.

Drop-in replacements for ndtools.fun_binary_graph functions that use igraph
(C-based) instead of NetworkX for graph operations. Typically 10-100x faster
for connectivity computations.

Requires: pip install python-igraph
"""
import igraph as ig
import networkx as nx
from typing import Dict, Tuple, Any, Optional, List


def nx_to_igraph(G: nx.Graph) -> Tuple[ig.Graph, Dict[str, int], Dict[str, int]]:
    """
    Convert a NetworkX graph to igraph, preserving node names and edge IDs.

    Returns:
        (ig_graph, node_to_idx, eid_to_edge_idx)
        - node_to_idx: {node_name: igraph vertex index}
        - eid_to_edge_idx: {edge_id: igraph edge index}
    """
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    ig_g = ig.Graph(n=len(nodes), directed=False)
    ig_g.vs["name"] = nodes

    edges = []
    eids = []
    for u, v, data in G.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        eids.append(data.get("eid"))

    ig_g.add_edges(edges)
    ig_g.es["eid"] = eids

    eid_to_edge_idx = {}
    for idx, eid in enumerate(eids):
        if eid is not None:
            eid_to_edge_idx[eid] = idx

    return ig_g, node_to_idx, eid_to_edge_idx


def eval_global_conn_k_igraph(
    comps_state: Dict[str, int],
    G_base: nx.Graph,
    ig_graph: ig.Graph,
    node_to_idx: Dict[str, int],
    eid_to_edge_idx: Dict[str, int],
) -> Tuple[int, int, None]:
    """
    igraph-accelerated version of eval_global_conn_k.

    Same semantics: build subgraph from component states, compute vertex connectivity.
    """
    node_off = {cid for cid, st in comps_state.items() if st == 0 and cid in node_to_idx}
    edge_on = {cid for cid, st in comps_state.items() if st == 1}

    # Build list of edge indices to keep
    edges_to_keep = []
    for eid, idx in eid_to_edge_idx.items():
        if eid not in edge_on:
            continue
        e = ig_graph.es[idx]
        u_name = ig_graph.vs[e.source]["name"]
        v_name = ig_graph.vs[e.target]["name"]
        if u_name not in node_off and v_name not in node_off:
            edges_to_keep.append(idx)

    # Create subgraph (keeps all vertices, only selected edges)
    H = ig_graph.subgraph_edges(edges_to_keep, delete_vertices=False)

    k_val = H.vertex_connectivity() if H.vcount() > 1 else 0
    return k_val, k_val, None


def eval_1od_connectivity_igraph(
    comps_state: Dict[str, int],
    G_base: nx.Graph,
    ig_graph: ig.Graph,
    node_to_idx: Dict[str, int],
    eid_to_edge_idx: Dict[str, int],
    orig_node: str,
    dest_node: str,
) -> Tuple[str, int, Dict[str, Any]]:
    """
    igraph-accelerated version of eval_1od_connectivity.

    Same semantics: check if path exists between orig and dest under component states.
    """
    node_off = {cid for cid, st in comps_state.items() if st == 0 and cid in node_to_idx}

    # Quick failure
    if orig_node in node_off or dest_node in node_off:
        return ("disconnected"), 0, {"connected": False, "path_nodes": None, "path_edge_ids": None}

    edge_on = {cid for cid, st in comps_state.items() if st == 1}

    # Build list of edge indices to keep
    edges_to_keep = []
    for eid, idx in eid_to_edge_idx.items():
        if eid not in edge_on:
            continue
        e = ig_graph.es[idx]
        u_name = ig_graph.vs[e.source]["name"]
        v_name = ig_graph.vs[e.target]["name"]
        if u_name not in node_off and v_name not in node_off:
            edges_to_keep.append(idx)

    H = ig_graph.subgraph_edges(edges_to_keep, delete_vertices=False)

    orig_idx = node_to_idx[orig_node]
    dest_idx = node_to_idx[dest_node]

    # Check connectivity and find shortest path
    paths = H.get_shortest_paths(orig_idx, to=dest_idx, output="vpath")
    path_v = paths[0] if paths else []

    if path_v:
        connected = True
        path_nodes = [ig_graph.vs[v]["name"] for v in path_v]
        # Extract edge IDs along the path
        path_eids = []
        for i in range(len(path_v) - 1):
            eid_found = H.get_eid(path_v[i], path_v[i + 1], error=False)
            if eid_found >= 0:
                path_eids.append(H.es[eid_found]["eid"])
            else:
                path_eids.append(None)
    else:
        connected = False
        path_nodes = None
        path_eids = None

    info = {
        "connected": connected,
        "path_nodes": path_nodes,
        "path_edge_ids": path_eids,
    }
    return ("connected" if connected else "disconnected"), (1 if connected else 0), info


def make_igraph_sfun_global_conn(G: nx.Graph, target_g_conn: int = 1):
    """
    Create an igraph-accelerated sfun for global connectivity.

    Returns a function with the same signature as the NetworkX-based version
    used by run_another.py.

    Usage:
        sfun = make_igraph_sfun_global_conn(G, target_g_conn=1)
        k, sys_st, _ = sfun(comps_st)
    """
    ig_graph, node_to_idx, eid_to_edge_idx = nx_to_igraph(G)

    def sfun(comps_st):
        k, _, _ = eval_global_conn_k_igraph(
            comps_st, G, ig_graph, node_to_idx, eid_to_edge_idx)
        sys_st = 1 if k >= target_g_conn else 0
        return k, sys_st, None

    return sfun


def make_igraph_sfun_conn(G: nx.Graph, orig_node: str, dest_node: str):
    """
    Create an igraph-accelerated sfun for 1OD connectivity.

    Returns a function with the same signature as the NetworkX-based version
    used by run_another.py.

    Usage:
        sfun = make_igraph_sfun_conn(G, hub, dest)
        fval, sys_st, info = sfun(comps_st)
    """
    ig_graph, node_to_idx, eid_to_edge_idx = nx_to_igraph(G)

    def sfun(comps_st):
        return eval_1od_connectivity_igraph(
            comps_st, G, ig_graph, node_to_idx, eid_to_edge_idx,
            orig_node, dest_node)

    return sfun
