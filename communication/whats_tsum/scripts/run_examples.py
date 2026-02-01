from pathlib import Path
from ndtools.network_generator import GenConfig, generate_and_save
from ndtools.io import load_json
from ndtools.graphs import build_graph
import networkx as nx
from ndtools.fun_binary_graph import eval_1od_connectivity, eval_global_conn_k
from mbnpy import brc

def generate_random_network_data(name: str = "rg", 
                                 generator = "rg",
                                 generator_params={"n_nodes": 60, "radius": 0.25, "p_fail": 0.1}, 
                                 target_g_conn = 1,
                                 out_base: Path = None,
                                 seed: int = 7) -> Path:
    print(f"Generating random network data {name} with params: {generator_params} ..")

    cfg = GenConfig(
        name=name,
        generator=generator,
        description=", ".join(f"{k}={v}" for k, v in generator_params.items()),
        generator_params=generator_params,
        seed=seed,
    )

    ds_root = generate_and_save(out_base, cfg, draw_graph=True)
    print("Wrote:", ds_root)

    # Load generated data
    nodes = load_json( ds_root / "data" / "nodes.json" )
    edges = load_json( ds_root / "data" / "edges.json" )
    probs = load_json( ds_root / "data" / "probs.json" )

    # Build a NetworkX graph
    G = build_graph(nodes, edges, probs)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Decide the hub node (node with highest degree)
    hub = max(G.degree, key=lambda x: x[1])
    print(f"Hub node is {hub[0]} with degree {hub[1]}")

    # Decide the destination node (node farthest from hub)
    dist = nx.single_source_shortest_path_length(G, hub[0])
    dest = max(dist.items(), key=lambda x: x[1])
    print(f"Destination node is {dest[0]} at distance {dest[1]} from hub")

    # Build system function
    ## Connectivity of one origin-destination pair
    sys_func_conn = lambda comps_st: eval_1od_connectivity(comps_st, G, hub[0], dest[0])

    ## Global connectivity
    def sys_func_global_conn_long(comps_st):
        _, k, _ = eval_global_conn_k(comps_st, G)
        if k >= target_g_conn:
            sys_st = 1
        else:
            sys_st = 0
        return k, sys_st, None
    sys_func_global_conn = lambda comps_st: sys_func_global_conn_long(comps_st)

    ## System functions for BRC (it accepts only 's'/'f' states for system event)
    def brc_wrapper(sys_func, func_option: str = "conn"):
        """Convert (k, sys_st (0/1), extra) -> (k, 's'/'f', None)."""
        assert func_option in ["conn", "global_conn"], f"Invalid func_option: {func_option}"
        
        if func_option == "conn":
            def f(comps_st):
                k, sys_st, info = sys_func(comps_st)
                if sys_st == 1: 
                    # survival case -> return minimum survival component state
                    # BRC's efficiency relies on the availability of minimum survival component state
                    min_comps_st = {comp: 1 for comp in info['path_edge_ids']}
                else:
                    min_comps_st = None
                return k, ('s' if sys_st == 1 else 'f'), min_comps_st
            return f
        else:
            def f(comps_st):
                k, sys_st, _ = sys_func(comps_st)
                return k, ('s' if sys_st == 1 else 'f'), None
            return f
    
    sys_func_conn_brc = brc_wrapper(sys_func_conn)
    sys_func_global_conn_brc = brc_wrapper(sys_func_global_conn)
    
    # Return random graph data
    rg_data = {
        "nodes": nodes,
        "edges": edges,
        "probs": probs,
        "hub": hub[0],
        "dest": dest[0],
        "sys_func_conn": sys_func_conn,
        "sys_func_global_conn": sys_func_global_conn,
        "sys_func_conn_brc": sys_func_conn_brc,
        "sys_func_global_conn_brc": sys_func_global_conn_brc,
        "graph": G
    }

    return ds_root, rg_data

if __name__ == "__main__":

    repo_root = Path(__file__).resolve().parents[1]
    out_base = repo_root / "results"

    # Generate a random network 
    # smaller example
    generator = "rg"
    name = "rg1" # Random Geometric Graph
    gen_params1 = {"n_nodes": 60, "radius": 0.25, "p_fail": 0.05}
    target_g_conn1 = 1
    ds_root1, rg_data1 = generate_random_network_data(name, out_base=out_base, 
                                                      generator=generator,
                                                      generator_params=gen_params1, 
                                                      target_g_conn=target_g_conn1,
                                                      seed=7)

    # Larger example
    name = "rg2" # Random Geometric Graph
    gen_params2 = {"n_nodes": 120, "radius": 0.25, "p_fail": 0.05}
    target_g_conn2 = 2
    ds_root2, rg_data2 = generate_random_network_data(name, out_base=out_base,
                                                      generator=generator,
                                                      generator_params=gen_params2, 
                                                      target_g_conn=target_g_conn2,
                                                      seed=7)

    # Run BRC algorithm on the generated data
    edges1 = rg_data1['edges']
    probs1 = rg_data1['probs']
    probs_brc1 = {e: {0: probs1[e]['0']['p'], 1: probs1[e]['1']['p']} for e in edges1}
    brs1, rules1, sys_res1, monitor1 = brc.run(probs_brc1, rg_data1['sys_func_conn_brc'], 
                                               max_nb = 40_000)
    brc_path1 = Path(ds_root1 / "brc")
    brc_path1_rel = brc_path1.relative_to(Path.cwd())
    brc.save_brc_data(rules1, brs1, sys_res1, monitor1, output_folder = str(brc_path1_rel), fname_suffix='conn')

    edges2 = rg_data2['edges']
    probs2 = rg_data2['probs']
    probs_brc2 = {e: {0: probs2[e]['0']['p'], 1: probs2[e]['1']['p']} for e in edges2}
    brs2, rules2, sys_res2, monitor2 = brc.run(probs_brc2, rg_data2['sys_func_conn_brc'], 
                                               max_nb = 40_000)
    brc_path2 = Path(ds_root2 / "brc")
    brc_path2_rel = brc_path2.relative_to(Path.cwd())
    brc.save_brc_data(rules2, brs2, sys_res2, monitor2, output_folder = str(brc_path2_rel), fname_suffix='conn')
    

