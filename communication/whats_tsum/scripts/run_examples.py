from pathlib import Path
from ndtools.network_generator import GenConfig, generate_and_save

def generate_random_network():

    cfg = GenConfig(
        name="ws_n60_k6_b015",
        generator="ws",
        description="WS n=60 k=6 beta=0.15",
        generator_params={"n_nodes": 60, "k": 6, "p_ws": 0.15, "p_fail": 0.1},
        seed=7,
    )
    repo_root = Path(__file__).resolve().parents[1]
    out_base = repo_root / "results"

    ds_root = generate_and_save(out_base, cfg, draw_graph=True)
    print("Wrote:", ds_root)

if __name__ == "__main__":
    generate_random_network()