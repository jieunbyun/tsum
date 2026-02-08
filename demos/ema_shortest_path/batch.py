import sys
import os
import pathlib
import torch
import json
import typer
import pdb

from pathlib import Path
import networkx as nx


HOME = Path(__file__).parent
sys.path.append(str(HOME.joinpath('../../../network-datasets/')))

from ndtools import fun_binary_graph as fbg # ndtools available at github.com/jieunbyun/network-datasets
from ndtools.graphs import build_graph

from tsum import tsum

app = typer.Typer()


def s_fun(comps_st):
    travel_time, sys_st, info = fbg.eval_travel_time_to_nearest(
            comps_st, G_base, origin, dests,
            avg_speed=60, # km/h
            target_max = 0.5, # hours: it shouldn't take longer than this compared to the original travel time
            length_attr = 'length_km')

    if sys_st == 's':
       path = info['path_filtered_edges']
       min_comps_st = {eid: ('>=', 1) for eid in path} # edges in the path are working
       min_comps_st['sys'] = ('>=', 1) # system edge is also working

    else:
        min_comps_st = None

    return travel_time, sys_st, min_comps_st


@app.command()
def main():

    global G_base, origin, dests

    DATASET = HOME.joinpath('./data')

    nodes = json.loads((DATASET / "nodes.json").read_text(encoding="utf-8"))
    edges = json.loads((DATASET / "edges.json").read_text(encoding="utf-8"))
    probs_dict = json.loads((DATASET / "probs_bin.json").read_text(encoding="utf-8"))

    G_base = build_graph(nodes, edges, probs_dict)

    #origin = 'n1'
    origin = 'n32'
    dests = ['n22', 'n66']

    row_names = list(edges.keys()) + ['sys']
    n_state = 2 # binary states

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs = [[probs_dict[n]['0']['p'], probs_dict[n]['1']['p']] for n in row_names[:-1]]
    probs = torch.tensor(probs, dtype=torch.float32, device=device)

    # run rule extraction: two options available: tsum.run_rule_extraction or tsum.run_rule_extraction_by_mcs
    result = tsum.run_rule_extraction(
    #result = tsum.run_rule_extraction_by_mcs(
        sfun=s_fun,
        probs=probs,
        row_names=row_names,
        n_state=n_state,
        output_dir="tsum_res",
        surv_json_name="rules_surv.json",
        fail_json_name="rules_fail.json",
        unk_prob_thres = 1e-6
    )


if __name__=='__main__':
    app()
