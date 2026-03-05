import sys
import os
import pathlib
import torch
import json
import typer
import pdb
import pandas as pd

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
            #target_max = 0.5, # hours: it shouldn't take longer than this compared to the original travel time
            target_max = [0.5, 0.25], # hours: it shouldn't take longer than this compared to the original travel time
            length_attr = 'length_km')

    if sys_st >= sys_surv_st:
       path = info['path_filtered_edges']
       min_comps_st = {eid: ('>=', 1) for eid in path} # edges in the path are working
       min_comps_st['sys'] = ('>=', sys_st) # system edge is also working

    else:
        min_comps_st = None

    return travel_time, sys_st, min_comps_st



@app.command()
def check_system():

    prerequites()

    comps_st = {eid: 1 for eid in edges.keys()}
    travel_time, sys_st, info = s_fun(comps_st)
    print(f"travel_time: {travel_time}, sys_st: {sys_st}, info: {info}")


def prerequites():

    global G_base, origin, dests, sys_surv_st, device, probs, edges, edge_names, n_state

    DATASET = HOME.joinpath('./data')

    nodes = json.loads((DATASET / "nodes.json").read_text(encoding="utf-8"))
    edges = json.loads((DATASET / "edges.json").read_text(encoding="utf-8"))
    probs_dict = json.loads((DATASET / "probs_bin.json").read_text(encoding="utf-8"))

    G_base = build_graph(nodes, edges, probs_dict)

    #origin = 'n1'
    origin = 'n52'
    dests = ['n22', 'n66']
    sys_surv_st = 2

    edge_names = list(edges.keys())
    n_state = 2 # binary states of components

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs = [[probs_dict[n]['0']['p'], probs_dict[n]['1']['p']] for n in edge_names]
    probs = torch.tensor(probs, dtype=torch.float32, device=device)


@app.command()
def find_rules():

    prerequites()

    # run rule extraction: two options available: tsum.run_rule_extraction or tsum.run_rule_extraction_by_mcs
    result = tsum.run_rule_extraction_by_mcs(
        sfun=s_fun,
        probs=probs,
        row_names=edge_names,
        n_state=n_state,
        sys_surv_st=sys_surv_st,
        output_dir="tsum_res",
        surv_json_name="rules_surv.json",
        fail_json_name="rules_fail.json",
        unk_prob_thres = 5e-3
    )


def load_results(tsum_path):

    prerequites()

    tsum_path = Path(tsum_path)
    assert tsum_path.exists(), f'tsum_path does not exisist: {tsum_path}'

    rules_mat_surv = torch.load(tsum_path / f"rules_geq_{sys_surv_st}.pt", map_location="cpu")
    rules_mat_surv = rules_mat_surv.to(device)
    rules_mat_fail = torch.load(tsum_path / f"rules_leq_{sys_surv_st-1}.pt", map_location="cpu")
    rules_mat_fail = rules_mat_fail.to(device)

    return rules_mat_surv, rules_mat_fail


def load_results_multi(tsum_path):

    prerequites()

    tsum_path = Path(tsum_path)
    assert tsum_path.exists(), f'tsum_path does not exisist: {tsum_path}'

    rules_dict_mat_surv = {}
    rules_dict_mat_fail = {}

    for sys_surv_st in [1, 2]:  # either 1 or 2
        rules_mat_surv = torch.load(tsum_path / f"rules_geq_{sys_surv_st}.pt", map_location="cpu")
        rules_mat_surv = rules_mat_surv.to(device)
        rules_dict_mat_surv[sys_surv_st] = rules_mat_surv
        rules_mat_fail = torch.load(tsum_path / f"rules_leq_{sys_surv_st-1}.pt", map_location="cpu")
        rules_mat_fail = rules_mat_fail.to(device)
        rules_dict_mat_fail[sys_surv_st] = rules_mat_fail

    return rules_dict_mat_surv, rules_dict_mat_fail


@app.command()
def cal_probs(tsum_path):

    rules_mat_surv, rules_mat_fail = load_results(tsum_path)

    # marginal probability
    pr_cond = tsum.get_comp_cond_sys_prob(
        rules_mat_surv,
        rules_mat_fail,
        probs,
        comps_st_cond = {},
        row_names = edge_names,
        s_fun = s_fun,
        sys_surv_st = sys_surv_st
    )
    print(f"P(sys >= {sys_surv_st}) = {pr_cond['survival']:.3e}")
    print(f"P(sys <= {sys_surv_st-1} ) = {pr_cond['failure']:.3e}\n")


@app.command()
def cal_cond_probs(tsum_path):

    rules_mat_surv, rules_mat_fail = load_results(tsum_path)

    # conditional probability given one components' survival
    for x in edge_names:
        print(f"Eval P(sys | {x}=1)")
        pr_cond = tsum.get_comp_cond_sys_prob(
            rules_mat_surv,
            rules_mat_fail,
            probs,
            comps_st_cond = {x: 1},
            row_names=edge_names,
            s_fun=s_fun,
            sys_surv_st=sys_surv_st
        )
        print(f"P(sys >= {sys_surv_st} | {x}=1) = {pr_cond['survival']:.3e}")
        print(f"P(sys <= {sys_surv_st-1} | {x}=1) = {pr_cond['failure']:.3e}\n")


@app.command()
def cal_probs_multi(tsum_path):

    rules_dict_mat_surv, rules_dict_mat_fail = load_results_multi(tsum_path)

    # marginal probability
    pr_cond = tsum.get_comp_cond_sys_prob_multi(
        rules_dict_mat_surv,
        rules_dict_mat_fail,
        probs,
        comps_st_cond = {},
        row_names = edge_names,
        s_fun = s_fun,
    )
    print(f"P(sys) = {pr_cond}")

@app.command()
def cal_cond_probs_multi(tsum_path):

    rules_dict_mat_surv, rules_dict_mat_fail = load_results_multi(tsum_path)

    results = []
    for x in edge_names:

        # Calculate probabilities
        cond_probs = tsum.get_comp_cond_sys_prob_multi(
                        rules_dict_mat_surv,
                        rules_dict_mat_fail,
                        probs,
                        comps_st_cond = {x: 0}, # 1: survival, 0: failure
                        row_names=edge_names,
                        s_fun=s_fun
                    )

        # Print results
        print(f"P(sys | {x}=0):", cond_probs)

        # Append data as a dictionary to the list
        results.append({"Component": x,
                        "System failure": cond_probs[0],
                        "Partial failure": cond_probs[1],
                        "Survival": cond_probs[2]
                        })

    # Convert the list to a DataFrame
    df_results = pd.DataFrame(results)

    # Save to a JSON file
    output_file = HOME.joinpath('post-processing/cond_sys_probs.json')
    df_results.to_json(output_file, orient="records", indent=4)

    print(f"\nData saved to {output_file}")


if __name__=='__main__':
    app()
