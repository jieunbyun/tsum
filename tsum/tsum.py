import torch
import itertools
from itertools import product, combinations
from math import prod
import numpy as np
import os, json, time
from typing import Callable, Dict, Any, List, Optional, Tuple, Sequence
from torch import Tensor

import tsum


def get_min_fail_comps_st(comps_st, max_st, sys_fail_st):
    """
    Get the minimal failing component states from a given state,
    by recording components in comps_st != max_st

    Args:
        comps_st (dict): {comp_name: state (int)}
        max_st (int): the highest state
        sys_fail_st (int): the system failure state

    Returns:
        (dict): {comp_name: ('comparison_operator', state (int))}

    """
    min_comps_st = {k: ('<=', v) for k, v in comps_st.items() if v < max_st}
    min_comps_st['sys'] = ('<=', sys_fail_st)
    return min_comps_st

def get_min_surv_comps_st(comps_st, sys_surv_st):
    """
    Get the minimal surviving component states from a given state,
    by recording components in comps_st != max_st

    Args:
        comps_st (dict): {comp_name: state (int)}
        sys_surv_st (int): the system survival state

    Returns:
        (dict): {comp_name: ('comparison_operator', state (int))}

    """
    min_comps_st = {k: ('>=', v) for k, v in comps_st.items() if v > 0}
    min_comps_st['sys'] = ('>=', sys_surv_st)
    return min_comps_st

def from_rule_dict_to_mat(rule_dict, row_names, max_st):
    """
    Convert a rule dictionary to a matrix representation.

    Args:
        rule_dict (dict): {name: ('comparison_operator', state (int))}
        row_names (list): list of component or system names associated with each row in order
        max_st (int): the highest state

    Returns:
        mat (list): binary matrix with shape (n_comp, max_st)

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mat = torch.zeros((len(row_names), max_st), dtype=torch.int32, device=device)

    for row, name in enumerate(row_names):  
        if name in rule_dict:
            op, state = rule_dict[name]
            if op == '<=':
                mat[row, :state + 1] = 1
            elif op == '<':
                mat[row, :state] = 1
            elif op == '>=':
                mat[row, state:] = 1
            elif op == '>':
                mat[row, state + 1:] = 1
            elif op == '==':
                mat[row, state] = 1
        else:
            mat[row, :] = 1

    return mat

def from_Bbound_to_comps_st(Bbound, row_names):
    """
    Extracts the index of the first non-zero state for each component (ignoring the system row).

    Args:
        Bbound (Tensor): shape (n_var, n_state)
        row_names (list): list of variable names including system

    Returns:
        comps_st (dict): {component_name: state_index}
    """
    n_var, n_state = Bbound.shape
    n_comp = n_var - 1  # exclude system event

    comps_st = {}
    for i in range(n_comp):
        row = Bbound[i]
        nz = torch.nonzero(row, as_tuple=False)
        if len(nz) > 0:
            comps_st[row_names[i]] = int(nz[0])
        else:
            comps_st[row_names[i]] = None  # or raise an error

    return comps_st

def get_branches_cap_branches(B1, B2, batch_size=64):
    """
    Memory-efficient intersection of branches with batching over the larger tensor (B1 or B2).
    Inputs:
        B1: (n_br1, n_var, n_state)
        B2: (n_br2, n_var, n_state)
    Returns:
        Bnew: (n_valid, n_var, n_state)
    """
    device = B1.device
    n_br1, n_var, n_state = B1.shape
    n_br2 = B2.shape[0]
    results = []

    if n_br1 >= n_br2:
        # Batch over B1
        for start in range(0, n_br1, batch_size):
            end = min(start + batch_size, n_br1)
            B1_batch = B1[start:end]                    # (batch_size, n_var, n_state)

            B1_exp = B1_batch.unsqueeze(1)              # (batch_size, 1, n_var, n_state)
            B2_exp = B2.unsqueeze(0)                    # (1, n_br2, n_var, n_state)
            Bnew = B1_exp & B2_exp                      # (batch_size, n_br2, n_var, n_state)
            Bnew = Bnew.view(-1, n_var, n_state)

            # Filter invalid
            invalid_mask = (Bnew == 0).all(dim=2)
            keep_mask = ~invalid_mask.any(dim=1)
            Bnew = Bnew[keep_mask]

            results.append(Bnew)
    else:
        # Batch over B2
        for start in range(0, n_br2, batch_size):
            end = min(start + batch_size, n_br2)
            B2_batch = B2[start:end]

            B1_exp = B1.unsqueeze(1)                    # (n_br1, 1, n_var, n_state)
            B2_exp = B2_batch.unsqueeze(0)              # (1, batch_size, n_var, n_state)
            Bnew = B1_exp & B2_exp                      # (n_br1, batch_size, n_var, n_state)
            Bnew = Bnew.view(-1, n_var, n_state)

            invalid_mask = (Bnew == 0).all(dim=2)
            keep_mask = ~invalid_mask.any(dim=1)
            Bnew = Bnew[keep_mask]

            results.append(Bnew)

    if results:
        return torch.cat(results, dim=0)
    else:
        return torch.empty((0, n_var, n_state), dtype=B1.dtype, device=device)

def get_complementary_events(mat):
    """
    Given a (n_vars, n_state) matrix with the last row as the system event,
    generate a set of complementary logical events (one per component).

    Returns:
        Bnew: (n_comps_kept, n_vars, n_state)
    """
    n_vars, n_state = mat.shape
    n_comps = n_vars - 1  # exclude the system event (last row)

    # Prepare output tensor
    B = torch.ones((n_comps, n_vars, n_state), dtype=mat.dtype, device=mat.device)

    # Broadcast mat for all i
    mat_exp = mat.unsqueeze(0).expand(n_comps, n_vars, n_state)

    # Create lower-triangular mask to copy rows before i
    mask = torch.arange(n_vars, device=mat.device).unsqueeze(0) < torch.arange(n_comps, device=mat.device).unsqueeze(1)  # (n_comps, n_vars)
    mask = mask.unsqueeze(-1).expand(-1, -1, n_state)  # (n_comps, n_vars, n_state)
    B[mask] = mat_exp[mask]  # copy rows before i

    # Flip row i in each batch
    flip_mask = torch.eye(n_comps, dtype=torch.bool, device=mat.device).unsqueeze(-1).expand(-1, -1, n_state)  # (n_comps, n_comps, n_state)
    B[:n_comps, :n_comps][flip_mask] = 1 - mat_exp[:n_comps, :n_comps][flip_mask]

    # Remove combinations where any row (excluding system) is all-zero across states
    invalid_mask = (B[:, :-1, :] == 0).all(dim=2)  # shape: (n_comps, n_comps)
    keep_mask = ~invalid_mask.any(dim=1)          # shape: (n_comps,)
    Bnew = B[keep_mask]

    return Bnew

def get_branch_probs(tensor, prob):
    """
    Computes the probability of each branch given a binary event tensor and state probabilities.

    Args:
        tensor: (n_br, n_var, n_state) - binary indicator of active states per variable per branch
        prob:   (n_var - 1, n_state)   - probability per state for each component variable

    Returns:
        Bprob: (n_br,) - probability per branch
    """
    n_br, n_var, n_state = tensor.shape
    device = tensor.device

    # Cut off system row (last one)
    tensor = tensor[:, :-1, :]  # shape: (n_br, n_comps, n_state)

    # Expand to match tensor: (n_br, n_comps, n_state)
    prob_exp = prob.unsqueeze(0).expand(n_br, -1, -1)

    # Element-wise multiplication and summing across states
    prob_selected = tensor * prob_exp  # (n_br, n_comps, n_state)
    prob_per_var = prob_selected.sum(dim=2)  # (n_br, n_comps)
    Bprob = prob_per_var.prod(dim=1)  # (n_br,)

    return Bprob

import torch

def get_boundary_branches(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute boundary branches for each input branch.

    Input:
        tensor: (n_vars, n_state)  OR  (n_br, n_vars, n_state)
                int/bool tensor with 0/1 entries.

                - n_vars includes the system row as the LAST row.
                - For each component row, the active state(s) are 1s along n_state.
                  We pick:
                    * 'lower' boundary: first active state (min index)
                    * 'upper' boundary: last  active state (max index)

    Output:
        (2, n_vars, n_state)             if input was (n_vars, n_state)
        (2*n_br, n_vars, n_state)        if input was (n_br, n_vars, n_state)

        The last row (system row) is set to all 1s in both upper and lower outputs.
    """
    assert tensor.ndim in (2, 3), "Input must be 2D (n_vars,n_state) or 3D (n_br,n_vars,n_state)"

    # Normalize to 3D (batch of branches)
    squeeze_back = (tensor.ndim == 2)
    if squeeze_back:
        x = tensor.unsqueeze(0)  # (1, n_vars, n_state)
    else:
        x = tensor               # (n_br, n_vars, n_state)

    n_br, n_vars, n_state = x.shape
    n_comps = n_vars - 1

    # Work only on component rows (exclude final system row)
    comp = x[:, :n_comps, :]             # (n_br, n_comps, n_state)
    mask = comp.bool()

    # First active state index per component
    first_hit = (mask.float().cumsum(dim=-1) == 1).float()
    first_idx = first_hit.argmax(dim=-1)  # (n_br, n_comps)

    # Last active state index per component
    rev = torch.flip(mask, dims=[-1])
    last_hit = (rev.float().cumsum(dim=-1) == 1).float()
    last_idx = last_hit.argmax(dim=-1)
    last_idx = (n_state - 1) - last_idx   # (n_br, n_comps)

    # (Optional) If a row has no 1s at all, both argmax above return 0.
    # If you want to *suppress* placing a 1 in such rows, detect and skip:
    # has_one = mask.any(dim=-1)                              # (n_br, n_comps)
    # first_idx = torch.where(has_one, first_idx, -1)         # -1 will be ignored by scatter_
    # last_idx  = torch.where(has_one,  last_idx,  -1)

    # Build lower/upper with one-hot at first/last indices
    lower = torch.zeros_like(comp)
    upper = torch.zeros_like(comp)

    lower.scatter_(-1, first_idx.unsqueeze(-1), 1)
    upper.scatter_(-1, last_idx.unsqueeze(-1), 1)

    # Append system row of all 1s
    system = torch.ones((n_br, 1, n_state), dtype=x.dtype, device=x.device)
    B_lower = torch.cat([lower, system], dim=1)  # (n_br, n_vars, n_state)
    B_upper = torch.cat([upper, system], dim=1)  # (n_br, n_vars, n_state)

    # Stack branches: [upper; lower] along branch dimension
    out = torch.cat([B_upper, B_lower], dim=0)   # (2*n_br, n_vars, n_state)

    # Squeeze back if original was 2D: return shape (2, n_vars, n_state)
    return out if not squeeze_back else out.view(2, n_vars, n_state)


def get_boundary_rules(tensor):
    n_br, n_vars, n_state = tensor.shape
    n_comps = n_vars - 1

    comp_tensor = tensor[:, :n_comps, :]  # (n_br, n_comps, n_state)

    # Create boolean mask of active entries
    mask = comp_tensor.bool()  # (n_br, n_comps, n_state)

    # Get first and last nonzero indices
    first_idx = mask.float().cumsum(dim=2)
    first_idx = (first_idx == 1).float()
    first_idx = first_idx.argmax(dim=2)  # (n_br, n_comps)

    # Reverse to find last
    reversed_mask = torch.flip(mask, dims=[2])
    last_idx = reversed_mask.float().cumsum(dim=2)
    last_idx = (last_idx == 1).float()
    last_idx = last_idx.argmax(dim=2)
    last_idx = n_state - 1 - last_idx  # reverse indices

    # Build upper and lower tensors
    ###### only this part is different from get_boundary_branches #####
    state_idx = torch.arange(n_state, device=last_idx.device).view(1, 1, -1).expand(n_br, n_comps, -1)
    upper = (state_idx >= last_idx.unsqueeze(-1)).to(tensor.dtype)
    lower = (state_idx <= first_idx.unsqueeze(-1)).to(tensor.dtype)
    ####################################################################

    # Append system row of all 1s
    system = torch.ones((n_br, 1, n_state), dtype=tensor.dtype, device=tensor.device)

    B_upper = torch.cat([upper, system], dim=1)
    B_lower = torch.cat([lower, system], dim=1)

    return torch.cat([B_upper, B_lower], dim=0)  # shape: (2*n_br, n_vars, n_state)

def is_intersect(events1, events2):
    """
    Determine whether each event in events1 intersects with any event in events2.

    Args:
        events1: (n_event1, n_vars, n_state)
        events2: (n_event2, n_vars, n_state)

    Returns:
        labels: (n_event1,) boolean tensor
    """
    n_event1, n_vars, n_state = events1.shape
    n_event2, _, _ = events2.shape

    # Expand for broadcasting
    events1_exp = events1.unsqueeze(1).expand(-1, n_event2, -1, -1)
    events2_exp = events2.unsqueeze(0).expand(n_event1, -1, -1, -1)

    # Compute intersection and check if any is non-zero per pair
    intersect = events1_exp & events2_exp  # logical AND
    is_empty = (intersect == 0).all(dim=3).any(dim=2)  # shape: (n_event1, n_event2)
    labels = ~is_empty.all(dim=1)  # if any intersected, mark True

    return labels

def is_subset(mat, tensor):
    """
    Checks if:
      1. `mat` is a subset of any of the events in `tensor`, and
      2. Any of the events in `tensor` is a subset of `mat`.

    Args:
        mat: Tensor of shape (n_var, n_state)
        tensor: Tensor of shape (n_event, n_var, n_state)

    Returns:
        is_mat_subset: bool
        is_tensor_subset: BoolTensor of shape (n_event,)
    """
    n_event, n_var, n_state = tensor.shape
    mat_e = mat.unsqueeze(0).expand(n_event, -1, -1)  # (n_event, n_var, n_state)

    intersect = mat_e & tensor  # (n_event, n_var, n_state)

    is_mat_subset = torch.any(torch.all(mat_e == intersect, dim=(1, 2))).item()
    is_tensor_subset = torch.all(tensor == intersect, dim=(1, 2))  # shape (n_event,)

    return bool(is_mat_subset), is_tensor_subset

import torch
from math import prod

@torch.no_grad()
def find_first_nonempty_combination(Rcs, batch_size=65536, verbose=False):
    """
    Rcs: list[Tensor] with shapes (n_i, n_vars, n_state), same (n_vars, n_state)
    Order: increasing sum of tuple indices, then lexicographic within that sum.
    Returns: (selected_mat: (n_vars, n_state), idx_tuple) or (None, None)
    """
    assert len(Rcs) > 0
    device = Rcs[0].device
    n_vars, n_state = Rcs[0].shape[1:]
    ns = torch.tensor([r.shape[0] for r in Rcs], device=device, dtype=torch.long)
    k = len(ns)
    assert all((r.device == device and r.shape[1]==n_vars and r.shape[2]==n_state) for r in Rcs)

    # total combinations and linear-index strides (mixed radix, right-to-left)
    n_combs = int(torch.prod(ns).item())
    if n_combs == 0:
        return None

    strides = torch.ones_like(ns)
    if k > 1:
        strides[:-1] = torch.cumprod(ns.flip(0)[:-1], dim=0).flip(0)  # lex rank weights too

    # maximum possible sum level
    max_sum = int((ns - 1).sum().item())

    # scan sum shells s = 0..max_sum
    for s in range(max_sum + 1):
        if verbose:
            print(f"[sum={s}] scanning...")
        start = 0

        best_lex_rank = None
        best_sel_mat = None
        best_tuple = None
        best_global_idx = None

        while start < n_combs:
            end = min(start + batch_size, n_combs)

            # linear indices (GPU)
            lin = torch.arange(start, end, device=device, dtype=torch.long)

            # decode to tuples (batch, k)
            idx = (lin[:, None] // strides) % ns

            # filter rows at the current sum level
            sum_mask = (idx.sum(dim=1) == s)
            if sum_mask.any():
                idx_s = idx[sum_mask]
                # gather the needed rows only (saves compute)
                mats = [r[idx_s[:, i]] for i, r in enumerate(Rcs)]
                mat = torch.stack(mats, dim=0).prod(dim=0)  # (batch_s, n_vars, n_state)

                # non-empty check (your original rule)
                is_empty = (mat == 0).all(dim=2).any(dim=1)
                valid = ~is_empty

                if valid.any():
                    # lexicographic rank within this sum shell
                    lex_rank = (idx_s * strides).sum(dim=1)  # dot with strides
                    # among valid, choose min lex
                    lex_rank_valid = lex_rank.clone()
                    # mask out invalid by setting to +inf
                    lex_rank_valid[~valid] = torch.iinfo(torch.int64).max

                    # candidate in this batch
                    batch_min_lex, batch_pos = torch.min(lex_rank_valid, dim=0)
                    if batch_min_lex != torch.iinfo(torch.int64).max:
                        # global best within this sum s (merge across batches)
                        if (best_lex_rank is None) or (batch_min_lex < best_lex_rank):
                            best_lex_rank = batch_min_lex
                            best_sel_mat = mat[batch_pos]              # (n_vars, n_state)
                            best_tuple = tuple(int(v) for v in idx_s[batch_pos].tolist())
                            best_global_idx = int((idx_s[batch_pos] * strides).sum().item())

            start = end

        if best_sel_mat is not None:
            if verbose:
                print(f"Selected index: {best_tuple} (sum={s}, lex_rank={int(best_lex_rank)}, lin={best_global_idx})")
            return best_sel_mat

    return None


def sum_sorted_tuples_limited(max_vals):
    """
    Generate all tuples of non-negative integers with len=max_vals,
    where each element i ≤ max_vals[i],
    ordered by increasing sum, then lexicographically.
    
    Args:
        max_vals (list or tuple): list of maximum values per position.
    
    Yields:
        tuple of ints
    """
    n = len(max_vals)
    sum_level = 0
    while True:
        found = False
        for t in itertools.product(*(range(v+1) for v in max_vals)):
            if sum(t) == sum_level:
                yield t
                found = True
        if not found:
            break  # no more combinations possible
        sum_level += 1

def merge_branches(B):
    "Use hashing for computational efficiency"

    is_merge = True

    while is_merge:
        B_com = bit_compress(B)
        groups_by_col = groups_by_column_remhash_dict(B_com)
        merges = plan_merges(groups_by_col, B.shape[0])
        B, _ = apply_merges(B, merges)

        is_merge = any(len(g) > 1 for g in groups_by_col)

    return B

def merge_branches_old(B, batch_size=100_000):
    device = B.device
    dtype = B.dtype

    B = B.clone()
    changed = True

    while changed:
        changed = False
        n_br, n_comp, n_state = B.shape
        keep_mask = torch.ones(n_br, dtype=torch.bool, device=device)
        new_branches = []

        # Generate all i < j combinations
        all_pairs = list(combinations(range(n_br), 2))
        total_pairs = len(all_pairs)

        used = torch.zeros(n_br, dtype=torch.bool, device=device)

        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)
            idx_i, idx_j = zip(*all_pairs[start:end])
            idx_i = torch.tensor(idx_i, device=device)
            idx_j = torch.tensor(idx_j, device=device)

            bi = B[idx_i]  # (n_pair, n_comp, n_state)
            bj = B[idx_j]

            # Step 1: Compare along components to count differing rows
            diffs = (bi != bj).any(dim=2)  # (n_pair, n_comp)
            num_diff_rows = diffs.sum(dim=1)  # (n_pair,)
            one_diff_mask = num_diff_rows == 1

            if one_diff_mask.sum() == 0:
                continue  # no valid pairs in this batch

            valid_idx_i = idx_i[one_diff_mask]
            valid_idx_j = idx_j[one_diff_mask]
            valid_diffs = diffs[one_diff_mask]
            valid_bi = bi[one_diff_mask]
            valid_bj = bj[one_diff_mask]

            diff_row_idx = valid_diffs.float().argmax(dim=1)  # (n_valid_pairs,)

            # Extract differing rows
            ri = torch.stack([valid_bi[k, diff_row_idx[k]] for k in range(len(diff_row_idx))])
            rj = torch.stack([valid_bj[k, diff_row_idx[k]] for k in range(len(diff_row_idx))])

            disjoint_mask = (ri & rj).sum(dim=1) == 0
            if disjoint_mask.sum() == 0:
                continue

            final_merge_indices = []
            for k in range(disjoint_mask.size(0)):
                if not disjoint_mask[k]:
                    continue
                i = valid_idx_i[k].item()
                j = valid_idx_j[k].item()
                if used[i] or used[j]:
                    continue
                used[i] = True
                used[j] = True
                final_merge_indices.append(k)

            if not final_merge_indices:
                continue

            changed = True
            final_merge_indices = torch.tensor(final_merge_indices, device=device)

            disjoint_i = valid_idx_i[final_merge_indices]
            disjoint_j = valid_idx_j[final_merge_indices]
            disjoint_bi = B[disjoint_i]
            disjoint_bj = B[disjoint_j]
            disjoint_diff_idx = diff_row_idx[disjoint_mask][final_merge_indices]

            for i in range(disjoint_i.size(0)):
                merged = disjoint_bi[i].clone()
                merged[disjoint_diff_idx[i]] = disjoint_bi[i][disjoint_diff_idx[i]] | disjoint_bj[i][disjoint_diff_idx[i]]
                new_branches.append(merged)

        keep_mask[used] = False
        if new_branches:
            B = torch.cat([B[keep_mask], torch.stack(new_branches)], dim=0)

    return B

def get_complementary_events_nondisjoint(mat: torch.Tensor) -> torch.Tensor:
    """
    Given a (n_vars, n_state) matrix with the last row as the system event,
    generate a set of complementary logical events by flipping each row.
    NOTE: The resulted events are not disjoint.

    Returns:
        Bnew: (n_comps_kept, n_vars, n_state)
    """
    n_vars, n_state = mat.shape
    n_comps = n_vars - 1  # exclude the system event (last row)

    # Prepare output tensor
    B = torch.ones((n_comps, n_vars, n_state), dtype=mat.dtype, device=mat.device)

    # Flip row i in batch i
    idx = torch.arange(n_comps, device=mat.device)
    if mat.dtype == torch.bool:
        B[idx, idx, :] = ~mat[idx, :]
    else:
        # assumes binary in {0,1}; works for float or int tensors
        B[idx, idx, :] = 1 - mat[idx, :]

    # Remove combinations where any row (excluding system) is all-zero across states
    invalid_mask = (B[:, :-1, :] == 0).all(dim=2)  # shape: (n_comps, n_comps)
    keep_mask = ~invalid_mask.any(dim=1)          # shape: (n_comps,)
    Bnew = B[keep_mask]

    return Bnew

def bit_compress(B: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor B of shape (n, m, k) of bits {0,1}
    into an integer tensor of shape (n, m),
    where each element is sum_k 2^k * B[i,j,k].
    """
    n, m, k = B.shape
    # weights = [1, 2, 4, ..., 2^(k-1)]
    weights = (2 ** torch.arange(k, device=B.device, dtype=torch.int32))
    return (B.to(torch.int32) * weights).sum(dim=2)


def groups_by_column_remhash_dict(X: torch.Tensor):
    """
    For each column j, return groups of row indices that are identical on all
    other columns but differ on column j. Uses removable hashes + CPU dict
    (expected O(m n)) and includes a collision-guard verification.
    """
    X = X.to(torch.long)
    device = X.device
    n, m = X.shape
    out = [[] for _ in range(m)]
    if n == 0 or m == 0:
        return out

    # Two primes + per-column coefficients
    p1 = 2_147_483_629
    p2 = 2_147_483_647
    a1 = torch.arange(1, m + 1, device=device, dtype=torch.long)
    a2 = (a1 * 1315423911) % p2

    # Precompute full-row hashes
    H1 = (X * a1).sum(dim=1) % p1
    H2 = (X * a2).sum(dim=1) % p2

    for j in range(m):
        H1_wo = (H1 - X[:, j] * a1[j]) % p1
        H2_wo = (H2 - X[:, j] * a2[j]) % p2

        # skinny keys to CPU dict
        keys = torch.stack((H1_wo, H2_wo), dim=1).cpu().tolist()
        buckets = {}
        for i, k in enumerate(keys):
            buckets.setdefault((int(k[0]), int(k[1])), []).append(i)

        for rows in buckets.values():
            if len(rows) < 2:
                continue
            rows_t = torch.tensor(rows, device=device)
            vals_j = X[rows_t, j]
            # must truly differ at column j
            if torch.unique(vals_j).numel() < 2:
                continue

            # --- collision guard: check equality on all other columns ---
            Xg = X[rows_t]  # (s, m)
            same_cols = (Xg == Xg[0]).all(dim=0)  # (m,) True if all rows equal in that column
            # require all columns except j to be identical
            if bool(same_cols[torch.arange(m, device=device) != j].all()):
                out[j].append(rows_t)

    return out


def plan_merges(groups_per_col, n_rows):
    """
    groups_per_col: list where groups_per_col[j] is a list of 1D LongTensors of row indices (same device)
    n_rows: total number of rows
    returns: list of (i, k, j) merges, greedy, non-overlapping across all columns
    """
    # Track rows already used in a merge
    # Keep this on CPU bool for simplicity; adjust to CUDA if you prefer
    used = torch.zeros(n_rows, dtype=torch.bool)
    merges = []

    for j, groups in enumerate(groups_per_col):
        for g in groups:
            # Greedily pair left-to-right inside this group, skipping used rows
            # Note: keep device of g, but we only read its indices here
            # Collect unused indices in order
            unused = [int(idx) for idx in g.tolist() if not used[int(idx)].item()]
            # Pair consecutive unused
            for t in range(0, len(unused) - 1, 2):
                i, k = unused[t], unused[t+1]
                if used[i] or used[k]:
                    continue
                merges.append((i, k, j))
                used[i] = True
                used[k] = True
            # If odd count, last one is left unmatched (as you wanted)

    return merges

def apply_merges(B, merges, reducer="or"):
    """
    B: (n, m, k) tensor (CPU or CUDA)
    merges: list of (i, k, j)
    reducer: "or" (clip sum to {0,1}), "sum" (raw sum), or "max"
    Returns: (B_merged, kept_indices)
      - B_merged: tensor with merged rows; second rows in pairs are removed
      - kept_indices: 1D LongTensor mapping new rows back to old indices
    """
    device = B.device
    n, m, k = B.shape
    keep = torch.ones(n, dtype=torch.bool, device=device)

    for (i, k_idx, j) in merges:
        i = int(i); k_idx = int(k_idx); j = int(j)
        if reducer == "or":
            B[i, j] = torch.clamp(B[i, j] + B[k_idx, j], min=0, max=1)
        elif reducer == "sum":
            B[i, j] = B[i, j] + B[k_idx, j]
        elif reducer == "max":
            B[i, j] = torch.maximum(B[i, j], B[k_idx, j])
        else:
            raise ValueError("reducer must be 'or', 'sum', or 'max'")
        keep[k_idx] = False  # drop the second row of the pair

    kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
    B_new = B[keep]
    return B_new, kept_indices

def sample_new_comp_st_to_test(probs, rules_mat, B=1_024, max_iters=1_000):

    device = probs.device
    n_comp, n_state = probs.shape
    n_var = n_comp + 1  # including system event

    if len(rules_mat) == 0:
        all_samples = torch.ones((1, n_var, n_state), dtype=torch.int32, device=device)
        return all_samples[0], all_samples

    all_samples = torch.empty((0, n_var, n_state), dtype=torch.int32, device=device)

    for iter in range(max_iters):

        # Start with all-ones batch
        samples_b = torch.ones((B, n_var, n_state), dtype=torch.int32, device=device)

        # Strategy 1: The same permutation applies within a batch
        rules_ord = np.random.permutation(len(rules_mat))
        # Strategy 2: Sort the rules by their probs
        #rules_probs = get_branch_probs(rules_mat, probs)
        #rules_ord = torch.argsort(rules_probs, descending=True)

        # Sampling starts.
        for r_idx in rules_ord:

            r_mat = rules_mat[r_idx]
            r_mat_c = get_complementary_events_nondisjoint(r_mat)

            # Decide whether to sample: skip samples that already contradicts r_mat (to obtain minimal rules)
            is_sampled = torch.ones((B,), dtype=torch.bool, device=device)
            for rc1 in r_mat_c:
                flag1, flag2 = is_subset(rc1, samples_b)

                is_sampled[flag2] = False

            # Select a r_mat_c
            r_mat_c_probs = get_branch_probs(r_mat_c, probs)
            r_mat_c_probs = r_mat_c_probs / r_mat_c_probs.sum()
            idx = torch.multinomial(r_mat_c_probs, num_samples=B, replacement=True)

            # Update samples if is_sampled == True
            samples_b[is_sampled] = samples_b[is_sampled] * r_mat_c[idx[is_sampled]].squeeze(0)

        # Check if there are events with positive prob
        real_prs = get_branch_probs(samples_b, probs)

        all_samples = torch.cat((all_samples, samples_b), dim=0)

        if (real_prs > 0).any():

            x = torch.randint(0, 2, (1,)).item() # which strategy to select?
            # Strategy 1: pick the rule with the highest probability
            if x == 0:
                s_idx = torch.argmax(real_prs)
            # Strategy 2: pick the lowest probability rule
            else:
                # Replace non-positives with +inf so they don't get picked
                masked = torch.where(real_prs > 0, real_prs, torch.inf)  # (B,1)
                s_idx = torch.argmin(masked)  # scalar index into the flattened tensor
            
            sample = samples_b[s_idx] 
            
            bound_br = get_boundary_branches(sample.unsqueeze(0))
            ## decide whether to check upper or lower bound first
            x = torch.randint(0, 2, (1,)).item()
            #x = 1 # check the upper bound first
            is_b_subset, _ = is_subset(bound_br[x], rules_mat) 
            if not is_b_subset:
                return bound_br[x], all_samples
            else:
                is_a_subset, _ = is_subset(bound_br[1-x], rules_mat) 
                if not is_a_subset:
                    return bound_br[1-x], all_samples
                else:
                    Warning("Both boundary branches are subsets of the existing rules. Something's wrong.")
            
            # Strategy 2: pick the branch with the highest probability
            """samples_b = samples_b[real_prs > 0]
            samples_br = get_boundary_rules(samples_b)
            br_prs = get_branch_probs(samples_br, probs)
            br_idx = torch.argsort(br_prs, descending=True)
            for b_idx in br_idx:
                bound_br = samples_br[b_idx]
                # decide whether to check upper or lower bound first
                is_b_subset, _ = is_subset(bound_br, rules_mat) 
                if not is_b_subset:
                    return bound_br, all_samples"""

        elif iter == max_iters - 1:
            print("Max iterations reached without finding a valid sample.")
            return None, all_samples
        
def sample_complementary_events(probs, rules_mat, rules_mat_other, rules_st = 'fail', B=1_024, max_iters=1_000):
    """
    Sample complementary events to the given rules_mat (that represents either fail or surv), ensuring they are not subsets of rules_mat_other.
    """
    device = probs.device
    n_comp, n_state = probs.shape
    n_var = n_comp + 1  # including system event

    n_samp = 0
    for iter in range(max_iters):
        n_samp += B

        # Start with all-ones batch
        samples_b = torch.ones((B, n_var, n_state), dtype=torch.int32, device=device)

        # Strategy 1: The same permutation applies within a batch
        # rules_ord = np.random.permutation(len(rules_mat))
        # Strategy 2: Sort the rules by their probs
        rules_probs = get_branch_probs(rules_mat, probs)
        rules_ord = torch.argsort(rules_probs, descending=True)

        # Sampling starts.
        for r_idx in rules_ord:

            r_mat = rules_mat[r_idx]
            r_mat_c = get_complementary_events_nondisjoint(r_mat)

            # Decide whether to sample: skip samples that already contradicts r_mat (to obtain minimal rules)
            is_sampled = torch.ones((B,), dtype=torch.bool, device=device)
            for rc1 in r_mat_c:
                flag1, flag2 = is_subset(rc1, samples_b)

                is_sampled[flag2] = False

            # Select a r_mat_c
            r_mat_c_probs = get_branch_probs(r_mat_c, probs)
            r_mat_c_probs = r_mat_c_probs / r_mat_c_probs.sum()
            idx = torch.multinomial(r_mat_c_probs, num_samples=B, replacement=True)

            # Update samples if is_sampled == True
            samples_b[is_sampled] = samples_b[is_sampled] * r_mat_c[idx[is_sampled]].squeeze(0)

        # Check if there are events with positive prob
        real_prs = get_branch_probs(samples_b, probs)
        samples_b = samples_b[real_prs > 0]
        real_prs = real_prs[real_prs > 0]

        if samples_b.shape[0] > 0:

            samples_b_bnd = get_boundary_branches(samples_b)
            n_samp_b = samples_b.shape[0]

            if rules_st == 'fail':
                # Get the candidate survival rules
                samples_b_bnd = samples_b_bnd[n_samp_b:]
                samples_b_rule = mask_from_first_one(samples_b_bnd, mode='after', keep_last_row=True)

                is_not_subset = torch.ones((samples_b_rule.shape[0],), dtype=torch.bool, device=device)
                for s_idx, samp in enumerate(samples_b_rule):
                    flag1, flag2 = is_subset(samp[:-1, :], rules_mat_other[:, :-1, :])
                    if flag1:
                        is_not_subset[s_idx] = False
                samples_b = samples_b[is_not_subset]
                real_prs = real_prs[is_not_subset]
                samples_b_bnd = samples_b_bnd[is_not_subset]
                samples_b_rule = samples_b_rule[is_not_subset]

                rule_prs = get_branch_probs(samples_b_rule, probs)
                if any(rule_prs > 0):
                    s_idx = torch.argmax(rule_prs)  
                    sample_bnd = samples_b_bnd[s_idx]

                # Get survival-likely branches with minimal survival conditions (i.e. lower bound)
                return sample_bnd, n_samp
            
            else: # rules_st == 'surv'
                # Get the candidate failure rules
                samples_b_bnd = samples_b_bnd[:n_samp_b]
                samples_b_rule = mask_from_first_one(samples_b_bnd, mode='before', keep_last_row=True)

                is_not_subset = torch.ones((samples_b_rule.shape[0],), dtype=torch.bool, device=device)
                for s_idx, samp in enumerate(samples_b_rule):
                    flag1, flag2 = is_subset(samp[:-1, :], rules_mat_other[:, :-1, :])
                    if flag1:
                        is_not_subset[s_idx] = False
                samples_b = samples_b[is_not_subset]
                real_prs = real_prs[is_not_subset]
                samples_b_bnd = samples_b_bnd[is_not_subset]
                samples_b_rule = samples_b_rule[is_not_subset]

                rule_prs = get_branch_probs(samples_b_rule, probs)
                if any(rule_prs > 0):
                    s_idx = torch.argmax(rule_prs)  
                    sample_bnd = samples_b_bnd[s_idx]

                # Get survival-likely branches with minimal survival conditions (i.e. lower bound)
                return sample_bnd, n_samp

    print("Max iterations reached without finding a valid sample.")
    return None, n_samp

def sample_complementary_events_max_prob_same_rule(probs, rules_mat, rules_mat_other, rules_st = 'fail', B=1_024, max_iters=1_000, stochastic_search = True, gamma = 1.2):
    """
    Sample complementary events to the given rules_mat (that represents either fail or surv), ensuring they are not subsets of rules_mat_other.
    This version selects the complementary event with the maximum probability of finding the rule with the same state.
    """
    device = probs.device
    n_comp, n_state = probs.shape
    n_var = n_comp + 1  # including system event

    n_samp = 0
    for iter in range(max_iters):
        n_samp += B

        # Start with all-ones batch
        samples_b = torch.ones((B, n_var, n_state), dtype=torch.int32, device=device)

        # Strategy 1: The same permutation applies within a batch
        # rules_ord = np.random.permutation(len(rules_mat))
        # Strategy 2: Sort the rules by their probs
        rules_probs = get_branch_probs(rules_mat, probs)
        rules_ord = torch.argsort(rules_probs, descending=True)

        # Sampling starts.
        for r_idx in rules_ord:

            r_mat = rules_mat[r_idx]
            r_mat_c = get_complementary_events_nondisjoint(r_mat)

            # Decide whether to sample: skip samples that already contradicts r_mat (to obtain minimal rules)
            is_sampled = torch.ones((B,), dtype=torch.bool, device=device)
            for rc1 in r_mat_c:
                flag1, flag2 = is_subset(rc1, samples_b)

                is_sampled[flag2] = False

            # Select a r_mat_c
            r_mat_c_probs = get_branch_probs(r_mat_c, probs)
            r_mat_c_probs = r_mat_c_probs / r_mat_c_probs.sum()
            idx = torch.multinomial(r_mat_c_probs, num_samples=B, replacement=True)

            # Update samples if is_sampled == True
            samples_b[is_sampled] = samples_b[is_sampled] * r_mat_c[idx[is_sampled]].squeeze(0)

        # Check if there are events with positive prob
        real_prs = get_branch_probs(samples_b, probs)
        samples_b = samples_b[real_prs > 0]
        real_prs = real_prs[real_prs > 0]

        if samples_b.shape[0] > 0:

            if rules_st == 'fail':
                #From failures' complementary events, one only gets [..0..1..] or [..1..]
                is_not_subset = torch.ones((samples_b.shape[0],), dtype=torch.bool, device=device)
                for s_idx, samp in enumerate(samples_b):
                    flag1, flag2 = is_subset(samp[:-1, :], rules_mat_other[:, :-1, :])
                    if flag1:
                        is_not_subset[s_idx] = False
                samples_b = samples_b[is_not_subset]
                real_prs = real_prs[is_not_subset]
                if any(real_prs > 0):
                    if stochastic_search:
                        min_val = real_prs[real_prs > 0].min() if (real_prs > 0).any() else torch.tensor(1.0)
                        eps = min_val / 1000.0
                        weights = 1.0 / (real_prs + eps) ** gamma # 0 < γ < 1 → flattens the distribution, small values of real_prs still get higher chance, but less extreme; if γ > 1 → sharpens the distribution, very small values of real_prs get much higher chance
                        weights = weights / weights.sum()
                        s_idx = torch.multinomial(weights, num_samples=1).item()
                    else:
                        s_idx = torch.argmin(real_prs)  # maximise the probability of 0's that would likely form the new rule
                        
                    sample_b = samples_b[s_idx]

                    # Get failure-likely branches with minimal failure conditions (i.e. lower bound)
                    sample_bnd = get_boundary_branches(sample_b.unsqueeze(0))
                    sample_bnd = sample_bnd[1] # get the lower bound

                    return sample_bnd, n_samp
            
            else: # rules_st == 'surv'
                # From survivals' complementary events, one only gets [..1..0..] or [..1..]
                is_not_subset = torch.ones((samples_b.shape[0],), dtype=torch.bool, device=device)
                for s_idx, samp in enumerate(samples_b):
                    flag1, flag2 = is_subset(samp[:-1, :], rules_mat_other[:, :-1, :]) # check if the sampled event is a subset of the existing failure rules
                    if flag1:
                        is_not_subset[s_idx] = False
                samples_b = samples_b[is_not_subset]
                real_prs = real_prs[is_not_subset]

                if any(real_prs > 0):
                    if stochastic_search:
                        min_val = real_prs[real_prs > 0].min() if (real_prs > 0).any() else torch.tensor(1.0)
                        eps = min_val / 1000.0
                        weights = 1.0 / (real_prs + eps) ** gamma # 0 < γ < 1 → flattens the distribution, small values of real_prs still get higher chance, but less extreme; if γ > 1 → sharpens the distribution, very small values of real_prs get much higher chance
                        weights = weights / weights.sum()
                        s_idx = torch.multinomial(weights, num_samples=1).item()
                    else:
                        s_idx = torch.argmin(real_prs)  # maximise the probability of 0's that would likely form the new rule

                    sample_b = samples_b[s_idx]

                    # Get survival-likely branches with minimal survival conditions (i.e. upper bound)
                    sample_bnd = get_boundary_branches(sample_b.unsqueeze(0))
                    sample_bnd = sample_bnd[0] # get the upper bound
                    return sample_bnd, n_samp

    print("Max iterations reached without finding a valid sample.")
    return None, n_samp

def classify_samples(samples, survival_rules, failure_rules):
    """
    Classify samples as survival, failure, or unknown using subset checks.

    Args:
        samples: (n_sample, n_var, n_state) sample tensor (binary)
        survival_rules: list of rule tensors, each shape (n_var, n_state)
        failure_rules: list of rule tensors, each shape (n_var, n_state)

    Returns:
        counts: dict with keys 'survival', 'failure', 'unknown'
    """

    device = samples.device
    n_sample = samples.shape[0]

    # Tracking masks
    classified = torch.zeros(n_sample, dtype=torch.bool, device=device)
    survival_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)
    failure_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)

    # Convert list to tensor stack
    all_rules = [(r[:-1, :], 'survival') for r in survival_rules] + [(r[:-1, :], 'failure') for r in failure_rules] # exclude the last row which represents the system state

    for rule_tensor, label in all_rules:
        # Only apply to unclassified samples
        unclassified_idx = ~classified
        if not unclassified_idx.any():
            break

        current_samples = samples[unclassified_idx]  # shape (n_curr, n_var, n_state)

        rule = rule_tensor.to(device).bool()
        current_samples = current_samples.bool()

        # Subset check
        is_subset = torch.all((current_samples & rule) == current_samples, dim=(1, 2))
        idx_all = torch.where(unclassified_idx)[0]
        matched_idx = idx_all[is_subset]

        if label == 'survival':
            survival_mask[matched_idx] = True
        else:
            failure_mask[matched_idx] = True

        classified[matched_idx] = True

    counts = {
        'survival': survival_mask.sum().item(),
        'failure': failure_mask.sum().item(),
        'unknown': (~classified).sum().item()
    }
    return counts

def sample_categorical(probs, n_sample):
    """
    Sample binary event tensors from categorical distributions.

    Args:
        probs: (n_var, n_state) - probabilities per state per variable.
        n_sample: Number of samples to draw.

    Returns:
        samples: (n_sample, n_var, n_state) - one-hot encoded state selection.
    """

    device = probs.device

    n_var, n_state = probs.shape

    # Step 1: Cumulative probability
    cum_probs = torch.cumsum(probs, dim=1)  # shape (n_var, n_state)

    # Step 2: Uniform random values for each variable
    rand_vals = torch.rand(n_sample, n_var, device=device)  # shape (n_sample, n_var)

    # Step 3: Use searchsorted to get index of selected state
    # cum_probs: (n_var, n_state) → expand to (n_sample, n_var, n_state)
    cum_probs_exp = cum_probs.unsqueeze(0).expand(n_sample, -1, -1)  # (n_sample, n_var, n_state)
    rand_vals_exp = rand_vals.unsqueeze(2)  # (n_sample, n_var, 1)

    # state_indices: (n_sample, n_var)
    state_indices = torch.sum(rand_vals_exp > cum_probs_exp, dim=2)

    # Step 4: One-hot encode
    samples = torch.nn.functional.one_hot(state_indices, num_classes=n_state).int()  # (n_sample, n_var, n_state)

    return samples

def mask_from_first_one(
    x: torch.Tensor,
    mode: str = "after",
    keep_last_row: bool = True,
) -> torch.Tensor:
    """
    Create masks relative to the first 1 in each row.

    Args:
        x: (n_row, n_col) or (batch, n_row, n_col) int/bool tensor with 0/1 entries
        mode:
            - "after"  → ones from first 1 (inclusive) to end
            - "before" → ones from start up to first 1 (inclusive)
        keep_last_row: if True, copy the last row from x into the mask (per batch item)

    Returns:
        Tensor of same shape as x, dtype=int32, device preserved.
    """
    assert x.ndim in (2, 3), "x must be 2D or 3D"
    device = x.device

    # Normalize to 3D: (B, N, M)
    squeeze_back = (x.ndim == 2)
    if squeeze_back:
        x3 = x.unsqueeze(0)
    else:
        x3 = x

    B, N, M = x3.shape

    # Column indices for broadcasting comparisons
    cols = torch.arange(M, device=device).view(1, 1, M).expand(B, N, M)

    # First index of "1" per row
    x_bool = (x3 == 1) if x3.dtype != torch.bool else x3
    has_one = x_bool.any(dim=2)                 # (B, N)
    first_idx = x_bool.int().argmax(dim=2)      # (B, N); 0 if none
    first_idx = torch.where(has_one, first_idx, torch.full_like(first_idx, M))

    if mode == "after":
        mask = cols >= first_idx.unsqueeze(-1)  # (B, N, M)
    elif mode == "before":
        mask = cols <= first_idx.unsqueeze(-1)  # (B, N, M)
    else:
        raise ValueError("mode must be 'after' or 'before'")

    if keep_last_row:
        # Preserve the last row exactly as in x
        mask[:, -1, :] = x3[:, -1, :].bool()

    mask = mask.to(torch.int32)

    return mask.squeeze(0) if squeeze_back else mask

def update_rules(min_comps_st, rules_dict, rules_mat, row_names, verbose=False):
    _, _, n_state = rules_mat.shape
    Rnew = from_rule_dict_to_mat(min_comps_st, row_names, n_state)
    is_Rnew_subset, are_Rset_subset = is_subset(Rnew, rules_mat)

    if is_Rnew_subset:
        if verbose:
            print("WARNING: New rule is a subset of existing rules. No update made.")
        return rules_dict, rules_mat
    
    rules_mat = rules_mat[~are_Rset_subset,:,:]
    rules_dict = [r for r, keep in zip(rules_dict, ~are_Rset_subset) if keep]

    rules_dict.append(min_comps_st)
    rules_mat = torch.cat((rules_mat, Rnew.unsqueeze(0)), dim=0)
    if verbose:
        print("No. of existing rules removed: ", int(sum(are_Rset_subset)))

    return rules_dict, rules_mat

def run_rule_extraction(
    *,
    # Problem-specific callables / data
    sfun: Callable[[Dict[str, int]], Tuple[Any, Any, Any]],
    probs: Tensor,
    row_names: List[str],
    n_state: int,
    rules_surv: List[Dict[str, Any]] = [],
    rules_fail: List[Dict[str, Any]] = [],
    rules_mat_surv: Tensor = None,
    rules_mat_fail: Tensor = None,
    # Analysis parameters
    stochastic_search: bool = True,
    gamma: float = 0.5, # if stochastic_search==False, ignored. 0 < γ < 1 → more emphasis on exploration; γ > 1 → more emphasis on exploitation
    # Termination / threshold settings
    unk_prob_thres: float = 5e-2,
    # Frequencies / sampling settings
    prob_update_every: int = 50,   # (2) how often to test system probabilities/bounds
    save_every: int = 10,          # (4) how often to persist logs/rules
    n_sample: int = 1_000_000,
    sample_batch_size: int = 1_000_000,
    rule_search_batch_size: int = 1_024,    # sampler batch for candidate rule search
    rule_search_max_iters: int = 10,
    # Display / verbose
    rule_update_verbose: bool = True,
    # Output control
    output_dir: str = "tsum_temp",
    surv_json_name: str = "rules_surv.json",
    fail_json_name: str = "rules_fail.json",
    surv_pt_name: str = "rules_surv.pt",
    fail_pt_name: str = "rules_fail.pt"
) -> Dict[str, Any]:
    """
    Runs the survival/failure rule discovery loop (steps 3 & 4 only),
    periodically evaluates unknown probability via sampling, and logs metrics.

    Returns a dict with updated rules, rule matrices, threshold lists, and the in-memory metrics log.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ---- helpers ----
    def _avg_rule_len(rule_store: Any) -> float:
        """
        Try to estimate average number of conditions in current rules.
        Length of rule dictionary minus system event: len(rule) - 1
        Works for list-of-dictionaries; returns 0.0 if unavailable.
        """
        try:
            if rule_store is None:
                return 0.0
            # If it's a list-like of rules:
            if hasattr(rule_store, "__len__") and len(rule_store) > 0:
                total = 0
                count = 0
                for r in rule_store:
                    total += len(r) - 1
                    count += 1
                return float(total) / count if count else 0.0
        except Exception:
            pass
        return 0.0

    def _save_json(obj, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)

    def _save_pt(t: torch.Tensor, path: str) -> None:
        torch.save(t.detach().cpu(), path)

    # ---- initial state ----
    device = probs.device
    n_sample_loop = max(int(n_sample // sample_batch_size), 1)

    unk_prob = 1.0
    n_round = 0
    metrics_log: List[Dict[str, Any]] = []

    n_vars = len(row_names)
    n_comps = n_vars - 1 # exclude system event
    if rules_mat_surv is None:
        rules_mat_surv = torch.empty((0,n_vars,n_state), dtype=torch.int32, device=device)
    if rules_mat_fail is None:
        rules_mat_fail = torch.empty((0,n_vars,n_state), dtype=torch.int32, device=device)

    # Threshold discovery bookkeeping
    sys_val_list = []

    # JSONL file for metrics (append-only)
    metrics_path = os.path.join(output_dir, f"metrics.jsonl")
    # snapshot rules paths
    rules_surv_path = os.path.join(output_dir, surv_json_name)
    rules_fail_path = os.path.join(output_dir, fail_json_name)
    rules_surv_pt_path = os.path.join(output_dir, surv_pt_name)
    rules_fail_pt_path = os.path.join(output_dir, fail_pt_name)

    # while flags: use only steps 3 & 4 flags
    is_new_surv_cand, is_new_fail_cand = True, True

    # last known probabilities (only updated when recomputed)
    last_probs = {"survival": None, "failure": None, "unknown": None}

    # ---- main loop ----
    while (is_new_surv_cand or is_new_fail_cand) and (unk_prob > unk_prob_thres):
        n_round += 1
        t0 = time.perf_counter()

        print(f"Round: {n_round}, Unk. prob.: {unk_prob:.3e}")
        print(f"No. of non-dominant rules: {len(rules_mat_surv)+len(rules_mat_fail)}, "
              f"Survival rules: {len(rules_mat_surv)}, Failure rules: {len(rules_mat_fail)}")

        # ---- 3) Get a survival candidate from survival rules ----
        surv_cand, n_samp2 = sample_complementary_events_max_prob_same_rule(
            probs, rules_mat_surv, rules_mat_fail, rules_st='surv', B=rule_search_batch_size, max_iters=rule_search_max_iters, stochastic_search=stochastic_search, gamma=gamma
        )
        if surv_cand is None:
            is_new_surv_cand = False
        else:
            is_new_surv_cand = True

            comps_st_test = from_Bbound_to_comps_st(surv_cand, row_names)
            fval, sys_st, min_comps_st = sfun(comps_st_test)
            if min_comps_st is None:
                if sys_st == 's':
                    min_comps_st = get_min_surv_comps_st(comps_st_test, sys_surv_st=1)
                else:
                    min_comps_st = get_min_fail_comps_st(comps_st_test, max_st=n_state-1, sys_fail_st=0)

            if sys_st == 's':
                print("Survival sample found from survival rules 👍")
                rules_surv, rules_mat_surv = update_rules(min_comps_st, rules_surv, rules_mat_surv, row_names, verbose=rule_update_verbose)
            else:
                print("Failure sample found from survival rules 👍👍")
                rules_fail, rules_mat_fail = update_rules(min_comps_st, rules_fail, rules_mat_fail, row_names, verbose=rule_update_verbose)

            print(f"New rule added. System state: {sys_st}, flow: {fval}. Total samples: {n_samp2}.")
            print(f"New rule (No. of conditions: {len(min_comps_st)-1}): {min_comps_st}")

            # ---- Track system values if new ----
            fval_int = int(fval)
            if fval_int not in sys_val_list:
                sys_val_list.append(fval_int)
                sys_val_list.sort(reverse=True) # sort by descending system value
                print(f"Updated sys_vals: {sys_val_list}")

        # ---- 4) Get a failure candidate from failure rules ----
        fail_cand, n_samp3 = sample_complementary_events_max_prob_same_rule(
            probs, rules_mat_fail, rules_mat_surv, rules_st='fail', B=rule_search_batch_size, max_iters=rule_search_max_iters, stochastic_search=stochastic_search, gamma=gamma
        )
        if fail_cand is None:
            is_new_fail_cand = False
        else:
            is_new_fail_cand = True

            comps_st_test = from_Bbound_to_comps_st(fail_cand, row_names)
            fval, sys_st, min_comps_st = sfun(comps_st_test)
            if min_comps_st is None:
                if sys_st == 's':
                    min_comps_st = get_min_surv_comps_st(comps_st_test, sys_surv_st=1)
                else:
                    min_comps_st = get_min_fail_comps_st(comps_st_test, max_st=n_state-1, sys_fail_st=0)

            if sys_st == 'f':
                print("Failure sample found from failure rules 👍")
                rules_fail, rules_mat_fail = update_rules(min_comps_st, rules_fail, rules_mat_fail, row_names, verbose=rule_update_verbose)
            else:
                print("Survival sample found from failure rules 👍👍")
                rules_surv, rules_mat_surv = update_rules(min_comps_st, rules_surv, rules_mat_surv, row_names, verbose=rule_update_verbose)

            print(f"New rule added. System state: {sys_st}, flow: {fval}. Total samples: {n_samp3}.")
            print(f"New rule (No. of conditions: {len(min_comps_st)-1}): {min_comps_st}")

            # ---- Track system values if new ----
            fval_int = int(fval)
            if fval_int not in sys_val_list:
                sys_val_list.append(fval_int)
                sys_val_list.sort(reverse=True)
                print(f"Unique system values: {sys_val_list}")

        # ---- Periodic probability (bound) test via sampling ----
        probs_updated = False
        if (n_round % prob_update_every) == 0:
            total_loops = max(n_sample // sample_batch_size, 1)
            counts = {"survival": 0, "failure": 0, "unknown": 0}
            for i in range(total_loops):
                samples = sample_categorical(probs, sample_batch_size)
                counts_i = classify_samples(samples, rules_mat_surv, rules_mat_fail)
                counts["survival"] += counts_i["survival"]
                counts["failure"] += counts_i["failure"]
                counts["unknown"] += counts_i["unknown"]

            samp_probs = {k: v / (sample_batch_size * total_loops) for k, v in counts.items()}
            print(f"Probs: 'surv': {samp_probs['survival']: .3e}, 'fail': {samp_probs['failure']: .3e}, 'unkn': {samp_probs['unknown']: .3e}")
            unk_prob = samp_probs["unknown"]
            last_probs.update(samp_probs)
            probs_updated = True

        # ---- metrics for this round ----
        dt = time.perf_counter() - t0
        entry = {
            "round": n_round,
            "time_sec": dt,
            "n_rules_surv": int(len(rules_mat_surv)),
            "n_rules_fail": int(len(rules_mat_fail)),
            "probs_updated": probs_updated,
            "p_survival": last_probs["survival"] if probs_updated else None,
            "p_failure": last_probs["failure"] if probs_updated else None,
            "p_unknown": last_probs["unknown"] if probs_updated else None,
            "avg_len_surv": _avg_rule_len(rules_surv),
            "avg_len_fail": _avg_rule_len(rules_fail),
        }
        metrics_log.append(entry)

        # ---- periodic persistence of metrics and rules ----
        if (n_round % save_every) == 0:
            # append metrics as JSONL
            with open(metrics_path, "a", encoding="utf-8") as mf:
                for e in metrics_log[-save_every:]:
                    mf.write(json.dumps(e) + "\n")
            # snapshot rules
            _save_json(rules_surv, rules_surv_path)
            _save_json(rules_fail, rules_fail_path)
            _save_pt(rules_mat_surv, rules_surv_pt_path)
            _save_pt(rules_mat_fail, rules_fail_pt_path)

    # Final flush of any remaining metrics not yet written by save_every
    last_flushed_rounds = (n_round // save_every) * save_every
    if last_flushed_rounds < n_round and metrics_log:
        with open(metrics_path, "a", encoding="utf-8") as mf:
            for e in metrics_log[last_flushed_rounds:]:
                mf.write(json.dumps(e) + "\n")
    # Final snapshot of rules
    _save_json(rules_surv, rules_surv_path)
    _save_json(rules_fail, rules_fail_path)
    _save_pt(rules_mat_surv, rules_surv_pt_path)
    _save_pt(rules_mat_fail, rules_fail_pt_path)
    # Final probability check
    total_loops = max(n_sample // sample_batch_size, 1)
    counts = {"survival": 0, "failure": 0, "unknown": 0}
    for i in range(total_loops):
        samples = sample_categorical(probs, sample_batch_size)
        counts_i = classify_samples(samples, rules_mat_surv, rules_mat_fail)
        counts["survival"] += counts_i["survival"]
        counts["failure"] += counts_i["failure"]
        counts["unknown"] += counts_i["unknown"]

    samp_probs = {k: v / (sample_batch_size * total_loops) for k, v in counts.items()}
    print(f"[Final results] Probs: 'surv': {samp_probs['survival']: .3e}, 'fail': {samp_probs['failure']: .3e}, 'unkn': {samp_probs['unknown']: .3e}")
    unk_prob = samp_probs["unknown"]
    last_probs.update(samp_probs)
    probs_updated = True
    # ---
    dt = time.perf_counter() - t0
    entry = {
        "round": n_round,
        "time_sec": dt,
        "n_rules_surv": int(len(rules_mat_surv)),
        "n_rules_fail": int(len(rules_mat_fail)),
        "probs_updated": probs_updated,
        "p_survival": last_probs["survival"] if probs_updated else None,
        "p_failure": last_probs["failure"] if probs_updated else None,
        "p_unknown": last_probs["unknown"] if probs_updated else None,
        "avg_len_surv": _avg_rule_len(rules_surv),
        "avg_len_fail": _avg_rule_len(rules_fail),
    }
    metrics_log.append(entry)

    return {
        "sys_vals": sys_val_list,
        "metrics_path": metrics_path,
        "rules_surv_path": rules_surv_path,
        "rules_fail_path": rules_fail_path,
        "rules_surv_pt_path": rules_surv_pt_path,   
        "rules_fail_pt_path": rules_fail_pt_path,
        "metrics_log": metrics_log,  # also returned in-memory
    }

def classify_samples_with_indices(
    samples: torch.Tensor,
    survival_rules: List[torch.Tensor],
    failure_rules: List[torch.Tensor],
    *,
    return_masks: bool = False
) -> Dict[str, Any]:
    """
    Classify samples as survival, failure, or unknown using subset checks,
    and return indices for each class.

    Args:
        samples: (n_sample, n_var, n_state) binary tensor
        survival_rules: list of rule tensors, each (n_var, n_state) or (n_var+1, n_state)
        failure_rules: list of rule tensors, each (n_var, n_state) or (n_var+1, n_state)
        return_masks: if True, also return boolean masks per class

    Returns:
        {
          'survival': int,
          'failure' : int,
          'unknown' : int,
          'idx_survival': LongTensor[ns],
          'idx_failure' : LongTensor[nf],
          'idx_unknown' : LongTensor[nu],
          # optionally:
          'mask_survival': BoolTensor[n_sample],
          'mask_failure' : BoolTensor[n_sample],
          'mask_unknown' : BoolTensor[n_sample],
        }
    """
    device = samples.device
    n_sample = samples.shape[0]

    # Tracking masks
    classified = torch.zeros(n_sample, dtype=torch.bool, device=device)
    survival_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)
    failure_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)

    # Build (rule_tensor, label) list; drop system row if requested
    def _prep_rules(rules, label):
        out = []
        for r in rules:
            r_ok = r[:-1, :]
            out.append((r_ok.to(device=device, dtype=torch.bool), label))
        return out

    all_rules = _prep_rules(survival_rules, 'survival') + _prep_rules(failure_rules, 'failure')

    # Classification loop
    samples_b = samples.to(device=device, dtype=torch.bool)
    for rule_tensor, label in all_rules:
        unclassified_idx = ~classified
        if not unclassified_idx.any():
            break

        current_samples = samples_b[unclassified_idx]  # (n_curr, n_var, n_state)
        # Subset check: sample ⊆ rule  <=>  (sample & rule) == sample  across (var, state)
        is_subset = torch.all((current_samples & rule_tensor) == current_samples, dim=(1, 2))

        # Map back to original indices
        idx_all = torch.where(unclassified_idx)[0]
        matched_idx = idx_all[is_subset]

        if matched_idx.numel() == 0:
            continue

        if label == 'survival':
            survival_mask[matched_idx] = True
        else:
            failure_mask[matched_idx] = True

        classified[matched_idx] = True

    unknown_mask = ~classified

    # Indices
    idx_survival = torch.where(survival_mask)[0]
    idx_failure  = torch.where(failure_mask)[0]
    idx_unknown  = torch.where(unknown_mask)[0]

    result: Dict[str, Any] = {
        'survival': int(survival_mask.sum().item()),
        'failure' : int(failure_mask.sum().item()),
        'unknown' : int(unknown_mask.sum().item()),
        'idx_survival': idx_survival,
        'idx_failure' : idx_failure,
        'idx_unknown' : idx_unknown,
    }

    if return_masks:
        result['mask_survival'] = survival_mask
        result['mask_failure']  = failure_mask
        result['mask_unknown']  = unknown_mask

    return result

def get_comp_cond_sys_prob(
    rules_mat_surv: Tensor,
    rules_mat_fail: Tensor,
    probs: Tensor,
    comps_st_cond: Dict[str, int],
    row_names: Sequence[str],
    s_fun,                          # Callable[[Dict[str,int]], tuple]
    n_sample: int = 1_000_000,
    n_batch:  int = 1_000_000,
    *,
    sys_row: Optional[int] = -1,    # index of the system row to exclude when building comps; set to None to include all
) -> Dict[str, float]:
    """
    P(system state | given component states).

    - 'probs' is (n_var, n_state) categorical; we condition rows listed in comps_st_cond to one-hot.
    - We classify samples using rules; for unknowns we call s_fun(comps_dict) to resolve.
    - Returns probabilities over {'survival','failure','unknown'} that sum ~ 1.0.

    Notes
    -----
    - If sys_row is not None, that row is excluded when building the comps dict for s_fun.
      Default -1 means the last row is the system variable.
    """
    # --- clone probs and apply conditioning ---
    if torch.is_tensor(probs):
        probs_cond = probs.clone()
        n_comps, n_states = probs_cond.shape
        n_vars = n_comps + 1 # system event
    else:
        raise TypeError("Expected 'probs' to be a torch.Tensor of shape (n_var, n_state).")

    if len(row_names) != n_vars:
        raise ValueError(f"row_names length ({len(row_names)}) must match probs rows ({n_vars}).")

    for x, s in comps_st_cond.items():
        try:
            row_idx = row_names.index(x)
        except ValueError:
            raise ValueError(f"Component {x} not found in row_names.")
        if not (0 <= int(s) < n_states):
            raise ValueError(f"State {s} for component {x} is out of bounds [0,{n_states-1}].")
        probs_cond[row_idx].zero_()
        probs_cond[row_idx, int(s)] = 1.0

    # --- sampling loop (exactly n_sample draws) ---
    batch_size = max(1, min(int(n_batch), int(n_sample)))
    remaining = int(n_sample)

    counts = {"survival": 0, "failure": 0, "unknown": 0}

    while remaining > 0:
        b = min(batch_size, remaining)
        # IMPORTANT: sample from the *conditioned* probs
        samples = sample_categorical(probs_cond, b)  # (b, n_var, n_state) one-hot

        res = classify_samples_with_indices(
            samples, rules_mat_surv, rules_mat_fail, return_masks=True
        )

        counts["survival"] += int(res["survival"])
        counts["failure"]  += int(res["failure"])

        # Resolve unknowns with s_fun
        idx_unknown = res["idx_unknown"]
        if idx_unknown.numel() > 0:
            # precompute the system row index if excluding
            sys_idx = None
            if sys_row is not None:
                sys_idx = sys_row if sys_row >= 0 else (n_vars + sys_row)

            for j in idx_unknown.tolist():
                sample_j = samples[j]  # (n_var, n_state)
                # convert one-hot row -> state index per var
                states = torch.argmax(sample_j, dim=1).tolist()

                # build comps dict for s_fun, excluding system row if requested
                if sys_idx is not None:
                    comps = {row_names[k]: int(states[k]) for k in range(n_vars) if k != sys_idx}
                else:
                    comps = {row_names[k]: int(states[k]) for k in range(n_vars)}

                _, sys_st, _ = s_fun(comps)

                if sys_st in ("s", "survival", 1, True):
                    counts["survival"] += 1
                elif sys_st in ("f", "failure", 0, False):
                    counts["failure"] += 1

        remaining -= b

    # --- normalize to probabilities (denominator = requested n_sample) ---
    total = float(n_sample)
    cond_probs = {k: counts[k] / total for k in counts}
    return cond_probs