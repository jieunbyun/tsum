from pathlib import Path
HOME = Path(__file__).absolute().parent

from tsum import tsum
#from tsum.utils import sys_fun_sum
from tsum import utils
import pytest
import torch

def test_get_min_fail_comps_st1():

    comps_st = {'x1': 2, 'x2': 1, 'x3': 0, 'x4': 0} # Example state that leads to system failure

    min_comps_st = tsum.get_min_fail_comps_st(comps_st, 2, 0)

    assert min_comps_st == {'x2': ('<=', 1), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)}, f"Expected {{'x2': ('<=', 1), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)}}, got {min_comps_st}"

def test_get_min_fail_comps_st2():

    comps_st = {'x1': 2, 'x2': 0, 'x3': 2, 'x4': 2} # Example state that leads to system failure

    min_comps_st = tsum.get_min_fail_comps_st(comps_st, 2, 0)

    assert min_comps_st == {'x2': ('<=', 0), 'sys': ('<=', 0)}, f"Expected {{'x2': ('<=', 0), 'sys': ('<=', 0)}}, got {min_comps_st}"

def test_get_min_fail_comps_st3():

    comps_st = {'x1': 2, 'x2': 1, 'x3': 1, 'x4': 3} # Example state that leads to system failure

    min_comps_st = tsum.get_min_fail_comps_st(comps_st, 3, 1)

    assert min_comps_st == {'x1': ('<=', 2), 'x2': ('<=', 1), 'x3': ('<=', 1), 'sys': ('<=', 1)}, f"Expected {{'x1': ('<=', 2), 'x2': ('<=', 1), 'x3': ('<=', 1), 'sys': ('<=', 1)}}, got {min_comps_st}"

def test_from_rule_dict_to_mat1():
    rule = {'x1': ('>=', 2), 'x2': ('>=', 2), 'sys': ('>=', 1)}
    col_names = ['x1', 'x2', 'x3', 'x4', 'sys']
    max_st = 3

    rule_mat = tsum.from_rule_dict_to_mat(rule, col_names, max_st)

    assert torch.equal(rule_mat, torch.tensor([[0, 0, 1],
                                               [0, 0, 1],
                                               [1, 1, 1],
                                               [1, 1, 1],
                                               [0, 1, 1]], device=rule_mat.device))

def test_from_rule_dict_to_mat2():
    rule = {'x1': ('>=', 2), 'x2': ('>=', 2), 'x3': ('>=', 2), 'x4': ('>=', 1), 'sys': ('>=', 1)}
    col_names = ['x1', 'x2', 'x3', 'x4', 'sys']
    max_st = 3

    rule_mat = tsum.from_rule_dict_to_mat(rule, col_names, max_st)

    assert torch.equal(rule_mat, torch.tensor([[0, 0, 1],
                                               [0, 0, 1],
                                               [0, 0, 1],
                                               [0, 1, 1],
                                               [0, 1, 1]], device=rule_mat.device))

def test_from_rule_dict_to_mat3():
    rule = {'x2': ('<=', 1), 'x3': ('<', 1), 'x4': ('<=', 0), 'sys': ('<=', 0)}
    col_names = ['x1', 'x2', 'x3', 'x4', 'sys']
    max_st = 4

    rule_mat = tsum.from_rule_dict_to_mat(rule, col_names, max_st)

    assert torch.equal(rule_mat, torch.tensor([[1, 1, 1, 1],
                                               [1, 1, 0, 0],
                                               [1, 0, 0, 0],
                                               [1, 0, 0, 0],
                                               [1, 0, 0, 0]], device=rule_mat.device))

def test_get_branches_cap_branches1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B1 = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,1,1],
         [1,1,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)
    utils.print_tensor(B1)

    B2 = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,0],
         [1,1,1]],
        [[1,1,1],
         [1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,1]
    ]], dtype=torch.int32, device=device)
    utils.print_tensor(B2)

    Bnew = tsum.get_branches_cap_branches(B1, B2)
    utils.print_tensor(Bnew)

    expected = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,0],
         [1,1,1]],
        [[1,1,1],
         [1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    Bnew = tsum.get_branches_cap_branches(B1, B2)
    utils.print_tensor(Bnew)
    assert torch.equal(Bnew, expected)


def test_get_branches_cap_branches2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B1 = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,0],
         [1,1,1]],
        [[1,1,1],
         [1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,1]
    ]], dtype=torch.int32, device=device)
    utils.print_tensor(B1)

    B2 = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,1,1],
         [0,1,1],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    Bnew = tsum.get_branches_cap_branches(B1, B2)
    expected = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,0,0],
         [0,1,0],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [1,0,0],
         [1,0,0],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    utils.print_tensor(Bnew)

    assert torch.equal(Bnew, expected)

def test_get_branches_cap_branches3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B1 = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,0,0],
         [0,1,0],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,0],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    B2 = torch.tensor([
        [[1,0,0],
         [1,1,1],
         [1,1,1],
         [0,1,1],
         [1,1,1]],
        [[0,1,1],
         [1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    Bnew = tsum.get_branches_cap_branches(B1, B2)
    expected = torch.tensor([
        [[1,0,0],
         [1,0,0],
         [1,0,0],
         [0,1,0],
         [1,1,1]],
        [[1,0,0],
         [0,1,1],
         [1,0,0],
         [0,1,0],
         [1,1,1]],
        [[0,1,1],
         [0,1,1],
         [1,0,0],
         [1,0,0],
         [1,1,1]],
        [[0,1,1],
         [0,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    assert torch.equal(Bnew, expected)

def test_get_complementary_events1():
    R = torch.tensor([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 1],
        [0, 1, 1],
    ], dtype=torch.int32)

    Bnew = tsum.get_complementary_events(R)

    expected = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,1,1],
         [1,1,0],
         [1,1,1]]
    ], dtype=torch.int32)

    utils.print_tensor(Bnew)

    assert torch.equal(Bnew, expected)

def test_get_complementary_events2():
    R = torch.tensor([
        [1,1,1],
        [1,1,1],
        [0,1,1],
        [0,1,1],
        [0,1,1],
    ], dtype=torch.int32)

    Bnew = tsum.get_complementary_events(R)

    expected = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,1],
         [1,1,1]],
        [[1,1,1],
         [1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,1]],
    ], dtype=torch.int32)

    assert torch.equal(Bnew, expected)

def test_get_complementary_events3():
    R = torch.tensor([
        [1,1,1],
        [0,1,1],
        [1,1,1],
        [0,1,1],
        [0,1,1],
    ], dtype=torch.int32)

    Bnew = tsum.get_complementary_events(R)

    expected = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,1,1],
         [1,1,1],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,1]],
    ], dtype=torch.int32)

    assert torch.equal(Bnew, expected)

def test_get_branch_probs1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B1 = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,0,0],
         [0,1,0],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,0],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    prob1 = torch.tensor([[0.3, 0.7, 0.0],
             [0.1, 0.2, 0.7],
             [0.2, 0.8, 0.0],
             [0.1, 0.3, 0.6]], dtype=torch.float32, device=device)

    Bprob = tsum.get_branch_probs(B1, prob1)
    print(Bprob)
    expected = torch.tensor([0.006, 0.072, 0.072], dtype=torch.float32, device=device)

    assert torch.allclose(Bprob, expected, atol=1e-5)

def test_get_branch_probs2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B1 = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,0,0],
         [0,1,0],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,0],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    prob1 = torch.tensor([[0.1, 0.9, 0.0],
             [0.3, 0.2, 0.5],
             [0.5, 0.5, 0.0],
             [0.5, 0.1, 0.4]], dtype=torch.float32, device=device)

    Bprob = tsum.get_branch_probs(B1, prob1)
    expected = torch.tensor([0.015, 0.21, 0.175], dtype=torch.float32, device=device)

    assert torch.allclose(Bprob, expected, atol=1e-5)

def test_get_branch_probs3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B1 = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,0],
         [1,1,1]],
        [[1,1,1],
         [1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    prob1 = torch.tensor([[0.3, 0.7, 0.0],
             [0.1, 0.2, 0.7],
             [0.2, 0.8, 0.0],
             [0.1, 0.3, 0.6]], dtype=torch.float32, device=device)

    Bprob = tsum.get_branch_probs(B1, prob1)
    expected = torch.tensor([0.08, 0.08], dtype=torch.float32, device=device)

    assert torch.allclose(Bprob, expected, atol=1e-5)

def test_get_boundary_branches1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B1 = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,0,0],
         [0,1,0],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,0],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    expected = torch.tensor([
        [[0,0,1],
         [1,0,0],
         [1,0,0],
         [0,1,0],
         [1,1,1]],
        [[0,0,1],
         [0,0,1],
         [1,0,0],
         [0,1,0],
         [1,1,1]],
        [[0,0,1],
         [0,0,1],
         [0,0,1],
         [1,0,0],
         [1,1,1]],
        [[1,0,0],
         [1,0,0],
         [1,0,0],
         [0,1,0],
         [1,1,1]],
        [[1,0,0],
         [0,1,0],
         [1,0,0],
         [1,0,0],
         [1,1,1]],
        [[1,0,0],
         [0,1,0],
         [0,1,0],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    Bbound = tsum.get_boundary_branches(B1)
    assert torch.equal(Bbound, expected)

def test_get_boundary_branches2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,0],
         [1,1,1]],

        [[1,1,1],
         [1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    expected = torch.tensor([
        [[0,0,1],
         [0,0,1],
         [1,0,0],
         [0,1,0],
         [1,1,1]],
        [[0,0,1],
         [0,0,1],
         [0,0,1],
         [1,0,0],
         [1,1,1]],
        [[1,0,0],
         [1,0,0],
         [1,0,0],
         [1,0,0],
         [1,1,1]],
        [[1,0,0],
         [1,0,0],
         [0,1,0],
         [1,0,0],
         [1,1,1]]
    ], dtype=torch.int32, device=device)

    Bbound = tsum.get_boundary_branches(B)
    assert torch.equal(Bbound, expected), "Test 2 failed: Bbound does not match expected output"

def test_is_intersect1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = torch.tensor([
        [[0,0,1],[1,0,0],[1,0,0],[0,1,0],[1,1,1]],
        [[0,0,1],[0,0,1],[1,0,0],[0,1,0],[1,1,1]],
        [[0,0,1],[0,0,1],[0,0,1],[1,0,0],[1,1,1]],
        [[1,0,0],[1,0,0],[1,0,0],[0,1,0],[1,1,1]],
        [[1,0,0],[0,1,0],[1,0,0],[1,0,0],[1,1,1]],
        [[1,0,0],[0,1,0],[0,1,0],[0,1,0],[1,1,1]]
    ], dtype=torch.int32, device=device)

    R = torch.tensor([
        [[1,1,1],[1,1,1],[1,1,1],[0,0,1],[0,1,1]],
        [[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)

    expected = torch.tensor([False, False, False, False, False, True], device=device)
    result = tsum.is_intersect(B, R)
    assert torch.equal(result, expected)

def test_is_intersect2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = torch.tensor([
        [[0,0,1],[0,0,1],[1,0,0],[0,1,0],[1,1,1]],
        [[0,0,1],[0,0,1],[0,0,1],[1,0,0],[1,1,1]],
        [[1,0,0],[1,0,0],[1,0,0],[1,0,0],[1,1,1]],
        [[1,0,0],[1,0,0],[0,1,0],[1,0,0],[1,1,1]]
    ], dtype=torch.int32, device=device)

    R = torch.tensor([
        [[1,1,1],[1,1,1],[1,1,1],[1,0,1],[0,1,1]],
        [[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)

    expected = torch.tensor([False, True, True, True], device=device)
    result = tsum.is_intersect(B, R)
    assert torch.equal(result, expected)

def test_from_Bbound_to_comps_st1():
    B = torch.tensor([
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 1]
    ], dtype=torch.int32)

    row_names = ['x1', 'x2', 'x3', 'x4', 'sys']
    expected = {'x1': 2, 'x2': 0, 'x3': 0, 'x4': 1}

    result = tsum.from_Bbound_to_comps_st(B, row_names)
    assert result == expected, f"Expected {expected}, but got {result}"
    print("Test passed.")

def test_is_subset1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Rnew = torch.tensor(
        [[0,1,1],[0,1,1],[0,0,1],[1,1,1],[0,1,1]],
        dtype=torch.int32, device=device
    )

    R = torch.tensor([
        [[1,1,1],[1,1,1],[1,1,1],[0,0,1],[0,1,1]],
        [[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)

    is_mat_subset, is_tensor_subset = tsum.is_subset(Rnew, R)

    assert is_mat_subset == False
    assert torch.equal(is_tensor_subset, torch.tensor([False, False], device=device))
    print("test_is_subset1 passed.")

def test_is_subset2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Rnew = torch.tensor(
        [[0,1,1],[0,1,1],[0,0,1],[1,1,1],[0,1,1]],
        dtype=torch.int32, device=device
    )

    R = torch.tensor([
        [[0,0,1],[0,0,1],[0,0,1],[0,1,1],[0,0,1]],
        [[1,1,1],[1,1,1],[0,1,1],[1,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)

    is_mat_subset, is_tensor_subset = tsum.is_subset(Rnew, R)

    assert is_mat_subset == True
    assert torch.equal(is_tensor_subset, torch.tensor([True, False], device=device))
    print("test_is_subset2 passed.")

def test_find_first_nonempty_combination1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R = torch.tensor([
        [[0,0,1],[0,0,1],[0,0,1],[0,1,1],[0,0,1]],
        [[1,1,1],[1,1,1],[0,1,1],[1,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)

    Rcs = []
    for i in range(R.shape[0]): 
        Ri = R[i,:,:]
        Ri_c = tsum.get_complementary_events(Ri)
        Rcs.append(Ri_c)

    mat = tsum.find_first_nonempty_combination(Rcs, verbose=False)
    expected = torch.tensor([[1,1,0], [1,1,1], [1,0,0], [1,1,1],[1,1,1]], dtype=torch.int32, device=device)
    assert torch.equal(mat, expected), f"Expected {expected}, but got {mat}"

def test_find_first_nonempty_combination2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R = torch.tensor([
        [[1,1,1],[1,1,1],[1,1,1],[0,0,1],[0,1,1]],
        [[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)

    Rcs = []
    for i in range(R.shape[0]):
        Ri = R[i,:,:]
        Ri_c = tsum.get_complementary_events(Ri)
        Rcs.append(Ri_c)

    mat = tsum.find_first_nonempty_combination(Rcs, verbose=False)
    expected = torch.tensor([[1,1,1], [1,1,1], [1,0,0], [1,1,0],[1,1,1]], dtype=torch.int32, device=device)
    assert torch.equal(mat, expected), f"Expected {expected}, but got {mat}"

def to_set_of_tuples(T: torch.Tensor):
    # T is (n, m, k), so each branch is (m, k)
    # To make the testing immune to the order of branches
    return {tuple(T[i].flatten().tolist()) for i in range(T.size(0))}

def test_merge_branches1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = torch.tensor([
    [[1,0,0],[0,0,1],[1,0,0],[0,0,1],[0,1,1]], # merged with 6th branch
    [[0,1,1],[0,1,0],[1,0,0],[0,0,1],[0,1,1]], 
    [[0,1,1],[1,0,0],[0,1,1],[0,0,1],[0,1,1]],
    [[1,0,0],[0,0,1],[0,1,1],[1,1,0],[0,1,1]], # merged with 7th branch
    [[0,1,1],[0,1,0],[0,1,1],[1,1,0],[0,1,1]], 
    [[0,1,1],[0,0,1],[1,0,0],[0,0,1],[0,1,1]],
    [[0,1,1],[0,0,1],[0,1,1],[1,1,0],[0,1,1]]
    ], dtype=torch.int32, device=device)


    expected = torch.tensor([
    [[0,1,1],[0,1,0],[1,0,0],[0,0,1],[0,1,1]],
    [[0,1,1],[1,0,0],[0,1,1],[0,0,1],[0,1,1]], 
    [[0,1,1],[0,1,0],[0,1,1],[1,1,0],[0,1,1]],
    [[1,1,1],[0,0,1],[1,0,0],[0,0,1],[0,1,1]],
    [[1,1,1],[0,0,1],[0,1,1],[1,1,0],[0,1,1]]
    ], dtype=torch.int32, device=device)

    result = tsum.merge_branches(B)

    for i in range(result.shape[0]):
        print(f"Result branch {i}: {result[i].cpu().numpy()}")

    expected_set = to_set_of_tuples(expected.cpu())
    result_set = to_set_of_tuples(result.cpu())

    assert expected_set == result_set, f"Expected {expected_set}, but got {result_set}"

def test_merge_branches2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = torch.tensor([
    [[1,0,0],[1,0,0],[0,1,1],[0,1,1],[0,1,1]], # merged with 3rd branch
    [[1,0,0],[0,1,1],[1,0,0],[0,1,1],[0,1,1]], # merged with 4th branch
    [[0,1,1],[1,0,0],[0,1,1],[0,1,1],[0,1,1]],
    [[0,1,1],[0,1,1],[1,0,0],[0,1,1],[0,1,1]],
    [[1,1,1],[0,1,1],[0,1,1],[1,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)


    expected = torch.tensor([
    [[1,1,1],[0,1,1],[0,1,1],[1,1,1],[0,1,1]],
    [[1,1,1],[1,0,0],[0,1,1],[0,1,1],[0,1,1]],
    [[1,1,1],[0,1,1],[1,0,0],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)


    result = tsum.merge_branches(B)

    expected_set = to_set_of_tuples(expected.cpu())
    result_set = to_set_of_tuples(result.cpu())

    assert expected_set == result_set, f"Expected {expected_set}, but got {result_set}"


def test_merge_branches3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = torch.tensor([
    [[1,1,1],[1,1,1],[1,0,0],[0,0,1],[0,1,1]],
    [[1,1,1],[0,1,1],[0,1,1],[0,1,1],[0,1,1]], # merged with 3rd branch
    [[1,1,1],[1,0,0],[0,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)


    expected = torch.tensor([
    [[1,1,1],[1,1,1],[1,0,0],[0,0,1],[0,1,1]],
    [[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)

    result = tsum.merge_branches(B)

    expected_set = to_set_of_tuples(expected.cpu())
    result_set = to_set_of_tuples(result.cpu())

    assert expected_set == result_set, f"Expected {expected_set}, but got {result_set}"

def test_get_complementary_events_nondisjoint1():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R1 = torch.tensor([[1,1,1], [1,1,1], [1,1,1], [0,0,1], [0,1,1]], dtype=torch.int32, device=device)

    expected = torch.tensor([[[1,1,1], [1,1,1], [1,1,1], [1,1,0], [1,1,1]]], dtype=torch.int32, device=device)

    result = tsum.get_complementary_events_nondisjoint(R1)

    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_get_complementary_events_nondisjoint2():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R2 = torch.tensor([[1,1,1], [1,1,0], [1,0,0], [1,0,0], [1,0,0]], dtype=torch.int32, device=device)

    expected = torch.tensor([[[1,1,1], [0,0,1], [1,1,1], [1,1,1], [1,1,1]],
                             [[1,1,1], [1,1,1], [0,1,1], [1,1,1], [1,1,1]],
                             [[1,1,1], [1,1,1], [1,1,1], [0,1,1], [1,1,1]]], dtype=torch.int32, device=device)

    result = tsum.get_complementary_events_nondisjoint(R2)

    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

@pytest.fixture
def def_B1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = torch.tensor([
        [[1,0,0],[1,0,0],[0,1,1],[0,1,1],[0,1,1]], # merged with 3rd branch
        [[1,0,0],[0,1,1],[1,0,0],[0,1,1],[0,1,1]], # merged with 4th branch
        [[0,1,1],[1,0,0],[0,1,1],[0,1,1],[0,1,1]],
        [[0,1,1],[0,1,1],[1,0,0],[0,1,1],[0,1,1]],
        [[1,1,1],[0,1,1],[0,1,1],[1,1,1],[0,1,1]]
        ], dtype=torch.int32, device=device)

    return B

@pytest.fixture
def def_B2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = torch.tensor([
    [[1,0,0],[0,0,1],[1,0,0],[0,0,1],[0,1,1]],
    [[0,1,1],[0,1,0],[1,0,0],[0,0,1],[0,1,1]],
    [[0,1,1],[1,0,0],[0,1,1],[0,0,1],[0,1,1]],
    [[1,0,0],[0,0,1],[0,1,1],[1,1,0],[0,1,1]],
    [[0,1,1],[0,1,0],[0,1,1],[1,1,0],[0,1,1]],
    [[0,1,1],[0,0,1],[1,0,0],[0,0,1],[0,1,1]],
    [[0,1,1],[0,0,1],[0,1,1],[1,1,0],[0,1,1]]
    ], dtype=torch.int32, device=device)
    return B

@pytest.fixture
def def_B3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = torch.tensor([
    [[1,1,1],[1,1,1],[1,0,0],[0,0,1],[0,1,1]],
    [[1,1,1],[0,1,1],[0,1,1],[0,1,1],[0,1,1]], # merged with 3rd branch
    [[1,1,1],[1,0,0],[0,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)
    return B

def test_bit_compress1(def_B1):
    result = tsum.bit_compress(def_B1)
    expected = torch.tensor([
        [1, 1, 6, 6, 6],
        [1, 6, 1, 6, 6],
        [6, 1, 6, 6, 6],
        [6, 6, 1, 6, 6],
        [7, 6, 6, 7, 6]
    ], dtype=torch.int32, device=def_B1.device)

    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_bit_compress2(def_B2):
    result = tsum.bit_compress(def_B2)
    expected = torch.tensor([
        [1, 4, 1, 4, 6],
        [6, 2, 1, 4, 6],
        [6, 1, 6, 4, 6],
        [1, 4, 6, 3, 6],
        [6, 2, 6, 3, 6],
        [6, 4, 1, 4, 6],
        [6, 4, 6, 3, 6]
    ], dtype=torch.int32, device=def_B2.device)

    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_bit_compress3(def_B3):
    result = tsum.bit_compress(def_B3)
    expected = torch.tensor([
        [7, 7, 1, 4, 6],
        [7, 6, 6, 6, 6],
        [7, 1, 6, 6, 6]
    ], dtype=torch.int32, device=def_B3.device)

    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def tensor_to_list(groups):
    # groups: list[list[tensor]] -> list[list[list]]
    return [[t.cpu().tolist() for t in col] for col in groups]

def test_groups_by_column_remhash_dict1(def_B1):
    B_com = tsum.bit_compress(def_B1)
    out = tsum.groups_by_column_remhash_dict(B_com)

    out = tensor_to_list(out)

    expected = [[torch.tensor([0,2], dtype=torch.int32, device=def_B1.device),torch.tensor([1,3], dtype=torch.int32, device=def_B1.device)],
                [], [], [], []]
    expected = tensor_to_list(expected)

    assert out == expected, f"Expected {expected}, but got {out}"

def test_groups_by_column_remhash_dict2(def_B2):
    B_com = tsum.bit_compress(def_B2)
    out = tsum.groups_by_column_remhash_dict(B_com)

    out = tensor_to_list(out)

    expected = [[torch.tensor([0,5], dtype=torch.int32, device=def_B2.device),torch.tensor([3,6], dtype=torch.int32, device=def_B2.device)],
                [torch.tensor([1, 5], dtype=torch.int32, device=def_B2.device), torch.tensor([4, 6], dtype=torch.int32, device=def_B2.device)],
                [], [], []]
    expected = tensor_to_list(expected)

    assert out == expected, f"Expected {expected}, but got {out}"

def test_groups_by_column_remhash_dict3(def_B3):
    B_com = tsum.bit_compress(def_B3)
    out = tsum.groups_by_column_remhash_dict(B_com)

    out = tensor_to_list(out)

    expected = [[],
                [torch.tensor([1, 2], dtype=torch.int32, device=def_B3.device)],
                [], [], []]
    expected = tensor_to_list(expected)

    assert out == expected, f"Expected {expected}, but got {out}"

def test_plan_merges1():
    groups_per_col = [
        [torch.tensor([0, 1, 2], dtype=torch.int32)],
        [torch.tensor([3, 4], dtype=torch.int32)],
        [torch.tensor([5, 6], dtype=torch.int32)]
    ]
    n_rows = 7
    expected = [(0, 1, 0), (3, 4, 1), (5, 6, 2)]
    out = tsum.plan_merges(groups_per_col, n_rows)
    assert out == expected, f"Expected {expected}, but got {out}"

def test_plan_merges2():
    groups_per_col = [
        [torch.tensor([0, 1, 2, 3], dtype=torch.int32)],
        [torch.tensor([3, 4], dtype=torch.int32)],
        [torch.tensor([5, 6], dtype=torch.int32)]
    ]
    n_rows = 7
    expected = [(0, 1, 0), (2, 3, 0), (5, 6, 2)]
    out = tsum.plan_merges(groups_per_col, n_rows)
    assert out == expected, f"Expected {expected}, but got {out}"

def test_plan_merges3():
    groups_per_col = [
        [torch.tensor([0, 1, 2, 3], dtype=torch.int32)],
        [torch.tensor([3, 4], dtype=torch.int32)],
        [],
        [torch.tensor([5, 6], dtype=torch.int32)]
    ]
    n_rows = 7
    expected = [(0, 1, 0), (2, 3, 0), (5, 6, 3)]
    out = tsum.plan_merges(groups_per_col, n_rows)
    assert out == expected, f"Expected {expected}, but got {out}"

def test_apply_merges1(def_B2):

    B = def_B2
    merge_plan = [(0, 5, 0), (3, 6, 0)]

    B_merged, kept_indices = tsum.apply_merges(B, merge_plan)

    expected = torch.tensor([
    [[0,1,1],[0,1,0],[1,0,0],[0,0,1],[0,1,1]],
    [[0,1,1],[1,0,0],[0,1,1],[0,0,1],[0,1,1]],
    [[0,1,1],[0,1,0],[0,1,1],[1,1,0],[0,1,1]],
    [[1,1,1],[0,0,1],[1,0,0],[0,0,1],[0,1,1]],
    [[1,1,1],[0,0,1],[0,1,1],[1,1,0],[0,1,1]]
    ], dtype=torch.int32, device=B.device)

    B_merged = to_set_of_tuples(B_merged.cpu())
    expected = to_set_of_tuples(expected)

    assert B_merged == expected, f"Expected {expected}, but got {B_merged}"

def test_apply_merges2(def_B1):

    B = def_B1
    merge_plan = [(0, 2, 0), (1, 3, 0)]

    B_merged, kept_indices = tsum.apply_merges(B, merge_plan)

    expected = torch.tensor([
    [[1,1,1],[0,1,1],[0,1,1],[1,1,1],[0,1,1]],
    [[1,1,1],[1,0,0],[0,1,1],[0,1,1],[0,1,1]],
    [[1,1,1],[0,1,1],[1,0,0],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=B.device)

    B_merged = to_set_of_tuples(B_merged.cpu())
    expected = to_set_of_tuples(expected)

    assert B_merged == expected, f"Expected {expected}, but got {B_merged}"

def test_apply_merges3(def_B3):

    B = def_B3
    merge_plan = [(1, 2, 1)]

    B_merged, kept_indices = tsum.apply_merges(B, merge_plan)

    expected = torch.tensor([
    [[1,1,1],[1,1,1],[1,0,0],[0,0,1],[0,1,1]],
    [[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=B.device)

    B_merged = to_set_of_tuples(B_merged.cpu())
    expected = to_set_of_tuples(expected)

    assert B_merged == expected, f"Expected {expected}, but got {B_merged}"

@pytest.fixture
def ex_surv_fail_rules():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rules_mat_surv = torch.tensor([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]]
    ], dtype=torch.int32, device=device)
    rules_mat_fail = torch.tensor([
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]]
    ], dtype=torch.int32, device=device)

    probs = torch.tensor([
        [0.3, 0.7, 0.0],
        [0.1, 0.2, 0.7],
        [0.2, 0.8, 0.0],
        [0.1, 0.3, 0.6]
    ], dtype=torch.float32, device=device)

    return rules_mat_surv, rules_mat_fail, probs

@pytest.fixture
def ex_surv_fail_rules_with_dict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    row_names = ['x1', 'x2', 'x3', 'x4', 'sys']
    rules_mat_surv = torch.tensor([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]]
    ], dtype=torch.int32, device=device)
    rules_mat_fail = torch.tensor([
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]]
    ], dtype=torch.int32, device=device)

    rules_surv = [{'x4': ('>=', 2), 'sys': ('>=', 1)},
                  {'x3': ('>=', 1), 'x4': ('>=', 1), 'sys': ('>=', 1)},
                  {'x2': ('>=', 1), 'x4': ('>=', 1), 'sys': ('>=', 1)}]
    rules_fail = [{'x2': ('<=', 1), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)},
                  {'x1': ('<=', 0),'x2': ('<=', 0), 'x3': ('<=', 0), 'sys': ('<=', 0)}]

    return rules_mat_surv, rules_mat_fail, rules_surv, rules_fail, row_names

def test_sample_complementary_events(ex_surv_fail_rules):

    rules_mat_surv, rules_mat_fail, probs = ex_surv_fail_rules

    fail_cand, n_samp0 = tsum.sample_complementary_events(probs, rules_mat_surv, rules_mat_fail, rules_st = 'surv')

    assert fail_cand.shape == (rules_mat_fail.shape[1], rules_mat_fail.shape[2])
    assert n_samp0 > 0

    surv_cand, n_samp1 = tsum.sample_complementary_events(probs, rules_mat_fail, rules_mat_surv, rules_st = 'fail')

    assert surv_cand.shape == (rules_mat_surv.shape[1], rules_mat_surv.shape[2])
    assert n_samp1 > 0

def test_mask_from_first_one1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], device='cuda:0', dtype=torch.int32)

    x_after = tsum.mask_from_first_one(x, mode="after")
    x_after_expected = torch.tensor([[1, 1, 1],
        [0, 1, 1],
        [1, 1, 1],
        [0, 1, 1],
        [1, 1, 1]], device='cuda:0', dtype=torch.int32)

    x_before = tsum.mask_from_first_one(x, mode="before")
    x_before_expected = torch.tensor([[1, 0, 0],
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]], device='cuda:0', dtype=torch.int32)

    assert torch.equal(x_after, x_after_expected), f"Expected {x_after_expected}, but got {x_after}"
    assert torch.equal(x_before, x_before_expected), f"Expected {x_before_expected}, but got {x_before}"

def test_mask_from_first_one2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[[1, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0],
                      [1, 1, 1]],
                      [[0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [1, 0, 0],
                      [1, 1, 1]]], device='cuda:0', dtype=torch.int32)

    x_after = tsum.mask_from_first_one(x, mode="after")
    x_after_expected = torch.tensor([[[1, 1, 1],
                      [0, 0, 1],
                      [0, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]],
                      [[0, 1, 1],
                      [0, 1, 1],
                      [0, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]], device='cuda:0', dtype=torch.int32)

    x_before = tsum.mask_from_first_one(x, mode="before")
    x_before_expected = torch.tensor([[[1, 0, 0],
                      [1, 1, 1],
                      [1, 1, 0],
                      [1, 0, 0],
                      [1, 1, 1]],
                      [[1, 1, 0],
                      [1, 1, 0],
                      [1, 1, 0],
                      [1, 0, 0],
                      [1, 1, 1]]], device='cuda:0', dtype=torch.int32)

    assert torch.equal(x_after, x_after_expected), f"Expected {x_after_expected}, but got {x_after}"
    assert torch.equal(x_before, x_before_expected), f"Expected {x_before_expected}, but got {x_before}"



def test_update_rules1(ex_surv_fail_rules_with_dict):
    rules_mat_surv, rules_mat_fail, rules_surv, rules_fail, row_names = ex_surv_fail_rules_with_dict

    min_comps_st = {'x1': ('<=', 0), 'x3': ('<=', 0), 'sys': ('<=', 0)}
    rules_dict, rules_mat = tsum.update_rules(min_comps_st, rules_fail, rules_mat_fail, row_names)

    expected_rules_dict = [{'x2': ('<=', 1), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)},
                           {'x1': ('<=', 0), 'x3': ('<=', 0), 'sys': ('<=', 0)}]

    expected_rules_mat = torch.tensor([
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1], [1, 0, 0]]
    ], dtype=torch.int32, device=rules_mat_fail.device)

    assert rules_dict == expected_rules_dict, f"Expected {expected_rules_dict}, but got {rules_dict}"
    assert torch.equal(rules_mat, expected_rules_mat), f"Expected {expected_rules_mat}, but got {rules_mat}"

def test_update_rules2(ex_surv_fail_rules_with_dict):
    rules_mat_surv, rules_mat_fail, rules_surv, rules_fail, row_names = ex_surv_fail_rules_with_dict

    min_comps_st = {'x1': ('>=', 1), 'x2': ('>=', 2), 'sys': ('>=', 1)}
    rules_dict, rules_mat = tsum.update_rules(min_comps_st, rules_surv, rules_mat_surv, row_names)

    expected_rules_dict = [{'x4': ('>=', 2), 'sys': ('>=', 1)},
                           {'x3': ('>=', 1), 'x4': ('>=', 1), 'sys': ('>=', 1)},
                           {'x2': ('>=', 1), 'x4': ('>=', 1), 'sys': ('>=', 1)},
                           {'x1': ('>=', 1), 'x2': ('>=', 2), 'sys': ('>=', 1)}]

    expected_rules_mat = torch.tensor([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]],
        [[0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1]]
    ], dtype=torch.int32, device=rules_mat_surv.device)

    assert rules_dict == expected_rules_dict, f"Expected {expected_rules_dict}, but got {rules_dict}"
    assert torch.equal(rules_mat, expected_rules_mat), f"Expected {expected_rules_mat}, but got {rules_mat}"

def test_update_rules3(ex_surv_fail_rules_with_dict):
    rules_mat_surv, rules_mat_fail, rules_surv, rules_fail, row_names = ex_surv_fail_rules_with_dict

    min_comps_st = {'x1': ('>=', 1), 'x4': ('>=', 2), 'sys': ('>=', 1)}
    rules_dict, rules_mat = tsum.update_rules(min_comps_st, rules_surv, rules_mat_surv, row_names)

    expected_rules_dict = [{'x4': ('>=', 2), 'sys': ('>=', 1)},
                           {'x3': ('>=', 1), 'x4': ('>=', 1), 'sys': ('>=', 1)},
                           {'x2': ('>=', 1), 'x4': ('>=', 1), 'sys': ('>=', 1)}]

    expected_rules_mat = torch.tensor([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]]
    ], dtype=torch.int32, device=rules_mat_surv.device)

    assert rules_dict == expected_rules_dict, f"Expected {expected_rules_dict}, but got {rules_dict}"
    assert torch.equal(rules_mat, expected_rules_mat), f"Expected {expected_rules_mat}, but got {rules_mat}"

def test_update_rules4(ex_surv_fail_rules_with_dict):
    rules_mat_surv, rules_mat_fail, rules_surv, rules_fail, row_names = ex_surv_fail_rules_with_dict

    min_comps_st = {'x2': ('<=', 0), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)}
    rules_dict, rules_mat = tsum.update_rules(min_comps_st, rules_fail, rules_mat_fail, row_names)

    expected_rules_dict = [{'x2': ('<=', 1), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)},
                           {'x1': ('<=', 0),'x2': ('<=', 0), 'x3': ('<=', 0), 'sys': ('<=', 0)}]

    expected_rules_mat = torch.tensor([
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 0, 0]]
    ], dtype=torch.int32, device=rules_mat_fail.device)

    assert rules_dict == expected_rules_dict, f"Expected {expected_rules_dict}, but got {rules_dict}"
    assert torch.equal(rules_mat, expected_rules_mat), f"Expected {expected_rules_mat}, but got {rules_mat}"

@pytest.fixture
def surv_fail_rules_ex_4comps():
    surv_rules = [{'x1': ('>=', 1), 'x2': ('>=', 2), 'x3': ('>=', 1), 'x4': ('>=', 2), 'sys': ('>=', 1)},
                  {'x1': ('>=', 2), 'x2': ('>=', 1), 'x3': ('>=', 2), 'x4': ('>=', 1), 'sys': ('>=', 1)}]
    fail_rules = [{'x1': ('<=', 0), 'sys': ('<=', 0)},
                  {'x1': ('<=', 1), 'x2': ('<=', 1), 'sys': ('<=', 0)},
                  {'x1': ('<=', 1), 'x4': ('<=', 1), 'sys': ('<=', 0)},
                  {'x3': ('<=', 1), 'x4': ('<=', 1), 'sys': ('<=', 0)},
                  {'x2': ('<=', 1), 'x3': ('<=', 1), 'sys': ('<=', 0)},
                  {'x2': ('<=', 0), 'sys': ('<=', 0)},
                  {'x3': ('<=', 0), 'sys': ('<=', 0)},
                  {'x4': ('<=', 0), 'sys': ('<=', 0)}]

    row_names = ['x1', 'x2', 'x3', 'x4', 'sys']

    return surv_rules, fail_rules, row_names

def test_minimise_surv_states_random1(surv_fail_rules_ex_4comps):
    surv_rules, fail_rules, row_names = surv_fail_rules_ex_4comps

    comps_st = {x: 2 for x in row_names if x != 'sys'}

    def sfun(comps_st):
        for s in surv_rules:
            if all(comps_st[k] >= v[1] for k, v in s.items() if k != 'sys'):
                return None, 's', None
        return None, 'f', None

    new_rule, info = tsum.minimise_surv_states_random(comps_st, sfun, 1)
    assert new_rule in surv_rules, f"Expected one of {surv_rules}, but got {new_rule}"

def test_minimise_surv_states_random2(surv_fail_rules_ex_4comps):
    surv_rules, fail_rules, row_names = surv_fail_rules_ex_4comps

    comps_st = {x: 2 for x in row_names if x != 'sys'}
    comps_st['x1'] = 1

    def sfun(comps_st):
        for s in surv_rules:
            if all(comps_st[k] >= v[1] for k, v in s.items() if k != 'sys'):
                return None, 's', None
        return None, 'f', None

    new_rule, info = tsum.minimise_surv_states_random(comps_st, sfun, 1)
    assert new_rule in surv_rules, f"Expected one of {surv_rules}, but got {new_rule}"

def test_minimise_fail_states_random1(surv_fail_rules_ex_4comps):
    surv_rules, fail_rules, row_names = surv_fail_rules_ex_4comps

    comps_st = {x: 0 for x in row_names if x != 'sys'}

    def sfun(comps_st):
        for s in surv_rules:
            if all(comps_st[k] >= v[1] for k, v in s.items() if k != 'sys'):
                return None, 's', None
        return None, 'f', None

    new_rule, info = tsum.minimise_fail_states_random(comps_st, sfun, max_state = 2, sys_fail_st = 0)
    assert new_rule in fail_rules, f"Got {new_rule}"

def test_minimise_fail_states_random2(surv_fail_rules_ex_4comps):
    surv_rules, fail_rules, row_names = surv_fail_rules_ex_4comps

    comps_st = {x: 0 for x in row_names if x != 'sys'}
    comps_st['x1'] = 1

    def sfun(comps_st):
        for s in surv_rules:
            if all(comps_st[k] >= v[1] for k, v in s.items() if k != 'sys'):
                return None, 's', None
        return None, 'f', None

    new_rule, info = tsum.minimise_fail_states_random(comps_st, sfun, max_state = 2, sys_fail_st = 0)
    assert new_rule in fail_rules, f"Got {new_rule}"
