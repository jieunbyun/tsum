#from tensor_uq_etc import sys_fun_sum
from tsum.utils import sys_fun_sum

def test_sys_fun_sum1():
    thres = 24
    comps_st = {'x1': 2, 'x2': 2, 'x3': 2, 'x4': 1}

    comps_capa = {'x1': [0, 5, 5],
                  'x2': [0, 10, 20],
                  'x3': [0, 15, 15],
                  'x4': [0, 20, 40]}

    fval, sys_st, main_comps_st = sys_fun_sum(comps_st, comps_capa, thres, 1)

    assert fval == 60, f"Total capacity should be 60, got {fval}"
    assert sys_st == 's', f"System state should be 's', got {sys_st}"
    expected_min = {'x1': ('>=', 2), 'x2': ('>=', 2), 'sys': ('>=', 1)}
    assert main_comps_st == expected_min, f"Expected {expected_min}, got {main_comps_st}"

def test_sys_fun_sum2():
    thres = 24
    comps_st = {'x1': 2, 'x2': 1, 'x3': 0, 'x4': 0}

    comps_capa = {'x1': [0, 5, 5],
                  'x2': [0, 10, 20],
                  'x3': [0, 15, 15],
                  'x4': [0, 20, 40]}

    fval, sys_st, main_comps_st = sys_fun_sum(comps_st, comps_capa, thres, 1)

    assert fval == 15, f"Total capacity should be 15, got {fval}"
    assert sys_st == 'f', f"System state should be 'f', got {sys_st}"
    assert main_comps_st == None, f"Expected None, got {main_comps_st}"

def test_sys_fun_sum3():
    thres = 56
    comps_st = {'x1': 2, 'x2': 2, 'x3': 2, 'x4': 1}

    comps_capa = {'x1': [0, 5, 5],
                  'x2': [0, 10, 20],
                  'x3': [0, 15, 15],
                  'x4': [0, 20, 40]}

    fval, sys_st, main_comps_st = sys_fun_sum(comps_st, comps_capa, thres, 1)

    assert fval == 60, f"Total capacity should be 60, got {fval}"
    assert sys_st == 's', f"System state should be 's', got {sys_st}"
    expected_min = {'x1': ('>=', 2), 'x2': ('>=', 2), 'x3': ('>=', 2), 'x4': ('>=', 1), 'sys': ('>=', 1)}
    assert main_comps_st == expected_min, f"Expected {expected_min}, got {main_comps_st}"
