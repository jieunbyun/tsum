def sys_fun_sum(comps_st, comps_capa, thres, sys_surv_st_int):

    assert sys_surv_st_int > 0, "System survival state must be positive."

    comps_sum = sum(comps_capa[k][v] for k, v in comps_st.items())

    if comps_sum < thres:
        sys_st = 'f'
        min_comps_st = None
    else:
        sys_st = 's'
        comps_sum_ = 0
        min_comps_st = {}
        for x in comps_st.keys():
            if comps_st[x] > 0:
                min_comps_st[x] = ('>=', comps_st[x])
                comps_sum_ += comps_capa[x][comps_st[x]]
            if comps_sum_ >= thres:
                break
        min_comps_st['sys'] = ('>=', sys_surv_st_int)

    return comps_sum, sys_st, min_comps_st


def print_tensor(tensor):
    print(f"(shape: {tensor.shape}):")
    n_br, _, _ = tensor.shape
    for i in range(n_br):
        print(tensor[i, :, :].int())
    print()
