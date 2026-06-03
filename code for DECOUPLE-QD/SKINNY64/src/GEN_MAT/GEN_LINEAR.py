import numpy as np
CELL_SIZE = 4                              
HALF_STATE_BITS = 16 * CELL_SIZE           
FULL_STATE_BITS = 2 * HALF_STATE_BITS      
KEY_BITS = HALF_STATE_BITS                 
Sbox = [0xc, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb,
           0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf]
M_EQ = np.load("./M_EQ.npy")
def key_schedule(round=0, key_index=0):
    key_permu = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]
    tmp = key_index
    for _ in range(round):
        tmp = key_permu[tmp]
    return tmp
def Global_mat(res, M_EQ, round_num):
    for i in range(np.shape(res)[0]):
        equ_num = i % 16
        rn = i // 16
        rn_k_ind = None
        for k in range(40):
            if M_EQ[equ_num][k] == 1:
                if k < 16:                       
                    res[i][(rn + 1) * 32 + k] = 1
                elif k < 32:                     
                    res[i][rn * 32 + k] = 1
                else:                            
                    rn_k_ind = k - 32
                    break
        if rn_k_ind is not None:
            k_ind = key_schedule(rn, rn_k_ind)
            res[i][32 * (round_num + 1) + k_ind] = 1
    return res
def Global_mat_bit(round_num):
    num_rows = HALF_STATE_BITS * round_num
    num_cols = FULL_STATE_BITS * (round_num + 1) + KEY_BITS
    res = np.zeros((num_rows, num_cols), dtype=int)
    for i in range(num_rows):
        equ_num_bit = i % HALF_STATE_BITS           
        rn = i // HALF_STATE_BITS                   
        equ_num_cell = equ_num_bit // CELL_SIZE     
        bit_idx = equ_num_bit % CELL_SIZE           
        for k in range(40):
            if M_EQ[equ_num_cell][k] != 1:
                continue
            if k < 16:
                res[i][(rn + 1) * FULL_STATE_BITS + k * CELL_SIZE + bit_idx] = 1
            elif k < 32:
                res[i][rn * FULL_STATE_BITS + k * CELL_SIZE + bit_idx] = 1
            else:
                rn_k_ind = k - 32
                k_ind = key_schedule(rn, rn_k_ind)
                res[i][FULL_STATE_BITS * (round_num + 1) + k_ind * CELL_SIZE + bit_idx] = 1
                break
    return res
def _fmt_state_var(j):
    rn = j // FULL_STATE_BITS
    ind = j % FULL_STATE_BITS
    if ind < HALF_STATE_BITS:
        return f"x_{rn}_{ind}"
    else:
        return f"y_{rn}_{ind - HALF_STATE_BITS}"
def show_L_equ_GIFT(lmat, active_bit_dic, round_num, FILE_FLG=False, file=None):
    F = file if file else "output.txt"
    L = ""
    out_handle = open(F, "w") if FILE_FLG else None
    state_cols = FULL_STATE_BITS * (round_num + 1)
    try:
        for i in range(np.shape(lmat)[0]):
            l_tmp = ""
            for j in range(state_cols):
                if lmat[i][j] == 1:
                    var_name = _fmt_state_var(j)
                    if str(j) in active_bit_dic:
                        l_tmp += f" + [{var_name}]"
                    else:
                        l_tmp += f" + {var_name}"
            has_key = False
            key_base = round_num * FULL_STATE_BITS + FULL_STATE_BITS  
            for k in range(KEY_BITS):
                if lmat[i][key_base + k] == 1:
                    has_key = True
                    l_tmp += f" + k_{k} = 0"
            if not has_key:
                l_tmp += "= 0   "
            if np.shape(lmat)[1] > key_base + KEY_BITS:
                l_tmp += "  SBOX: " + str(lmat[i][-1])
            print(l_tmp)
            if out_handle:
                print(l_tmp, file=out_handle)
            L += l_tmp + "\n"
    finally:
        if out_handle:
            out_handle.close()
    return L
def show_L_equ_GIFT_extract(lmat, round_num, FILE_FLG=False, file=None):
    F = file if file else "output.txt"
    out_handle = open(F, "w") if FILE_FLG else None
    state_cols = FULL_STATE_BITS * (round_num + 1)
    key_base = (round_num + 1) * FULL_STATE_BITS
    try:
        for i in range(np.shape(lmat)[0]):
            l_tmp = ""
            for j in range(state_cols):
                if lmat[i][j] == 2:
                    var_name = _fmt_state_var(j)
                    l_tmp += f" + [{var_name}]"
                elif lmat[i][j] == 1:
                    var_name = _fmt_state_var(j)
                    l_tmp += f" + {var_name}"
            has_key = False
            for k in range(KEY_BITS):
                if lmat[i][key_base + k] == 1:
                    has_key = True
                    l_tmp += f" + k_{k} = 0"
            if not has_key:
                l_tmp += "= 0   "
            print(l_tmp)
            if out_handle:
                print(l_tmp, file=out_handle)
    finally:
        if out_handle:
            out_handle.close()
if __name__ == "__main__":
    res = Global_mat_bit(2)
    show_L_equ_GIFT_extract(res, 2)