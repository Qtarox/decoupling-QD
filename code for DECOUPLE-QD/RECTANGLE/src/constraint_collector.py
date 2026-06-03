"""
constraint_collector_RECT.py

Entry script that orchestrates the full RECTANGLE quasidifferential pipeline:
  1) Run the Gurobi-based MILP to enumerate quasidifferential trails and produce
     a frequency-weighted importance mask  (RECTANGLE_msk.RECTANGLE_MILP_Quasi_Diff).
  2) Load that mask, combine it with the per-S-box active-bit dictionary, and
     run the cluster decomposition  (MASK_DIVIDER.generate_constraints_RECT).
  3) Write three artifacts:
        - text dump of all clusters (CONS*, Z*) + dic_cons dictionary
        - a .py file holding FULL_MSK (list of per-cluster mask variables)
        - the .npy mask file (already produced by step 1).

Index convention (must match utils.py and MASK_DIVIDER.py):
    flat index j = 4*col + row,  row = j % 4,  col = j // 4,  row 0 = LSB.
The frequency mask saved by RECTANGLE_msk.py uses exactly this convention,
so we DO NOT do a (63 - i) flip like the GIFT version.
"""

import os
import numpy as np

from RECTANGLE_msk import *           
from MASK_DIVIDER import (
    genLinear_RECT,
    extract_diff_trail_cell_RECT_from_diffs,
    creat_dic_RECT,
    get_active_bit,
    generate_constraints_RECT,
    mask_trans,
)
from diffs import *                    
diffs = DIF[TRAIL_ID]

if __name__ == "__main__":
    THRES = 1

    
    masks, sol_num, save_pth = RECTANGLE_MILP_Quasi_Diff(NB_ROUNDS, THRES)
    print("mask's solution number:", sol_num)

    if save_pth is None:
        raise RuntimeError("MILP did not return any mask path; aborting.")

    
    data = np.load(save_pth).tolist()
    print("Number of rounds loaded:", len(data))
    print("================================")

    diff_trail_cell = extract_diff_trail_cell_RECT_from_diffs(diffs)
    dic_x, dic_y    = creat_dic_RECT(diff_trail_cell)
    active_bit_dic  = get_active_bit(dic_x, dic_y)

    masked_bit_dic = active_bit_dic.copy()
    for r in range(NB_ROUNDS):
        for s in range(2):
            for i in range(64):
                if data[r][s][i] == 1:
                    
                    masked_bit_dic[str(r * 128 + s * 64 + i)] = 1

    print("number of masked positions:", len(masked_bit_dic))

    
    L = genLinear_RECT(NB_ROUNDS)
    L_mat = L.copy()
    cons_str, Z_lst, mask_lst = generate_constraints_RECT(
        L_mat, dic_x, active_bit_dic, masked_bit_dic, NB_ROUNDS, L
    )

    
    FULL_MSK = []
    res_str = ""

    out_dir     = "./constraints"
    os.makedirs(out_dir, exist_ok=True)
    file_prefix = f"{out_dir}/CONS_{NB_ROUNDS}R_{MIN_CORR}{NAME}_TH{THRES}"
    cons_txt    = f"{file_prefix}.txt"
    mask_py     = f"{file_prefix}_FULL_MSK.py"

    
    with open(cons_txt, 'w', encoding='utf-8') as f:
        pass
    cons_lst = "dic_cons={\n \n"
    for i in range(len(cons_str)):
        res_str += f'CONS{i}="""\n' + str(cons_str[i])
        print(f'CONS{i}="""\n', cons_str[i])

        res_str += '"""'
        print('"""\n')

        res_str += f'\nZ{i}=' + str(Z_lst[i]) + '\n\n'
        print(f'Z{i}=', Z_lst[i])
        cons_lst += f"'CONS{i}': (CONS{i},Z{i}),\n"

        formatted_mask = mask_trans(mask_lst[i])
        print(f'MASK{i}=', formatted_mask)
        FULL_MSK.append(formatted_mask)

        with open(cons_txt, 'a', encoding='utf-8') as f:
            f.write(res_str)
        res_str = ""

    cons_lst += '}'

    with open(cons_txt, 'a', encoding='utf-8') as f:
        f.write(cons_lst)

    
    with open(mask_py, 'w', encoding='utf-8') as f:
        f.write(f'FULL_MSK = {FULL_MSK}\n')
