import os
import importlib
import numpy as np
from utils import (
    NB_ROUNDS, SBOX_SIZE, MIN_CORR, THRESH, ADV_MODEL,
    extract_diff_trail_flat,
)
from SKINNYMILP_msk import SKINNY_MILP_Quasi_Diff
_MASK_DIVIDER_BY_MODE = {
    "SK":  "MASK_DIVIDER_TK1",
    "TK1": "MASK_DIVIDER_TK1",
    "TK2": "MASK_DIVIDER_TK2",
}
def _load_mask_divider():
    if ADV_MODEL not in _MASK_DIVIDER_BY_MODE:
        raise ValueError(
            f"unsupported: ADV_MODEL={ADV_MODEL}。"
        )
    name = _MASK_DIVIDER_BY_MODE[ADV_MODEL]
    print(f"[loader] ADV_MODEL={ADV_MODEL}  ->  import {name}")
    return importlib.import_module(name)
def run_milp_if_needed():
    mask_path = f'./freq_msk/masks_freq_{NB_ROUNDS}RD_CORR{MIN_CORR}_T{THRESH}.npy'
    if os.path.exists(mask_path):
        print(f"[skip MILP] mask exists:{mask_path}")
    else:
        print(f"[run MILP] mask non-exists, groubi start ...")
        SKINNY_MILP_Quasi_Diff(NB_ROUNDS)
    return mask_path
def main():
    ROUNDS = NB_ROUNDS
    MD = _load_mask_divider()
    mask_path = run_milp_if_needed()
    data = np.load(mask_path)
    expected = (ROUNDS, 2, MD.HALF_STATE_BITS)
    assert data.shape == expected, (
        f"wrong shape "
    )
    data = data.tolist()
    print(f"[load] {mask_path}  shape={expected}")
    diff_file = (
        f"../data/differential_trails/SKINNY{16 * SBOX_SIZE}_{ADV_MODEL}_R{ROUNDS}.txt"
    )
    diff_trail = extract_diff_trail_flat(diff_file, ROUNDS)
    dic_x, dic_y = MD.creat_dic_GIFT(diff_trail)
    active_bit_dic = MD.get_active_bit(dic_x, dic_y) 
    masked_bit_dic = active_bit_dic.copy()
    for r in range(ROUNDS):
        for s in range(2):
            for i in range(MD.HALF_STATE_BITS):
                if data[r][s][i] == 1:
                    g = r * MD.FULL_STATE_BITS + s * MD.HALF_STATE_BITS + i
                    masked_bit_dic[str(g)] = 1
    L = MD.Global_mat_bit(ROUNDS)
    L_mat = L.copy()
    cons_str, Z_lst = MD.generate_constraints(
        L_mat, dic_x, active_bit_dic, masked_bit_dic, ROUNDS, L
    )
    out_file = (
        f"./constraints/CONS_{ADV_MODEL}_{ROUNDS}R_{MIN_CORR}_TH{THRESH}.txt"
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    cons_lst_str = "dic_cons={\n \n"
    with open(out_file, 'w', encoding='utf-8') as fh:
        for i, (cstr, Z) in enumerate(zip(cons_str, Z_lst)):
            block = f'CONS{i}="""\n{cstr}"""\nZ{i}={Z}\n\n'
            fh.write(block)
            cons_lst_str += f"'CONS{i}': (CONS{i},Z{i}),\n"
        cons_lst_str += "}"
        fh.write(cons_lst_str)
    print(f"\nConstraints written: {out_file}")
if __name__ == "__main__":
    main()