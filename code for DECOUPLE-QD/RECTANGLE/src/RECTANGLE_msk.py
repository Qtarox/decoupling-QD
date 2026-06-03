from gurobipy import *
from utils import *
from tqdm import tqdm
import time
import numpy as np
from math import log2
import sys
np.set_printoptions(threshold=np.inf, linewidth=sys.maxsize)
from diffs import *
diffs= DIF[TRAIL_ID]
diff_trail        = extract_diff_trail_from_list(diffs)
blocks            = get_transitions_from_list(diffs)
QDTM_RECT         = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)
sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)
def print_mask(mask_trail, nb_rounds):
    for n in range(nb_rounds):
        for side in range(2):
            for row in range(NB_ROWS):
                for col in range(NB_COLS):
                    print(mask_trail[n][side][4*col + row], end='')
                    if col % 4 == 3:
                        print(' ', end='')
                print()
        print()
def compute_correlation(diff_trail, mask_trail, nb_rounds):
    conditions = [[[] for _ in range(64)] for _ in range(nb_rounds)]
    corr = 1
    for k in range(nb_rounds):
        for col in range(NB_COLS):
            a_bits = [diff_trail[k][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            b_bits = [diff_trail[k][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            u_bits = [mask_trail[k][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            v_bits = [mask_trail[k][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            a = bin_to_int(a_bits, SBOX_SIZE)
            b = bin_to_int(b_bits, SBOX_SIZE)
            u = bin_to_int(u_bits, SBOX_SIZE)
            v = bin_to_int(v_bits, SBOX_SIZE)
            entry = QDTM_RECT[b][a][v][u]
            if entry == 0:
                print(f"[Error] Zero QDTM at r={k} col={col}: a={a} b={b} u={u} v={v}")
                return "Error"
            corr *= entry
    for k in range(nb_rounds):
        for j in range(64):
            if mask_trail[k][0][j] != 0:
                conditions[k][j] = mask_trail[k][0][j]
    if corr > 0:
        return  1, log2( corr), conditions
    elif corr < 0:
        return -1, log2(-corr), conditions
    return 1, 0, []
def RECTANGLE_MILP_Quasi_Diff(nb_rounds, THRESH=1, save_pth=None):
    model = Model("RECTANGLE_Quasi_Diff_MILP")
    u = model.addVars(nb_rounds, 2, 64, vtype=GRB.BINARY, name="m")
    Q = model.addVars(nb_rounds, NB_COLS, CORR_RANGE, vtype=GRB.BINARY, name="c")
    model.addConstrs(u[0,          0, j] == 0 for j in range(64))
    model.addConstrs(u[nb_rounds-1, 1, j] == 0 for j in range(64))
    for r in tqdm(range(nb_rounds), desc="S-box constraints"):
        for col in range(NB_COLS):
            a_bits = [diff_trail[r][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            b_bits = [diff_trail[r][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            a = bin_to_int(a_bits, SBOX_SIZE)
            b = bin_to_int(b_bits, SBOX_SIZE)
            model.addConstr(quicksum(Q[r, col, corr] for corr in CORR_RANGE) == 1)
            for corr in CORR_RANGE:
                if sbox_inequalities[b][a][corr] == []:
                    model.addConstr(Q[r, col, corr] == 0)
                    continue
                for ineq in sbox_inequalities[b][a][corr]:
                    model.addConstr(
                        quicksum(ineq[l]           * u[r, 1, 4*col + (SBOX_SIZE-1-l)]
                                 for l in BIT_RANGE) +
                        quicksum(ineq[SBOX_SIZE+l] * u[r, 0, 4*col + (SBOX_SIZE-1-l)]
                                 for l in BIT_RANGE) +
                        ineq[2*SBOX_SIZE] + 50000*(1 - Q[r, col, corr]) >= 0
                    )
    for r in range(nb_rounds - 1):
        model.addConstrs(
            u[r, 1, j] == u[r+1, 0, RECT_PERM[j]]
            for j in range(64)
        )
    model.write("rectangle_quasi_diff.lp")
    total_corr = quicksum(
        Q[r, col, corr] * corr
        for r in range(nb_rounds)
        for col in range(NB_COLS)
        for corr in CORR_RANGE
    )
    model.addConstr(total_corr >= -MIN_CORR)
    model.setObjective(total_corr, GRB.MAXIMIZE)
    model.params.PoolSearchMode = 2
    model.params.PoolSolutions  = 2000000
    t1 = time.time()
    model.optimize()
    t = time.time() - t1
    print(f"Time used: {t:.2f}s")
    print(f"Found {model.SolCount} trails")
    sol_num = model.SolCount
    print("Computing sign and conditions for each trail...")
    signs = []
    correlations = []
    trails_conditions = []
    corr_dict = {}
    MASK = []
    ref_corr = None     
    for m in tqdm(range(model.SolCount)):
        model.params.SolutionNumber = m
        mask_trail = [
            [
                [round(u[r, side, j].Xn) for j in range(64)]
                for side in range(2)
            ]
            for r in range(nb_rounds)
        ]
        result = compute_correlation(diff_trail, mask_trail, nb_rounds)
        if result == "Error":
            continue
        sign, corr, conditions = result
        if ref_corr is None:
            ref_corr = corr
        corr_dict[corr] = corr_dict.get(corr, 0) + 1
        signs.append(sign)
        correlations.append(corr)
        trails_conditions.append(conditions)
        MASK.append(mask_trail)
    print(f"Total valid masks: {len(MASK)}")
    if len(MASK) == 0:
        print("no masks identified!")
        return None, sol_num, None
    res = np.zeros_like(np.array(MASK[0], dtype=np.float64))
    for m in range(len(MASK)):
        weight = 2 ** (correlations[m] - ref_corr)
        res += np.array(MASK[m]) * weight
    MSK = res.copy()
    for r in range(nb_rounds):
        for i in range(2):
            for j in range(64):
                if MSK[r][i][j] < THRESH:
                    MSK[r][i][j] = 0
                else:
                    MSK[r][i][j] = 1
    if save_pth is None:
        save_pth = f'./freq_msk/masks_freq_RECT_{nb_rounds}RD_CORR{MIN_CORR}'
    np.save(save_pth + f'_T{THRESH}.npy', np.array(MSK))
    print(f"Filtered mask saved at: {save_pth}_T{THRESH}.npy")
    np.save(save_pth + '.npy', np.array(res))
    print(f"Original (weighted) mask saved at: {save_pth}.npy")
    return MSK, sol_num, save_pth + f'_T{THRESH}.npy'
if __name__ == "__main__":
    masks = RECTANGLE_MILP_Quasi_Diff(NB_ROUNDS)