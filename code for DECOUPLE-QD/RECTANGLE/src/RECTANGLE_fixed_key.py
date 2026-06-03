from gurobipy import *
from utils import *
from tqdm import tqdm
import time
import numpy as np
from math import log2
diffs = [
    0x0020000600000000,
    0x0060000200000000,
    0x0200006000000000,
    0x0600002000000000,
    0x2000060000000000,
    0x6000020000000000,
    0x0000600000000002,
    0x0000200000000006,
    0x0006000000000020,
    0x0002000000000060,
    0x0060000000000200,
    0x0020000000000600,
    0x0600000000002000,
    0x0200000000006000,
    0x6000000000020000,
    0x2000000000060000,
    0x0000000000200006,
    0x0000000000600002,
    0x0000000002000060,
    0x000000000c000020,
    0x0000000000008600,
    0x0000000000001200,
    0x0000000000003000,
    0x0000000000008000,
    0x0000000000000008,
    0x0000000000000001,
    0x0000000000000001,
    0x0000000000000006,
    0x0004000000000020,
]
diff_trail        = extract_diff_trail_from_list(diffs)   
blocks            = get_transitions_from_list(diffs)       
QDTM_RECT         = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)
sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)
def print_mask(mask_trail, nb_rounds):
    for n in range(nb_rounds):
        for side, label in enumerate(['in ', 'out']):
            print(f"  r{n} {label}:", end='')
            for row in range(NB_ROWS):
                s = ''.join(
                    str(mask_trail[n][side][4 * col + row])
                    for col in range(NB_COLS)
                )
                print(f"  row{row}=[{s}]", end='')
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
            print(f'position in mat [{a}][{b}][{v}][{u}]: ')
            entry = QDTM_RECT[b][a][v][u]
            if entry == 0:
                print(f"  [Error] Zero QDTM at r={k} col={col}: a={a} b={b} u={u} v={v}")
                return "Error"
            corr *= entry
    for k in range(nb_rounds):
        corr *= rect_rc_corr_factor(mask_trail[k][0], RECTANGLE_RC[k])
    for k in range(nb_rounds):
        for j in range(64):
            if mask_trail[k][1][j] != 0:
                conditions[k][j] = mask_trail[k][1][j]
    if corr > 0:
        return  1, log2( corr), conditions
    elif corr < 0:
        return -1, log2(-corr), conditions
    return 1, 0, []
def RECTANGLE_MILP_Quasi_Diff(nb_rounds):
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
                        quicksum(ineq[l]          * u[r, 1, 4*col + (SBOX_SIZE-1-l)]
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
    print(f"\nTime used: {time.time()-t1:.2f}s")
    print(f"Found {model.SolCount} trails")
    signs, correlations, trails_conditions = [], [], []
    corr_dict = {}
    MSKS = []
    for m in tqdm(range(model.SolCount), desc="Computing correlations"):
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
        corr_dict[corr] = corr_dict.get(corr, 0) + 1
        signs.append(sign)
        correlations.append(corr)
        trails_conditions.append(conditions)
        MSKS.append(mask_trail)
    print("=" * 60)
    for cnt, (mk, corr_val) in enumerate(zip(MSKS, correlations)):
        print(f"Trail {cnt}  |  log2(|corr|) = {corr_val:.4f}")
        print_mask(mk, nb_rounds)
    if corr_dict:
        factor_dict = {
            corr: int(2 ** (corr - min(corr_dict)))
            for corr in sorted(corr_dict)
        }
        trail_indic(trails_conditions, correlations, signs, factor_dict, nb_rounds)
    out_mask = ""
    for m in tqdm(range(model.SolCount), desc="Exporting"):
        model.params.SolutionNumber = m
        for r in range(nb_rounds):
            for side in range(2):
                for col in range(NB_COLS):
                    nibble = bin_to_int(
                        [round(u[r, side, 4*col + row].Xn)
                         for row in range(NB_ROWS-1, -1, -1)],
                        SBOX_SIZE
                    )
                    out_mask += str(nibble) + " "
        if m != model.SolCount - 1:
            out_mask += "\n"
    with open(RESULTS_FILE + "_mask", 'w') as f:
        f.write(out_mask)
    solutions_to_readable(RESULTS_FILE + "_mask")
    return model.SolCount
if __name__ == "__main__":
    nb_sol = RECTANGLE_MILP_Quasi_Diff(NB_ROUNDS)
    print(f"number of trails: {nb_sol}")