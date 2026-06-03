from gurobipy import *
from utils import *
from tqdm import tqdm
import time
import numpy as np
from math import log2
from key_master_rectangle import build_round_keys_80, build_round_keys_128, extract_master_key_conditions
from diffs import *
KEY_SIZE = 80    
diffs= diffs14_2
diff_trail        = extract_diff_trail_from_list(diffs)
blocks            = get_transitions_from_list(diffs)
QDTM_RECT         = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)
sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)
if KEY_SIZE == 80:
    round_keys_sym, nl_log = build_round_keys_80(NB_ROUNDS)
elif KEY_SIZE == 128:
    round_keys_sym, nl_log = build_round_keys_128(NB_ROUNDS)
else:
    raise ValueError("KEY_SIZE must be 80 or 128")
print(f"Built round keys for RECTANGLE-{KEY_SIZE}, NL events in key schedule: {len(nl_log)}")
def compute_correlation(diff_trail, mask_trail, nb_rounds):
    corr = 1
    for k in range(nb_rounds):
        for col in range(NB_COLS):
            a_bits = [diff_trail[k][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            b_bits = [diff_trail[k][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            u_bits = [mask_trail[k][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            v_bits = [mask_trail[k][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            a, b = bin_to_int(a_bits, SBOX_SIZE), bin_to_int(b_bits, SBOX_SIZE)
            u, v = bin_to_int(u_bits, SBOX_SIZE), bin_to_int(v_bits, SBOX_SIZE)
            entry = QDTM_RECT[b][a][v][u]
            if entry == 0:
                return "Error"
            corr *= entry
    for k in range(nb_rounds):
        corr *= rect_rc_corr_factor(mask_trail[k][0], RECTANGLE_RC[k])
    if corr > 0:
        return  1, log2( corr)
    elif corr < 0:
        return -1, log2(-corr)
    return 1, 0
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
    print(f"\nTime used: {time.time()-t1:.2f}s")
    print(f"Found {model.SolCount} trails")
    T = []
    avg_prob = None
    for m in tqdm(range(model.SolCount), desc="Extracting trails"):
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
        sign, corr_log = result
        if avg_prob is None:
            avg_prob = corr_log
        keys_list = extract_master_key_conditions(mask_trail, round_keys_sym, nb_rounds)
        T.append({
            'sign': sign,
            'corr': corr_log,
            'keys': keys_list,
        })
    return T, avg_prob
if __name__ == "__main__":
    T, avg_prob = RECTANGLE_MILP_Quasi_Diff(NB_ROUNDS)
    print(f"\nFound {len(T)} valid trails")
    print(f"Reference correlation: {avg_prob}")
    print("\nFirst 10 trails:")
    for i, t in enumerate(T[:10]):
        print(f"Trail {i}: sign={t['sign']:+d}, corr={t['corr']:.4f}, keys={t['keys']}")
    from fast_distribut import plot_quasi_distribut
    plot_quasi_distribut(T, appendix=f'_RECT{KEY_SIZE}_{NB_ROUNDS}RD', avg_p=-avg_prob)