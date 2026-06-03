import os
import numpy as np
from gurobipy import Model, GRB, quicksum
from tqdm import tqdm
from utils import (
    NB_ROUNDS, MIN_CORR, THRESH, SBOX_SIZE,
    STATE_RANGE, BIT_RANGE, CORR_RANGE,
    DIFF_TRAIL_FILE, SBOX_INEQUALITIES_DIR,
    bin_to_int, extract_diff_trail, extract_inequalities_by_corr,
)
def add_xor_constraints(model, x1, x2, y):
    """y = x1 XOR x2 (binary)"""
    model.addConstr(-x1 + x2 + y >= 0)
    model.addConstr( x1 - x2 + y >= 0)
    model.addConstr( x1 + x2 - y >= 0)
    model.addConstr(-x1 - x2 - y >= -2)
def add_xor_constraints2(model, x1, x2, x3, y):
    """y = x1 XOR x2 XOR x3 (binary)"""
    model.addConstr(-x1 + x2 + x3 + y >= 0)
    model.addConstr( x1 - x2 + x3 + y >= 0)
    model.addConstr( x1 + x2 - x3 + y >= 0)
    model.addConstr( x1 + x2 + x3 - y >= 0)
    model.addConstr( x1 - x2 - x3 - y >= -2)
    model.addConstr(-x1 + x2 - x3 - y >= -2)
    model.addConstr(-x1 - x2 + x3 - y >= -2)
    model.addConstr(-x1 - x2 - x3 + y >= -2)
def SKINNY_MILP_Quasi_Diff(nb_rounds, save_pth=None):
    diff_trail = extract_diff_trail(DIFF_TRAIL_FILE, nb_rounds)
    sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)
    model = Model("SKINNY_SK_Quasi_Diff_MILP")
    u = model.addVars(nb_rounds, 2, 4, 4, SBOX_SIZE, vtype=GRB.BINARY)
    Q = model.addVars(nb_rounds, 4, 4, CORR_RANGE, vtype=GRB.BINARY)
    for i in STATE_RANGE:
        for j in STATE_RANGE:
            model.addConstrs(u[0, 0, i, j, l] == 0 for l in BIT_RANGE)
            model.addConstrs(u[nb_rounds - 1, 1, i, j, l] == 0 for l in BIT_RANGE)
    for r in tqdm(range(nb_rounds), desc="building SBox constraints"):
        for i in STATE_RANGE:
            for j in STATE_RANGE:
                b = bin_to_int(
                    [diff_trail[r][1][i][j][l] for l in BIT_RANGE], SBOX_SIZE
                )
                a = bin_to_int(
                    [diff_trail[r][0][i][j][l] for l in BIT_RANGE], SBOX_SIZE
                )
                model.addConstr(quicksum(Q[r, i, j, c] for c in CORR_RANGE) == 1)
                for corr in CORR_RANGE:
                    if sbox_inequalities[b][a][corr] == []:
                        model.addConstr(Q[r, i, j, corr] == 0)
                        continue
                    for ineq in sbox_inequalities[b][a][corr]:
                        model.addConstr(
                            quicksum(ineq[2 * SBOX_SIZE - l - 1] * u[r, 1, i, j, l]
                                     for l in BIT_RANGE) +
                            quicksum(ineq[1 * SBOX_SIZE - l - 1] * u[r, 0, i, j, l]
                                     for l in BIT_RANGE) -
                            ineq[2 * SBOX_SIZE]
                            + 50000 * (1 - Q[r, i, j, corr]) >= 0
                        )
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:
            model.addConstrs(u[r, 1, 3, (j - 3) % 4, l] == u[r + 1, 0, 0, j, l]
                             for l in BIT_RANGE)
            for l in BIT_RANGE:
                add_xor_constraints2(
                    model,
                    u[r, 1, 0, j, l],
                    u[r, 1, 1, (j - 1) % 4, l],
                    u[r, 1, 2, (j - 2) % 4, l],
                    u[r + 1, 0, 1, j, l],
                )
            model.addConstrs(u[r, 1, 1, (j - 1) % 4, l] == u[r + 1, 0, 2, j, l]
                             for l in BIT_RANGE)
            for l in BIT_RANGE:
                add_xor_constraints2(
                    model,
                    u[r, 1, 1, (j - 1) % 4, l],
                    u[r, 1, 2, (j - 2) % 4, l],
                    u[r, 1, 3, (j - 3) % 4, l],
                    u[r + 1, 0, 3, j, l],
                )
    total_corr = quicksum(Q[r, i, j, c] * c
                          for r in range(nb_rounds)
                          for i in STATE_RANGE for j in STATE_RANGE
                          for c in CORR_RANGE)
    model.addConstr(total_corr >= -MIN_CORR)
    model.setObjective(total_corr, GRB.MAXIMIZE)
    print("Searching for quasi-differential trails...\n")
    model.params.PoolSearchMode = 2
    model.params.PoolSolutions = 2_000_000
    model.optimize()
    n_sol = model.SolCount
    print(f"Found {n_sol} trails")
    if n_sol == 0:
        print("no masks identified!")
        return
    print("Collecting masks and correlations from solution pool...")
    masks = []
    correlations = []
    for m in tqdm(range(n_sol), desc="reading pool"):
        model.params.SolutionNumber = m
        mt = [[[[[round(u[r, side, i, j, l].Xn) for l in reversed(BIT_RANGE)]
                  for j in STATE_RANGE] for i in STATE_RANGE]
                  for side in range(2)] for r in range(nb_rounds)]
        arr = np.array(mt)
        masks.append(arr.reshape(*arr.shape[:2], 16 * SBOX_SIZE))
        correlations.append(model.PoolObjVal)
    avg_prob = max(correlations)
    if not np.all(masks[0] == 0):
        print(f"warning: pool[0] not all zero mask")
    res = np.zeros_like(masks[0], dtype=np.float64)
    for m in range(n_sol):
        w = 2 ** (correlations[m] - avg_prob)
        res += masks[m] * w
    MSK = (res >= THRESH).astype(np.int8)
    for r in range(nb_rounds):
        print(f'round {r}')
        for side in range(2):
            row = ','.join(f'{v:.3g}' for v in res[r][side])
            print(f'[{row}]')
    print('=' * 50)
    if save_pth is None:
        save_pth = f'./freq_msk/masks_freq_{NB_ROUNDS}RD_CORR{MIN_CORR}'
    os.makedirs(os.path.dirname(save_pth) or '.', exist_ok=True)
    np.save(save_pth + f'_T{THRESH}.npy', MSK)
    np.save(save_pth + '.npy', res)
    print(f"thresholded MSK saved at: {save_pth}_T{THRESH}.npy")
    print(f"raw freq saved at:        {save_pth}.npy")
    print("+++++ MSK +++++")
    print(MSK)
if __name__ == "__main__":
    SKINNY_MILP_Quasi_Diff(NB_ROUNDS)