from gurobipy import *
from utils import *
from tqdm import tqdm
import time
import numpy as np
from math import log2
import sys
np.set_printoptions(threshold=np.inf, linewidth=sys.maxsize)

blocks = get_transitions(DIFF_TRAIL_FILE)
QDTM_GIFT_SBOX = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)

AC_STATES = [0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,0x37,0x2F,0x1E,0x3C,0x39,0x33,0x27,0x0E,
0x1D,0x3A,0x35,0x2B,0x16,0x2C,0x18,0x30,0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38,
0x31,0x23,0x06,0x0D,0x1B,0x36,0x2D,0x1A,0x34,0x29,0x12,0x24,0x08,0x11,0x22,0x04]


def print_mask(mask_trail, nb_rounds):
    for n in range(nb_rounds):
        for i in range(BLOCK_SIZE):
            print(mask_trail[n][0][i], end='')
            if i % 4 == 3:
                print(' ', end='')
        print()
        for i in range(BLOCK_SIZE):
            print(mask_trail[n][1][i], end='')
            if i % 4 == 3:
                print(' ', end='')
        print()
    print()


def compute_correlation(diff_trail, mask_trail, nb_rounds):
    conditions = [[[] for _ in range(BLOCK_SIZE)] for _ in range(nb_rounds)]
    corr = 1
    
    
    for k in range(nb_rounds):
        for i in range(NB_SBOXES):
            a_bits = diff_trail[k][0][4*i:4*i+4]
            b_bits = diff_trail[k][1][4*i:4*i+4]
            u_bits = mask_trail[k][0][4*i:4*i+4]
            v_bits = mask_trail[k][1][4*i:4*i+4]
            
            a = bin_to_int(a_bits, SBOX_SIZE)
            b = bin_to_int(b_bits, SBOX_SIZE)
            u = bin_to_int(u_bits, SBOX_SIZE)
            v = bin_to_int(v_bits, SBOX_SIZE)
            
            corr *= QDTM_GIFT_SBOX[b][a][v][u]
            if QDTM_GIFT_SBOX[b][a][v][u] == 0:
                return "Error"
    
    
    
    
    for k in range(nb_rounds - 1):
        
        corr *= character([1], [mask_trail[k+1][0][0]], 1)                            
        corr *= character([(AC_STATES[k] >> 5) & 1], [mask_trail[k+1][0][104]], 1)    
        corr *= character([(AC_STATES[k] >> 4) & 1], [mask_trail[k+1][0][108]], 1)    
        corr *= character([(AC_STATES[k] >> 3) & 1], [mask_trail[k+1][0][112]], 1)    
        corr *= character([(AC_STATES[k] >> 2) & 1], [mask_trail[k+1][0][116]], 1)    
        corr *= character([(AC_STATES[k] >> 1) & 1], [mask_trail[k+1][0][120]], 1)    
        corr *= character([(AC_STATES[k] >> 0) & 1], [mask_trail[k+1][0][124]], 1)    
        
        
        for j in range(BLOCK_SIZE):
            if mask_trail[k+1][0][j] == 0:
                continue
            
            elif j % 4 == 1 or j % 4 == 2:
                conditions[k][j] = mask_trail[k+1][0][j]

    if corr > 0:
        return  1, log2( corr), conditions
    elif corr < 0:
        return -1, log2(-corr), conditions
    return 1, 0, []


def add_xor_constraints(model, x1, x2, y):
    model.addConstr(-x1 + x2 + y >= 0)
    model.addConstr( x1 - x2 + y >= 0)
    model.addConstr( x1 + x2 - y >= 0)
    model.addConstr(-x1 - x2 - y >= -2)


def add_xor_constraints2(model, x1, x2, x3, y):
    model.addConstr(-x1 + x2 + x3 + y >= 0)
    model.addConstr( x1 - x2 + x3 + y >= 0)
    model.addConstr( x1 + x2 - x3 + y >= 0)
    model.addConstr( x1 + x2 + x3 - y >= 0)
    model.addConstr( x1 - x2 - x3 - y >= -2)
    model.addConstr(-x1 + x2 - x3 - y >= -2)
    model.addConstr(-x1 - x2 + x3 - y >= -2)
    model.addConstr(-x1 - x2 - x3 + y >= -2)


def GIFT_MILP_Quasi_Diff(nb_rounds, THRESH=1, save_pth=None):
    diff_trail = extract_diff_trail(DIFF_TRAIL_FILE, nb_rounds)
    sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)
    model = Model("GIFT128_SK_Quasi_Diff_MILP")
    
    
    u = model.addVars(nb_rounds, 2, BLOCK_SIZE, vtype=GRB.BINARY, name="m")
    Q = model.addVars(nb_rounds, NB_SBOXES, CORR_RANGE, vtype=GRB.BINARY, name="c")
    
    
    model.addConstrs(u[0, 0, l] == 0 for l in range(BLOCK_SIZE))
    model.addConstrs(u[nb_rounds - 1, 1, l] == 0 for l in range(BLOCK_SIZE))
    
    
    for r in tqdm(range(nb_rounds)):
        for i in range(NB_SBOXES):
            b = bin_to_int([diff_trail[r][1][4*i+l] for l in range(4)], SBOX_SIZE)
            a = bin_to_int([diff_trail[r][0][4*i+l] for l in range(4)], SBOX_SIZE)
            
            model.addConstr(quicksum(Q[r, i, corr] for corr in CORR_RANGE) == 1)
            for corr in CORR_RANGE:
                if sbox_inequalities[b][a][corr] == []:
                    model.addConstr(Q[r, i, corr] == 0)
                    continue
                for ineq in sbox_inequalities[b][a][corr]:
                    model.addConstr(
                        quicksum(ineq[0*SBOX_SIZE + l] * u[r, 1, 4*i + l] for l in BIT_RANGE) +
                        quicksum(ineq[1*SBOX_SIZE + l] * u[r, 0, 4*i + l] for l in BIT_RANGE) +
                        ineq[2*SBOX_SIZE] + 50000 * (1 - Q[r, i, corr]) >= 0
                    )
    
    
    for r in range(nb_rounds - 1):
        model.addConstrs(
            u[r, 1, l] == u[r + 1, 0, (BLOCK_SIZE - 1) - BIT_PERM[(BLOCK_SIZE - 1) - l]]
            for l in range(BLOCK_SIZE)
        )
    
    model.write("gift128_fixed_key.lp")
    
    
    model.addConstr(
        quicksum(Q[r, i, corr] * corr
                 for r in range(nb_rounds)
                 for i in range(NB_SBOXES)
                 for corr in CORR_RANGE) >= -MIN_CORR
    )
    model.setObjective(
        quicksum(Q[r, i, corr] * corr
                 for r in range(nb_rounds)
                 for i in range(NB_SBOXES)
                 for corr in CORR_RANGE),
        GRB.MAXIMIZE
    )
    
    model.params.PoolSearchMode = 2
    model.params.PoolSolutions = 2000000
    
    t1 = time.time()
    model.optimize()
    t = time.time() - t1
    print(f"Time used: {t:.2f} s")
    print(f"Found {model.SolCount} trails")
    sol_num = model.SolCount
    
    
    print("Computing sign and conditions for each trail...")
    signs = []
    correlations = []
    trails_conditions = []
    corr_dict = {}
    MASK = []
    avg_prob = None
    
    for m in tqdm(range(model.SolCount)):
        model.params.SolutionNumber = m
        mask_trail = [
            [
                [round(u[r, side, l].Xn) for l in range(BLOCK_SIZE)]
                for side in range(2)
            ]
            for r in range(nb_rounds)
        ]
        
        
        result = compute_correlation(diff_trail, mask_trail, nb_rounds)
        if result == "Error":
            continue
        sign, corr, conditions = result
        
        if avg_prob is None:
            avg_prob = corr
        
        corr_dict[corr] = corr_dict.get(corr, 0) + 1
        signs.append(sign)
        correlations.append(corr)
        trails_conditions.append(conditions)
        MASK.append(mask_trail)
    
    print(f"Number of valid masks: {len(MASK)}")
    masks = MASK
    
    if len(masks) == 0:
        print("no masks identified!")
        return None, sol_num, None
    
    
    
    res = np.zeros((nb_rounds, 2, BLOCK_SIZE), dtype=np.float64)
    for m in range(len(masks)):
        res += np.array(masks[m]) * (2 ** ((-1) * avg_prob + correlations[m]))
    
    
    MSK = res.copy()
    for r in range(nb_rounds):
        for i in range(2):
            for j in range(BLOCK_SIZE):
                if MSK[r][i][j] < THRESH:
                    MSK[r][i][j] = 0
                else:
                    MSK[r][i][j] = 1
    
    
    if save_pth is None:
        save_pth = f'./freq_msk/masks_freq_GIFT128_{nb_rounds}RD{NAME}_CORR{MIN_CORR}'
    np.save(save_pth + f'_T{THRESH}.npy', np.array(MSK))
    print(f"mask saved at: {save_pth + f'_T{THRESH}.npy'}")
    np.save(save_pth + '.npy', np.array(res))
    
    return MSK, sol_num, save_pth + f'_T{THRESH}.npy'


if __name__ == "__main__":
    masks = GIFT_MILP_Quasi_Diff(NB_ROUNDS)