from gurobipy import *
from utils import *
import random, sys, numpy as np
from tqdm      import tqdm
from math      import log2
from itertools import product
from fast_distribut import *
blocks = get_transitions(DIFF_TRAIL_FILE)
QDTM_SKINNY_SBOX = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)
AC_STATES = compute_ac_states(NB_ROUNDS, SBOX_SIZE)
PT =[9,15,8,13,10,14,12,11,0,1,2,3,4,5,6,7]
def compute_correlation(diff_trail, mask_trail, nb_rounds):
    conditions = [[[[] for j in range(4)] for i in range(4)] for k in range(nb_rounds)]
    corr = 1
    for k in range(nb_rounds):
        for i in STATE_RANGE:
            for j in STATE_RANGE:
                a, b = diff_trail[k][0][i][j], diff_trail[k][1][i][j]
                u, v = mask_trail[k][0][i][j], mask_trail[k][1][i][j]
                a, b = bin_to_int(a, SBOX_SIZE), bin_to_int(b, SBOX_SIZE)
                u, v = bin_to_int(u, SBOX_SIZE), bin_to_int(v, SBOX_SIZE)
                corr *= QDTM_SKINNY_SBOX[b][a][v][u]
                if QDTM_SKINNY_SBOX[b][a][v][u] == 0:
                    print(a, b)
                    print(u, v)
                    return "Error"    
    for k in range(nb_rounds):
        corr *= character(AC_STATES[k][0],  mask_trail[k][1][0][0], SBOX_SIZE)
        corr *= character(AC_STATES[k][1],  mask_trail[k][1][1][0], SBOX_SIZE)
        corr *= character(AC_STATES[k][2],  mask_trail[k][1][2][0], SBOX_SIZE)
    for k in range(nb_rounds):
        for i in range(2):
            for j in STATE_RANGE:
                if mask_trail[k][1][i][j] == [0 for _ in BIT_RANGE]:
                    continue
                conditions[k][i][j] = mask_trail[k][1][i][j]
    if corr > 0:
        return  1, log2( corr), conditions
    elif corr < 0:
        return -1, log2(-corr), conditions
    return 1, 0, []
def add_xor_constraints(model, x1, x2, y):
    model.addConstr(-x1 + x2 + y >=  0)
    model.addConstr( x1 - x2 + y >=  0)
    model.addConstr( x1 + x2 - y >=  0)
    model.addConstr(-x1 - x2 - y >= -2)
def add_xor_constraints2(model, x1, x2, x3, y):
    model.addConstr(-x1 + x2 + x3 + y >=  0)
    model.addConstr( x1 - x2 + x3 + y >=  0)
    model.addConstr( x1 + x2 - x3 + y >=  0)
    model.addConstr( x1 + x2 + x3 - y >=  0)
    model.addConstr( x1 - x2 - x3 - y >= -2)
    model.addConstr(-x1 + x2 - x3 - y >= -2)
    model.addConstr(-x1 - x2 + x3 - y >= -2)    
    model.addConstr(-x1 - x2 - x3 + y >= -2)
def msk4x4_16(trail):
    mask = []
    for rd_layer in trail:
        tmp_mask = []
        for row in rd_layer:
            for cell in row:
                if (SBOX_SIZE==4):
                    tmp_mask.extend((cell or [0, 0, 0, 0]))  
                else:
                    tmp_mask.extend((cell or [0, 0, 0, 0,0, 0, 0, 0]))  
        mask.append(tmp_mask) 
    return mask
def master_key(ind,rd):
    for i in range(rd):
        ind=PT[ind]
    return ind
def SKINNY_MILP_Quasi_Diff(nb_rounds,save_pth=None):
    diff_trail = extract_diff_trail(DIFF_TRAIL_FILE, nb_rounds)
    sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)
    model = Model("SKINNY_SK_Quasi_Diff_MILP")
    u = model.addVars(nb_rounds, 2, 4, 4, SBOX_SIZE, vtype=GRB.BINARY)
    Q = model.addVars(nb_rounds, 4, 4, CORR_RANGE, vtype=GRB.BINARY)
    for i in STATE_RANGE:
        for j in STATE_RANGE:
            model.addConstrs(u[0, 0, i, j, l] == 0 for l in BIT_RANGE)
            model.addConstrs(u[nb_rounds - 1, 1, i, j, l] == 0 for l in BIT_RANGE)
    for r in tqdm(range(nb_rounds)):
        for i in STATE_RANGE:
            for j in STATE_RANGE:                
                b = bin_to_int([diff_trail[r][1][i][j][l] for l in BIT_RANGE], SBOX_SIZE) 
                a = bin_to_int([diff_trail[r][0][i][j][l] for l in BIT_RANGE], SBOX_SIZE)
                model.addConstr(quicksum(Q[r, i, j, corr] for corr in CORR_RANGE) == 1)
                for corr in CORR_RANGE:
                    if sbox_inequalities[b][a][corr] == []:
                        model.addConstr(Q[r, i, j, corr] == 0)
                        continue
                    for ineq in sbox_inequalities[b][a][corr]:
                        model.addConstr(quicksum(ineq[2 * SBOX_SIZE - l - 1] * u[r, 1, i, j, l] for l in BIT_RANGE) + \
                                        quicksum(ineq[1 * SBOX_SIZE - l - 1] * u[r, 0, i, j, l] for l in BIT_RANGE) - \
                                        ineq[2 * SBOX_SIZE] + 500 * (1 - Q[r, i, j, corr]) >= 0)  
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:     
                model.addConstrs(u[r, 1, 3, (j - 3) % 4, l] == u[r + 1, 0, 0, j, l] for l in BIT_RANGE)
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:
            for l in BIT_RANGE:
                add_xor_constraints2(model, u[r, 1, 0, j, l],
                                            u[r, 1, 1, (j - 1) % 4, l], 
                                            u[r, 1, 2, (j - 2) % 4, l], 
                                            u[r + 1, 0, 1, j, l]) 
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:
            model.addConstrs(u[r, 1, 1, (j - 1) % 4, l] == u[r + 1, 0, 2, j, l] for l in BIT_RANGE)
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:
            for l in BIT_RANGE:                
                add_xor_constraints2(model, u[r, 1, 1, (j - 1) % 4, l], 
                                            u[r, 1, 2, (j - 2) % 4, l], 
                                            u[r, 1, 3, (j - 3) % 4, l], 
                                            u[r + 1, 0, 3, j, l])
    model.addConstr(quicksum(Q[r, i, j, corr] * corr for r in range(NB_ROUNDS) for i in STATE_RANGE for j in STATE_RANGE for corr in CORR_RANGE) >= -MIN_CORR)
    model.setObjective(quicksum(Q[r, i, j, corr] * corr for r in range(NB_ROUNDS) for i in STATE_RANGE for j in STATE_RANGE for corr in CORR_RANGE), GRB.MAXIMIZE)
    print("Searching for quasi-differential trails...\n")
    model.params.PoolSearchMode = 2
    model.params.PoolSolutions = 2000000
    model.optimize()    
    print("Found ", model.SolCount, "trails")
    print("Computing sign and conditions for each trail...  ")
    signs = []
    correlations = []
    trails_conditions = []
    corr_dict = {}
    masks=[]
    for m in tqdm(range(model.SolCount)):
        model.params.SolutionNumber = m
        mask_trail = [[[[[round(u[r, before, i, j, l].Xn) for l in BIT_RANGE]
                            for j in STATE_RANGE] for i in STATE_RANGE] 
                            for before in range(2)] for r in range(nb_rounds)]
        mask_trail1 = [[[[[round(u[r, before, i, j, l].Xn) for l in reversed(BIT_RANGE)]
                            for j in STATE_RANGE] for i in STATE_RANGE] 
                            for before in range(2)] for r in range(nb_rounds)]
        arr = np.array(mask_trail1)
        new_arr = arr.reshape(*arr.shape[:2], 16*SBOX_SIZE)
        masks.append(new_arr.copy())
        sign, corr, conditions = compute_correlation(diff_trail, mask_trail, nb_rounds)
        if m == 0:
            avg_prob = corr
        if corr not in corr_dict:
            corr_dict[corr] = 0
        corr_dict[corr] += 1
        signs.append(sign)
        correlations.append(corr)
        trails_conditions.append(conditions)
    LST_RES=[]
    keys=[]
    for t in trails_conditions:
        condi = msk4x4_16(t) 
        key = []
        for r in range(NB_ROUNDS):
            for state_bit in range(8*SBOX_SIZE): 
                if condi[r][state_bit] == 1:
                    nibble_idx = state_bit // SBOX_SIZE
                    bit_offset = SBOX_SIZE - 1 - (state_bit % SBOX_SIZE) 
                    mk_nibble_idx = master_key(nibble_idx, r)
                    mk_bit_idx = mk_nibble_idx * SBOX_SIZE + bit_offset 
                    key.append(f'MK_{mk_bit_idx}')
        keys.append(key.copy())
    res=np.array(masks[0], dtype=np.float64) 
    if(len(res)==0):
        print("no masks identified!")
    else:
        for m in range(len(masks)):
            res+=np.array(masks[m])*(2**((-1)*avg_prob+correlations[m]))
            print(f"weight{m}: {2**((-1)*avg_prob+correlations[m])}")
        MSK=res.copy()
        for r in range(NB_ROUNDS):
            print(f'round{r}')
            for i in range(2):
                print('[',end="")
                for j in range(16 * SBOX_SIZE):
                    print(res[r][i][j],end=",")
                    if(MSK[r][i][j]<THRESH):
                        MSK[r][i][j]=0
                    else:
                        MSK[r][i][j]=1
                print(']')
        print("="*50)
    if save_pth is None:
        save_pth=f'./freq_msk/masks_freq_{NB_ROUNDS}RD_CORR{MIN_CORR}'
    np.save(save_pth+f'_T{THRESH}.npy', np.array(MSK))
    print(f"mask saved at :{save_pth+f'_T{THRESH}.npy'}")
    np.save(save_pth+'.npy', np.array(res))
    print("+++++ mask ++++++")
    print(MSK)
    for l in range(len(signs)):
        dic_tmp={}
        dic_tmp['sign']=signs[l]
        dic_tmp['corr']=correlations[l]
        dic_tmp['keys']=keys[l]
        LST_RES.append(dic_tmp.copy())
    print(LST_RES)
    return LST_RES
if __name__=="__main__":
    LST_RES=SKINNY_MILP_Quasi_Diff(NB_ROUNDS)
    plot_quasi_distribut(LST_RES,f'{NB_ROUNDS}_{MIN_CORR}')