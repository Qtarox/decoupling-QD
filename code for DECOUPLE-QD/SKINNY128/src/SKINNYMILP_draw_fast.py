from gurobipy import *
from utils import *
import random, sys, numpy as np
import json, os
from tqdm      import tqdm
from math      import log2
from itertools import product
from fast_distribut import *

blocks = get_transitions(DIFF_TRAIL_FILE)
QDTM_SKINNY_SBOX = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)
AC_STATES = compute_ac_states(NB_ROUNDS, SBOX_SIZE)
PT =[9,15,8,13,10,14,12,11,0,1,2,3,4,5,6,7]
# blocks = [(15,15)]
# QDTM_SKINNY_SBOX = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)
# for i in range(16):
#     for j in range(16):
#         print(QDTM_SKINNY_SBOX[15][15][i][j],end='\t')
#     print()

# assert False

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

                q = QDTM_SKINNY_SBOX[b][a][v][u]
                if q == 0:
                    # FIX 1: Previously returned the string "Error", which
                    # crashed the caller's `sign, corr, conditions = ...`
                    # unpack. Return a sentinel tuple instead so the caller
                    # can skip this trail cleanly.
                    print(a, b)
                    print(u, v)
                    return 0, float('-inf'), []
                corr *= q

                        # Linear layer
    # AddConstants
    for k in range(nb_rounds):
        corr *= character(AC_STATES[k][0],  mask_trail[k][1][0][0], SBOX_SIZE)
        corr *= character(AC_STATES[k][1],  mask_trail[k][1][1][0], SBOX_SIZE)
        corr *= character(AC_STATES[k][2],  mask_trail[k][1][2][0], SBOX_SIZE)
        
    # AddRoundKey
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

# Add constraints to the model so that y = x1 ^ x2
def add_xor_constraints(model, x1, x2, y):
    model.addConstr(-x1 + x2 + y >=  0)
    model.addConstr( x1 - x2 + y >=  0)
    model.addConstr( x1 + x2 - y >=  0)
    model.addConstr(-x1 - x2 - y >= -2)
    
# Add constraints to the model so that y = x1 ^ x2 ^ x3
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
    # 直接遍历列表元素，比使用 range(len(...)) 然后再按索引取值快得多
    mask = []
    for rd_layer in trail:
        tmp_mask = []
        for row in rd_layer:
            for cell in row:
                # 核心魔法：如果 cell 是空列表 []，它在 Python 里相当于 False
                # 所以 `cell or [0, 0, 0, 0]` 会在为空时直接返回后面的 4 个 0 列表
                # .extend() 一次性把 4 个元素塞进去，比 append 4次快得多
                if (SBOX_SIZE==4):
                    tmp_mask.extend((cell or [0, 0, 0, 0]))  
                else:
                    tmp_mask.extend((cell or [0, 0, 0, 0,0, 0, 0, 0]))  
                    
        # 因为 tmp_mask 是每一轮重新创建的，直接 append 即可，不需要 .copy()
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

                        # Definition of Variables
    # State variables
    u = model.addVars(nb_rounds, 2, 4, 4, SBOX_SIZE, vtype=GRB.BINARY)
    Q = model.addVars(nb_rounds, 4, 4, CORR_RANGE, vtype=GRB.BINARY)
    
                        # Adding Constraints

                        # Starting/Ending with zero masks contraints"""
    for i in STATE_RANGE:
        for j in STATE_RANGE:
            model.addConstrs(u[0, 0, i, j, l] == 0 for l in BIT_RANGE)
            model.addConstrs(u[nb_rounds - 1, 1, i, j, l] == 0 for l in BIT_RANGE)
            
    

                        # Non-linear layer constraints
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
                                        ineq[2 * SBOX_SIZE] + 500 * (1 - Q[r, i, j, corr]) >= 0)  # M = 500
                        
                        # Linear layer constraints
    # MixColumns
    # First row 
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:     
                model.addConstrs(u[r, 1, 3, (j - 3) % 4, l] == u[r + 1, 0, 0, j, l] for l in BIT_RANGE)
            
    
    # Second row
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:
            for l in BIT_RANGE:
                add_xor_constraints2(model, u[r, 1, 0, j, l],
                                            u[r, 1, 1, (j - 1) % 4, l], 
                                            u[r, 1, 2, (j - 2) % 4, l], 
                                            u[r + 1, 0, 1, j, l]) 

    # Third row
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:
            model.addConstrs(u[r, 1, 1, (j - 1) % 4, l] == u[r + 1, 0, 2, j, l] for l in BIT_RANGE)

    # Fourth row
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:
            for l in BIT_RANGE:                
                add_xor_constraints2(model, u[r, 1, 1, (j - 1) % 4, l], 
                                            u[r, 1, 2, (j - 2) % 4, l], 
                                            u[r, 1, 3, (j - 3) % 4, l], 
                                            u[r + 1, 0, 3, j, l])


    # Correlation constraints
    model.addConstr(quicksum(Q[r, i, j, corr] * corr for r in range(NB_ROUNDS) for i in STATE_RANGE for j in STATE_RANGE for corr in CORR_RANGE) >= -MIN_CORR)
    model.setObjective(quicksum(Q[r, i, j, corr] * corr for r in range(NB_ROUNDS) for i in STATE_RANGE for j in STATE_RANGE for corr in CORR_RANGE), GRB.MAXIMIZE)

                        # Gurobi options
    print("Searching for quasi-differential trails...\n")
    model.params.PoolSearchMode = 2
    model.params.PoolSolutions = 2000000
                        # Resolution of the model
    model.optimize()    

    print("Found ", model.SolCount, "trails")
                        # Computation of correlation
    print("Computing sign and conditions for each trail...  ")
    signs = []
    correlations = []
    trails_conditions = []
    
    corr_dict = {}
    masks=[]
    avg_prob = None     # FIX: initialise so we can detect "no valid trail" later
    for m in tqdm(range(model.SolCount)):
        model.params.SolutionNumber = m

        mask_trail = [[[[[round(u[r, before, i, j, l].Xn) for l in BIT_RANGE]
                            for j in STATE_RANGE] for i in STATE_RANGE] 
                            for before in range(2)] for r in range(nb_rounds)]
        mask_trail1 = [[[[[round(u[r, before, i, j, l].Xn) for l in reversed(BIT_RANGE)]
                            for j in STATE_RANGE] for i in STATE_RANGE] 
                            for before in range(2)] for r in range(nb_rounds)]

        sign, corr, conditions = compute_correlation(diff_trail, mask_trail, nb_rounds)

        # FIX 1 (caller side): skip trails where compute_correlation hit a
        # zero QDTM entry. This used to crash because the function returned
        # the string "Error". Now we get (0, -inf, []) and can skip cleanly.
        if sign == 0:
            continue

        # Only push the mask AFTER we know this trail is valid, so that
        # `masks`, `signs`, `correlations`, `trails_conditions` stay
        # index-aligned.
        arr = np.array(mask_trail1)
        new_arr = arr.reshape(*arr.shape[:2], 16*SBOX_SIZE)
        masks.append(new_arr.copy())

        if avg_prob is None:
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
        # condi 的形状是 (NB_ROUNDS, 64)
        condi = msk4x4_16(t) 
        key = []
        for r in range(NB_ROUNDS):
            # SKINNY 的轮密钥只加到前两行，即前 8 个 nibbles，共计 32 个 bits
            for state_bit in range(8*SBOX_SIZE): 
                if condi[r][state_bit] == 1:
                    # 1. 计算当前状态位对应的 Nibble 索引 (0-7) 和 位偏移 (0-3)
                                        # 正确逻辑：端序修正
                    nibble_idx = state_bit // SBOX_SIZE

                    # 如果 l=0 在 bin_to_int 中是 MSB(bit 3)，这里我们要把 0 映射为 3，1 映射为 2
                    bit_offset = SBOX_SIZE - 1 - (state_bit % SBOX_SIZE) 

                    mk_nibble_idx = master_key(nibble_idx, r)
                    # 这里还要注意，主密钥通常也是 MSB 在左 LSB 在右，
                    # 所以按常规大端定义，推导出来的绝对比特索引如下：
                    mk_bit_idx = mk_nibble_idx * SBOX_SIZE + bit_offset 
                    key.append(f'MK_{mk_bit_idx}')

                    
        keys.append(key.copy())
    
    #TODO1. GENERATE THE MASK FREQ
    #TODO2. RETURN DISTRIBUTION INFO

    # FIX 2: previously `res = np.array(masks[0], ...)` and then the loop
    # also added masks[0] in -- masks[0] was double-counted unless it
    # happened to be the all-zero solution. Use np.zeros_like instead.
    # Also: the original `if(len(res)==0)` was checking the wrong thing
    # (res is always shape (nb_rounds, 2, 16*SBOX_SIZE), never length 0);
    # what we really want to check is whether any valid mask was found.
    if len(masks) == 0:
        print("no masks identified!")
        MSK = np.zeros((NB_ROUNDS, 2, 16 * SBOX_SIZE), dtype=np.float64)
        res = MSK.copy()
    else:
        res = np.zeros_like(np.array(masks[0]), dtype=np.float64)
        for m in range(len(masks)):
            # Mask-frequency accumulator (NOT the signed correlation sum
            # -- that one is computed in plot_quasi_distribut). Weighting
            # by 2^(corr - avg_prob) keeps the highest-correlation trail
            # at weight 1 and damps weaker ones.
            res += np.array(masks[m]) * (2**(correlations[m] - avg_prob))
            # FIX 4 (partial): commented out to avoid millions of prints
            # when PoolSolutions is large. Uncomment for small runs.
            # print(f"weight{m}: {2**(correlations[m] - avg_prob)}")
        MSK = res.copy()
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

        # MSK IS THE FILTERED ONES 
    
    if save_pth is None:
        save_pth=f'./freq_msk/masks_freq_{NB_ROUNDS}RD_CORR{MIN_CORR}'
    # Make sure the output directory exists so np.save doesn't blow up.
    os.makedirs(os.path.dirname(save_pth) or '.', exist_ok=True)
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

    # FIX 4: don't dump 2,000,000 dicts to stdout. Show a summary and
    # persist the full list to disk in case downstream tooling needs it.
    print(f"Collected {len(LST_RES)} trails (corr histogram: {corr_dict})")
    lst_res_path = f'./lst_res_{NB_ROUNDS}RD_CORR{MIN_CORR}.json'
    try:
        with open(lst_res_path, 'w') as f:
            json.dump(LST_RES, f)
        print(f"LST_RES dumped to {lst_res_path}")
    except Exception as e:
        print(f"(could not dump LST_RES to disk: {e})")
    
    
    return LST_RES

if __name__=="__main__":
    LST_RES=SKINNY_MILP_Quasi_Diff(NB_ROUNDS)
    plot_quasi_distribut(LST_RES,f'{NB_ROUNDS}_{MIN_CORR}')