from gurobipy import *
from utils import *
import random, sys, numpy as np
import json, os
from tqdm      import tqdm
from math      import log2
from itertools import product
from fast_distribut import *
from gurobipy import *
from utils import *
from tqdm import tqdm
import time
import numpy as np
from math import log2
from collections import Counter
import itertools

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

def compute_distribution_from_trails(T, avg_p):
    """
    基于 trail 列表 T 计算概率分布（直接枚举密钥空间或随机采样）
    返回形如 {0.0: 1007, 2.6715645349807984: 68, ...} 的字典
    """
    print(avg_p)
    if len(T) == 0:
        return {}
    
    base_corr = (1) * avg_p
    
    unique_keys = set()
    for t in T:
        unique_keys.update(t['keys'])
    unique_keys = sorted(list(unique_keys))
    num_keys = len(unique_keys)
    
    print(f"  涉及密钥变量数: {num_keys}")
    
    if num_keys == 0:
        # 没有密钥条件，所有 trail 的贡献固定
        total = sum(t['sign'] * (2 ** (t['corr'] - base_corr)) for t in T)
        return {round(total, 10): 1}
    
    key_to_idx = {k: i for i, k in enumerate(unique_keys)}
    total_space = 2 ** num_keys
    
    # 智能判断：全枚举还是采样
    num_samples = 80000
    if total_space <= num_samples:
        X = np.array(list(itertools.product([0, 1], repeat=num_keys)))
        actual = total_space
    else:
        X = np.random.randint(0, 2, size=(num_samples, num_keys))
        actual = num_samples
    
    total_sum = np.zeros(actual)
    for t in T:
        k_indices = [key_to_idx[k] for k in t['keys']]
        if len(k_indices) > 0:
            xor_sum = np.sum(X[:, k_indices], axis=1) % 2
        else:
            xor_sum = np.zeros(actual)
        term_val = t['sign'] * (2 ** (t['corr'] - base_corr)) * ((-1) ** xor_sum)
        total_sum += term_val
    
    # 聚合（四舍五入到 10 位以避免浮点噪声）
    rounded = np.round(total_sum, 10)
    return dict(sorted(Counter(rounded.tolist()).items()))

def SKINNY_MILP_Quasi_Diff(nb_rounds,mask_range,T=250,save_pth=None):
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
    allowed_masks = set(mask_range)
    print(f"Applying mask range constraints (Allowed bits: {len(allowed_masks)})...")
    
    CELL_BITS = SBOX_SIZE          # 每个 cell 的比特数（SKINNY-128 = 8）
    for r in range(nb_rounds):
        for before in range(2):
            for l in range(16 * SBOX_SIZE):
                if (r, before, l) not in allowed_masks:
                    s_ind = l // CELL_BITS
                    i = s_ind // 4
                    j = s_ind % 4
                    k = SBOX_SIZE - 1 - (l % CELL_BITS)   # ← 端序翻回去
                    model.addConstr(u[r, before, i, j, k] == 0,
                                    name=f"mask_limit_{r}_{before}_{l}")    
    

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
                                        ineq[2 * SBOX_SIZE] + 50000 * (1 - Q[r, i, j, corr]) >= 0)  # M = 500
                        
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
    model.addConstr(quicksum(Q[r, i, j, corr] * corr for r in range(NB_ROUNDS) for i in STATE_RANGE for j in STATE_RANGE for corr in CORR_RANGE) >= -T)
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
    
    distribution = compute_distribution_from_trails(LST_RES, avg_prob)
    print(f"Cluster distribution: {distribution}")
    
    return model.SolCount, distribution
def transform_var(var_str):
    """
    将类似 'y_10_53' 的字符串转换为元组 (10, 1, 53)
    """
    # 按下划线拆分字符串
    parts = var_str.split('_')
    
    # 解析各个部分
    prefix = parts[0]
    num1 = int(parts[1])
    num2 = int(parts[2])
    
    # 根据前缀判定中间的值（'x' 为 0，'y' 为 1）
    middle_val = 0 if prefix == 'x' else 1
    
    # 按照 (num1, 字母对应数字, num2) 的顺序返回
    return (num1, middle_val, num2)

# 使用嵌套的列表推导式处理整个二维列表
def trans_lst(MSK):
    transformed_list = [[transform_var(item) for item in sublist] for sublist in MSK]
    return transformed_list
if __name__=="__main__":
    
    # LST_RES=SKINNY_MILP_Quasi_Diff(NB_ROUNDS)
    # plot_quasi_distribut(LST_RES,f'{NB_ROUNDS}_{MIN_CORR}')
    
    MSK_16=[['y_4_2', 'x_5_34', 'y_3_110', 'x_4_6', 'x_4_102', 'y_3_106', 'x_4_2', 'x_4_98', 'y_4_98', 'x_5_26', 'x_5_122', 'y_4_99', 'x_5_27', 'x_5_123', 'x_4_0', 'x_4_1', 'x_4_3', 'x_4_4', 'x_4_5', 'x_4_7', 'y_4_0', 'y_4_1', 'y_4_3', 'y_4_4', 'y_4_5', 'y_4_6', 'y_4_7', 'x_4_96', 'x_4_97', 'x_4_99', 'x_4_100', 'x_4_101', 'x_4_103', 'y_4_96', 'y_4_97', 'y_4_100', 'y_4_101', 'y_4_102', 'y_4_103'],
            ['y_2_12', 'x_3_44', 'y_2_14', 'x_3_46', 'y_4_11', 'x_5_43', 'y_4_12', 'x_5_44', 'y_4_13', 'x_5_45', 'y_4_14', 'x_5_46', 'y_4_15', 'x_5_47', 'y_4_59', 'y_4_83', 'x_5_67', 'y_5_20', 'y_5_68', 'y_5_124', 'x_6_20', 'y_5_22', 'y_5_70', 'y_5_126', 'x_6_22', 'x_6_52', 'x_6_54', 'x_6_116', 'x_6_118', 'y_5_44', 'x_6_84', 'y_5_46', 'x_6_86', 'y_5_47', 'y_5_71', 'x_6_87', 'y_6_6', 'x_7_38', 'y_6_22', 'y_6_70', 'x_7_118', 'x_6_16', 'x_6_17', 'x_6_18', 'x_6_19', 'x_6_21', 'x_6_23', 'y_6_16', 'y_6_17', 'y_6_18', 'y_6_19', 'y_6_20', 'y_6_21', 'y_6_23', 'x_5_40', 'x_5_41', 'x_5_42', 'y_5_40', 'y_5_41', 'y_5_42', 'y_5_43', 'y_5_45', 'x_4_8', 'x_4_9', 'x_4_10', 'x_4_11', 'x_4_12', 'x_4_13', 'x_4_14', 'x_4_15', 'y_4_8', 'y_4_9', 'y_4_10', 'x_5_64', 'x_5_65', 'x_5_66', 'x_5_68', 'x_5_69', 'x_5_70', 'x_5_71', 'y_5_64', 'y_5_65', 'y_5_66', 'y_5_67', 'y_5_69', 'x_6_112', 'x_6_113', 'x_6_114', 'x_6_115', 'x_6_117', 'x_6_119', 'y_6_112', 'y_6_113', 'y_6_114', 'y_6_115', 'y_6_116', 'y_6_117', 'y_6_118', 'y_6_119', 'x_6_80', 'x_6_81', 'x_6_82', 'x_6_83', 'x_6_85', 'y_6_80', 'y_6_81', 'y_6_82', 'y_6_83', 'y_6_84', 'y_6_85', 'y_6_86', 'y_6_87', 'x_6_48', 'x_6_49', 'x_6_50', 'x_6_51', 'x_6_53', 'x_6_55', 'y_6_48', 'y_6_49', 'y_6_50', 'y_6_51', 'y_6_52', 'y_6_53', 'y_6_54', 'y_6_55', 'x_3_40', 'x_3_41', 'x_3_42', 'x_3_43', 'x_3_45', 'x_3_47', 'y_3_40', 'y_3_41', 'y_3_42', 'y_3_43', 'y_3_44', 'y_3_45', 'y_3_46', 'y_3_47'],
            ['y_5_12', 'x_6_44', 'y_5_14', 'x_6_46', 'x_6_40', 'x_6_41', 'x_6_42', 'x_6_43', 'x_6_45', 'x_6_47', 'y_6_40', 'y_6_41', 'y_6_42', 'y_6_43', 'y_6_44', 'y_6_45', 'y_6_46', 'y_6_47'],
            ['y_5_28', 'x_6_60', 'y_5_30', 'x_6_62', 'x_6_56', 'x_6_57', 'x_6_58', 'x_6_59', 'x_6_61', 'x_6_63', 'y_6_56', 'y_6_57', 'y_6_58', 'y_6_59', 'y_6_60', 'y_6_61', 'y_6_62', 'y_6_63'],
            ['y_5_36', 'y_5_92', 'x_6_76', 'y_5_38', 'y_5_94', 'x_6_78', 'y_6_50', 'y_6_74', 'x_7_90', 'y_6_54', 'y_6_78', 'x_7_94', 'y_7_8', 'y_7_88', 'x_8_104', 'y_7_9', 'y_7_89', 'x_8_105', 'y_7_10', 'y_7_90', 'x_8_106', 'y_7_11', 'y_7_91', 'x_8_107', 'y_7_12', 'y_7_92', 'x_8_108', 'y_7_13', 'y_7_93', 'x_8_109', 'y_7_14', 'y_7_94', 'x_8_110', 'x_7_8', 'x_7_9', 'x_7_10', 'x_7_11', 'x_7_12', 'x_7_13', 'x_7_14', 'x_7_15', 'y_7_15', 'x_8_111', 'y_8_104', 'y_8_105', 'y_8_106', 'y_8_107', 'y_8_108', 'y_8_109', 'y_8_110', 'y_8_111', 'x_7_88', 'x_7_89', 'x_7_91', 'x_7_92', 'x_7_93', 'x_7_95', 'y_7_95', 'x_6_72', 'x_6_73', 'x_6_74', 'x_6_75', 'x_6_77', 'x_6_79', 'y_6_72', 'y_6_73', 'y_6_75', 'y_6_76', 'y_6_77', 'y_6_79'],
            ['y_4_26', 'x_5_58', 'x_4_24', 'x_4_25', 'x_4_26', 'x_4_27', 'x_4_28', 'x_4_29', 'x_4_30', 'x_4_31', 'y_4_24', 'y_4_25', 'y_4_27', 'y_4_28', 'y_4_29', 'y_4_30', 'y_4_31'],
            ['y_4_3', 'x_5_35', 'y_4_83', 'x_5_99'],
            ['y_1_19', 'x_2_51'],            
            ]
    MSK_16_05=[['y_4_2', 'x_5_34', 'y_4_99', 'x_5_27', 'x_5_123', 'y_3_106', 'x_4_2', 'x_4_98', 'y_3_110', 'x_4_6', 'x_4_102', 'y_4_98', 'x_5_26', 'x_5_122', 'x_4_0', 'x_4_1', 'x_4_3', 'x_4_4', 'x_4_5', 'x_4_7', 'y_4_0', 'y_4_1', 'y_4_3', 'y_4_4', 'y_4_5', 'y_4_6', 'y_4_7', 'x_4_96', 'x_4_97', 'x_4_99', 'x_4_100', 'x_4_101', 'x_4_103', 'y_4_96', 'y_4_97', 'y_4_100', 'y_4_101', 'y_4_102', 'y_4_103'], 
               ['y_5_36', 'y_5_92', 'x_6_76', 'y_5_38', 'y_5_94', 'x_6_78', 'y_6_50', 'y_6_74', 'x_7_90', 'y_6_54', 'y_6_78', 'x_7_94', 'y_7_10', 'y_7_90', 'x_8_106', 'x_6_72', 'x_6_73', 'x_6_74', 'x_6_75', 'x_6_77', 'x_6_79', 'y_6_72', 'y_6_73', 'y_6_75', 'y_6_76', 'y_6_77', 'y_6_79', 'x_7_88', 'x_7_89', 'x_7_91', 'x_7_92', 'x_7_93', 'x_7_95', 'y_7_88', 'y_7_89', 'y_7_91', 'y_7_92', 'y_7_93', 'y_7_94', 'y_7_95'], 
               ['y_4_12', 'x_5_44', 'y_4_14', 'x_5_46', 'y_4_15', 'x_5_47', 'y_6_6', 'x_7_38', 'x_5_40', 'x_5_41', 'x_5_42', 'x_5_43', 'x_5_45', 'y_5_40', 'y_5_41', 'y_5_42', 'y_5_43', 'y_5_44', 'y_5_45', 'y_5_46', 'y_5_47', 'x_4_8', 'x_4_9', 'x_4_10', 'x_4_11', 'x_4_12', 'x_4_13', 'x_4_14', 'x_4_15', 'y_4_8', 'y_4_9', 'y_4_10', 'y_4_11', 'y_4_13'], 
               ['y_2_12', 'x_3_44', 'y_2_14', 'x_3_46', 'y_4_59', 'y_4_83', 'x_5_67', 'y_5_20', 'y_5_68', 'y_5_124', 'x_6_20', 'y_5_22', 'y_5_70', 'y_5_126', 'x_6_22', 'x_6_52', 'x_6_54', 'x_6_116', 'x_6_118', 'y_5_44', 'x_6_84', 'y_5_46', 'x_6_86', 'y_6_22', 'y_6_70', 'x_7_118', 'x_6_16', 'x_6_17', 'x_6_18', 'x_6_19', 'x_6_21', 'x_6_23', 'y_6_16', 'y_6_17', 'y_6_18', 'y_6_19', 'y_6_20', 'y_6_21', 'y_6_23', 'x_5_64', 'x_5_65', 'x_5_66', 'x_5_68', 'x_5_69', 'x_5_70', 'x_5_71', 'y_5_64', 'y_5_65', 'y_5_66', 'y_5_67', 'y_5_69', 'y_5_71', 'x_6_112', 'x_6_113', 'x_6_114', 'x_6_115', 'x_6_117', 'x_6_119', 'y_6_112', 'y_6_113', 'y_6_114', 'y_6_115', 'y_6_116', 'y_6_117', 'y_6_118', 'y_6_119', 'x_6_80', 'x_6_81', 'x_6_82', 'x_6_83', 'x_6_85', 'x_6_87', 'y_6_80', 'y_6_81', 'y_6_82', 'y_6_83', 'y_6_84', 'y_6_85', 'y_6_86', 'y_6_87', 'x_6_48', 'x_6_49', 'x_6_50', 'x_6_51', 'x_6_53', 'x_6_55', 'y_6_48', 'y_6_49', 'y_6_50', 'y_6_51', 'y_6_52', 'y_6_53', 'y_6_54', 'y_6_55', 'x_3_40', 'x_3_41', 'x_3_42', 'x_3_43', 'x_3_45', 'x_3_47', 'y_3_40', 'y_3_41', 'y_3_42', 'y_3_43', 'y_3_44', 'y_3_45', 'y_3_46', 'y_3_47'], 
               ['y_4_26', 'x_5_58', 'x_4_24', 'x_4_25', 'x_4_26', 'x_4_27', 'x_4_28', 'x_4_29', 'x_4_30', 'x_4_31', 'y_4_24', 'y_4_25', 'y_4_27', 'y_4_28', 'y_4_29', 'y_4_30', 'y_4_31'], 
               ['y_5_12', 'x_6_44', 'y_5_14', 'x_6_46', 'x_6_40', 'x_6_41', 'x_6_42', 'x_6_43', 'x_6_45', 'x_6_47', 'y_6_40', 'y_6_41', 'y_6_42', 'y_6_43', 'y_6_44', 'y_6_45', 'y_6_46', 'y_6_47'], 
               ['y_4_3', 'x_5_35', 'y_4_83', 'x_5_99'], 
               ['y_5_28', 'x_6_60', 'y_5_30', 'x_6_62', 'x_6_56', 'x_6_57', 'x_6_58', 'x_6_59', 'x_6_61', 'x_6_63', 'y_6_56', 'y_6_57', 'y_6_58', 'y_6_59', 'y_6_60', 'y_6_61', 'y_6_62', 'y_6_63'], 
               ['y_1_19', 'x_2_51']]
    MSK_16_131=[['y_4_2', 'x_5_34', 'y_3_110', 'x_4_6', 'x_4_102', 'y_3_106', 'x_4_2', 'x_4_98', 'y_4_98', 'x_5_26', 'x_5_122', 'y_4_99', 'x_5_27', 'x_5_123', 'x_4_0', 'x_4_1', 'x_4_3', 'x_4_4', 'x_4_5', 'x_4_7', 'y_4_0', 'y_4_1', 'y_4_3', 'y_4_4', 'y_4_5', 'y_4_6', 'y_4_7', 'x_4_96', 'x_4_97', 'x_4_99', 'x_4_100', 'x_4_101', 'x_4_103', 'y_4_96', 'y_4_97', 'y_4_100', 'y_4_101', 'y_4_102', 'y_4_103'], ['y_2_12', 'x_3_44', 'y_2_14', 'x_3_46', 'y_4_11', 'x_5_43', 'y_4_12', 'x_5_44', 'y_4_13', 'x_5_45', 'y_4_14', 'x_5_46', 'y_4_15', 'x_5_47', 'y_4_59', 'y_4_83', 'x_5_67', 'y_5_20', 'y_5_68', 'y_5_124', 'x_6_20', 'y_5_22', 'y_5_70', 'y_5_126', 'x_6_22', 'x_6_52', 'x_6_54', 'x_6_116', 'x_6_118', 'y_5_44', 'x_6_84', 'y_5_46', 'x_6_86', 'y_5_47', 'y_5_71', 'x_6_87', 'y_6_6', 'x_7_38', 'y_6_22', 'y_6_70', 'x_7_118', 'x_6_16', 'x_6_17', 'x_6_18', 'x_6_19', 'x_6_21', 'x_6_23', 'y_6_16', 'y_6_17', 'y_6_18', 'y_6_19', 'y_6_20', 'y_6_21', 'y_6_23', 'x_5_40', 'x_5_41', 'x_5_42', 'y_5_40', 'y_5_41', 'y_5_42', 'y_5_43', 'y_5_45', 'x_4_8', 'x_4_9', 'x_4_10', 'x_4_11', 'x_4_12', 'x_4_13', 'x_4_14', 'x_4_15', 'y_4_8', 'y_4_9', 'y_4_10', 'x_5_64', 'x_5_65', 'x_5_66', 'x_5_68', 'x_5_69', 'x_5_70', 'x_5_71', 'y_5_64', 'y_5_65', 'y_5_66', 'y_5_67', 'y_5_69', 'x_6_112', 'x_6_113', 'x_6_114', 'x_6_115', 'x_6_117', 'x_6_119', 'y_6_112', 'y_6_113', 'y_6_114', 'y_6_115', 'y_6_116', 'y_6_117', 'y_6_118', 'y_6_119', 'x_6_80', 'x_6_81', 'x_6_82', 'x_6_83', 'x_6_85', 'y_6_80', 'y_6_81', 'y_6_82', 'y_6_83', 'y_6_84', 'y_6_85', 'y_6_86', 'y_6_87', 'x_6_48', 'x_6_49', 'x_6_50', 'x_6_51', 'x_6_53', 'x_6_55', 'y_6_48', 'y_6_49', 'y_6_50', 'y_6_51', 'y_6_52', 'y_6_53', 'y_6_54', 'y_6_55', 'x_3_40', 'x_3_41', 'x_3_42', 'x_3_43', 'x_3_45', 'x_3_47', 'y_3_40', 'y_3_41', 'y_3_42', 'y_3_43', 'y_3_44', 'y_3_45', 'y_3_46', 'y_3_47'], ['y_5_12', 'x_6_44', 'y_5_14', 'x_6_46', 'x_6_40', 'x_6_41', 'x_6_42', 'x_6_43', 'x_6_45', 'x_6_47', 'y_6_40', 'y_6_41', 'y_6_42', 'y_6_43', 'y_6_44', 'y_6_45', 'y_6_46', 'y_6_47'], ['y_5_28', 'x_6_60', 'y_5_30', 'x_6_62', 'x_6_56', 'x_6_57', 'x_6_58', 'x_6_59', 'x_6_61', 'x_6_63', 'y_6_56', 'y_6_57', 'y_6_58', 'y_6_59', 'y_6_60', 'y_6_61', 'y_6_62', 'y_6_63'], ['y_5_36', 'y_5_92', 'x_6_76', 'y_5_38', 'y_5_94', 'x_6_78', 'y_6_50', 'y_6_74', 'x_7_90', 'y_6_54', 'y_6_78', 'x_7_94', 'y_7_8', 'y_7_88', 'x_8_104', 'y_7_9', 'y_7_89', 'x_8_105', 'y_7_10', 'y_7_90', 'x_8_106', 'y_7_11', 'y_7_91', 'x_8_107', 'y_7_12', 'y_7_92', 'x_8_108', 'y_7_13', 'y_7_93', 'x_8_109', 'y_7_14', 'y_7_94', 'x_8_110', 'x_7_8', 'x_7_9', 'x_7_10', 'x_7_11', 'x_7_12', 'x_7_13', 'x_7_14', 'x_7_15', 'y_7_15', 'x_8_111', 'y_8_104', 'y_8_105', 'y_8_106', 'y_8_107', 'y_8_108', 'y_8_109', 'y_8_110', 'y_8_111', 'x_7_88', 'x_7_89', 'x_7_91', 'x_7_92', 'x_7_93', 'x_7_95', 'y_7_95', 'x_6_72', 'x_6_73', 'x_6_74', 'x_6_75', 'x_6_77', 'x_6_79', 'y_6_72', 'y_6_73', 'y_6_75', 'y_6_76', 'y_6_77', 'y_6_79'], ['y_4_26', 'x_5_58', 'x_4_24', 'x_4_25', 'x_4_26', 'x_4_27', 'x_4_28', 'x_4_29', 'x_4_30', 'x_4_31', 'y_4_24', 'y_4_25', 'y_4_27', 'y_4_28', 'y_4_29', 'y_4_30', 'y_4_31'], ['y_4_3', 'x_5_35', 'y_4_83', 'x_5_99'], ['y_1_19', 'x_2_51']]
    
    MSK_14_05=[['y_0_0', 'x_1_32'], ['y_2_127', 'x_3_23', 'x_3_119', 'y_1_101', 'x_2_29', 'x_2_125', 'x_2_120', 'x_2_121', 'x_2_122', 'x_2_123', 'x_2_124', 'x_2_126', 'x_2_127', 'y_2_120', 'y_2_121', 'y_2_122', 'y_2_123', 'y_2_124', 'y_2_125', 'y_2_126'], ['y_3_17', 'x_4_49', 'y_3_57', 'y_3_81', 'x_4_65', 'y_3_63', 'y_3_87', 'x_4_71', 'y_4_7', 'x_5_39', 'y_4_19', 'x_5_51', 'y_4_67', 'x_5_115', 'y_4_23', 'y_4_71', 'x_5_119', 'y_5_11', 'y_5_91', 'y_5_115', 'x_6_11', 'y_5_15', 'y_5_95', 'y_5_119', 'x_6_15', 'x_6_43', 'y_5_12', 'x_6_44', 'y_5_13', 'x_6_45', 'x_6_47', 'y_5_35', 'x_6_75', 'y_6_9', 'x_7_41', 'y_6_11', 'x_7_43', 'y_4_65', 'x_5_49', 'x_5_113', 'y_5_37', 'y_5_117', 'x_6_13', 'x_6_77', 'y_5_33', 'y_5_113', 'x_6_9', 'x_6_41', 'x_6_73', 'x_6_40', 'x_6_42', 'x_6_46', 'y_6_40', 'y_6_41', 'y_6_42', 'y_6_43', 'y_6_44', 'y_6_45', 'y_6_46', 'y_6_47', 'x_6_8', 'x_6_10', 'x_6_12', 'x_6_14', 'y_6_8', 'y_6_10', 'y_6_12', 'y_6_13', 'y_6_14', 'y_6_15', 'x_5_32', 'x_5_33', 'x_5_34', 'x_5_35', 'x_5_36', 'x_5_37', 'x_5_38', 'y_5_32', 'y_5_34', 'y_5_36', 'y_5_38', 'y_5_39', 'x_5_112', 'x_5_114', 'x_5_116', 'x_5_117', 'x_5_118', 'y_5_112', 'y_5_114', 'y_5_116', 'y_5_118', 'x_5_8', 'x_5_9', 'x_5_10', 'x_5_11', 'x_5_12', 'x_5_13', 'x_5_14', 'x_5_15', 'y_5_8', 'y_5_9', 'y_5_10', 'y_5_14', 'x_4_64', 'x_4_66', 'x_4_67', 'x_4_68', 'x_4_69', 'x_4_70', 'y_4_64', 'y_4_66', 'y_4_68', 'y_4_69', 'y_4_70', 'x_6_72', 'x_6_74', 'x_6_76', 'x_6_78', 'x_6_79', 'y_6_72', 'y_6_73', 'y_6_74', 'y_6_75', 'y_6_76', 'y_6_77', 'y_6_78', 'y_6_79'], ['y_5_19', 'x_6_51', 'y_5_20', 'x_6_52', 'y_5_21', 'x_6_53', 'y_5_23', 'x_6_55', 'x_6_48', 'x_6_49', 'x_6_50', 'x_6_54', 'y_6_48', 'y_6_49', 'y_6_50', 'y_6_51', 'y_6_52', 'y_6_53', 'y_6_54', 'y_6_55', 'x_5_16', 'x_5_17', 'x_5_18', 'x_5_19', 'x_5_20', 'x_5_21', 'x_5_22', 'x_5_23', 'y_5_16', 'y_5_17', 'y_5_18', 'y_5_22'], ['y_0_24', 'x_1_56'], ['y_6_27', 'x_7_59', 'y_6_28', 'x_7_60', 'y_6_29', 'x_7_61', 'y_6_31', 'x_7_63', 'x_7_56', 'x_7_57', 'x_7_58', 'x_7_62', 'y_7_56', 'y_7_57', 'y_7_58', 'y_7_59', 'y_7_60', 'y_7_61', 'y_7_62', 'y_7_63', 'x_6_24', 'x_6_25', 'x_6_26', 'x_6_27', 'x_6_28', 'x_6_29', 'x_6_30', 'x_6_31', 'y_6_24', 'y_6_25', 'y_6_26', 'y_6_30'], ['y_2_29', 'x_3_61', 'y_2_31', 'x_3_63', 'y_2_77', 'x_3_125', 'y_2_79', 'x_3_127', 'y_3_121', 'x_4_17', 'x_4_113', 'x_3_56', 'x_3_57', 'x_3_58', 'x_3_59', 'x_3_60', 'x_3_62', 'y_3_56', 'y_3_57', 'y_3_58', 'y_3_59', 'y_3_60', 'y_3_61', 'y_3_62', 'y_3_63', 'x_3_120', 'x_3_121', 'x_3_122', 'x_3_123', 'x_3_124', 'x_3_126', 'y_3_120', 'y_3_122', 'y_3_123', 'y_3_124', 'y_3_125', 'y_3_126', 'y_3_127'], ['y_0_48', 'y_0_72', 'x_1_88', 'y_1_37', 'y_1_93', 'x_2_77', 'x_1_89', 'x_1_90', 'x_1_91', 'x_1_92', 'x_1_93', 'x_1_94', 'x_1_95', 'y_1_88', 'y_1_89', 'y_1_90', 'y_1_91', 'y_1_92', 'y_1_94', 'y_1_95'], ['y_4_3', 'x_5_35']]
    MSK_LST = trans_lst(MSK_16_131)
    # MSK_LST = MSK_12_60
    
    # file_prefix = f"./constraints/CONS_{NB_ROUNDS}R_{MIN_CORR}{NAME}_TH{THRES}"
    # file_py=mask_py_file = f"{file_prefix}_FULL_MSK.py"
    solu = []
    distributions = []
    
    for i in range(len(MSK_LST)):
        print(f"\n========== Cluster {i} ==========")
        nb_sol, dist = SKINNY_MILP_Quasi_Diff(NB_ROUNDS, MSK_LST[i],T=300)
        solu.append(nb_sol)
        distributions.append(dist)
    
    print("\n" + "=" * 50)
    print("Trail counts per cluster:", solu)
    print("\nDistributions per cluster:")
    print(distributions)
    
    # 保存
    import json
    with open(f"distributions_{NB_ROUNDS}R_{MIN_CORR}.json", 'w') as f:
        # JSON 不支持浮点 key，转字符串
        json_safe = [{str(k): v for k, v in d.items()} for d in distributions]
        json.dump({
            'sol_counts': solu,
            'distributions': json_safe,
        }, f, indent=2)
    print(f"\n分布已保存到 distributions_{NB_ROUNDS}R_{MIN_CORR}.json")