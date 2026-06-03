from gurobipy import *
from utils import *
import random, sys, numpy as np
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


def SKINNY_MILP_Quasi_Diff(nb_rounds,mask_range, T=150,save_pth=None):
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
    res=np.array(masks[0], dtype=np.float64) # mask[0] is all zero mask, corresponding to prob_avg
    
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

        # MSK IS THE FILTERED ONES 
    
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
    MSK_7=[['y_0_20', 'y_0_32', 'x_1_40', 'y_0_22', 'y_0_34', 'x_1_42', 'y_1_13', 'y_1_37', 'x_2_61', 'y_1_15', 'y_1_39', 'x_2_63', 'y_1_28', 'y_1_40', 'x_2_32', 'y_1_29', 'y_1_41', 'x_2_33', 'y_1_30', 'y_1_42', 'x_2_34', 'y_1_31', 'y_1_43', 'x_2_35', 'y_2_8', 'x_3_24', 'y_2_10', 'x_3_26', 'y_2_32', 'x_3_56', 'y_2_34', 'x_3_58', 'y_0_48', 'x_1_12', 'x_1_60', 'y_0_38', 'y_0_50', 'x_1_14', 'x_1_30', 'x_1_62', 'y_0_36', 'x_1_28', 'x_1_29', 'x_1_31', 'x_1_41', 'x_1_43', 'x_1_13', 'x_1_15', 'y_1_12', 'y_1_14', 'y_2_33', 'y_2_35'],
           ['y_0_8', 'x_1_24', 'y_0_10', 'x_1_26', 'y_1_0', 'x_2_16', 'y_1_1', 'x_2_17', 'y_1_2', 'x_2_18', 'y_1_3', 'x_2_19', 'y_1_25', 'y_1_37', 'x_2_45', 'y_1_27', 'y_1_39', 'x_2_47', 'y_2_16', 'y_2_44', 'x_3_36', 'y_2_17', 'y_2_45', 'x_3_37', 'y_2_18', 'y_2_46', 'x_3_38', 'y_2_19', 'y_2_47', 'x_3_39', 'y_3_25', 'y_3_37', 'x_4_45', 'y_3_27', 'y_3_39', 'x_4_47', 'y_0_52', 'x_1_0', 'x_1_48', 'y_0_54', 'x_1_2', 'x_1_50', 'y_3_36', 'y_3_38', 'x_1_1', 'x_1_3', 'x_2_44', 'x_2_46'],
           ['y_4_4', 'x_5_20'], ['y_4_6', 'x_5_22']]
    MSK_5=[['y_0_3', 'x_1_19', 'y_1_18', 'y_1_46', 'x_2_38', 'y_2_9', 'x_3_25', 'y_2_11', 'x_3_27', 'y_3_24', 'y_3_36', 'x_4_44', 'y_3_26', 'y_3_38', 'x_4_46', 'x_1_16', 'x_1_17', 'x_1_18', 'y_1_16', 'y_1_17', 'y_1_19', 'x_3_24', 'x_3_26', 'y_3_25', 'y_3_27'], ['y_0_13', 'x_1_29'], ['y_1_0', 'x_2_16', 'y_1_2', 'x_2_18', 'y_1_3', 'x_2_19', 'y_1_40', 'x_2_48', 'y_1_42', 'x_2_50', 'y_1_43', 'x_2_51', 'y_2_49', 'x_3_13', 'x_3_61', 'x_2_17', 'y_2_16', 'y_2_17', 'y_2_18', 'y_2_19', 'x_2_49', 'y_2_48', 'y_2_50', 'y_2_51', 'x_1_40', 'x_1_41', 'x_1_42', 'x_1_43', 'y_1_41', 'x_1_0', 'x_1_1', 'x_1_2', 'x_1_3', 'y_1_1'], ['y_3_12', 'x_4_28'], ['y_3_14', 'x_4_30'], ['y_1_6', 'x_2_22']]
    MSK_15=[['y_11_4', 'x_12_20', 'y_11_5', 'x_12_21', 'y_11_6', 'x_12_22', 'y_11_7', 'x_12_23', 'x_11_4', 'x_11_5', 'x_11_6', 'x_11_7', 'y_12_20', 'y_12_21', 'y_12_22', 'y_12_23'], ['y_2_4', 'x_3_20', 'y_2_6', 'x_3_22', 'y_3_20', 'y_3_32', 'x_4_40', 'y_3_21', 'y_3_33', 'x_4_41', 'y_3_22', 'y_3_34', 'x_4_42', 'y_3_23', 'y_3_35', 'x_4_43', 'y_4_40', 'y_4_41', 'y_4_42', 'y_4_43', 'x_3_32', 'x_3_33', 'x_3_34', 'x_3_35', 'x_3_21', 'x_3_23'], ['y_3_8', 'x_4_24', 'y_3_9', 'x_4_25', 'y_3_10', 'x_4_26', 'y_3_11', 'x_4_27', 'y_4_24', 'y_4_25', 'y_4_26', 'y_4_27', 'x_3_8', 'x_3_9', 'x_3_10', 'x_3_11']]
    MSK_LST = trans_lst(MSK_15)
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