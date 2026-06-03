from gurobipy import *
from utils import *
from tqdm import tqdm
import time
import numpy as np
from math import log2
from key_master_rectangle import build_round_keys_80, build_round_keys_128, extract_master_key_conditions
from diffs import *
from gurobipy import *
from utils import *
from tqdm import tqdm
import time
import numpy as np
from math import log2
from collections import Counter
import itertools
# ─── 选择密钥版本 ─────────────────────────────────────────────────
KEY_SIZE = 80    # 设为 80 或 128

# ─── 你的 diffs 列表（保持不变）──────────────────────────────────


diffs= diffs14_2
# ─── 预加载 ──────────────────────────────────────────────────────
diff_trail        = extract_diff_trail_from_list(diffs)
blocks            = get_transitions_from_list(diffs)
QDTM_RECT         = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)
sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)

# 构建轮密钥的符号表示（密钥扩展线性追踪）
if KEY_SIZE == 80:
    round_keys_sym, nl_log = build_round_keys_80(NB_ROUNDS)
elif KEY_SIZE == 128:
    round_keys_sym, nl_log = build_round_keys_128(NB_ROUNDS)
else:
    raise ValueError("KEY_SIZE must be 80 or 128")

print(f"Built round keys for RECTANGLE-{KEY_SIZE}, NL events in key schedule: {len(nl_log)}")


# ─── compute_correlation：保持你之前给出的逻辑 ────────────────────
def compute_correlation(diff_trail, mask_trail, nb_rounds):
    corr = 1
    # S盒贡献
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
    # 轮常数
    for k in range(nb_rounds):
        corr *= rect_rc_corr_factor(mask_trail[k][0], RECTANGLE_RC[k])
    
    if corr > 0:
        return  1, log2( corr)
    elif corr < 0:
        return -1, log2(-corr)
    return 1, 0

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


# ─── 主MILP函数 ──────────────────────────────────────────────────
def RECTANGLE_MILP_Quasi_Diff(nb_rounds,mask_range,T=150):
    model = Model("RECTANGLE_Quasi_Diff_MILP")
    u = model.addVars(nb_rounds, 2, 64, vtype=GRB.BINARY, name="m")
    Q = model.addVars(nb_rounds, NB_COLS, CORR_RANGE, vtype=GRB.BINARY, name="c")

    model.addConstrs(u[0,          0, j] == 0 for j in range(64))
    model.addConstrs(u[nb_rounds-1, 1, j] == 0 for j in range(64))
    allowed_masks = set(mask_range)
    print(f"Applying mask range constraints (Allowed bits: {len(allowed_masks)})...")
    
    for r in range(nb_rounds):
        for before in range(2):
            for l in range(64):
                if (r, before, l) not in allowed_masks:
                    model.addConstr(u[r, before,  l] == 0,
                                    name=f"mask_limit_{r}_{before}_{l}")
    
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
    model.addConstr(total_corr >= -T)
    model.setObjective(total_corr, GRB.MAXIMIZE)

    model.params.PoolSearchMode = 2
    model.params.PoolSolutions  = 2000000

    t1 = time.time()
    model.optimize()
    print(f"\nTime used: {time.time()-t1:.2f}s")
    print(f"Found {model.SolCount} trails")

    # ── 提取每条 trail 并构建 T 列表 ─────────────────────────────
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
        
        # 提取主密钥条件
        keys_list = extract_master_key_conditions(mask_trail, round_keys_sym, nb_rounds)
        
        T.append({
            'sign': sign,
            'corr': corr_log,
            'keys': keys_list,
        })

    if(len(T)<10):
        print("DITRIBUTION: ",T)
    
    # ── 计算该 cluster 的分布 ─────────────────────────────────
    if avg_prob is None:
        return model.SolCount, {}
    
    distribution = compute_distribution_from_trails(T, avg_prob)
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

if __name__ == "__main__":
    MSK_LST2 = [[(3, 1, 0), (4, 0, 0), (4, 1, 3), (5, 0, 55)], [(7, 1, 19), (8, 0, 7), (8, 0, 20), (8, 0, 22), (8, 0, 23), (8, 1, 4), (8, 1, 5), (8, 1, 20), (8, 1, 21), (8, 1, 22), (8, 1, 23), (9, 0, 4), (9, 0, 6), (9, 0, 7), (9, 0, 9), (9, 0, 11), (9, 0, 24), (9, 0, 25), (9, 0, 26), (9, 0, 27), (9, 1, 5), (9, 1, 8), (9, 1, 9), (9, 1, 11), (9, 1, 24), (9, 1, 25), (9, 1, 26), (9, 1, 27), (10, 0, 8), (10, 0, 9), (10, 0, 10), (10, 0, 11), (10, 0, 12), (10, 0, 13), (10, 0, 14), (10, 0, 15), (10, 0, 63), (10, 1, 9), (10, 1, 11), (10, 1, 12), (10, 1, 13), (10, 1, 14), (10, 1, 15), (10, 1, 60), (10, 1, 61), (11, 0, 0), (11, 0, 1), (11, 0, 2), (11, 0, 3), (11, 0, 12), (11, 0, 13), (11, 0, 14), (11, 0, 15), (11, 0, 60), (11, 0, 62), (11, 0, 63), (11, 1, 0), (11, 1, 1), (11, 1, 2), (11, 1, 3), (11, 1, 12), (11, 1, 13), (11, 1, 14), (11, 1, 15), (11, 1, 60), (11, 1, 61), (12, 0, 0), (12, 0, 1), (12, 0, 2), (12, 0, 3), (12, 0, 60), (12, 0, 62), (12, 1, 1), (12, 1, 2), (12, 1, 3), (12, 1, 61), (13, 0, 1)], [(8, 1, 20), (9, 0, 20), (9, 1, 23), (10, 0, 11)], [(7, 1, 16), (8, 0, 16), (8, 1, 19), (9, 0, 7)], [(9, 1, 24), (10, 0, 24), (10, 1, 27), (11, 0, 15)], [(0, 1, 52), (1, 0, 52), (1, 1, 55), (2, 0, 43)], [(1, 1, 56), (2, 0, 56), (2, 1, 59), (3, 0, 47)], [(2, 1, 60), (3, 0, 60), (3, 1, 63), (4, 0, 51)], [(4, 1, 4), (5, 0, 4), (5, 1, 7), (6, 0, 59)], [(5, 1, 8), (6, 0, 8), (6, 1, 11), (7, 0, 63)], [(6, 1, 12), (7, 0, 12), (7, 1, 15), (8, 0, 3)]]    # MSK_LST = MSK_12_60
    MSK_LST1 =[[(3, 1, 0), (4, 0, 0), (4, 1, 3), (5, 0, 55)], [(6, 1, 15), (7, 0, 3), (7, 1, 0), (7, 1, 1), (7, 1, 19), (8, 0, 0), (8, 0, 5), (8, 0, 7), (8, 1, 4), (8, 1, 5), (8, 1, 23), (9, 0, 4), (9, 0, 9), (9, 0, 11), (9, 1, 8), (9, 1, 9), (10, 0, 8), (10, 0, 13)], [(8, 0, 20), (8, 0, 22), (8, 0, 23), (8, 1, 20), (8, 1, 21), (8, 1, 22), (8, 1, 23), (9, 0, 4), (9, 0, 6), (9, 0, 7), (9, 0, 24), (9, 0, 25), (9, 0, 26), (9, 0, 27), (9, 1, 5), (9, 1, 24), (9, 1, 25), (9, 1, 26), (9, 1, 27), (10, 0, 8), (10, 0, 9), (10, 0, 10), (10, 0, 11), (10, 0, 12), (10, 0, 13), (10, 0, 14), (10, 0, 15), (10, 1, 9), (10, 1, 11), (10, 1, 12), (10, 1, 13), (10, 1, 14), (10, 1, 15), (11, 0, 12), (11, 0, 13), (11, 0, 14), (11, 0, 15), (11, 0, 62), (11, 0, 63), (11, 1, 12), (11, 1, 13), (11, 1, 14), (11, 1, 60), (11, 1, 61), (12, 0, 1), (12, 0, 60), (12, 0, 62), (12, 1, 61), (13, 0, 1)], [(8, 1, 20), (9, 0, 20), (9, 1, 23), (10, 0, 11)], [(7, 1, 16), (8, 0, 16), (8, 1, 19), (9, 0, 7)], [(9, 1, 24), (10, 0, 24), (10, 1, 27), (11, 0, 15)], [(0, 1, 52), (1, 0, 52), (1, 1, 55), (2, 0, 43)], [(1, 1, 56), (2, 0, 56), (2, 1, 59), (3, 0, 47)], [(2, 1, 60), (3, 0, 60), (3, 1, 63), (4, 0, 51)], [(4, 1, 4), (5, 0, 4), (5, 1, 7), (6, 0, 59)], [(5, 1, 8), (6, 0, 8), (6, 1, 11), (7, 0, 63)], [(6, 1, 12), (7, 0, 12), (7, 1, 15), (8, 0, 3)]]
    # file_prefix = f"./constraints/CONS_{NB_ROUNDS}R_{MIN_CORR}{NAME}_TH{THRES}"
    # file_py=mask_py_file = f"{file_prefix}_FULL_MSK.py"
    solu = []
    distributions = []
    MSK_LST=MSK_LST1
    for i in range(len(MSK_LST)):
        print(f"\n========== Cluster {i} ==========")
        nb_sol, dist = RECTANGLE_MILP_Quasi_Diff(NB_ROUNDS, MSK_LST[i])
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