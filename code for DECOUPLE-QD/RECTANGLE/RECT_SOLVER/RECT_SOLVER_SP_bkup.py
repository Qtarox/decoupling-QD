import re
import time
from ortools.sat.python import cp_model
from solution_plt import *
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict
import importlib
import argparse
import numpy as np
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════════
# RECTANGLE 参数
# ════════════════════════════════════════════════════════════════════
RECTANGLE_SBOX = [0x6, 0x5, 0xC, 0xA, 0x1, 0xE, 0x7, 0x9,
                  0xB, 0x0, 0x3, 0xD, 0x8, 0xF, 0x4, 0x2]

RECTANGLE_RC = [0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16,
                0x0C, 0x19, 0x13, 0x07, 0x0F, 0x1F, 0x1E, 0x1C,
                0x18, 0x11, 0x03, 0x06, 0x0D, 0x1B, 0x17, 0x0E, 0x1D]

my_sbox = RECTANGLE_SBOX


# ════════════════════════════════════════════════════════════════════
# RECTANGLE 密钥扩展（数值版，输入主密钥得到所有轮密钥位）
# ════════════════════════════════════════════════════════════════════

def key_schedule_80(mk_bits, nb_rounds):
    """
    mk_bits: 长度 80 的 0/1 列表（主密钥位）
    返回 round_keys[r][j] = 0/1，j = 4*col+row ∈ [0, 64)
    
    密钥状态 5×16，K[row][col]，mk[i] → K[i // 16][i % 16]
    """
    K = [[mk_bits[row * 16 + col] for col in range(16)] for row in range(5)]
    round_keys = []
    
    for r in range(nb_rounds):
        # 提取轮密钥（前4行），j = 4*col+row
        rk = [0] * 64
        for col in range(16):
            for row in range(4):
                rk[4 * col + row] = K[row][col]
        round_keys.append(rk)
        
        if r >= nb_rounds - 1:
            break
        
        # ── Step 1: SubColumn 在最右4列前4行 ──────────────────
        new_K = [row[:] for row in K]
        for col in range(4):
            nibble = (K[0][col] | (K[1][col] << 1) |
                      (K[2][col] << 2) | (K[3][col] << 3))
            out = RECTANGLE_SBOX[nibble]
            new_K[0][col] = (out >> 0) & 1
            new_K[1][col] = (out >> 1) & 1
            new_K[2][col] = (out >> 2) & 1
            new_K[3][col] = (out >> 3) & 1
        
        # ── Step 2: 行移位 ────────────────────────────────────
        shifts = [8, 0, 0, 0, 0]
        rotated = [[0]*16 for _ in range(5)]
        for row in range(5):
            sh = shifts[row]
            for col in range(16):
                rotated[row][col] = new_K[row][(col - sh) % 16]
        
        # ── Step 3: Feistel 重组 ───────────────────────────────
        K0_old = new_K[0][:]
        K = [[0]*16 for _ in range(5)]
        for col in range(16):
            K[0][col] = rotated[0][col] ^ new_K[1][col]
            K[1][col] = new_K[2][col]
            K[2][col] = new_K[3][col]
            K[3][col] = new_K[4][col]
            K[4][col] = rotated[4][col] ^ K0_old[col]
        
        # ── Step 4: 轮常数 XOR 到 K[0] 低5位 ───────────────────
        rc = RECTANGLE_RC[r]
        for i in range(5):
            K[0][i] ^= (rc >> i) & 1
    
    return round_keys


def key_schedule_128(mk_bits, nb_rounds):
    """
    mk_bits: 长度 128 的 0/1 列表
    返回 round_keys[r][j]，j = 4*col+row ∈ [0, 64)
    
    密钥状态 4×32，mk[i] → K[i // 32][i % 32]
    """
    K = [[mk_bits[row * 32 + col] for col in range(32)] for row in range(4)]
    round_keys = []
    
    for r in range(nb_rounds):
        rk = [0] * 64
        for col in range(16):
            for row in range(4):
                rk[4 * col + row] = K[row][col]
        round_keys.append(rk)
        
        if r >= nb_rounds - 1:
            break
        
        # SubColumn 在最右8列所有4行
        new_K = [row[:] for row in K]
        for col in range(8):
            nibble = (K[0][col] | (K[1][col] << 1) |
                      (K[2][col] << 2) | (K[3][col] << 3))
            out = RECTANGLE_SBOX[nibble]
            new_K[0][col] = (out >> 0) & 1
            new_K[1][col] = (out >> 1) & 1
            new_K[2][col] = (out >> 2) & 1
            new_K[3][col] = (out >> 3) & 1
        
        # 行移位
        shifts = [8, 16, 24, 0]
        K = [[0]*32 for _ in range(4)]
        for row in range(4):
            sh = shifts[row]
            for col in range(32):
                K[row][col] = new_K[row][(col - sh) % 32]
        
        # 轮常数
        rc = RECTANGLE_RC[r]
        for i in range(5):
            K[0][i] ^= (rc >> i) & 1
    
    return round_keys


# ════════════════════════════════════════════════════════════════════
# S盒元组预计算
# ════════════════════════════════════════════════════════════════════

def get_sbox_tuples(sbox):
    valid_tuples = []
    for i in range(16):
        out = sbox[i]
        valid_tuples.append(
            [(i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1,
             (out >> 0) & 1, (out >> 1) & 1, (out >> 2) & 1, (out >> 3) & 1]
        )
    return valid_tuples


# ════════════════════════════════════════════════════════════════════
# 求解器：给定轮密钥的实际值，求解状态变量的解数
# ════════════════════════════════════════════════════════════════════

class SolutionCounter(cp_model.CpSolverSolutionCallback):
    """只计数解的数量，不存储具体解"""
    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.count = 0
    
    def on_solution_callback(self):
        self.count += 1


def solve_for_fixed_key(input_text, fixed_vars, round_keys):
    """
    给定主密钥经密钥扩展后的轮密钥（数值），求解状态变量的解数量。
    
    input_text: 约束文本（含 k_r_j）
    fixed_vars: Z 字典（其他被强制固定的状态变量）
    round_keys: round_keys[r][j] = 0/1
    
    返回：状态变量的可行解数
    """
    model = cp_model.CpModel()
    var_dict = {}
    
    # ── 提取状态变量 ──────────────────────────────────────────
    state_vars = set(re.findall(r'[xy]_\d+_\d+', input_text))
    for v in fixed_vars.keys():
        if v.startswith('x_') or v.startswith('y_'):
            state_vars.add(v)
    
    for v in state_vars:
        var_dict[v] = model.NewBoolVar(v)
    
    sbox_valid_tuples = get_sbox_tuples(my_sbox)
    dummy_counter = 0
    
    # ── 解析约束 ──────────────────────────────────────────────
    for line in input_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        
        sbox_match = re.match(r'S(?:_\[([\d_]+)\])?\((.*?)\)\s*=\s*\((.*?)\)', line)
        
        if sbox_match:
            valid_x_str = sbox_match.group(1)
            in_vars = [var_dict[v.strip()] for v in sbox_match.group(2).split(',')]
            out_vars = [var_dict[v.strip()] for v in sbox_match.group(3).split(',')]
            
            if valid_x_str:
                allowed_x_vals = set(int(v) for v in valid_x_str.split('_'))
                subset_tuples = []
                for x in allowed_x_vals:
                    out = my_sbox[x]
                    subset_tuples.append([
                        (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1, (x >> 3) & 1,
                        (out >> 0) & 1, (out >> 1) & 1, (out >> 2) & 1, (out >> 3) & 1
                    ])
                model.AddAllowedAssignments(in_vars + out_vars, subset_tuples)
            else:
                model.AddAllowedAssignments(in_vars + out_vars, sbox_valid_tuples)
        else:
            # 线性方程：先解析右端常数
            rhs_const = 0
            line_clean = re.sub(r'[\[\]]', '', line)
            if '= 1' in line_clean:
                rhs_const = 1
                line_clean = line_clean.replace('= 1', '').strip()
            elif '= 0' in line_clean:
                line_clean = line_clean.replace('= 0', '').strip()
            
            # 找出所有变量（状态变量 + k 变量）
            all_tokens = re.findall(r'[xy]_\d+_\d+|k_\d+_\d+', line_clean)
            
            # ── 关键：把 k_r_j 替换为它的数值 ─────────────────
            constant_val = rhs_const  # 右端常数
            state_terms = []
            for tok in all_tokens:
                if tok.startswith('k_'):
                    # 从 round_keys 取出具体值
                    parts = tok.split('_')
                    r_k = int(parts[1])
                    j_k = int(parts[2])
                    if r_k < len(round_keys) and j_k < len(round_keys[r_k]):
                        k_val = round_keys[r_k][j_k]
                        # k 项移到右端 → 等价于把 k 值 XOR 到 constant_val
                        constant_val ^= k_val
                    # 若越界则忽略
                else:
                    state_terms.append(var_dict[tok])
            
            if state_terms:
                # sum(state_terms) ≡ constant_val (mod 2)
                dummy = model.NewIntVar(0, len(state_terms) // 2 + 1, f'd_{dummy_counter}')
                model.Add(sum(state_terms) + constant_val == 2 * dummy)
                dummy_counter += 1
            else:
                # 所有项都是 k（已变常数），约束变成 constant_val == 0
                # 如果不满足，模型无解
                if constant_val != 0:
                    # 添加一个永假约束
                    f = model.NewBoolVar('false_const')
                    model.Add(f == 1)
                    model.Add(f == 0)
    
    # ── 加固定变量约束 ─────────────────────────────────────────
    for var, vals in fixed_vars.items():
        if var in var_dict:
            val = list(vals)[0]
            model.Add(var_dict[var] == val)
    
    # ── 求解，枚举状态变量的解 ─────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.log_search_progress = False
    
    counter = SolutionCounter()
    solver.Solve(model, counter)
    
    return counter.count


# ════════════════════════════════════════════════════════════════════
# 蒙特卡洛主流程
# ════════════════════════════════════════════════════════════════════

def monte_carlo_distribution(dic_cons, key_size, nb_rounds, num_samples,
                             res_name_prefix, seed=42):
    """
    对每个 cluster：
      1. 采样 N 个主密钥
      2. 对每个主密钥跑密钥扩展得到轮密钥
      3. 把轮密钥的实际值喂给求解器，求解状态变量解数
      4. 统计每个主密钥的解数 → 概率分布
    
    返回：cluster_dists[cluster_id] = [解数1, 解数2, ..., 解数N]
    """
    rng = np.random.default_rng(seed)
    
    if key_size == 80:
        ks_func = key_schedule_80
    elif key_size == 128:
        ks_func = key_schedule_128
    else:
        raise ValueError(f"Unsupported key size: {key_size}")
    
    # 预采样所有主密钥（保证不同 cluster 用相同的 key 集合）
    master_keys = rng.integers(0, 2, size=(num_samples, key_size)).tolist()
    
    # 预计算所有主密钥对应的轮密钥（一次性，避免重复）
    print(f"\n预计算 {num_samples} 个主密钥的密钥扩展...")
    all_round_keys = []
    for mk in tqdm(master_keys, desc="Key schedule"):
        all_round_keys.append(ks_func(mk, nb_rounds))
    
    # 对每个 cluster 求解
    cluster_dists = {}
    cluster_summaries = {}
    
    for cluster_name in sorted(dic_cons.keys()):
        cons_text, z_dict = dic_cons[cluster_name]
        print(f"\n========== {cluster_name} ==========")
        
        sol_counts = []
        t0 = time.time()
        
        for sample_idx in tqdm(range(num_samples),
                               desc=f"Sampling {cluster_name}"):
            rk = all_round_keys[sample_idx]
            cnt = solve_for_fixed_key(cons_text, z_dict, rk)
            sol_counts.append(cnt)
        
        elapsed = time.time() - t0
        sol_counts = np.array(sol_counts)
        
        # 写文件
        res_file = f"{res_name_prefix}_{cluster_name}.txt"
        with open(res_file, "w") as fw:
            fw.write(f"Cluster: {cluster_name}\n")
            fw.write(f"Master key size: {key_size}\n")
            fw.write(f"Number of samples: {num_samples}\n")
            fw.write(f"Time: {elapsed:.2f}s\n")
            fw.write(f"Mean # solutions: {sol_counts.mean():.4f}\n")
            fw.write(f"Std # solutions: {sol_counts.std():.4f}\n")
            fw.write(f"Min: {sol_counts.min()}, Max: {sol_counts.max()}\n")
            fw.write(f"Distinct values: {len(np.unique(sol_counts))}\n")
            fw.write("-" * 30 + "\n")
            # 频次直方图
            unique, counts = np.unique(sol_counts, return_counts=True)
            fw.write("# solutions : # master keys with this count\n")
            for u, c in zip(unique, counts):
                fw.write(f"{u}: {c}\n")
        
        # 简短统计
        unique, counts = np.unique(sol_counts, return_counts=True)
        hist = dict(zip(unique.tolist(), counts.tolist()))
        cluster_summaries[cluster_name] = hist
        cluster_dists[cluster_name] = sol_counts
        
        print(f"耗时: {elapsed:.2f}s, 平均解数: {sol_counts.mean():.2f}")
        print(f"分布直方图（解数: 主密钥个数）: {hist}")
    
    return cluster_dists, cluster_summaries


# ════════════════════════════════════════════════════════════════════
# 主程序
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RECTANGLE SAT Solver (Monte Carlo)")
    parser.add_argument('-m', '--module', type=str, default='CONS.cons_14R',
                        help='约束模块名')
    parser.add_argument('-k', '--keysize', type=int, default=80, choices=[80, 128],
                        help='主密钥长度')
    parser.add_argument('-r', '--rounds', type=int, required=True,
                        help='轮数（用于密钥扩展）')
    parser.add_argument('-N', '--samples', type=int, default=2000,
                        help='采样的主密钥数量')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()
    
    print(f"加载模块: {args.module}")
    print(f"主密钥长度: {args.keysize}-bit")
    print(f"密钥扩展轮数: {args.rounds}")
    print(f"采样数量: {args.samples}")
    
    cons_module = importlib.import_module(args.module)
    dic_cons = getattr(cons_module, 'dic_cons')
    str_n = cons_module.__name__.split('.')[-1]
    
    cluster_dists, cluster_summaries = monte_carlo_distribution(
        dic_cons, args.keysize, args.rounds, args.samples,
        res_name_prefix=f"mc_results_rect{args.keysize}_{str_n}",
        seed=args.seed,
    )
    
    # 保存所有解数到 npy 便于后续可视化
    npy_path = f"mc_dists_rect{args.keysize}_{str_n}.npz"
    np.savez(npy_path, **{k: v for k, v in cluster_dists.items()})
    print(f"\n所有分布数据已保存到 {npy_path}")
    
    # 简短汇总
    RES_DIC={}
    dic_lst=[]
    print("\n========== 各 Cluster 汇总 ==========")
    for cname, hist in cluster_summaries.items():
        print(f"{cname}: {hist}")
        RES_DIC[cname]=hist
        dic_lst.append(cname)

    #plot code
    normalized_dic_lst=[dic_normalize(RES_DIC[key]) for key in dic_lst]
    print(normalized_dic_lst)
    distributions = normalized_dic_lst
    final_dist = defaultdict(int)
    final_dist[1.0] = 1  # 初始状态：乘积为1，组合数为1

    for current_dict in distributions:
        new_dist = defaultdict(int)
        # 将当前已累积的结果，与下一个字典进行交叉相乘
        for current_val, current_count in final_dist.items():
            for val, count in current_dict.items():
                new_dist[current_val * val] += current_count * count
        final_dist = new_dist

    # 2. 准备画图数据
    # 将最终的不同乘积结果从小到大排序
    sorted_items = sorted(final_dist.items())

    x_indices = []
    y_values = []
    current_index = 0

    # 为了在图上表现出 5500 亿个点排布的效果，我们通过记录区间的起点和终点来画图
    for val, count in sorted_items:
        x_indices.append(current_index)      # 区间起点
        y_values.append(val)
        current_index += count               # 加上这种乘积出现的次数
        x_indices.append(current_index - 1)  # 区间终点
        y_values.append(val)

    print(f"总共有 {len(sorted_items)} 种不同的乘积结果！")
    print(f"总共计算了 {current_index} 种组合！")

    # 3. 画图 (由于 X 轴刻度极大，matplotlib 会自动使用科学计数法)
    plt.figure(figsize=(10, 6))
    plt.plot(x_indices, y_values, linewidth=2)
    plt.title("Sorted Products of Variable Combinations (Optimized)")
    plt.xlabel("Combination Index")
    plt.ylabel("Product Value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig("./distribut_prob_solution.png", dpi=300)
    plt.show()