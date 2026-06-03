"""
SKINNY-128 Monte-Carlo joint distribution solver
==================================================
对应 MASK_DIVIDER_{SK,TK1,TK2} 输出的 cluster 约束格式。

与 GIFT-64 版本的核心差别：
  - cell 8-bit、state 128-bit、半态 cell 编号 0..15
  - S-box 是 SKINNY-128 的 256 项表
  - AddConstants 注入到 cell 0 / cell 4 / cell 8 的特定 bit
  - 主密钥维度按 ADV_MODEL 切换：
        SK   -> 0
        TK1  -> 128  (变量名 k_0..k_127)
        TK2  -> 256  (变量名 k1_0..k1_127, k2_0..k2_127)

线性方程示例（MASK_DIVIDER 输出格式）：
     + [x_3_5] + y_4_12 + k_17 = 0
     + [y_2_8] + k1_5 + k2_77 = 0          (TK2 时可能两层都出现)
S-box 行示例：
     S(x_3_8,...,x_3_15) = (y_3_8,...,y_3_15)
     S_[12_34_...](x_3_8,...) = (y_3_8,...)   (允许的输入子集)
"""

import re
import time
import argparse
import importlib
import numpy as np
from tqdm import tqdm
from ortools.sat.python import cp_model

# 复用绘图工具（与 GIFT 版本相同）
try:
    from solution_plt import dic_normalize
except ImportError:
    # 退化实现：用最大公约数归一化频次
    from math import gcd
    from functools import reduce
    def dic_normalize(d):
        if not d:
            return {}
        g = reduce(gcd, d.values())
        return {k: v // g for k, v in d.items()}

import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════════════
# 1) SKINNY-128 常量：S-box / round constants
# ════════════════════════════════════════════════════════════════════
SKINNY128_SBOX = [
    0x65, 0x4c, 0x6a, 0x42, 0x4b, 0x63, 0x43, 0x6b, 0x55, 0x75, 0x5a, 0x7a, 0x53, 0x73, 0x5b, 0x7b,
    0x35, 0x8c, 0x3a, 0x81, 0x89, 0x33, 0x80, 0x3b, 0x95, 0x25, 0x98, 0x2a, 0x90, 0x23, 0x99, 0x2b,
    0xe5, 0xcc, 0xe8, 0xc1, 0xc9, 0xe0, 0xc0, 0xe9, 0xd5, 0xf5, 0xd8, 0xf8, 0xd0, 0xf0, 0xd9, 0xf9,
    0xa5, 0x1c, 0xa8, 0x12, 0x1b, 0xa0, 0x13, 0xa9, 0x05, 0xb5, 0x0a, 0xb8, 0x03, 0xb0, 0x0b, 0xb9,
    0x32, 0x88, 0x3c, 0x85, 0x8d, 0x34, 0x84, 0x3d, 0x91, 0x22, 0x9c, 0x2c, 0x94, 0x24, 0x9d, 0x2d,
    0x62, 0x4a, 0x6c, 0x45, 0x4d, 0x64, 0x44, 0x6d, 0x52, 0x72, 0x5c, 0x7c, 0x54, 0x74, 0x5d, 0x7d,
    0xa1, 0x1a, 0xac, 0x15, 0x1d, 0xa4, 0x14, 0xad, 0x02, 0xb1, 0x0c, 0xbc, 0x04, 0xb4, 0x0d, 0xbd,
    0xe1, 0xc8, 0xec, 0xc5, 0xcd, 0xe4, 0xc4, 0xed, 0xd1, 0xf1, 0xdc, 0xfc, 0xd4, 0xf4, 0xdd, 0xfd,
    0x36, 0x8e, 0x38, 0x82, 0x8b, 0x30, 0x83, 0x39, 0x96, 0x26, 0x9a, 0x28, 0x93, 0x20, 0x9b, 0x29,
    0x66, 0x4e, 0x68, 0x41, 0x49, 0x60, 0x40, 0x69, 0x56, 0x76, 0x58, 0x78, 0x50, 0x70, 0x59, 0x79,
    0xa6, 0x1e, 0xaa, 0x11, 0x19, 0xa3, 0x10, 0xab, 0x06, 0xb6, 0x08, 0xba, 0x00, 0xb3, 0x09, 0xbb,
    0xe6, 0xce, 0xea, 0xc2, 0xcb, 0xe3, 0xc3, 0xeb, 0xd6, 0xf6, 0xda, 0xfa, 0xd3, 0xf3, 0xdb, 0xfb,
    0x31, 0x8a, 0x3e, 0x86, 0x8f, 0x37, 0x87, 0x3f, 0x92, 0x21, 0x9e, 0x2e, 0x97, 0x27, 0x9f, 0x2f,
    0x61, 0x48, 0x6e, 0x46, 0x4f, 0x67, 0x47, 0x6f, 0x51, 0x71, 0x5e, 0x7e, 0x57, 0x77, 0x5f, 0x7f,
    0xa2, 0x18, 0xae, 0x16, 0x1f, 0xa7, 0x17, 0xaf, 0x01, 0xb2, 0x0e, 0xbe, 0x07, 0xb7, 0x0f, 0xbf,
    0xe2, 0xca, 0xee, 0xc6, 0xcf, 0xe7, 0xc7, 0xef, 0xd2, 0xf2, 0xde, 0xfe, 0xd7, 0xf7, 0xdf, 0xff,
]
assert sorted(SKINNY128_SBOX) == list(range(256))

CELL_SIZE = 8
HALF_STATE_BITS = 16 * CELL_SIZE         # 128
CELLS_PER_HALF = 16

# 6-bit LFSR 用于 round constant：rc_{t+1} = (rc_t << 1) | (rc5 XOR rc4 XOR 1)
# 与论文 / extract_lfsr_states 函数一致。这里直接列出前若干轮的 LFSR 状态
# (rc5, rc4, rc3, rc2, rc1, rc0)，按位整数表示，即 rc5*32 + rc4*16 + ... + rc0
def _compute_lfsr_states(n_rounds):
    state = 0  # 6-bit value
    out = []
    for _ in range(n_rounds):
        rc0 = (state >> 5) & 1
        rc1 = (state >> 4) & 1
        new_lsb = rc0 ^ rc1 ^ 1
        state = ((state << 1) & 0x3F) | new_lsb
        out.append(state)
    return out

# 最多支持 64 轮的 LFSR 状态（够用）
_LFSR_STATES = _compute_lfsr_states(64)


def get_skinny_constant(var_name):
    """
    解析 x_r_j 形式的状态变量，返回该位上自带的 SKINNY-128 round constant 贡献。

    SKINNY 论文中 AddConstants 把第 r 轮 LFSR 的 6 bit 注入到 3 个 cell：
        cell 0  (row 0, col 0):  低 4 bit 异或 (rc3, rc2, rc1, rc0)
        cell 4  (row 1, col 0):  低 2 bit 异或 (rc5, rc4)
        cell 8  (row 2, col 0):  整 cell 异或 0x02

    注意：MASK_DIVIDER 的全局位编码是 `x_r_b` 其中 b ∈ [0, 128)，
    cell c 的 8 个 bit 位于 [c*8, c*8+8)，bit 内部 LSB-first。
    AddConstants 在 SubCells **之后**作用于 x 半态（论文里 AC 加在 SC 之后），
    所以这里只对 `x_r_*` 起效，y 不动。
    """
    if not var_name.startswith('x_'):
        return 0
    parts = var_name.split('_')
    r = int(parts[1])
    bit = int(parts[2])

    if r <= 0 or r > len(_LFSR_STATES):
        return 0
    lfsr = _LFSR_STATES[r - 1]   # 第 r 轮使用第 r-1 次更新后的状态

    rc0 = (lfsr >> 0) & 1
    rc1 = (lfsr >> 1) & 1
    rc2 = (lfsr >> 2) & 1
    rc3 = (lfsr >> 3) & 1
    rc4 = (lfsr >> 4) & 1
    rc5 = (lfsr >> 5) & 1

    cell = bit // CELL_SIZE
    bit_in_cell = bit % CELL_SIZE

    # cell 0 低 4 bit
    if cell == 0:
        if bit_in_cell == 0: return rc0
        if bit_in_cell == 1: return rc1
        if bit_in_cell == 2: return rc2
        if bit_in_cell == 3: return rc3
        return 0
    # cell 4 低 2 bit
    if cell == 4:
        if bit_in_cell == 0: return rc4
        if bit_in_cell == 1: return rc5
        return 0
    # cell 8 整 cell ^= 0x02 (LSB-first: bit 1 是 1)
    if cell == 8:
        return 1 if bit_in_cell == 1 else 0

    return 0


# ════════════════════════════════════════════════════════════════════
# 2) S-box tuple 预计算（16 元 0/1 向量列表）
# ════════════════════════════════════════════════════════════════════
def get_sbox_tuples(sbox):
    """全部 256 个合法 (in, out) 比特组合，每个 16 长度（8 in + 8 out）。"""
    valid_tuples = []
    n = CELL_SIZE
    for i in range(1 << n):
        out = sbox[i]
        tup = [(i >> b) & 1 for b in range(n)] + [(out >> b) & 1 for b in range(n)]
        valid_tuples.append(tup)
    return valid_tuples


_SBOX_TUPLES_CACHE = None
def _sbox_tuples():
    global _SBOX_TUPLES_CACHE
    if _SBOX_TUPLES_CACHE is None:
        _SBOX_TUPLES_CACHE = get_sbox_tuples(SKINNY128_SBOX)
    return _SBOX_TUPLES_CACHE


def _sbox_subset_tuples(allowed_inputs):
    """指定的输入集合 -> 对应 tuple 子表。"""
    n = CELL_SIZE
    out = []
    for x in allowed_inputs:
        y = SKINNY128_SBOX[x]
        out.append([(x >> b) & 1 for b in range(n)] + [(y >> b) & 1 for b in range(n)])
    return out


# ════════════════════════════════════════════════════════════════════
# 3) CP-SAT 求解：单个 (cluster, fixed key) -> 状态变量解数
# ════════════════════════════════════════════════════════════════════
class SolutionCounter(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.count = 0
    def on_solution_callback(self):
        self.count += 1


# 匹配的 token：状态变量 x_r_b / y_r_b，以及 key 变量 k_i / k1_i / k2_i
_TOKEN_RE = re.compile(r'[xy]_\d+_\d+|k[12]?_\d+')


def _key_value(token, key_values):
    """token 形如 k_5 / k1_5 / k2_5；查 key_values 字典。"""
    if token in key_values:
        return key_values[token]
    return 0


def solve_for_fixed_key(input_text, fixed_vars, key_values):
    """
    给定主密钥取值 key_values（dict: 'k_5' -> 0/1, 或 'k1_5'/'k2_5' -> 0/1），
    求该 cluster 在该密钥下状态变量的可行解数。

    key_values 是一个 dict 而不是 list，原因：TK2 有两层（k1_/k2_），
    用 dict 比"按下标分段"更稳健；SK/TK1 也兼容（只用 k_*）。
    """
    model = cp_model.CpModel()
    var_dict = {}

    # 收集所有出现过的状态变量（包含 fixed_vars 中的）
    state_vars = set(re.findall(r'[xy]_\d+_\d+', input_text))
    for v in fixed_vars.keys():
        if v.startswith('x_') or v.startswith('y_'):
            state_vars.add(v)
    for v in state_vars:
        var_dict[v] = model.NewBoolVar(v)

    full_sbox_tuples = _sbox_tuples()
    dummy_counter = 0

    # ── 解析每条约束 ──
    sbox_pat = re.compile(r'S(?:_\[([\d_]+)\])?\((.*?)\)\s*=\s*\((.*?)\)')

    for line in input_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        m = sbox_pat.match(line)
        if m:
            # S-box 行
            subset_str = m.group(1)
            in_vars  = [var_dict[v.strip()] for v in m.group(2).split(',')]
            out_vars = [var_dict[v.strip()] for v in m.group(3).split(',')]
            assert len(in_vars) == CELL_SIZE and len(out_vars) == CELL_SIZE, \
                f"S-box arity 不对（in={len(in_vars)}, out={len(out_vars)}）：{line}"

            if subset_str:
                allowed = [int(v) for v in subset_str.split('_')]
                tuples = _sbox_subset_tuples(allowed)
            else:
                tuples = full_sbox_tuples
            model.AddAllowedAssignments(in_vars + out_vars, tuples)
            continue

        # 线性方程：把 [..] 去掉，再处理 RHS
        line_clean = re.sub(r'[\[\]]', '', line)
        rhs = 0
        if '= 1' in line_clean:
            rhs = 1
            line_clean = line_clean.replace('= 1', '').strip()
        elif '= 0' in line_clean:
            line_clean = line_clean.replace('= 0', '').strip()

        tokens = _TOKEN_RE.findall(line_clean)
        constant_val = rhs
        state_terms = []
        for tok in tokens:
            if tok.startswith('k'):
                constant_val ^= _key_value(tok, key_values)
            else:
                # x_r_b / y_r_b
                constant_val ^= get_skinny_constant(tok)
                state_terms.append(var_dict[tok])

        if state_terms:
            # sum(state_terms) ≡ constant_val (mod 2)
            dummy = model.NewIntVar(0, len(state_terms) // 2 + 1, f'd_{dummy_counter}')
            model.Add(sum(state_terms) + constant_val == 2 * dummy)
            dummy_counter += 1
        else:
            if constant_val != 0:
                # 死方程：使 model infeasible
                f = model.NewBoolVar('false_const')
                model.Add(f == 1)
                model.Add(f == 0)

    # ── 固定 active 状态比特 ──
    for var, vals in fixed_vars.items():
        if var in var_dict:
            val = list(vals)[0]
            model.Add(var_dict[var] == val)

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.log_search_progress = False

    counter = SolutionCounter()
    solver.Solve(model, counter)
    return counter.count


# ════════════════════════════════════════════════════════════════════
# 4) 蒙特卡洛主流程
# ════════════════════════════════════════════════════════════════════
# def _scan_key_namespace(dic_cons):
#     """
#     扫描所有 cluster 文本，记录出现过的 key 变量及其最大下标。
#     返回 dict：'k' -> max_idx, 'k1' -> max_idx, 'k2' -> max_idx （缺则不存在）。
#     """
#     seen = {}   # prefix ('k'|'k1'|'k2') -> max idx
#     pat = re.compile(r'\b(k[12]?)_(\d+)\b')
#     for _, (text, _) in dic_cons.items():
#         for m in pat.finditer(text):
#             prefix, idx = m.group(1), int(m.group(2))
#             seen[prefix] = max(seen.get(prefix, -1), idx)
#     return seen

def _scan_key_namespace(dic_cons):
    """
    扫描所有 cluster 文本，记录所有【真正出现过】的 key 变量名。
    返回一个包含所有活跃密钥变量名的列表，例如: ['k_11', 'k_15', 'k_111', ...]
    """
    active_keys = set()
    pat = re.compile(r'\b(k[12]?_\d+)\b')  # 直接抓取整个变量名
    for _, (text, _) in dic_cons.items():
        for m in pat.finditer(text):
            active_keys.add(m.group(1))
    
    # 按照前缀和数字大小排序，保证每次输出稳定
    return sorted(list(active_keys), key=lambda x: (x.split('_')[0], int(x.split('_')[1])))

# def _sample_master_key(rng, key_spaces):
#     """
#     对每个 key prefix 采样足够长的 0/1 数组，返回一个 dict 'k_5' -> 0/1。
#     key_spaces 例： {'k': 127, 'k1': 127, 'k2': 127}
#     """
#     d = {}
#     for prefix, max_idx in key_spaces.items():
#         bits = rng.integers(0, 2, size=max_idx + 1)
#         for i, b in enumerate(bits):
#             d[f"{prefix}_{i}"] = int(b)
#     return d

def _sample_master_key(rng, active_keys):
    """
    只为真正出现的密钥变量进行 0/1 随机采样。
    """
    d = {}
    bits = rng.integers(0, 2, size=len(active_keys))
    for key_name, b in zip(active_keys, bits):
        d[key_name] = int(b)
    return d
def monte_carlo_distribution(dic_cons, num_samples, res_name_prefix, seed=42):
    rng = np.random.default_rng(seed)

    # key_spaces = _scan_key_namespace(dic_cons)
    # if not key_spaces:
    #     print("[info] cluster 里不含任何 k_* / k1_* / k2_*（SK 模式）")
    #     key_dim_total = 0
    # else:
    #     key_dim_total = sum(v + 1 for v in key_spaces.values())
    # print(f"主密钥维度: {key_dim_total} bits（按 prefix: {key_spaces}）")

    # cluster_names = sorted(dic_cons.keys(),
    #                        key=lambda x: int(re.search(r'\d+', x).group()))
    active_keys = _scan_key_namespace(dic_cons)
    if not active_keys:
        print("[info] cluster 里不含任何 k_* / k1_* / k2_*（SK 模式）")
        key_dim_total = 0
    else:
        key_dim_total = len(active_keys)
        
    print(f"主密钥维度: {key_dim_total} bits")
    print(f"实际采样的密钥: {active_keys}")
    # === 替换到这里结束 ===

    cluster_names = sorted(dic_cons.keys(),
                           key=lambda x: int(re.search(r'\d+', x).group()))
    n_clusters = len(cluster_names)

    per_sample_cluster_counts = np.zeros((num_samples, n_clusters), dtype=np.int64)
    per_sample_total_counts   = np.zeros(num_samples, dtype=np.float64)

    # 记录每次采样的 master key（仅当存在 key 变量时）
    master_keys_record = []

    t_start = time.time()
    for sample_idx in tqdm(range(num_samples), desc="MC samples"):
        k_vals = _sample_master_key(rng, active_keys) if active_keys else {}
        if active_keys:
            master_keys_record.append(
                {k: v for k, v in sorted(k_vals.items())}
            )

        total = 1
        is_zero = False
        for c_idx, cname in enumerate(cluster_names):
            cons_text, z_dict = dic_cons[cname]
            cnt = solve_for_fixed_key(cons_text, z_dict, k_vals)
            per_sample_cluster_counts[sample_idx, c_idx] = cnt

            if cnt == 0:
                is_zero = True
                total = 0
            elif not is_zero:
                total *= cnt

        per_sample_total_counts[sample_idx] = total

    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed:.2f}s ({elapsed / max(num_samples,1):.3f}s/sample)")

    # 边际分布
    cluster_marginal = {}
    for c_idx, cname in enumerate(cluster_names):
        counts = per_sample_cluster_counts[:, c_idx]
        unique, freq = np.unique(counts, return_counts=True)
        cluster_marginal[cname] = dict(zip(unique.tolist(), freq.tolist()))

    unique_total, freq_total = np.unique(per_sample_total_counts, return_counts=True)
    joint_dist = dict(zip(unique_total.tolist(), freq_total.tolist()))

    summary_file = f"{res_name_prefix}_summary.txt"
    with open(summary_file, "w") as fw:
        fw.write("SKINNY-128 MC Joint Distribution\n")
        fw.write(f"Master key size: {key_dim_total}\n")
        fw.write(f"Key prefixes / max idx: {active_keys}\n")
        fw.write(f"Number of samples: {num_samples}\n")
        fw.write(f"Total time: {elapsed:.2f}s\n")
        fw.write("=" * 60 + "\n\n")

        fw.write("# Per-cluster MARGINAL distribution (诊断用)\n")
        for cname, hist in cluster_marginal.items():
            fw.write(f"{cname}: {hist}\n")
        fw.write("\n")

        fw.write("# JOINT distribution (真实的密钥相关分布)\n")
        fw.write(f"Mean: {per_sample_total_counts.mean():.4f}\n")
        fw.write(f"Std:  {per_sample_total_counts.std():.4f}\n")
        fw.write(f"Min: {per_sample_total_counts.min()}, "
                 f"Max: {per_sample_total_counts.max()}\n")
        fw.write(f"# Master keys with 0 total solutions: "
                 f"{int((per_sample_total_counts == 0).sum())}\n")
        fw.write(f"# Distinct total values: {len(unique_total)}\n\n")
        fw.write("Total solutions : # master keys\n")
        for u, c in zip(unique_total, freq_total):
            fw.write(f"{u:.0f}: {c}\n")
    print(f"汇总写入 {summary_file}")

    return {
        'per_sample_cluster_counts': per_sample_cluster_counts,
        'per_sample_total_counts': per_sample_total_counts,
        'cluster_marginal': cluster_marginal,
        'joint_distribution': joint_dist,
        'cluster_names': cluster_names,
        'master_keys': master_keys_record,
        'key_spaces': active_keys,
    }


# ════════════════════════════════════════════════════════════════════
# 5) CLI
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SKINNY-128 MC Solver")
    parser.add_argument('-m', '--module', type=str, default='CONS.cons_14R',
                        help='约束模块名（python import 路径）')
    parser.add_argument('-N', '--samples', type=int, default=2000,
                        help='采样数量')
    parser.add_argument('-s', '--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    print(f"加载模块: {args.module}")
    print(f"采样数量: {args.samples}")

    cons_module = importlib.import_module(args.module)
    dic_cons = getattr(cons_module, 'dic_cons')
    str_n = cons_module.__name__.split('.')[-1]

    result = monte_carlo_distribution(
        dic_cons, args.samples,
        res_name_prefix=f"mc_results_skinny128_{str_n}",
        seed=args.seed,
    )

    npz_path = f"mc_dists_skinny128_{str_n}.npz"
    save_kwargs = dict(
        cluster_counts=result['per_sample_cluster_counts'],
        total_counts=result['per_sample_total_counts'],
        cluster_names=np.array(result['cluster_names']),
    )
    # master_keys 在 SK 模式下为空，存一个空数组占位
    if result['master_keys']:
        # 把 list[dict] 转成可保存的结构：先排序 key 名拿到列序
        sorted_keys = sorted(result['master_keys'][0].keys())
        mk_mat = np.array(
            [[mk[k] for k in sorted_keys] for mk in result['master_keys']],
            dtype=np.int8,
        )
        save_kwargs['master_keys'] = mk_mat
        save_kwargs['master_key_names'] = np.array(sorted_keys)
    np.savez(npz_path, **save_kwargs)
    print(f"\n所有数据保存到 {npz_path}")

    # ── 联合分布归一化 ──
    total_counts = result['per_sample_total_counts']
    raw_total_dist = {}
    for v in total_counts:
        raw_total_dist[v] = raw_total_dist.get(v, 0) + 1
    final_dist = dic_normalize(raw_total_dist)

    print("\n========== 联合分布（归一化后）==========")
    print(final_dist)

    lst_res = []
    for key, cnt in final_dist.items():
        lst_res.extend([key] * cnt)
    lst_res.sort()
    import os
    os.makedirs('./TRACE_distri', exist_ok=True)
    np.save(f'./TRACE_distri/TRACE_{str_n}.npy', np.array(lst_res))

    sorted_items = sorted(final_dist.items())
    x_indices, y_values = [], []
    cur = 0
    for val, count in sorted_items:
        x_indices.append(cur); y_values.append(val)
        cur += count
        x_indices.append(cur - 1); y_values.append(val)

    print(f"\n总共 {len(sorted_items)} 种不同的解数值")
    print(f"总共 {cur} 个数据点（应等于采样数 {args.samples}）")

    plt.figure(figsize=(10, 6))
    plt.plot(x_indices, y_values, linewidth=2)
    plt.title(f"Sorted Total Solutions per Master Key — SKINNY-128, "
              f"{args.samples} samples")
    plt.xlabel("Master Key Index (sorted)")
    plt.ylabel("Total Solutions (normalized)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    fig_path = f"./distribut_prob_solution_skinny128_{str_n}.png"
    plt.savefig(fig_path, dpi=300)
    print(f"图已保存到 {fig_path}")
    plt.show()

    # ── 独立性诊断 ──
    marginal_means = result['per_sample_cluster_counts'].mean(axis=0)
    indep_expected = float(np.prod(marginal_means))
    actual_mean = float(total_counts.mean())
    print("\n========== 独立性诊断 ==========")
    print(f"各 cluster 边际均值: {marginal_means}")
    print(f"独立性假设下 ∏E[c_i] = {indep_expected:.4f}")
    print(f"实际 E[∏c_i]       = {actual_mean:.4f}")
    if indep_expected > 0:
        print(f"比值 = {actual_mean / indep_expected:.4f}")