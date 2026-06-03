"""
SKINNY-128 Monte-Carlo joint distribution solver  (并行版)
===========================================================
对原版的修改集中在主流程：用 multiprocessing.Pool 把 2000 次采样分摊到多个
进程，每个进程独立调用 CP-SAT。

并行策略
--------
- 外层 sample 维度并行：每个 sample 一份 master key + 跑所有 cluster -> 一个总解数
- 子进程初始化时一次性加载 dic_cons / SBOX_TUPLES，避免每个 task 重新构造
- Pool.imap_unordered + chunksize 控制粒度
- OR-Tools 内部不开 num_search_workers (enumerate_all_solutions 不允许多 worker)
- 用 fork（Linux 默认）避免拷贝大对象到子进程

CLI 新增参数
------------
  -p / --processes  进程数，默认 = min(CPU 数, samples)。设 1 = 串行。
  --chunksize       Pool imap 的 chunksize，默认按 samples / (processes*4) 自适应

典型加速比：32 核机器上 8-16 倍（受 OR-Tools 建模开销影响，不一定线性）
"""

import os
import re
import time
import argparse
import importlib
import multiprocessing as mp
from functools import partial

import numpy as np
from tqdm import tqdm
from ortools.sat.python import cp_model

try:
    from solution_plt import dic_normalize
except ImportError:
    from math import gcd
    from functools import reduce
    def dic_normalize(d):
        if not d:
            return {}
        g = reduce(gcd, d.values())
        return {k: v // g for k, v in d.items()}

import matplotlib
matplotlib.use("Agg")     # 服务器无显示环境
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
HALF_STATE_BITS = 16 * CELL_SIZE


def _compute_lfsr_states(n_rounds):
    state = 0
    out = []
    for _ in range(n_rounds):
        rc0 = (state >> 5) & 1
        rc1 = (state >> 4) & 1
        new_lsb = rc0 ^ rc1 ^ 1
        state = ((state << 1) & 0x3F) | new_lsb
        out.append(state)
    return out


_LFSR_STATES = _compute_lfsr_states(64)


def get_skinny_constant(var_name):
    """与原版一致：返回该状态比特上自带的 round constant 贡献。"""
    if not var_name.startswith('x_'):
        return 0
    parts = var_name.split('_')
    r = int(parts[1])
    bit = int(parts[2])
    if r <= 0 or r > len(_LFSR_STATES):
        return 0
    lfsr = _LFSR_STATES[r - 1]
    rc0 = (lfsr >> 0) & 1; rc1 = (lfsr >> 1) & 1
    rc2 = (lfsr >> 2) & 1; rc3 = (lfsr >> 3) & 1
    rc4 = (lfsr >> 4) & 1; rc5 = (lfsr >> 5) & 1
    cell = bit // CELL_SIZE
    bit_in_cell = bit % CELL_SIZE
    if cell == 0:
        if bit_in_cell == 0: return rc0
        if bit_in_cell == 1: return rc1
        if bit_in_cell == 2: return rc2
        if bit_in_cell == 3: return rc3
        return 0
    if cell == 4:
        if bit_in_cell == 0: return rc4
        if bit_in_cell == 1: return rc5
        return 0
    if cell == 8:
        return 1 if bit_in_cell == 1 else 0
    return 0


# ════════════════════════════════════════════════════════════════════
# 2) S-box tuples（每个进程进程级缓存）
# ════════════════════════════════════════════════════════════════════
def _build_sbox_tuples():
    n = CELL_SIZE
    out = []
    for i in range(1 << n):
        y = SKINNY128_SBOX[i]
        out.append([(i >> b) & 1 for b in range(n)] + [(y >> b) & 1 for b in range(n)])
    return out


def _sbox_subset_tuples(allowed_inputs):
    n = CELL_SIZE
    out = []
    for x in allowed_inputs:
        y = SKINNY128_SBOX[x]
        out.append([(x >> b) & 1 for b in range(n)] + [(y >> b) & 1 for b in range(n)])
    return out


# ════════════════════════════════════════════════════════════════════
# 3) CP-SAT 单 cluster 求解（与原版同）
# ════════════════════════════════════════════════════════════════════
class _SolutionCounter(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        super().__init__()
        self.count = 0
    def on_solution_callback(self):
        self.count += 1


_TOKEN_RE = re.compile(r'[xy]_\d+_\d+|k[12]?_\d+')
_SBOX_PAT = re.compile(r'S(?:_\[([\d_]+)\])?\((.*?)\)\s*=\s*\((.*?)\)')


def solve_for_fixed_key(input_text, fixed_vars, key_values, full_sbox_tuples):
    model = cp_model.CpModel()
    var_dict = {}
    state_vars = set(re.findall(r'[xy]_\d+_\d+', input_text))
    for v in fixed_vars.keys():
        if v.startswith('x_') or v.startswith('y_'):
            state_vars.add(v)
    for v in state_vars:
        var_dict[v] = model.NewBoolVar(v)

    dummy_counter = 0
    for line in input_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _SBOX_PAT.match(line)
        if m:
            subset_str = m.group(1)
            in_vars  = [var_dict[v.strip()] for v in m.group(2).split(',')]
            out_vars = [var_dict[v.strip()] for v in m.group(3).split(',')]
            assert len(in_vars) == CELL_SIZE and len(out_vars) == CELL_SIZE
            tuples = _sbox_subset_tuples([int(v) for v in subset_str.split('_')]) \
                if subset_str else full_sbox_tuples
            model.AddAllowedAssignments(in_vars + out_vars, tuples)
            continue

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
                constant_val ^= key_values.get(tok, 0)
            else:
                constant_val ^= get_skinny_constant(tok)
                state_terms.append(var_dict[tok])
        if state_terms:
            dummy = model.NewIntVar(0, len(state_terms) // 2 + 1, f'd_{dummy_counter}')
            model.Add(sum(state_terms) + constant_val == 2 * dummy)
            dummy_counter += 1
        else:
            if constant_val != 0:
                f = model.NewBoolVar('false_const')
                model.Add(f == 1)
                model.Add(f == 0)

    for var, vals in fixed_vars.items():
        if var in var_dict:
            val = list(vals)[0]
            model.Add(var_dict[var] == val)

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.log_search_progress = False
    # 重要：enumerate_all_solutions 模式下 OR-Tools 不允许 num_search_workers>1，
    # 所以我们不在 solver 这一层并行；并行做在外层 multiprocessing。
    counter = _SolutionCounter()
    solver.Solve(model, counter)
    return counter.count


# ════════════════════════════════════════════════════════════════════
# 4) key 扫描 / 采样
# ════════════════════════════════════════════════════════════════════
def _scan_key_namespace(dic_cons):
    """扫所有 cluster，记录真正出现过的 k_* / k1_* / k2_* 变量名（排序后返回）。"""
    active = set()
    pat = re.compile(r'\b(k[12]?_\d+)\b')
    for _, (text, _) in dic_cons.items():
        active.update(pat.findall(text))
    return sorted(active, key=lambda x: (x.split('_')[0], int(x.split('_')[1])))


def _sample_master_key(rng, active_keys):
    if not active_keys:
        return {}
    bits = rng.integers(0, 2, size=len(active_keys))
    return {k: int(b) for k, b in zip(active_keys, bits)}


# ════════════════════════════════════════════════════════════════════
# 5) 并行 worker
# ════════════════════════════════════════════════════════════════════
# 每个子进程在初始化时把 dic_cons / cluster_names / sbox_tuples 缓存到 globals
_WORKER_CTX = {}


def _worker_init(dic_cons, cluster_names):
    """Pool initializer：每个进程缓存一次大对象。"""
    _WORKER_CTX['dic_cons'] = dic_cons
    _WORKER_CTX['cluster_names'] = cluster_names
    _WORKER_CTX['sbox_tuples'] = _build_sbox_tuples()


def _run_one_sample(args):
    """
    单次 sample 的 worker 入口。
    args = (sample_idx, key_values_dict)
    返回 (sample_idx, cluster_counts_list, total_count)
    """
    sample_idx, k_vals = args
    dic_cons      = _WORKER_CTX['dic_cons']
    cluster_names = _WORKER_CTX['cluster_names']
    sbox_tuples   = _WORKER_CTX['sbox_tuples']

    cluster_counts = []
    total = 1
    is_zero = False
    for cname in cluster_names:
        cons_text, z_dict = dic_cons[cname]
        cnt = solve_for_fixed_key(cons_text, z_dict, k_vals, sbox_tuples)
        cluster_counts.append(cnt)
        if cnt == 0:
            is_zero = True
            total = 0
        elif not is_zero:
            total *= cnt
    return sample_idx, cluster_counts, total


# ════════════════════════════════════════════════════════════════════
# 6) 主流程
# ════════════════════════════════════════════════════════════════════
def monte_carlo_distribution(dic_cons, num_samples, res_name_prefix,
                             seed=42, processes=None, chunksize=None):
    rng = np.random.default_rng(seed)
    active_keys = _scan_key_namespace(dic_cons)
    key_dim_total = len(active_keys)

    print(f"主密钥维度: {key_dim_total} bits")
    if key_dim_total:
        preview = active_keys[:8] + (['...'] if len(active_keys) > 8 else [])
        print(f"  采样的 key 变量(前 8 个): {preview}")
    else:
        print("[info] cluster 里不含 k_* / k1_* / k2_*（SK 模式）")

    cluster_names = sorted(
        dic_cons.keys(), key=lambda x: int(re.search(r'\d+', x).group())
    )
    n_clusters = len(cluster_names)
    print(f"#clusters = {n_clusters}")

    # 预生成所有 master keys（在主进程做，避免子进程间随机性冲突）
    all_keys = [_sample_master_key(rng, active_keys) for _ in range(num_samples)]
    tasks = list(enumerate(all_keys))

    # 进程数 / chunksize 自适应
    cpu = os.cpu_count() or 1
    if processes is None:
        processes = min(cpu, num_samples)
    processes = max(1, min(processes, num_samples))

    if chunksize is None:
        # 经验值：把任务切成 processes*4 块左右，给 worker 之间留点 work-stealing 空间
        chunksize = max(1, num_samples // (processes * 4))
    print(f"并行配置: processes={processes}, chunksize={chunksize}")

    per_sample_cluster_counts = np.zeros((num_samples, n_clusters), dtype=np.int64)
    per_sample_total_counts   = np.zeros(num_samples, dtype=np.float64)

    t_start = time.time()
    if processes == 1:
        # 串行兜底（方便 debug / 小样本 / Windows）
        _worker_init(dic_cons, cluster_names)
        for args in tqdm(tasks, desc="MC samples (serial)"):
            idx, counts, total = _run_one_sample(args)
            per_sample_cluster_counts[idx] = counts
            per_sample_total_counts[idx]   = total
    else:
        # fork 启动可以让子进程零拷贝继承 dic_cons；但 macOS/Windows 必须用 spawn，
        # 此时 initializer 会把对象传过去。Linux 服务器默认 fork，最佳路径。
        # 强制使用 spawn，彻底避开底层 C++ 多线程锁死的 Bug
        # 如果 spawn 嫌慢，Linux 下可以尝试使用 forkserver
        ctx = mp.get_context('forkserver') if hasattr(mp, 'get_context') and 'forkserver' in mp.get_all_start_methods() else mp.get_context('spawn')
        with ctx.Pool(processes=processes,
                      initializer=_worker_init,
                      initargs=(dic_cons, cluster_names)) as pool:
            for idx, counts, total in tqdm(
                pool.imap_unordered(_run_one_sample, tasks, chunksize=chunksize),
                total=num_samples, desc=f"MC samples (×{processes})"
            ):
                per_sample_cluster_counts[idx] = counts
                per_sample_total_counts[idx]   = total

    elapsed = time.time() - t_start
    print(f"\n总耗时: {elapsed:.2f}s ({elapsed/max(num_samples,1):.3f}s/sample"
          f"; {num_samples/max(elapsed,1e-9):.2f} samples/s)")

    # ---- 统计、汇总文件 ----
    cluster_marginal = {}
    for c_idx, cname in enumerate(cluster_names):
        counts = per_sample_cluster_counts[:, c_idx]
        u, f = np.unique(counts, return_counts=True)
        cluster_marginal[cname] = dict(zip(u.tolist(), f.tolist()))

    u_total, f_total = np.unique(per_sample_total_counts, return_counts=True)
    joint = dict(zip(u_total.tolist(), f_total.tolist()))

    summary_file = f"{res_name_prefix}_summary.txt"
    with open(summary_file, "w") as fw:
        fw.write("SKINNY-128 MC Joint Distribution (parallel)\n")
        fw.write(f"Master key size: {key_dim_total}\n")
        fw.write(f"Active key vars: {active_keys}\n")
        fw.write(f"Number of samples: {num_samples}\n")
        fw.write(f"Processes: {processes}\n")
        fw.write(f"Total time: {elapsed:.2f}s\n")
        fw.write("=" * 60 + "\n\n")
        fw.write("# Per-cluster MARGINAL distribution\n")
        for cname, hist in cluster_marginal.items():
            fw.write(f"{cname}: {hist}\n")
        fw.write("\n# JOINT distribution\n")
        fw.write(f"Mean: {per_sample_total_counts.mean():.4f}\n")
        fw.write(f"Std:  {per_sample_total_counts.std():.4f}\n")
        fw.write(f"Min: {per_sample_total_counts.min()}, "
                 f"Max: {per_sample_total_counts.max()}\n")
        fw.write(f"# zero-total samples: "
                 f"{int((per_sample_total_counts == 0).sum())}\n")
        fw.write(f"# distinct totals: {len(u_total)}\n\n")
        fw.write("Total solutions : # master keys\n")
        for u, c in zip(u_total, f_total):
            fw.write(f"{u:.0f}: {c}\n")
    print(f"汇总写入 {summary_file}")

    return {
        'per_sample_cluster_counts': per_sample_cluster_counts,
        'per_sample_total_counts':   per_sample_total_counts,
        'cluster_marginal': cluster_marginal,
        'joint_distribution': joint,
        'cluster_names': cluster_names,
        'master_keys': all_keys,
        'active_keys': active_keys,
    }


# ════════════════════════════════════════════════════════════════════
# 7) CLI
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SKINNY-128 MC Solver (parallel)")
    parser.add_argument('-m', '--module',   type=str, default='CONS.cons_14R')
    parser.add_argument('-N', '--samples',  type=int, default=2000)
    parser.add_argument('-s', '--seed',     type=int, default=42)
    parser.add_argument('-p', '--processes', type=int, default=None,
                        help='进程数，默认 = min(CPU 数, samples)；设 1 = 串行')
    parser.add_argument('--chunksize', type=int, default=None,
                        help='Pool imap chunksize，默认按 processes/samples 自适应')
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
        processes=args.processes,
        chunksize=args.chunksize,
    )

    # ---- 保存 npz ----
    npz_path = f"mc_dists_skinny128_{str_n}.npz"
    save_kwargs = dict(
        cluster_counts=result['per_sample_cluster_counts'],
        total_counts=result['per_sample_total_counts'],
        cluster_names=np.array(result['cluster_names']),
    )
    if result['master_keys'] and result['active_keys']:
        sorted_keys = result['active_keys']
        mk_mat = np.array(
            [[mk.get(k, 0) for k in sorted_keys] for mk in result['master_keys']],
            dtype=np.int8,
        )
        save_kwargs['master_keys'] = mk_mat
        save_kwargs['master_key_names'] = np.array(sorted_keys)
    np.savez(npz_path, **save_kwargs)
    print(f"\n所有数据保存到 {npz_path}")

    # ---- 联合分布归一化 / 画图（与原版一致） ----
    total_counts = result['per_sample_total_counts']
    raw_total = {}
    for v in total_counts:
        raw_total[v] = raw_total.get(v, 0) + 1
    final_dist = dic_normalize(raw_total)

    print("\n========== 联合分布（归一化后）==========")
    print(final_dist)

    lst_res = []
    for k, c in final_dist.items():
        lst_res.extend([k] * c)
    lst_res.sort()
    os.makedirs('./TRACE_distri', exist_ok=True)
    np.save(f'./TRACE_distri/TRACE_{str_n}.npy', np.array(lst_res))

    sorted_items = sorted(final_dist.items())
    x_idx, y_val = [], []
    cur = 0
    for val, count in sorted_items:
        x_idx.append(cur);       y_val.append(val)
        cur += count
        x_idx.append(cur - 1);   y_val.append(val)

    print(f"\n#distinct totals = {len(sorted_items)}")
    print(f"#points = {cur}  (应等于 {args.samples})")

    plt.figure(figsize=(10, 6))
    plt.plot(x_idx, y_val, linewidth=2)
    plt.title(f"Sorted Total Solutions per Master Key — SKINNY-128, "
              f"{args.samples} samples")
    plt.xlabel("Master Key Index (sorted)")
    plt.ylabel("Total Solutions (normalized)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    fig_path = f"./distribut_prob_solution_skinny128_{str_n}.png"
    plt.savefig(fig_path, dpi=300)
    print(f"图已保存到 {fig_path}")

    # ---- 独立性诊断 ----
    marg_means = result['per_sample_cluster_counts'].mean(axis=0)
    indep_exp  = float(np.prod(marg_means))
    actual_mean = float(total_counts.mean())
    print("\n========== 独立性诊断 ==========")
    print(f"各 cluster 边际均值: {marg_means}")
    print(f"独立性假设下 ∏E[c_i] = {indep_exp:.4f}")
    print(f"实际 E[∏c_i]       = {actual_mean:.4f}")
    if indep_exp > 0:
        print(f"比值 = {actual_mean / indep_exp:.4f}")