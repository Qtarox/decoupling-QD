import re
import time
from ortools.sat.python import cp_model
from solution_plt import *
import matplotlib.pyplot as plt
import importlib
import argparse
import numpy as np
from tqdm import tqdm
GIFT_SBOX = [1, 10, 4, 12, 6, 15, 3, 9, 2, 13, 11, 7, 5, 0, 8, 14]
AC_STATES = [0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,0x37,0x2F,0x1E,0x3C,0x39,0x33,0x27,0x0E,
0x1D,0x3A,0x35,0x2B,0x16,0x2C,0x18,0x30,0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38,
0x31,0x23,0x06,0x0D,0x1B,0x36,0x2D,0x1A,0x34,0x29,0x12,0x24,0x08,0x11,0x22,0x04]
MASTER_KEY_SIZE = 128
STATE_SIZE = 128


def get_gift_constant(var_name):

    if not var_name.startswith('x_'):
        return 0
    parts = var_name.split('_')
    r = int(parts[1])
    bit = int(parts[2])
    r_idx = r - 1
    if r_idx < 0 or r_idx >= len(AC_STATES):
        return 0
    ac = AC_STATES[r_idx]
    
    if bit == STATE_SIZE - 1: return 1
    if bit == 23: return (ac >> 5) & 1
    if bit == 19: return (ac >> 4) & 1
    if bit == 15: return (ac >> 3) & 1
    if bit == 11: return (ac >> 2) & 1
    if bit == 7:  return (ac >> 1) & 1
    if bit == 3:  return (ac >> 0) & 1
    return 0


def get_sbox_tuples(sbox):
    valid_tuples = []
    for i in range(16):
        out = sbox[i]
        valid_tuples.append(
            [(i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1,
             (out >> 0) & 1, (out >> 1) & 1, (out >> 2) & 1, (out >> 3) & 1]
        )
    return valid_tuples






class SolutionCounter(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.count = 0
    def on_solution_callback(self):
        self.count += 1


def solve_for_fixed_key(input_text, fixed_vars, mk_values):
    model = cp_model.CpModel()
    var_dict = {}

    
    state_vars = set(re.findall(r'[xy]_\d+_\d+', input_text))
    for v in fixed_vars.keys():
        if v.startswith('x_') or v.startswith('y_'):
            state_vars.add(v)
    for v in state_vars:
        var_dict[v] = model.NewBoolVar(v)

    sbox_valid_tuples = get_sbox_tuples(GIFT_SBOX)
    dummy_counter = 0

    
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
                allowed = set(int(v) for v in valid_x_str.split('_'))
                subset_tuples = []
                for x in allowed:
                    out = GIFT_SBOX[x]
                    subset_tuples.append([
                        (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1, (x >> 3) & 1,
                        (out >> 0) & 1, (out >> 1) & 1, (out >> 2) & 1, (out >> 3) & 1
                    ])
                model.AddAllowedAssignments(in_vars + out_vars, subset_tuples)
            else:
                model.AddAllowedAssignments(in_vars + out_vars, sbox_valid_tuples)
        else:
            
            line_clean = re.sub(r'[\[\]]', '', line)
            rhs = 0
            if '= 1' in line_clean:
                rhs = 1
                line_clean = line_clean.replace('= 1', '').strip()
            elif '= 0' in line_clean:
                line_clean = line_clean.replace('= 0', '').strip()

            tokens = re.findall(r'[a-z]_\d+_\d+|k_\d+', line_clean)

            constant_val = rhs
            state_terms = []
            for tok in tokens:
                if tok.startswith('k_'):
                    
                    idx = int(tok.split('_')[1])
                    if idx < len(mk_values):
                        constant_val ^= mk_values[idx]
                else:
                    
                    constant_val ^= get_gift_constant(tok)
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

    counter = SolutionCounter()
    solver.Solve(model, counter)
    return counter.count



def monte_carlo_distribution(dic_cons, num_samples, res_name_prefix, seed=42):
    rng = np.random.default_rng(seed)

    
    all_k_indices = set()
    for cname, (cons_text, _) in dic_cons.items():
        for m in re.finditer(r'k_(\d+)', cons_text):
            all_k_indices.add(int(m.group(1)))
    max_k = max(all_k_indices) if all_k_indices else -1
    key_dim = max(MASTER_KEY_SIZE, max_k + 1)

    
    master_keys = rng.integers(0, 2, size=(num_samples, key_dim)).tolist()

    cluster_names = sorted(dic_cons.keys(),
                           key=lambda x: int(re.search(r'\d+', x).group()))

    per_sample_cluster_counts = np.zeros((num_samples, len(cluster_names)),
                                          dtype=np.int64)
    per_sample_total_counts = np.zeros(num_samples, dtype=np.float64)

    t_start = time.time()

    for sample_idx in tqdm(range(num_samples), desc="MC samples"):
        k_vals = master_keys[sample_idx]

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
    print(f"\nTime consumption: {elapsed:.2f}s ({elapsed/num_samples:.3f}s/sample)")

    
    cluster_marginal = {}
    for c_idx, cname in enumerate(cluster_names):
        counts = per_sample_cluster_counts[:, c_idx]
        unique, freq = np.unique(counts, return_counts=True)
        cluster_marginal[cname] = dict(zip(unique.tolist(), freq.tolist()))

    
    unique_total, freq_total = np.unique(per_sample_total_counts, return_counts=True)
    joint_dist = dict(zip(unique_total.tolist(), freq_total.tolist()))

    
    summary_file = f"{res_name_prefix}_summary.txt"
    with open(summary_file, "w") as fw:
        fw.write(f"GIFT-128 MC Joint Distribution\n")
        fw.write(f"Master key size: {key_dim}\n")
        fw.write(f"State size: {STATE_SIZE}\n")
        fw.write(f"Number of samples: {num_samples}\n")
        fw.write(f"Total time: {elapsed:.2f}s\n")
        fw.write("=" * 60 + "\n\n")

        for cname, hist in cluster_marginal.items():
            fw.write(f"{cname}: {hist}\n")
        fw.write("\n")


        fw.write(f"Mean: {per_sample_total_counts.mean():.4f}\n")
        fw.write(f"Std: {per_sample_total_counts.std():.4f}\n")
        fw.write(f"Min: {per_sample_total_counts.min()}, Max: {per_sample_total_counts.max()}\n")
        fw.write( f"{int((per_sample_total_counts == 0).sum())}\n")
        for u, c in zip(unique_total, freq_total):
            fw.write(f"{u:.0f}: {c}\n")

    return {
        'per_sample_cluster_counts': per_sample_cluster_counts,
        'per_sample_total_counts': per_sample_total_counts,
        'cluster_marginal': cluster_marginal,
        'joint_distribution': joint_dist,
        'cluster_names': cluster_names,
        'master_keys': master_keys,
    }






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GIFT-128 MC Solver")
    parser.add_argument('-m', '--module', type=str, default='CONS.cons_6R',
                        help='module name')
    parser.add_argument('-N', '--samples', type=int, default=2000,
                        help='sample number')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='seed selection')
    args = parser.parse_args()

    cons_module = importlib.import_module(args.module)
    dic_cons = getattr(cons_module, 'dic_cons')
    str_n = cons_module.__name__.split('.')[-1]

    result = monte_carlo_distribution(
        dic_cons, args.samples,
        res_name_prefix=f"mc_results_gift128_{str_n}",
        seed=args.seed,
    )

    
    npz_path = f"mc_dists_gift128_{str_n}.npz"
    np.savez(
        npz_path,
        cluster_counts=result['per_sample_cluster_counts'],
        total_counts=result['per_sample_total_counts'],
        master_keys=np.array(result['master_keys']),
        cluster_names=np.array(result['cluster_names']),
    )
    total_counts = result['per_sample_total_counts']

    raw_total_dist = {}
    for v in total_counts:
        raw_total_dist[v] = raw_total_dist.get(v, 0) + 1

    final_dist = dic_normalize(raw_total_dist)
    print(final_dist)
    lst_res = []
    for key in final_dist:
        for i in range(final_dist[key]):
            lst_res.append(key)
    lst_res.sort()
    arr_res = np.array(lst_res)
    np.save(f'./TRACE_distri/TRACE_{str_n}', arr_res)

    sorted_items = sorted(final_dist.items())
    x_indices = []
    y_values = []
    current_index = 0
    for val, count in sorted_items:
        x_indices.append(current_index)
        y_values.append(val)
        current_index += count
        x_indices.append(current_index - 1)
        y_values.append(val)

    plt.figure(figsize=(10, 6))
    plt.plot(x_indices, y_values, linewidth=2)
    plt.title(f"Sorted Total Solutions per Master Key — GIFT-128, "
              f"{args.samples} samples")
    plt.xlabel("Master Key Index (sorted)")
    plt.ylabel("Total Solutions (normalized)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    fig_path = f"./distribut_prob_solution_gift128_{str_n}.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()

    
    marginal_means = result['per_sample_cluster_counts'].mean(axis=0)
    indep_expected = np.prod(marginal_means)
    actual_mean = total_counts.mean()
