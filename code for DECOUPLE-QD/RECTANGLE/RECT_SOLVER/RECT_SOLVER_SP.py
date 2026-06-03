import re
import time
from ortools.sat.python import cp_model
from solution_plt import *
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import importlib
import argparse
import numpy as np
from tqdm import tqdm

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




RECTANGLE_SBOX = [0x6, 0x5, 0xC, 0xA, 0x1, 0xE, 0x7, 0x9,
                  0xB, 0x0, 0x3, 0xD, 0x8, 0xF, 0x4, 0x2]

RECTANGLE_RC = [0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16,
                0x0C, 0x19, 0x13, 0x07, 0x0F, 0x1F, 0x1E, 0x1C,
                0x18, 0x11, 0x03, 0x06, 0x0D, 0x1B, 0x17, 0x0E, 0x1D]

my_sbox = RECTANGLE_SBOX

def key_schedule_80(mk_bits, nb_rounds):
    K = [[mk_bits[row * 16 + col] for col in range(16)] for row in range(5)]
    round_keys = []
    
    for r in range(nb_rounds):
        
        rk = [0] * 64
        for col in range(16):
            for row in range(4):
                rk[4 * col + row] = K[row][col]
        round_keys.append(rk)
        
        if r >= nb_rounds - 1:
            break
        
        
        new_K = [row[:] for row in K]
        for col in range(4):
            nibble = (K[0][col] | (K[1][col] << 1) |
                      (K[2][col] << 2) | (K[3][col] << 3))
            out = RECTANGLE_SBOX[nibble]
            new_K[0][col] = (out >> 0) & 1
            new_K[1][col] = (out >> 1) & 1
            new_K[2][col] = (out >> 2) & 1
            new_K[3][col] = (out >> 3) & 1
        
        
        shifts = [8, 0, 0, 0, 0]
        rotated = [[0]*16 for _ in range(5)]
        for row in range(5):
            sh = shifts[row]
            for col in range(16):
                rotated[row][col] = new_K[row][(col - sh) % 16]
        
        
        K0_old = new_K[0][:]
        K = [[0]*16 for _ in range(5)]
        for col in range(16):
            K[0][col] = rotated[0][col] ^ new_K[1][col]
            K[1][col] = new_K[2][col]
            K[2][col] = new_K[3][col]
            K[3][col] = new_K[4][col]
            K[4][col] = rotated[4][col] ^ K0_old[col]
        
        
        rc = RECTANGLE_RC[r]
        for i in range(5):
            K[0][i] ^= (rc >> i) & 1
    
    return round_keys


def key_schedule_128(mk_bits, nb_rounds):
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
        
        
        new_K = [row[:] for row in K]
        for col in range(8):
            nibble = (K[0][col] | (K[1][col] << 1) |
                      (K[2][col] << 2) | (K[3][col] << 3))
            out = RECTANGLE_SBOX[nibble]
            new_K[0][col] = (out >> 0) & 1
            new_K[1][col] = (out >> 1) & 1
            new_K[2][col] = (out >> 2) & 1
            new_K[3][col] = (out >> 3) & 1
        
        
        shifts = [8, 16, 24, 0]
        K = [[0]*32 for _ in range(4)]
        for row in range(4):
            sh = shifts[row]
            for col in range(32):
                K[row][col] = new_K[row][(col - sh) % 32]
        
        
        rc = RECTANGLE_RC[r]
        for i in range(5):
            K[0][i] ^= (rc >> i) & 1
    
    return round_keys






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


def solve_for_fixed_key(input_text, fixed_vars, round_keys):
    model = cp_model.CpModel()
    var_dict = {}
    
    
    state_vars = set(re.findall(r'[xy]_\d+_\d+', input_text))
    for v in fixed_vars.keys():
        if v.startswith('x_') or v.startswith('y_'):
            state_vars.add(v)
    
    for v in state_vars:
        var_dict[v] = model.NewBoolVar(v)
    
    sbox_valid_tuples = get_sbox_tuples(my_sbox)
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
            
            rhs_const = 0
            line_clean = re.sub(r'[\[\]]', '', line)
            if '= 1' in line_clean:
                rhs_const = 1
                line_clean = line_clean.replace('= 1', '').strip()
            elif '= 0' in line_clean:
                line_clean = line_clean.replace('= 0', '').strip()
            
            
            all_tokens = re.findall(r'[xy]_\d+_\d+|k_\d+_\d+', line_clean)
            
            
            constant_val = rhs_const  
            state_terms = []
            for tok in all_tokens:
                if tok.startswith('k_'):
                    
                    parts = tok.split('_')
                    r_k = int(parts[1])
                    j_k = int(parts[2])
                    if r_k < len(round_keys) and j_k < len(round_keys[r_k]):
                        k_val = round_keys[r_k][j_k]
                        
                        constant_val ^= k_val
                    
                else:
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






def monte_carlo_distribution(dic_cons, key_size, nb_rounds, num_samples,
                             res_name_prefix, seed=42):
    rng = np.random.default_rng(seed)
    
    if key_size == 80:
        ks_func = key_schedule_80
    elif key_size == 128:
        ks_func = key_schedule_128
    else:
        raise ValueError(f"Unsupported key size: {key_size}")
    
    
    master_keys = rng.integers(0, 2, size=(num_samples, key_size)).tolist()
    
    all_round_keys = []
    for mk in tqdm(master_keys, desc="Key schedule"):
        all_round_keys.append(ks_func(mk, nb_rounds))
    
    cluster_names = sorted(dic_cons.keys(),
                           key=lambda x: int(re.search(r'\d+', x).group()))
    
    
    per_sample_cluster_counts = np.zeros(
        (num_samples, len(cluster_names)), dtype=np.int64
    )
    
    per_sample_total_counts = np.zeros(num_samples, dtype=np.float64)
    
    t_start = time.time()
    
    
    for sample_idx in tqdm(range(num_samples), desc="MC samples"):
        rk = all_round_keys[sample_idx]
        
        total = 1
        is_zero = False
        for c_idx, cname in enumerate(cluster_names):
            cons_text, z_dict = dic_cons[cname]
            cnt = solve_for_fixed_key(cons_text, z_dict, rk)
            per_sample_cluster_counts[sample_idx, c_idx] = cnt
            
            if cnt == 0:
                is_zero = True
                total = 0
            elif not is_zero:
                total *= cnt
        
        per_sample_total_counts[sample_idx] = total
    
    elapsed = time.time() - t_start
    print(f"\ntime: {elapsed:.2f}s ({elapsed/num_samples:.3f}s/sample)")
    
    
    cluster_marginal = {}
    for c_idx, cname in enumerate(cluster_names):
        counts = per_sample_cluster_counts[:, c_idx]
        unique, freq = np.unique(counts, return_counts=True)
        cluster_marginal[cname] = dict(zip(unique.tolist(), freq.tolist()))
    
    
    unique_total, freq_total = np.unique(per_sample_total_counts,
                                          return_counts=True)
    joint_dist = dict(zip(unique_total.tolist(), freq_total.tolist()))
    
    
    summary_file = f"{res_name_prefix}_summary.txt"
    with open(summary_file, "w") as fw:
        fw.write(f"Monte Carlo Joint Distribution\n")
        fw.write(f"Master key size: {key_size}\n")
        fw.write(f"Number of samples: {num_samples}\n")
        fw.write(f"Total time: {elapsed:.2f}s\n")
        fw.write("=" * 60 + "\n\n")
        
        for cname, hist in cluster_marginal.items():
            fw.write(f"{cname}: {hist}\n")
        fw.write("\n")
        
        fw.write(f"Mean total solutions: {per_sample_total_counts.mean():.4f}\n")
        fw.write(f"Std: {per_sample_total_counts.std():.4f}\n")
        fw.write(f"Min: {per_sample_total_counts.min()}, "
                 f"Max: {per_sample_total_counts.max()}\n")
        fw.write(f"{int((per_sample_total_counts == 0).sum())}\n")

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
    parser = argparse.ArgumentParser(description="RECTANGLE MC Solver (Joint)")
    parser.add_argument('-m', '--module', type=str, default='CONS.cons_14R')
    parser.add_argument('-k', '--keysize', type=int, default=80, choices=[80, 128])
    parser.add_argument('-r', '--rounds', type=int, required=True)
    parser.add_argument('-N', '--samples', type=int, default=2000)
    parser.add_argument('-s', '--seed', type=int, default=42)
    args = parser.parse_args()
    
    
    cons_module = importlib.import_module(args.module)
    dic_cons = getattr(cons_module, 'dic_cons')
    str_n = cons_module.__name__.split('.')[-1]
    
    result = monte_carlo_distribution(
        dic_cons, args.keysize, args.rounds, args.samples,
        res_name_prefix=f"mc_results_rect{args.keysize}_{str_n}",
        seed=args.seed,
    )
    
    
    npz_path = f"mc_dists_rect{args.keysize}_{str_n}.npz"
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
    plt.title(f"Sorted Total Solutions per Master Key — "
              f"RECT-{args.keysize}, {args.samples} samples")
    plt.xlabel("Master Key Index (sorted by total solutions)")
    plt.ylabel("Total Solutions (normalized)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    fig_path = f"./distribut_prob_solution_rect{args.keysize}_{str_n}.png"
    plt.savefig(fig_path, dpi=300)
    plt.show()
    
    
