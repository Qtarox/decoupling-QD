import re
import time
from ortools.sat.python import cp_model
import importlib
import argparse

my_sbox = [0x6, 0x5, 0xC, 0xA, 0x1, 0xE, 0x7, 0x9, 0xB, 0x0, 0x3, 0xD, 0x8, 0xF, 0x4, 0x2] 
AC_STATES = [0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,0x37,0x2F,0x1E,0x3C,0x39,0x33,0x27,0x0E,
0x1D,0x3A,0x35,0x2B,0x16,0x2C,0x18,0x30,0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38,
0x31,0x23,0x06,0x0D,0x1B,0x36,0x2D,0x1A,0x34,0x29,0x12,0x24,0x08,0x11,0x22,0x04]
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
    
    
    if bit == 63: return 1
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

class KeyDistributionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, k_vars):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.k_vars = k_vars
        self.count_k = {}
        self.solution_count = 0

    def on_solution_callback(self):
        self.solution_count += 1
        key_tuple = tuple(self.Value(v) for v in self.k_vars)
        self.count_k[key_tuple] = self.count_k.get(key_tuple, 0) + 1

def solve_with_ortools(input_text, fixed_vars, res_name):
    model = cp_model.CpModel()
    var_dict = {}

    
    raw_vars = set(re.findall(r'[a-z]_\d+_\d+|k_\d+', input_text))
    for v in fixed_vars.keys():
        raw_vars.add(v)
    for v in raw_vars:
        var_dict[v] = model.NewBoolVar(v)
        

    k_names = sorted([v for v in raw_vars if v.startswith('k_')], key=lambda x: int(x.split('_')[1]))
    k_vars = [var_dict[k] for k in k_names]

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
            
            line = re.sub(r'[\[\]]', '', line).replace("= 0", "").strip()
            vars_in_eq = re.findall(r'[a-z]_\d+_\d+|k_\d+', line)
            
            if vars_in_eq:
                eq_vars = [var_dict[v] for v in vars_in_eq]
                
                
                
                constant_val = 0
                for v in vars_in_eq:
                    constant_val ^= get_gift_constant(v)
                

                
                
                dummy = model.NewIntVar(0, len(eq_vars) // 2 + 1, f'dummy_{dummy_counter}')
                model.Add(sum(eq_vars) + constant_val == 2 * dummy)
                dummy_counter += 1

    
    for var, vals in fixed_vars.items():
        if var in var_dict:
            val = list(vals)[0] 
            model.Add(var_dict[var] == val)

    
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True 
    
    model.AddDecisionStrategy(k_vars, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)

    collector = KeyDistributionCollector(k_vars)
    time_start = time.time()
    status = solver.Solve(model, collector)
    time_end = time.time()
    

    total_k_combinations = 1 << len(k_vars) 
    keys_with_solutions = len(collector.count_k)
    keys_with_zero_solutions = total_k_combinations - keys_with_solutions

    
    KEY_HASH = {}
    
    if keys_with_zero_solutions > 0:
        KEY_HASH['0'] = keys_with_zero_solutions

    with open(res_name, "w") as fw:
        fw.write(f"Total possible key combinations (2^{len(k_vars)}): {total_k_combinations}\n")
        fw.write(f"Combinations with 0 solutions: {keys_with_zero_solutions}\n")
        fw.write("-" * 30 + "\n")
        fw.write("Count of combinations\n")
        for k, count in collector.count_k.items():
            fw.write(f"{k}: {count}\n")
            KEY_HASH[str(count)] = KEY_HASH.get(str(count), 0) + 1
        fw.write(f"time: {time_end - time_start:.4f}s\n")

    return KEY_HASH
parser = argparse.ArgumentParser(description="GIFT SAT Solver")
parser.add_argument('-m', '--module', type=str, default='CONS.cons_6R', 
                    help='module name, for exp: CONS.cons_6R')
args = parser.parse_args()

print(f"loding: {args.module}")


try:
    cons_module = importlib.import_module(args.module)
    
    dic_cons = getattr(cons_module, 'dic_cons') 
except ImportError:
    print(f"error: {args.module}")
    exit(1)
except AttributeError:
    print(f"erroe: {args.module} no dic_cons")
    exit(1)
str_n=cons_module.__name__.split('.')[-1]  


if __name__ == "__main__":
    
    str_res=""
    dic_lst=[]
    for i in range(len(dic_cons)):
        
        name = f"CONS{i}" 
        if name not in dic_cons: continue 
        
        cons_pair = dic_cons[name]
        cons_t = cons_pair[0]
        z = cons_pair[1]
        solu_txt = f"solve_results_gift_{name}.txt"
        
        t = time.time()
        dist = solve_with_ortools(cons_t, z, solu_txt)
        print("distribution):", dist)
        str_res+= f"c{i} = {dist}\n"
        print("Total time used:", time.time() - t)
    print(str_res)