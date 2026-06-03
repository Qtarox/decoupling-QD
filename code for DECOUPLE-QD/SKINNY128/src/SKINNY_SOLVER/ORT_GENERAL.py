import re
import time
from ortools.sat.python import cp_model
import importlib
import argparse
# 确认使用的是正确的 GIFT S盒
my_sbox= [0x65,0x4c,0x6a,0x42,0x4b,0x63,0x43,0x6b,0x55,0x75,0x5a,0x7a,0x53,0x73,0x5b,0x7b,
        0x35,0x8c,0x3a,0x81,0x89,0x33,0x80,0x3b,0x95,0x25,0x98,0x2a,0x90,0x23,0x99,0x2b,
        0xe5,0xcc,0xe8,0xc1,0xc9,0xe0,0xc0,0xe9,0xd5,0xf5,0xd8,0xf8,0xd0,0xf0,0xd9,0xf9,
        0xa5,0x1c,0xa8,0x12,0x1b,0xa0,0x13,0xa9,0x05,0xb5,0x0a,0xb8,0x03,0xb0,0x0b,0xb9,
        0x32,0x88,0x3c,0x85,0x8d,0x34,0x84,0x3d,0x91,0x22,0x9c,0x2c,0x94,0x24,0x9d,0x2d,
        0x62,0x4a,0x6c,0x45,0x4d,0x64,0x44,0x6d,0x52,0x72,0x5c,0x7c,0x54,0x74,0x5d,0x7d,
        0xa1,0x1a,0xac,0x15,0x1d,0xa4,0x14,0xad,0x02,0xb1,0x0c,0xbc,0x04,0xb4,0x0d,0xbd,
        0xe1,0xc8,0xec,0xc5,0xcd,0xe4,0xc4,0xed,0xd1,0xf1,0xdc,0xfc,0xd4,0xf4,0xdd,0xfd,
        0x36,0x8e,0x38,0x82,0x8b,0x30,0x83,0x39,0x96,0x26,0x9a,0x28,0x93,0x20,0x9b,0x29,
        0x66,0x4e,0x68,0x41,0x49,0x60,0x40,0x69,0x56,0x76,0x58,0x78,0x50,0x70,0x59,0x79,
        0xa6,0x1e,0xaa,0x11,0x19,0xa3,0x10,0xab,0x06,0xb6,0x08,0xba,0x00,0xb3,0x09,0xbb,
        0xe6,0xce,0xea,0xc2,0xcb,0xe3,0xc3,0xeb,0xd6,0xf6,0xda,0xfa,0xd3,0xf3,0xdb,0xfb,
        0x31,0x8a,0x3e,0x86,0x8f,0x37,0x87,0x3f,0x92,0x21,0x9e,0x2e,0x97,0x27,0x9f,0x2f,
        0x61,0x48,0x6e,0x46,0x4f,0x67,0x47,0x6f,0x51,0x71,0x5e,0x7e,0x57,0x77,0x5f,0x7f,
        0xa2,0x18,0xae,0x16,0x1f,0xa7,0x17,0xaf,0x01,0xb2,0x0e,0xbe,0x07,0xb7,0x0f,0xbf,
        0xe2,0xca,0xee,0xc6,0xcf,0xe7,0xc7,0xef,0xd2,0xf2,0xde,0xfe,0xd7,0xf7,0xdf,0xff]
AC_STATES = [0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,0x37,0x2F,0x1E,0x3C,0x39,0x33,0x27,0x0E,
0x1D,0x3A,0x35,0x2B,0x16,0x2C,0x18,0x30,0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38,
0x31,0x23,0x06,0x0D,0x1B,0x36,0x2D,0x1A,0x34,0x29,0x12,0x24,0x08,0x11,0x22,0x04]
def get_skinny_constant(var_name):
    """
    针对 SKINNY-128 的常数注入逻辑：
    每个 Cell 为 8-bit。
    常数 c0 (0,0,rc3,rc2,rc1,rc0) 注入 Cell 0 (bits 0-7)
    常数 c1 (0,0,0,0,rc5,rc4) 注入 Cell 4 (bits 32-39)
    常数 c2 (0,0,0,0,0,0,1,0) 注入 Cell 8 (bits 64-71)
    """
    if not var_name.startswith('x_'):
        return 0
    
    parts = var_name.split('_')
    r = int(parts[1])
    bit = int(parts[2])
    
    r_idx = r - 1 # 假设 x_1 对应第一轮后的状态，注入第 0 组常数
    if r_idx < 0 or r_idx >= len(AC_STATES):
        return 0
        
    rc = AC_STATES[r_idx]
    
    # Cell 0 (bits 0-7): 注入 rc0, rc1, rc2, rc3
    if bit == 0: return (rc >> 0) & 1
    if bit == 1: return (rc >> 1) & 1
    if bit == 2: return (rc >> 2) & 1
    if bit == 3: return (rc >> 3) & 1
    
    # Cell 4 (bits 32-39): 注入 rc4, rc5
    if bit == 32: return (rc >> 4) & 1
    if bit == 33: return (rc >> 5) & 1
    
    # Cell 8 (bits 64-71): 注入固定常数 0x02，即第 1 位 (对应全局的 bit 65)
    if bit == 65: return 1
    
    return 0
def get_sbox_tuples(sbox):
    """预计算 S 盒的所有合法 16 位输入输出组合 (8-bit in, 8-bit out)"""
    valid_tuples = []
    # SKINNY-128 S盒大小为 256
    for i in range(256): 
        out = sbox[i]
        valid_tuples.append(
            [(i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1,
             (i >> 4) & 1, (i >> 5) & 1, (i >> 6) & 1, (i >> 7) & 1,
             (out >> 0) & 1, (out >> 1) & 1, (out >> 2) & 1, (out >> 3) & 1,
             (out >> 4) & 1, (out >> 5) & 1, (out >> 6) & 1, (out >> 7) & 1]
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

    # 1. 提取变量
    raw_vars = set(re.findall(r'[a-z]_\d+_\d+|k_\d+', input_text))
    for v in fixed_vars.keys():
        raw_vars.add(v)
    for v in raw_vars:
        var_dict[v] = model.NewBoolVar(v)
        

    k_names = sorted([v for v in raw_vars if v.startswith('k_')], key=lambda x: int(x.split('_')[1]))
    k_vars = [var_dict[k] for k in k_names]

    sbox_valid_tuples = get_sbox_tuples(my_sbox)
    dummy_counter = 0

    # 2. 解析约束
    for line in input_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
            
        # 兼容 S(x)=y 和 S_[...](x)=y
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
                        # 8-bit 输入
                        (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1, (x >> 3) & 1,
                        (x >> 4) & 1, (x >> 5) & 1, (x >> 6) & 1, (x >> 7) & 1,
                        # 8-bit 输出
                        (out >> 0) & 1, (out >> 1) & 1, (out >> 2) & 1, (out >> 3) & 1,
                        (out >> 4) & 1, (out >> 5) & 1, (out >> 6) & 1, (out >> 7) & 1
                    ])
                model.AddAllowedAssignments(in_vars + out_vars, subset_tuples)
            else:
                model.AddAllowedAssignments(in_vars + out_vars, sbox_valid_tuples)
                
        else:
            # 还原你原来的纯正 XOR 解析逻辑
            line = re.sub(r'[\[\]]', '', line).replace("= 0", "").strip()
            vars_in_eq = re.findall(r'[a-z]_\d+_\d+|k_\d+', line)
            
            if vars_in_eq:
                eq_vars = [var_dict[v] for v in vars_in_eq]
                
                # ================= 新增逻辑开始 =================
                # 动态计算这条方程中是否包含常数注入
                constant_val = 0
                for v in vars_in_eq:
                    constant_val ^= get_skinny_constant(v)
                # ================= 新增逻辑结束 =================

                # 修改原有的 dummy 约束，把 constant_val 加进去
                # 这样如果 constant_val 是 1，等式就变成了 sum(vars) + 1 == 2 * dummy (即异或和为1)
                dummy = model.NewIntVar(0, len(eq_vars) // 2 + 1, f'dummy_{dummy_counter}')
                model.Add(sum(eq_vars) + constant_val == 2 * dummy)
                dummy_counter += 1

    # 3. 添加固定变量约束
    for var, vals in fixed_vars.items():
        if var in var_dict:
            val = list(vals)[0] 
            model.Add(var_dict[var] == val)

    # 4. 求解配置 (完全还原！绝对不加多线程！)
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True 
    
    model.AddDecisionStrategy(k_vars, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)

    collector = KeyDistributionCollector(k_vars)
    
    print("开始极速求解...")
    time_start = time.time()
    status = solver.Solve(model, collector)
    time_end = time.time()
    
    print(f"求解状态: {solver.StatusName(status)}")
    print(f"共发现有效解: {collector.solution_count} 个")

    total_k_combinations = 1 << len(k_vars) 
    keys_with_solutions = len(collector.count_k)
    keys_with_zero_solutions = total_k_combinations - keys_with_solutions

    # 5. 写入结果
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
parser.add_argument('-m', '--module', type=str, default='CONS.cons_7R', 
                    help='指定要导入的模块名，例如 CONS.cons_6R')
args = parser.parse_args()

print(f"正在加载模块: {args.module}")

# 2. 动态导入模块
try:
    cons_module = importlib.import_module(args.module)
    # 显式获取模块中的 dic_cons 变量 (避免使用 import *)
    dic_cons = getattr(cons_module, 'dic_cons') 
except ImportError:
    print(f"错误: 找不到模块 {args.module}")
    exit(1)
except AttributeError:
    print(f"错误: 模块 {args.module} 中没有定义 dic_cons")
    exit(1)
str_n=cons_module.__name__.split('.')[-1]  # 获取模块名最后一部分作为标识

# ================= 测试代码保持不变 =================
if __name__ == "__main__":
    # 假设 dic_cons 已经加载
    str_res=""
    dic_lst=[]
    for i in range(len(dic_cons)):
        # 注意这里获取字典 key 的方式适配了你的 'CONS0' 字符串格式
        name = f"CONS{i}" 
        if name not in dic_cons: continue # 容错
        
        cons_pair = dic_cons[name]
        cons_t = cons_pair[0]
        z = cons_pair[1]
        solu_txt = f"solve_results_gift_{name}.txt"
        
        t = time.time()
        dist = solve_with_ortools(cons_t, z, solu_txt)
        print("分布哈希 (数量: 出现次数):", dist)
        str_res+= f"c{i} = {dist}\n"
        print("Total time used:", time.time() - t)
    print(str_res)