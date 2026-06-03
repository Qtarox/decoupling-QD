"""
ORT_SKINNY64.py
================

SKINNY-64 求解器（OR-Tools 版），与 MASK_DIVIDER/SKINNYMILP_msk/utils 
所产生的约束格式对齐。

【比特序约定 —— LSB-first】
依据 MASK_DIVIDER.get_active_bit 的实现:
    initial = (X_lst[0] >> i) & 1
    global_idx = rn * FULL_STATE_BITS + cell_idx * CELL_SIZE + i

其中 i 是从 0 开始递增的偏移，对应 (x >> i) & 1，即 i=0 是整数 LSB。
所以变量 x_r_b 中 b = cell_idx*4 + i：
    b = 4*c + 0  -> cell c 的 LSB
    b = 4*c + 3  -> cell c 的 MSB

【轮编号约定】
MASK_DIVIDER 中轮号 r 从 0 开始，x_r_* 是第 r 轮 S-box 输入
(AddConstants 已在此前注入)。所以使用 LFSR 第 r 步（从 0 起）的状态。
"""

import re
import time
import importlib
import argparse
from ortools.sat.python import cp_model


# ============================================================
# SKINNY-64 官方 S 盒 (Beierle et al., CRYPTO 2016)
# S = c 6 9 0 1 a 2 b 3 8 5 d 4 e 7 f
# ============================================================
my_sbox = [0xc, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb,
           0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf]


# ============================================================
# SKINNY 6-bit 仿射 LFSR (与 utils.py: lfsr_ac 完全等价)
# 状态 [rc5, rc4, rc3, rc2, rc1, rc0]，初值全 0
# 更新规则: tmp = rc5 ^ rc4
#          new_state = [rc4, rc3, rc2, rc1, rc0, tmp ^ 1]
# 第 r 次更新后的状态用于第 r 轮 (r 从 0 起)
# 这里我们把状态打包为 6-bit 整数 (rc0 在低位): rc = rc5*32 + ... + rc0
# ============================================================
def _compute_ac_states(n):
    """返回 n 个 6-bit 整数，rc0 在最低位"""
    s = [0, 0, 0, 0, 0, 0]   # [rc5, rc4, rc3, rc2, rc1, rc0]
    out = []
    for _ in range(n):
        tmp = s[0] ^ s[1]
        s = s[1:] + [tmp ^ 1]
        # 打包: rc0 在 bit 0, rc5 在 bit 5
        rc = (s[5] << 0) | (s[4] << 1) | (s[3] << 2) | (s[2] << 3) | (s[1] << 4) | (s[0] << 5)
        out.append(rc)
    return out

AC_STATES = _compute_ac_states(64)


def get_skinny_constant(var_name):
    """
    SKINNY-64 AddConstants 注入 (LSB-first 比特序):
      c0 = (rc3, rc2, rc1, rc0)  整数值 = rc & 0xF        -> cell 0  (bits 0..3)
      c1 = (0,   0,   rc5, rc4)  整数值 = (rc >> 4) & 0x3 -> cell 4  (bits 16..19)
      c2 = 0x2 = (0, 0, 1, 0)                              -> cell 8  (bits 32..35)

    在 LSB-first 编号下，每个 cell 的 4 个 bit 对应整数值的 bit 0..3:
      cell 0:
        x_r_0 = rc0  (LSB)
        x_r_1 = rc1
        x_r_2 = rc2
        x_r_3 = rc3  (MSB of cell)

      cell 4:
        x_r_16 = rc4  (LSB)
        x_r_17 = rc5
        x_r_18 = 0
        x_r_19 = 0

      cell 8 (c2 = 0x2 = 二进制 0010):
        x_r_32 = 0
        x_r_33 = 1
        x_r_34 = 0
        x_r_35 = 0
    """
    if not var_name.startswith('x_'):
        return 0

    parts = var_name.split('_')
    r = int(parts[1])
    bit = int(parts[2])

    if r < 0 or r >= len(AC_STATES):
        return 0
    rc = AC_STATES[r]

    # Cell 0: bits 0..3 对应 rc0..rc3
    if bit == 0: return (rc >> 0) & 1   # rc0
    if bit == 1: return (rc >> 1) & 1   # rc1
    if bit == 2: return (rc >> 2) & 1   # rc2
    if bit == 3: return (rc >> 3) & 1   # rc3

    # Cell 4: bits 16..19 对应 rc4, rc5, 0, 0
    if bit == 16: return (rc >> 4) & 1  # rc4
    if bit == 17: return (rc >> 5) & 1  # rc5
    if bit == 18: return 0
    if bit == 19: return 0

    # Cell 8: c2 = 0x2 = 0010 (LSB-first)
    if bit == 32: return 0
    if bit == 33: return 1
    if bit == 34: return 0
    if bit == 35: return 0

    return 0


def get_sbox_tuples(sbox):
    """
    预计算 S 盒的所有合法 8 位输入输出组合 (0/1)。
    返回的每个 tuple 顺序与 MASK_DIVIDER 生成的变量顺序对齐:
      S(x_r_{c*4+0}, x_r_{c*4+1}, x_r_{c*4+2}, x_r_{c*4+3}) = (y_r_{c*4+0}, ...)
      其中 x_r_{c*4+0} 是 cell 的 LSB (整数的 bit 0)
    所以 tuple = (x_bit0, x_bit1, x_bit2, x_bit3, y_bit0, y_bit1, y_bit2, y_bit3)
    每个 bit i 由 (val >> i) & 1 给出 (LSB-first)。
    """
    valid_tuples = []
    for i in range(16):
        out = sbox[i]
        valid_tuples.append([
            (i >> 0) & 1, (i >> 1) & 1, (i >> 2) & 1, (i >> 3) & 1,
            (out >> 0) & 1, (out >> 1) & 1, (out >> 2) & 1, (out >> 3) & 1
        ])
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

    # 1. 提取所有变量
    raw_vars = set(re.findall(r'[a-z]_\d+_\d+|k_\d+', input_text))
    for v in fixed_vars.keys():
        raw_vars.add(v)
    for v in raw_vars:
        var_dict[v] = model.NewBoolVar(v)

    k_names = sorted([v for v in raw_vars if v.startswith('k_')],
                     key=lambda x: int(x.split('_')[1]))
    k_vars = [var_dict[k] for k in k_names]

    sbox_valid_tuples = get_sbox_tuples(my_sbox)
    dummy_counter = 0

    # 2. 解析约束
    for line in input_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # S(...) = (...) 或 S_[v1_v2_...](...) = (...)
        sbox_match = re.match(r'S(?:_\[([\d_]+)\])?\((.*?)\)\s*=\s*\((.*?)\)', line)

        if sbox_match:
            valid_x_str = sbox_match.group(1)
            in_vars = [var_dict[v.strip()] for v in sbox_match.group(2).split(',')]
            out_vars = [var_dict[v.strip()] for v in sbox_match.group(3).split(',')]

            if valid_x_str:
                # 仅允许指定的 x 取值（如 S_[10_11_14_15](...) = (...)）
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
            # 线性 XOR 方程
            line_clean = re.sub(r'[\[\]]', '', line).replace("= 0", "").strip()
            vars_in_eq = re.findall(r'[a-z]_\d+_\d+|k_\d+', line_clean)

            if vars_in_eq:
                eq_vars = [var_dict[v] for v in vars_in_eq]

                # 动态收集这条方程中的常数注入总和
                constant_val = 0
                for v in vars_in_eq:
                    constant_val ^= get_skinny_constant(v)

                # XOR 约束：sum(eq_vars) + constant_val ≡ 0 (mod 2)
                dummy = model.NewIntVar(0, len(eq_vars) // 2 + 1,
                                        f'dummy_{dummy_counter}')
                model.Add(sum(eq_vars) + constant_val == 2 * dummy)
                dummy_counter += 1

    # 3. 添加固定变量约束
    for var, vals in fixed_vars.items():
        if var in var_dict:
            val = list(vals)[0]
            model.Add(var_dict[var] == val)

    # 4. 求解配置
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    model.AddDecisionStrategy(k_vars, cp_model.CHOOSE_FIRST,
                              cp_model.SELECT_MIN_VALUE)

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
        fw.write(f"Total possible key combinations (2^{len(k_vars)}): "
                 f"{total_k_combinations}\n")
        fw.write(f"Combinations with 0 solutions: {keys_with_zero_solutions}\n")
        fw.write("-" * 30 + "\n")
        fw.write("Count of combinations\n")
        for k, count in collector.count_k.items():
            fw.write(f"{k}: {count}\n")
            KEY_HASH[str(count)] = KEY_HASH.get(str(count), 0) + 1
        fw.write(f"time: {time_end - time_start:.4f}s\n")

    return KEY_HASH


# ============================================================
# 命令行入口
# ============================================================
parser = argparse.ArgumentParser(description="SKINNY-64 SAT Solver")
parser.add_argument('-m', '--module', type=str, default='CONS.cons_7R',
                    help='指定要导入的模块名，例如 CONS.cons_6R')
args = parser.parse_args()

print(f"正在加载模块: {args.module}")

try:
    cons_module = importlib.import_module(args.module)
    dic_cons = getattr(cons_module, 'dic_cons')
except ImportError:
    print(f"错误: 找不到模块 {args.module}")
    exit(1)
except AttributeError:
    print(f"错误: 模块 {args.module} 中没有定义 dic_cons")
    exit(1)

str_n = cons_module.__name__.split('.')[-1]


if __name__ == "__main__":
    str_res = ""
    for i in range(len(dic_cons)):
        name = f"CONS{i}"
        if name not in dic_cons:
            continue

        cons_pair = dic_cons[name]
        cons_t = cons_pair[0]
        z = cons_pair[1]
        solu_txt = f"solve_results_skinny_{name}.txt"

        t = time.time()
        dist = solve_with_ortools(cons_t, z, solu_txt)
        print("分布哈希 (数量: 出现次数):", dist)
        str_res += f"c{i} = {dist}\n"
        print("Total time used:", time.time() - t)
    print(str_res)