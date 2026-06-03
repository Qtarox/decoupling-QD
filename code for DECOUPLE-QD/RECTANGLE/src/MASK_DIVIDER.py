"""
MASK_DIVIDER for RECTANGLE.

把 quasidifferential trail 搜索阶段产出的 masks_freq_*.npy 转化为可读的
独立 cluster：每个 cluster 给出
    - 线性等式（含子密钥 k_{r,j}），
    - S 盒等式，
    - 涉及的 mask 变量列表，
    - Z 字典（差分要求该比特取的确定值）。

索引约定（必须与 utils.py / RECTANGLE_msk.py 完全一致）：
    flat index j  = 4*col + row   ，row = j % 4 ， col = j // 4
    row 0 = LSB，row 3 = MSB
    Nibble Col(c) = a[3][c]<<3 | a[2][c]<<2 | a[1][c]<<1 | a[0][c]

L 矩阵列编址：
    第 r 轮 x_{r,j}    →  r * 128 + j           （j = 0..63）
    第 r 轮 y_{r,j}    →  r * 128 + 64 + j       （j = 0..63）
    第 r 轮 k_{r,j}    →  STATE_COLS + r * 64 + j（j = 0..63）
        其中 STATE_COLS = 128 * (rounds + 1)。
"""

import re
import os
import numpy as np

from diffs import *
from utils import (
    NB_ROUNDS, NB_ROWS, NB_COLS, RECTANGLE_SBOX as Sbox,
    ADV_MODEL, MIN_CORR, SBOX_SIZE, RECT_PERM,
)


# ════════════════════════════════════════════════════════════════
# 线性层矩阵
# ════════════════════════════════════════════════════════════════
def genLinear_RECT(rounds):
    """
    生成 RECTANGLE 线性层 + AddRoundKey 的约束矩阵 L。

    每轮的状态变量：
        x_{r,j} : 第 r 轮 SubColumn 输入 = AddRoundKey 输出  （j = 0..63）
        y_{r,j} : 第 r 轮 SubColumn 输出 = ShiftRow 输入     （j = 0..63）
        k_{r,j} : 在 x_r 之前注入的轮密钥                    （j = 0..63）

    每个约束（对 r = 0 .. rounds-1，对 j = 0 .. 63）：
        y_{r,j}  XOR  x_{r+1, RECT_PERM[j]}  XOR  k_{r+1, RECT_PERM[j]}  =  0

    注意：
      - 没有显式 k_{0,*}（trail 边界条件 u[0,0,*]=0 让它无法出现在 mask 中）。
      - 也没有 final-AddRoundKey 的 k_{rounds,*}（边界 u[last,1,*]=0 同理）。
      - 因此 k_round 实际取值范围是 1 .. rounds（也就是说 k_0 列被分配但永远不会被引用）。
    """
    STATE_COLS = 128 * (rounds + 1)
    KEY_COLS   = 64 * (rounds + 1)     # 让 k_{r,j} 偏移恰为 STATE_COLS + r*64 + j
    TOTAL_COLS = STATE_COLS + KEY_COLS

    rows = []
    for r in range(rounds):
        for j in range(64):
            row = np.zeros(TOTAL_COLS, dtype=np.int8)

            # y_{r, j}
            row[r * 128 + 64 + j] = 1

            # ShiftRow 把 j 送到 RECT_PERM[j]
            target_bit = RECT_PERM[j]

            # x_{r+1, target_bit}
            row[(r + 1) * 128 + target_bit] = 1

            # k_{r+1, target_bit}
            row[STATE_COLS + (r + 1) * 64 + target_bit] = 1

            rows.append(row)

    return np.array(rows, dtype=np.int8)


# ════════════════════════════════════════════════════════════════
# 兼容旧文件格式的 mask 读取（可选）
# ════════════════════════════════════════════════════════════════
def load_masks_from_file(file_path, round_num):
    mask_list = [[[0 for _ in range(64)] for _ in range(2)] for _ in range(round_num)]
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        round_blocks = re.findall(r'--- Round (\d+) ---(.*?)(?=--- Round|$)', content, re.DOTALL)
        for r_idx_str, block_text in round_blocks:
            r_idx = int(r_idx_str)
            if r_idx >= round_num:
                continue
            side_matches = re.findall(r'Side (\d+) \(.*?\): ([\s01]+)', block_text)
            for s_idx_str, bit_raw in side_matches:
                s_idx = int(s_idx_str)
                if s_idx > 1:
                    continue
                clean_bits = bit_raw.replace(" ", "").replace("\n", "").replace("\r", "").strip()
                for bit_pos, bit_val in enumerate(clean_bits):
                    if bit_pos < 64:
                        mask_list[r_idx][s_idx][bit_pos] = int(bit_val)
        return mask_list
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


# ════════════════════════════════════════════════════════════════
# 差分 -> active bit 字典
# ════════════════════════════════════════════════════════════════
def xddt_list(input_diff, output_diff):
    res = []
    for x in range(16):
        if input_diff != 0 and (Sbox[x] ^ Sbox[x ^ input_diff]) == output_diff:
            res.append(x)
    return res


def yddt_list(input_diff, output_diff):
    res = []
    for x in range(16):
        if input_diff != 0 and (Sbox[x] ^ Sbox[x ^ input_diff]) == output_diff:
            res.append(Sbox[x])
    return res


def creat_dic_RECT(diff_trail_cell):
    """
    diff_trail_cell[r][0][col] = 第 r 轮 SubColumn 输入差分（nibble，行号 LSB-first）
    diff_trail_cell[r][1][col] = 第 r 轮 SubColumn 输出差分
    """
    x_dic, y_dic = {}, {}
    for r in range(len(diff_trail_cell)):
        for col in range(NB_COLS):
            in_diff  = diff_trail_cell[r][0][col]
            out_diff = diff_trail_cell[r][1][col]
            if in_diff == 0:
                continue
            x_dic[f'x_{r}_{col}'] = xddt_list(in_diff, out_diff)
            y_dic[f'y_{r}_{col}'] = yddt_list(in_diff, out_diff)
    return x_dic, y_dic


def get_active_bit(x_dic, y_dic):
    """
    对每个活跃 S 盒，找出在该 S 盒的"右对(x,y)集合"中取值恒定的 row。
    这些 row 上的 bit 在所有右对里是确定值，我们将之放进 active_bit_dic。

    返回字典：键为字符串形式的 flat 索引 r*128 + side*64 + (4*col + row)，值为该 bit。
    """
    res = {}

    # 输入侧：x_dic[f'x_{r}_{col}'] 是 nibble value 列表，bit i = (value >> i) & 1 = row i。
    for key, X_lst in x_dic.items():
        m = re.match(r"x_(\d+)_(\d+)", key)
        rn, col = int(m.group(1)), int(m.group(2))
        for i in range(NB_ROWS):                          # i = row
            ref = (X_lst[0] >> i) & 1
            if all(((x >> i) & 1) == ref for x in X_lst):
                # flat 位置 = 4*col + row（row 0 = LSB），与 utils 一致
                res[str(rn * 128 + 4 * col + i)] = ref

    # 输出侧
    for key, Y_lst in y_dic.items():
        m = re.match(r"y_(\d+)_(\d+)", key)
        rn, col = int(m.group(1)), int(m.group(2))
        for i in range(NB_ROWS):
            ref = (Y_lst[0] >> i) & 1
            if all(((y >> i) & 1) == ref for y in Y_lst):
                res[str(rn * 128 + 64 + 4 * col + i)] = ref

    return res


# ════════════════════════════════════════════════════════════════
# Union-Find
# ════════════════════════════════════════════════════════════════
class UnionFind:
    def __init__(self, elements):
        self.parent = {el: el for el in elements}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


# ════════════════════════════════════════════════════════════════
# S 盒等式 + 线性等式格式化
# ════════════════════════════════════════════════════════════════
def generate_sb_equ_RECT(r, col, active_bit_dic, dic_x):
    """
    生成形如  S_[x0_x1_...](x_r_{4col},x_r_{4col+1},x_r_{4col+2},x_r_{4col+3})
                       = (y_r_{4col},...,y_r_{4col+3})  的等式字符串。
    同时返回这条等式里"取确定值"的活跃 bit 全局索引列表。
    """
    x_cons = ""
    if f'x_{r}_{col}' in dic_x:
        x_cons += "_["
        for x_v in dic_x[f'x_{r}_{col}']:
            x_cons += f'{x_v}_'
        x_cons = x_cons[:-1] + ']'

    l_tmp = f"S{x_cons}("
    var_lst = []

    # 输入侧 x: 4*col + i，i 是 row
    for i in range(NB_ROWS):
        l_tmp += f"x_{r}_{4*col + i},"
        gidx = r * 128 + 4 * col + i
        if str(gidx) in active_bit_dic:
            var_lst.append(gidx)
    l_tmp = l_tmp[:-1] + ") = ("

    # 输出侧 y: 同样 4*col + i
    for i in range(NB_ROWS):
        l_tmp += f"y_{r}_{4*col + i},"
        gidx = r * 128 + 64 + 4 * col + i
        if str(gidx) in active_bit_dic:
            var_lst.append(gidx)
    l_tmp = l_tmp[:-1] + ')'
    return l_tmp, var_lst


def show_L_equ_RECT(cons_mat, active_bit_dic, rounds):
    """
    把线性等式矩阵转成「var1 + var2 + ... + k_r_j + ... = rhs」字符串列表。
    """
    STATE_COLS = 128 * (rounds + 1)
    lines = []
    for r_idx in range(cons_mat.shape[0]):
        row = cons_mat[r_idx]
        terms = []
        rhs = 0

        # 状态比特列
        for j in range(STATE_COLS):
            if row[j] == 0:
                continue
            r = j // 128
            ind = j % 128
            var_name = f"x_{r}_{ind}" if ind < 64 else f"y_{r}_{ind - 64}"

            if str(j) in active_bit_dic:
                # 该 bit 在所有右对里取确定值 active_bit_dic[str(j)]，作为常数移到右端
                rhs ^= active_bit_dic[str(j)]
            else:
                terms.append(var_name)

        # 密钥比特列
        for j in range(STATE_COLS, cons_mat.shape[1]):
            if row[j] == 1:
                k_idx = j - STATE_COLS
                k_round = k_idx // 64
                k_bit = k_idx % 64
                terms.append(f"k_{k_round}_{k_bit}")

        if terms:
            lines.append(" + ".join(terms) + f" = {rhs}")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# 主流程：构造 cluster 化的等式
# ════════════════════════════════════════════════════════════════
def generate_constraints_RECT(L_mat, dic_x, active_bit_dic, masked_bit_dic, ROUNDS, L_original):
    STATE_COLS = 128 * (ROUNDS + 1)
    TOTAL_COLS = np.shape(L_mat)[1]

    # 1. 状态列标注：
    #      值 1 → 中间未知比特（既不在 active 也不在 mask 字典中）
    #      值 2 → masked inactive（在 mask 中，不在 active 中）—— 真正的"未知数"
    #      值 3 → active（取确定值，在两侧字典中可能都出现，但只关心 active）
    #    密钥列保持值 1（"未知" key 比特），不参与上面的状态分类。
    for r in range(np.shape(L_mat)[0]):
        for j in range(STATE_COLS):
            if L_mat[r][j] == 1:
                if str(j) in masked_bit_dic:
                    if str(j) in active_bit_dic:
                        L_mat[r][j] = 3
                    else:
                        L_mat[r][j] = 2

    # 2. 筛选目标行
    target_rows = []
    independent_rows = []
    for r in range(np.shape(L_mat)[0]):
        state_part = L_mat[r, :STATE_COLS]
        if np.any(state_part == 1):     # 含中间未知 bit → 不要
            continue
        if np.all(L_mat[r] == 0):       # 空行
            continue
        if np.any(state_part == 2):     # 有 mask 未知数
            target_rows.append(r)
        elif np.any(state_part == 3):   # 只有 active bit + 可能的 key bit
            independent_rows.append(r)

    # 3. 收集每行的未知数（mask 状态位 + 密钥位）
    unknown_vars = set()
    row_to_unknowns = {}
    for r in target_rows:
        un_vars = []
        for j in range(STATE_COLS):
            if L_mat[r][j] == 2:
                un_vars.append(j)
        for j in range(STATE_COLS, TOTAL_COLS):
            if L_mat[r][j] == 1:
                un_vars.append(j)
        row_to_unknowns[r] = un_vars
        unknown_vars.update(un_vars)

    # 找出未知数所在的 S 盒
    involved_sboxes = set()
    for v in unknown_vars:
        if v < STATE_COLS:
            r_idx = v // 128
            ind   = v % 128
            local = ind % 64           # 跨 x/y 都取该 side 内的位
            col   = local // 4         # 列号
            involved_sboxes.add((r_idx, col))

    # 4. Union-Find 聚类
    uf = UnionFind(list(unknown_vars))

    # 4a. 同一行内的所有未知数互连
    for r, un_vars in row_to_unknowns.items():
        if len(un_vars) > 1:
            first = un_vars[0]
            for other in un_vars[1:]:
                uf.union(first, other)

    # 4b. 同一 S 盒内的未知数互连（S 盒方程里 x 和 y 不独立）
    sbox_to_unknowns = {}
    for (r_idx, col) in involved_sboxes:
        sb_un_vars = []
        for i in range(NB_ROWS):
            x_var = r_idx * 128 + 4 * col + i
            y_var = r_idx * 128 + 64 + 4 * col + i
            if x_var in unknown_vars:
                sb_un_vars.append(x_var)
            if y_var in unknown_vars:
                sb_un_vars.append(y_var)
        sbox_to_unknowns[(r_idx, col)] = sb_un_vars
        if len(sb_un_vars) > 1:
            first = sb_un_vars[0]
            for other in sb_un_vars[1:]:
                uf.union(first, other)

    # 5. 输出各 cluster
    cluster_dict = {}
    for v in unknown_vars:
        root = uf.find(v)
        cluster_dict.setdefault(root, []).append(v)

    cons_str = []
    Z_lst = []
    mask_lst = []

    for cluster_id, root in enumerate(cluster_dict.keys()):
        c_un_vars = set(cluster_dict[root])
        c_rows    = [r for r, un_vars in row_to_unknowns.items() if c_un_vars.intersection(un_vars)]
        c_sboxes  = [sb for sb, sb_un_vars in sbox_to_unknowns.items() if c_un_vars.intersection(sb_un_vars)]

        c_active_vars_for_Z = set()

        # 5a. 线性层等式
        l_tmp = ""
        if c_rows:
            cons = L_original[c_rows, :]
            l_tmp = show_L_equ_RECT(cons, active_bit_dic, ROUNDS)
            for r in c_rows:
                for j in range(STATE_COLS):
                    if L_mat[r][j] == 3:
                        c_active_vars_for_Z.add(j)

        # 5b. S 盒等式
        SB_CONS = ""
        for sb in c_sboxes:
            sb_con, var_S = generate_sb_equ_RECT(sb[0], sb[1], active_bit_dic, dic_x)
            SB_CONS += sb_con + "\n"
            c_active_vars_for_Z.update(var_S)

        # 5c. 属于该簇的 mask 比特
        c_mask_vars = set()
        for v in c_un_vars:
            if v < STATE_COLS and str(v) in masked_bit_dic:
                c_mask_vars.add(v)
        for v in c_active_vars_for_Z:
            if v < STATE_COLS and str(v) in masked_bit_dic:
                c_mask_vars.add(v)

        c_mask_formatted = []
        for v in sorted(list(c_mask_vars)):
            r_idx = v // 128
            ind = v % 128
            c_mask_formatted.append(f"x_{r_idx}_{ind}" if ind < 64 else f"y_{r_idx}_{ind-64}")

        final_str = l_tmp + "\n" + SB_CONS
        if final_str.strip():
            cons_str.append(final_str.strip())
            Z = generate_Z(list(c_active_vars_for_Z), [], active_bit_dic)
            Z_lst.append(Z)
            mask_lst.append(c_mask_formatted)

    # 收集无未知数的纯独立物理等式（即整行只剩 active + key）
    if independent_rows:
        indep_active_vars_for_Z = set()
        indep_mask_vars = set()

        indep_cons_mat = L_original[independent_rows, :]
        indep_str = show_L_equ_RECT(indep_cons_mat, active_bit_dic, ROUNDS)

        for r_eq in independent_rows:
            for j in range(STATE_COLS):
                if L_original[r_eq][j] == 1:
                    if str(j) in active_bit_dic:
                        indep_active_vars_for_Z.add(j)
                    if str(j) in masked_bit_dic:
                        indep_mask_vars.add(j)

        indep_mask_formatted = []
        for v in sorted(list(indep_mask_vars)):
            r_idx = v // 128
            ind = v % 128
            indep_mask_formatted.append(f"x_{r_idx}_{ind}" if ind < 64 else f"y_{r_idx}_{ind-64}")

        Z_indep = generate_Z(list(indep_active_vars_for_Z), [], active_bit_dic)
        if indep_str.strip():
            cons_str.append(indep_str.strip())
            Z_lst.append(Z_indep)
            mask_lst.append(indep_mask_formatted)

    return cons_str, Z_lst, mask_lst


# ════════════════════════════════════════════════════════════════
# 辅助
# ════════════════════════════════════════════════════════════════
def generate_Z(var_S, active_vars, active_bit_dic):
    Z = {}
    for v in set(var_S + active_vars):
        if str(v) in active_bit_dic:
            r = v // 128
            ind = v % 128
            name = f"x_{r}_{ind}" if ind < 64 else f"y_{r}_{ind-64}"
            Z[name] = {active_bit_dic[str(v)]}
    return Z


def mask_trans(MASK):
    MSK_LST = []
    side_map = {'x': 0, 'y': 1}
    for m_b in MASK:
        parts = m_b.split('_')
        # parts = ['x'/'y', round, flat_idx]
        MSK_LST.append((int(parts[1]), side_map[parts[0]], int(parts[2])))
    return MSK_LST


def extract_diff_trail_cell_RECT_from_diffs(diffs_list):
    """
    diffs_list[2r]  = 第 r 轮 SubColumn 输入差分（64-bit 整数）
    diffs_list[2r+1]= 第 r 轮 SubColumn 输出差分
    返回 diff_trail[r][side][col] = nibble，按 (val >> (4*col)) & 0xf。
    （这就是与 utils.py 一致的 LSB-first nibble 提取。）
    """
    nb_rounds = len(diffs_list) // 2
    diff_trail = [[[0] * NB_COLS for _ in range(2)] for _ in range(nb_rounds)]
    for r in range(nb_rounds):
        in_val, out_val = diffs_list[2 * r], diffs_list[2 * r + 1]
        for col in range(NB_COLS):
            diff_trail[r][0][col] = (in_val  >> (4 * col)) & 0xf
            diff_trail[r][1][col] = (out_val >> (4 * col)) & 0xf
    return diff_trail


# ════════════════════════════════════════════════════════════════
# Entry
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    diffs = diffs14_2

    ROUNDS = NB_ROUNDS
    THRESH = 1

    npy_path = f'./freq_msk/masks_freq_RECT_{NB_ROUNDS}RD_CORR{MIN_CORR}_T{THRESH}.npy'
    data = np.load(npy_path).tolist()

    diff_trail_cell = extract_diff_trail_cell_RECT_from_diffs(diffs)
    dic_x, dic_y    = creat_dic_RECT(diff_trail_cell)
    active_bit_dic  = get_active_bit(dic_x, dic_y)

    masked_bit_dic = active_bit_dic.copy()
    for r in range(ROUNDS):
        for s in range(2):
            for j in range(64):
                if data[r][s][j] == 1:
                    masked_bit_dic[str(r * 128 + s * 64 + j)] = 1

    L = genLinear_RECT(ROUNDS)
    L_mat = L.copy()

    cons_str, Z_lst, mask_lst = generate_constraints_RECT(
        L_mat, dic_x, active_bit_dic, masked_bit_dic, ROUNDS, L
    )

    FULL_MSK = []
    print("\n#========= 最终分离的独立 Cluster =========")
    for i in range(len(cons_str)):
        print(f"\n #[ Cluster {i} ]")
        print(f'CONS{i}="""\n{cons_str[i]}\n"""')
        print(f'Z{i}=', Z_lst[i])
        print(f'MASK{i}=', mask_trans(mask_lst[i]))
        FULL_MSK.append(mask_trans(mask_lst[i]))
        print("#----------------------------------------")