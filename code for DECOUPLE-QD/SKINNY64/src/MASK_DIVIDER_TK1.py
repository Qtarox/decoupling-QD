

import re
import os
import numpy as np

from GEN_MAT.GEN_LINEAR import (
    Sbox, M_EQ, key_schedule,
    Global_mat, Global_mat_bit,
    show_L_equ_GIFT, show_L_equ_GIFT_extract,
    CELL_SIZE, HALF_STATE_BITS, FULL_STATE_BITS, KEY_BITS,
)
from utils import *  # NB_ROUNDS / MIN_CORR / THRESH / ADV_MODEL / SBOX_SIZE / extract_diff_trail_flat 等

# S-box 输入空间大小
SBOX_DOMAIN = 1 << CELL_SIZE   # = 256 for SKINNY-128


def extract_hidden_constraints(L_original, active_bit_dic, masked_bit_dic, ROUNDS):
    """
    用 GF(2) 上的高斯消元，把所有中间 / 密钥变量消去，提取出只涉及
    masked 与 active 状态比特的隐藏方程。逻辑与 64-bit 版完全一致，
    只是状态列宽改为 FULL_STATE_BITS * (ROUNDS+1)。
    """
    STATE_COLS = FULL_STATE_BITS * (ROUNDS + 1)
    TOTAL_COLS = np.shape(L_original)[1]

    keep_cols, elim_cols = [], []
    for j in range(STATE_COLS):
        if str(j) in masked_bit_dic or str(j) in active_bit_dic:
            keep_cols.append(j)
        else:
            elim_cols.append(j)
    for j in range(STATE_COLS, TOTAL_COLS):
        elim_cols.append(j)   # 所有 key 列也消去

    col_order = elim_cols + keep_cols
    mat = L_original[:, col_order].copy()

    rows, _ = mat.shape
    elim_count = len(elim_cols)
    print(f"[*] GF(2) 高斯消元: 试图消去 {elim_count} 个冗余变量 ...")

    r = 0
    for c in range(elim_count):
        if r >= rows:
            break
        pivot = r
        while pivot < rows and mat[pivot, c] == 0:
            pivot += 1
        if pivot == rows:
            continue
        mat[[r, pivot]] = mat[[pivot, r]]
        for i in range(rows):
            if i != r and mat[i, c] == 1:
                mat[i] = (mat[i] + mat[r]) % 2
        r += 1

    pure_equations = []
    for i in range(rows):
        if np.all(mat[i, :elim_count] == 0) and np.any(mat[i, elim_count:] == 1):
            pure_row = np.zeros(TOTAL_COLS, dtype=int)
            pure_row[keep_cols] = mat[i, elim_count:]
            pure_equations.append(pure_row)

    print(f"[+] 提取出 {len(pure_equations)} 条纯状态约束")
    return np.array(pure_equations)


def load_masks_from_file(file_path, round_num):
    """
    解析每轮 Side 0 / Side 1 的 bit 掩码到 [round_num][2][HALF_STATE_BITS] 列表。
    与旧版 64 bit 唯一的区别在于每个 Side 现在期望 HALF_STATE_BITS=128 位。
    """
    mask_list = [[[0 for _ in range(HALF_STATE_BITS)] for _ in range(2)] for _ in range(round_num)]
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
                    if bit_pos < HALF_STATE_BITS:
                        mask_list[r_idx][s_idx][bit_pos] = int(bit_val)
        print(f"成功加载 {len(round_blocks)} 轮数据。")
        return mask_list
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


def xddt_list(input_diff, output_diff):
    """对 8-bit S-box 的 X-DDT：穷举输入空间 0..255。"""
    res = []
    if input_diff == 0:
        return res
    for x in range(SBOX_DOMAIN):
        x1 = x
        x2 = x ^ input_diff
        y1 = Sbox[x1]
        y2 = Sbox[x2]
        if (y1 ^ y2) == output_diff:
            res.append(x)
    return res


def yddt_list(input_diff, output_diff):
    """对 8-bit S-box 的 Y-DDT。"""
    res = []
    if input_diff == 0:
        return res
    for x in range(SBOX_DOMAIN):
        x1 = x
        x2 = x ^ input_diff
        y1 = Sbox[x1]
        y2 = Sbox[x2]
        if (y1 ^ y2) == output_diff:
            res.append(y1)
    return res


def creat_dic_GIFT(rd):
    """
    rd[r][0][i] / rd[r][1][i]: 第 r 轮第 i 个 cell 的输入/输出差分（整数，
    SKINNY-128 下取值 0..255）。
    """
    x_dic, y_dic = {}, {}
    print("running!")
    for r in range(len(rd)):
        for cell_idx in range(16):
            if rd[r][0][cell_idx] == 0:
                continue
            x_tmp = f"x_{r}_{cell_idx}"
            y_tmp = f"y_{r}_{cell_idx}"
            x_dic[x_tmp] = xddt_list(rd[r][0][cell_idx], rd[r][1][cell_idx]).copy()
            y_dic[y_tmp] = yddt_list(rd[r][0][cell_idx], rd[r][1][cell_idx]).copy()
    return x_dic, y_dic


def get_active_bit(x_dic, y_dic):
    """
    对每个 cell 的 X-DDT / Y-DDT 取值集合，逐 bit 检查是否所有元素该 bit 都相同；
    若相同，则该 bit 在该轨迹上是 "active"（取值固定），把它登记到结果字典中。
    SKINNY-128 下每 cell 8 个 bit，循环到 CELL_SIZE。
    """
    res = {}

    pattern_x = r"x_(\d+)_([\d]+)"
    for key, X_lst in x_dic.items():
        m = re.match(pattern_x, key)
        rn = int(m.group(1))
        cell_idx = int(m.group(2))
        for i in range(CELL_SIZE):
            initial = (X_lst[0] >> i) & 1
            active_flag = all(((x >> i) & 1) == initial for x in X_lst)
            if active_flag:
                # 全局 bit 索引：x 段从 rn*FULL_STATE_BITS 起
                global_idx = rn * FULL_STATE_BITS + cell_idx * CELL_SIZE + i
                res[str(global_idx)] = initial

    pattern_y = r"y_(\d+)_([\d]+)"
    for key, Y_lst in y_dic.items():
        m = re.match(pattern_y, key)
        rn = int(m.group(1))
        cell_idx = int(m.group(2))
        for i in range(CELL_SIZE):
            initial = (Y_lst[0] >> i) & 1
            active_flag = all(((y >> i) & 1) == initial for y in Y_lst)
            if active_flag:
                # y 段从 rn*FULL_STATE_BITS + HALF_STATE_BITS 起
                global_idx = rn * FULL_STATE_BITS + HALF_STATE_BITS + cell_idx * CELL_SIZE + i
                res[str(global_idx)] = initial
    return res


def get_var(L_mat, rounds):
    """
    收集所有 "masked" (值=2) 列；并把同一个 S-box (相同 round / cell) 的
    masked 比特放到一个 relation 组里。
    """
    var_lst = []
    var_relation = []
    state_cols = FULL_STATE_BITS * (rounds + 1)
    for r in range(np.shape(L_mat)[0]):
        tmp_rela = []
        for j in range(state_cols):
            if L_mat[r][j] == 2:
                var_lst.append(j)
                tmp_rela.append(j)
        var_relation.append(tmp_rela)
    var_lst = sorted(set(var_lst))
    print("Variables corresponding to masked bits:", var_lst)

    sb_lst = {}
    for v in var_lst:
        r = v // FULL_STATE_BITS
        # 在一轮内：先判 x/y 半态，再按 CELL_SIZE 取 cell 索引
        within = v % FULL_STATE_BITS
        cell_in_half = (within % HALF_STATE_BITS) // CELL_SIZE
        sb_lst.setdefault((r, cell_in_half), []).append(v)

    for sb, members in sb_lst.items():
        if len(members) > 1:
            var_relation.append(members)
    return var_lst, var_relation


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


def generate_sb_equ(r, sb_ind, active_bit_dic, dic_x):
    """
    打印第 r 轮第 sb_ind 个 S-box 的输入/输出 cell 方程；
    SKINNY-128 下每个 cell 有 CELL_SIZE=8 个 bit 变量。
    返回构造好的字符串与其涉及的 active 变量列表。
    """
    x_cons = ""
    key = f"x_{r}_{sb_ind}"
    if key in dic_x:
        x_cons = "_["
        for x_v in dic_x[key]:
            x_cons += f"{x_v}_"
        x_cons = x_cons[:-1] + "]"

    l_tmp = f"S{x_cons}("
    var_lst = []

    for i in range(CELL_SIZE):
        l_tmp += f"x_{r}_{sb_ind * CELL_SIZE + i},"
        global_idx = r * FULL_STATE_BITS + sb_ind * CELL_SIZE + i
        if str(global_idx) in active_bit_dic:
            var_lst.append(global_idx)
    l_tmp = l_tmp[:-1] + ") = ("

    for i in range(CELL_SIZE):
        l_tmp += f"y_{r}_{sb_ind * CELL_SIZE + i},"
        global_idx = r * FULL_STATE_BITS + HALF_STATE_BITS + sb_ind * CELL_SIZE + i
        if str(global_idx) in active_bit_dic:
            var_lst.append(global_idx)
    l_tmp = l_tmp[:-1] + ")"
    print(l_tmp)
    return l_tmp, var_lst


def get_clusters(var_lst, var_relation):
    var_lst = list(set(var_lst))
    uf = UnionFind(var_lst)
    for r in var_relation:
        if len(r) > 1:
            first = r[0]
            for other in r[1:]:
                uf.union(first, other)
    clusters = {}
    for v in var_lst:
        root = uf.find(v)
        clusters.setdefault(root, []).append(v)
    return list(clusters.values())


def generate_constraints(L_mat, dic_x, active_bit_dic, masked_bit_dic, ROUNDS, L_original):
    STATE_COLS = FULL_STATE_BITS * (ROUNDS + 1)
    TOTAL_COLS = np.shape(L_mat)[1]

    # 1) 把原矩阵里 state 段的 1 重标记为 2 (masked) 或 3 (active)
    for r in range(np.shape(L_mat)[0]):
        for j in range(STATE_COLS):
            if L_mat[r][j] == 1:
                if str(j) in masked_bit_dic:
                    if str(j) in active_bit_dic:
                        L_mat[r][j] = 3
                    else:
                        L_mat[r][j] = 2

    # 2) 高斯消元抽取隐藏约束
    hidden_eqs = extract_hidden_constraints(L_original, active_bit_dic, masked_bit_dic, ROUNDS)
    if len(hidden_eqs) > 0:
        L_original = np.vstack((L_original, hidden_eqs))
        L_mat_hidden = hidden_eqs.copy()
        for i in range(len(hidden_eqs)):
            for j in range(STATE_COLS):
                if L_mat_hidden[i][j] == 1:
                    if str(j) in active_bit_dic:
                        L_mat_hidden[i][j] = 3
                    elif str(j) in masked_bit_dic:
                        L_mat_hidden[i][j] = 2
        L_mat = np.vstack((L_mat, L_mat_hidden))

    # 3) 找出有效行
    target_rows = []
    for r in range(np.shape(L_mat)[0]):
        state_part = L_mat[r, :STATE_COLS]
        if np.any(state_part == 1):
            continue
        if np.all(L_mat[r] == 0):
            continue
        if np.any((state_part == 2) | (state_part == 3)):
            target_rows.append(r)

    # 4) 收集未知数（masked state bits + 涉及的所有 key bits）
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

    # 5) 找出涉及的 S-box (round, cell_in_half)
    involved_sboxes = set()
    for v in unknown_vars:
        if v < STATE_COLS:
            r_idx = v // FULL_STATE_BITS
            within = v % FULL_STATE_BITS
            cell_in_half = (within % HALF_STATE_BITS) // CELL_SIZE
            involved_sboxes.add((r_idx, cell_in_half))

    # 6) Union-Find 聚类
    uf = UnionFind(list(unknown_vars))
    for r, un_vars in row_to_unknowns.items():
        if len(un_vars) > 1:
            first = un_vars[0]
            for other in un_vars[1:]:
                uf.union(first, other)

    # 把同一个 S-box 的未知变量并到一起
    sbox_to_unknowns = {}
    for (r_idx, cell_in_half) in involved_sboxes:
        sb_un_vars = []
        for i in range(CELL_SIZE):
            x_var = r_idx * FULL_STATE_BITS + cell_in_half * CELL_SIZE + i
            y_var = r_idx * FULL_STATE_BITS + HALF_STATE_BITS + cell_in_half * CELL_SIZE + i
            if x_var in unknown_vars:
                sb_un_vars.append(x_var)
            if y_var in unknown_vars:
                sb_un_vars.append(y_var)
        sbox_to_unknowns[(r_idx, cell_in_half)] = sb_un_vars
        if len(sb_un_vars) > 1:
            first = sb_un_vars[0]
            for other in sb_un_vars[1:]:
                uf.union(first, other)

    # 7) 按 cluster 输出
    cluster_dict = {}
    for v in unknown_vars:
        root = uf.find(v)
        cluster_dict.setdefault(root, []).append(v)

    cons_str = []
    Z_lst = []

    for cluster_id, root in enumerate(cluster_dict.keys()):
        c_un_vars = set(cluster_dict[root])
        c_rows = [r for r, un_vars in row_to_unknowns.items() if c_un_vars.intersection(un_vars)]
        c_sboxes = [sb for sb, sb_un_vars in sbox_to_unknowns.items() if c_un_vars.intersection(sb_un_vars)]
        c_active_vars_for_Z = set()

        l_tmp = ""
        if c_rows:
            cons = L_original[c_rows, :]
            l_tmp = show_L_equ_GIFT(cons, active_bit_dic, ROUNDS)
            for r in c_rows:
                for j in range(STATE_COLS):
                    if L_mat[r][j] == 3:
                        c_active_vars_for_Z.add(j)

        SB_CONS = ""
        for sb in c_sboxes:
            sb_con, var_S = generate_sb_equ(sb[0], sb[1], active_bit_dic, dic_x)
            SB_CONS += sb_con + "\n"
            c_active_vars_for_Z.update(var_S)

        final_str = l_tmp + "\n" + SB_CONS
        if final_str.strip():
            cons_str.append(final_str.strip())
            Z = generate_Z(list(c_active_vars_for_Z), [], active_bit_dic)
            Z_lst.append(Z)

    return cons_str, Z_lst


def generate_Z(var_S, active_vars, active_bit_dic):
    """根据全局 bit 索引重建可读的 x_r_b / y_r_b 标签。"""
    Z = {}
    for v in var_S:
        if str(v) in active_bit_dic:
            r = v // FULL_STATE_BITS
            ind = v % FULL_STATE_BITS
            if ind < HALF_STATE_BITS:
                Z[f"x_{r}_{ind}"] = {active_bit_dic[str(v)]}
            else:
                Z[f"y_{r}_{ind - HALF_STATE_BITS}"] = {active_bit_dic[str(v)]}
    for v in active_vars:
        r = v // FULL_STATE_BITS
        ind = v % FULL_STATE_BITS
        if ind < HALF_STATE_BITS:
            Z[f"x_{r}_{ind}"] = {active_bit_dic[str(v)]}
        else:
            Z[f"y_{r}_{ind - HALF_STATE_BITS}"] = {active_bit_dic[str(v)]}
    print("Variables involved in the constraints (Z):", Z)
    return Z


if __name__ == "__main__":
    ROUNDS = NB_ROUNDS

    # 加载 freq mask（形状: [ROUNDS][2][HALF_STATE_BITS]）
    data = np.load(
        f'./freq_msk/masks_freq_{NB_ROUNDS}RD_CORR{MIN_CORR}_T{THRESH}.npy'
    ).tolist()
    print("Loaded mask data:", data)
    print("Number of rounds loaded:", len(data))
    print("================================")

    DIFF_TRAIL_FILE = f"../data/differential_trails/SKINNY{16 * SBOX_SIZE}_{ADV_MODEL}_R{NB_ROUNDS}.txt"
    diff_trail = extract_diff_trail_flat(DIFF_TRAIL_FILE, ROUNDS)
    print("trails ", diff_trail)

    dic_x, dic_y = creat_dic_GIFT(diff_trail)
    print("x_dic:", dic_x)
    print("y_dic:", dic_y)

    active_bit_dic = get_active_bit(dic_x, dic_y)
    print("active_bit_dic:", active_bit_dic)

    masked_bit_dic = active_bit_dic.copy()

    # 把 freq mask 标记到 masked_bit_dic 中
    # 每个 round 有 2 个 side (x/y 半态)，每个 side HALF_STATE_BITS 位
    for r in range(ROUNDS):
        for s in range(2):
            for i in range(HALF_STATE_BITS):
                if data[r][s][i] == 1:
                    global_idx = r * FULL_STATE_BITS + s * HALF_STATE_BITS + i
                    masked_bit_dic[str(global_idx)] = 1
    print("masked_bit_dic after merging masks:", masked_bit_dic)

    # ---------------- 生成约束并写文件 ----------------
    L = Global_mat_bit(ROUNDS)
    L_mat = L.copy()

    cons_str, Z_lst = generate_constraints(
        L_mat, dic_x, active_bit_dic, masked_bit_dic, ROUNDS, L
    )

    out_file = f"./constraints/CONS_{NB_ROUNDS}R_{MIN_CORR}_TH{THRESH}.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    open(out_file, 'w', encoding='utf-8').close()

    print("\n#========= 最终分离的独立 Cluster =========")
    cons_lst = "dic_cons={\n \n"
    for i in range(len(cons_str)):
        res_str = f'CONS{i}="""\n' + str(cons_str[i]) + '"""'
        res_str += f"\nZ{i}=" + str(Z_lst[i]) + "\n\n"

        print(f"\n#[ Cluster {i} ]")
        print(f'CONS{i}="""\n', cons_str[i])
        print('"""')
        print(f"Z{i}=", Z_lst[i])
        print("#----------------------------------------")

        cons_lst += f"'CONS{i}': (CONS{i},Z{i}),\n"
        with open(out_file, 'a', encoding='utf-8') as fh:
            fh.write(res_str)

    cons_lst += "}"
    with open(out_file, 'a', encoding='utf-8') as fh:
        fh.write(cons_lst)