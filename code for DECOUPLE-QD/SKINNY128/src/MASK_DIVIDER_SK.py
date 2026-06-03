"""
MASK_DIVIDER_SK.py  --  SKINNY single-key (SK) 模式
====================================================

差别点（相对 TK1 版本）
-----------------------
SK 模式下密钥是固定的、对攻击者未知但对所有 round 都已经"用完"。在 quasi-
differential 约束抽取这一步里，攻击者并不需要把每轮的 round-key 比特作为
独立未知数来求解 —— 它们与 state 比特之间的 XOR 关系等价于一个**常量项**，
不影响 cluster 划分。

具体处理：
1) `Global_mat_bit` 不再在矩阵右侧追加 KEY_BITS 列 —— 矩阵宽度只到
   `FULL_STATE_BITS * (R+1)`。
2) 原本 `k < 40` 中 `32..39` 这段表示"该 cell 方程引入了一个 round-key
   贡献"，SK 下直接跳过（不向矩阵里写任何 1）。
3) 因为没有 key 列，cluster 抽取里所有 "TOTAL_COLS > STATE_COLS" 相关
   的分支自然 no-op，结果就只包含 state-state 的 cluster。
"""

import os
import re
import numpy as np

from utils import (
    NB_ROUNDS, MIN_CORR, THRESH, ADV_MODEL, SBOX_SIZE,
    extract_diff_trail_flat,
)

# ---------------- 配置常量 -------------------------------------------------
# 这里**不再**从 GEN_LINEAR 导入 KEY_BITS / Global_mat_bit；
# SK 模式下我们用本文件里专门的 zero-key 版本。
CELL_SIZE       = SBOX_SIZE                    # 4 (SKINNY-64) 或 8 (SKINNY-128)
HALF_STATE_BITS = 16 * CELL_SIZE
FULL_STATE_BITS = 2 * HALF_STATE_BITS
KEY_BITS        = 0                            # SK：无 key 列
SBOX_DOMAIN     = 1 << CELL_SIZE

# S-box（与 GEN_LINEAR 保持一致）
from GEN_MAT.GEN_LINEAR import Sbox, M_EQ


# ============== 0) SK 专用：构造 bit-level 线性方程矩阵 =====================
def Global_mat_bit(round_num):
    """
    SK 模式版本：矩阵只有 state 列，没有 key 列。
    每轮 M_EQ 的 cell 方程展开成 CELL_SIZE 条 bit 方程。
    K 段 (k >= 32) 直接跳过，相当于把 round-key 贡献当作已知常量丢弃。
    """
    num_rows = HALF_STATE_BITS * round_num
    num_cols = FULL_STATE_BITS * (round_num + 1)        # 无 key 列
    res = np.zeros((num_rows, num_cols), dtype=int)

    for i in range(num_rows):
        equ_num_bit  = i % HALF_STATE_BITS
        rn           = i // HALF_STATE_BITS
        equ_num_cell = equ_num_bit // CELL_SIZE
        bit_idx      = equ_num_bit % CELL_SIZE

        for k in range(40):
            if M_EQ[equ_num_cell][k] != 1:
                continue
            if k < 16:        # x_{r+1}
                res[i][(rn + 1) * FULL_STATE_BITS + k * CELL_SIZE + bit_idx] = 1
            elif k < 32:      # y_r
                res[i][rn * FULL_STATE_BITS + k * CELL_SIZE + bit_idx] = 1
            else:             # round-key —— SK 模式：跳过
                pass
    return res


# ================ 1) GF(2) 高斯消元，抽取隐藏纯状态约束 ====================
def extract_hidden_constraints(L_original, active_bit_dic, masked_bit_dic, ROUNDS):
    STATE_COLS = FULL_STATE_BITS * (ROUNDS + 1)
    TOTAL_COLS = L_original.shape[1]                    # SK: == STATE_COLS

    keep_cols, elim_cols = [], []
    for j in range(STATE_COLS):
        if str(j) in masked_bit_dic or str(j) in active_bit_dic:
            keep_cols.append(j)
        else:
            elim_cols.append(j)
    # SK: 没有 key 列要消，TOTAL_COLS == STATE_COLS，下面循环自然为空
    for j in range(STATE_COLS, TOTAL_COLS):
        elim_cols.append(j)

    col_order = elim_cols + keep_cols
    mat = L_original[:, col_order].astype(np.uint8).copy()
    rows = mat.shape[0]
    elim_count = len(elim_cols)
    print(f"[SK] GF(2) 高斯消元: 消去 {elim_count} 个冗余变量 ...")

    r = 0
    for c in range(elim_count):
        if r >= rows:
            break
        nz = np.flatnonzero(mat[r:, c])
        if len(nz) == 0:
            continue
        piv = r + nz[0]
        if piv != r:
            mat[[r, piv]] = mat[[piv, r]]
        tgts = np.flatnonzero(mat[:, c])
        tgts = tgts[tgts != r]
        if len(tgts):
            mat[tgts] ^= mat[r]
        r += 1

    front_zero = ~mat[:, :elim_count].any(axis=1)
    back_nz    = mat[:, elim_count:].any(axis=1)
    chosen = np.flatnonzero(front_zero & back_nz)

    pure = np.zeros((len(chosen), TOTAL_COLS), dtype=int)
    pure[:, keep_cols] = mat[np.ix_(chosen, np.arange(elim_count, mat.shape[1]))]
    print(f"[SK] 提取出 {len(pure)} 条纯状态约束")
    return pure


# ================ 2) Sbox DDT 推导 active bit ===============================
def xddt_list(input_diff, output_diff):
    if input_diff == 0:
        return []
    return [x for x in range(SBOX_DOMAIN)
            if (Sbox[x] ^ Sbox[x ^ input_diff]) == output_diff]


def yddt_list(input_diff, output_diff):
    if input_diff == 0:
        return []
    return [Sbox[x] for x in range(SBOX_DOMAIN)
            if (Sbox[x] ^ Sbox[x ^ input_diff]) == output_diff]


def creat_dic_GIFT(rd):
    x_dic, y_dic = {}, {}
    for r in range(len(rd)):
        for c in range(16):
            if rd[r][0][c] == 0:
                continue
            x_dic[f"x_{r}_{c}"] = xddt_list(rd[r][0][c], rd[r][1][c])
            y_dic[f"y_{r}_{c}"] = yddt_list(rd[r][0][c], rd[r][1][c])
    return x_dic, y_dic


def get_active_bit(x_dic, y_dic):
    res = {}
    for key, lst in x_dic.items():
        m = re.match(r"x_(\d+)_(\d+)", key)
        rn, c = int(m.group(1)), int(m.group(2))
        for i in range(CELL_SIZE):
            init = (lst[0] >> i) & 1
            if all(((x >> i) & 1) == init for x in lst):
                res[str(rn * FULL_STATE_BITS + c * CELL_SIZE + i)] = init
    for key, lst in y_dic.items():
        m = re.match(r"y_(\d+)_(\d+)", key)
        rn, c = int(m.group(1)), int(m.group(2))
        for i in range(CELL_SIZE):
            init = (lst[0] >> i) & 1
            if all(((y >> i) & 1) == init for y in lst):
                res[str(rn * FULL_STATE_BITS + HALF_STATE_BITS + c * CELL_SIZE + i)] = init
    return res


# ================ 3) 打印方程（state-only） =================================
def _fmt_state_var(j):
    rn = j // FULL_STATE_BITS
    ind = j % FULL_STATE_BITS
    return (f"x_{rn}_{ind}" if ind < HALF_STATE_BITS
            else f"y_{rn}_{ind - HALF_STATE_BITS}")


def show_L_equ(lmat, active_bit_dic, ROUNDS):
    """SK 版：方程里不会出现 k_*。"""
    STATE_COLS = FULL_STATE_BITS * (ROUNDS + 1)
    L = ""
    for i in range(lmat.shape[0]):
        l_tmp = ""
        for j in range(STATE_COLS):
            if lmat[i][j] == 1:
                var = _fmt_state_var(j)
                l_tmp += f" + [{var}]" if str(j) in active_bit_dic else f" + {var}"
        l_tmp += " = 0"
        print(l_tmp)
        L += l_tmp + "\n"
    return L


# ================ 4) Sbox 行内方程 ==========================================
def generate_sb_equ(r, sb_ind, active_bit_dic, dic_x):
    x_cons = ""
    key = f"x_{r}_{sb_ind}"
    if key in dic_x:
        x_cons = "_[" + "_".join(str(v) for v in dic_x[key]) + "]"
    parts_in, parts_out, var_lst = [], [], []
    for i in range(CELL_SIZE):
        parts_in.append(f"x_{r}_{sb_ind * CELL_SIZE + i}")
        g = r * FULL_STATE_BITS + sb_ind * CELL_SIZE + i
        if str(g) in active_bit_dic:
            var_lst.append(g)
    for i in range(CELL_SIZE):
        parts_out.append(f"y_{r}_{sb_ind * CELL_SIZE + i}")
        g = r * FULL_STATE_BITS + HALF_STATE_BITS + sb_ind * CELL_SIZE + i
        if str(g) in active_bit_dic:
            var_lst.append(g)
    l_tmp = f"S{x_cons}({','.join(parts_in)}) = ({','.join(parts_out)})"
    print(l_tmp)
    return l_tmp, var_lst


# ================ 5) Union-Find =============================================
class UnionFind:
    def __init__(self, elements):
        self.parent = {el: el for el in elements}

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj


# ================ 6) cluster 约束生成 =======================================
def generate_constraints(L_mat, dic_x, active_bit_dic, masked_bit_dic,
                         ROUNDS, L_original):
    STATE_COLS = FULL_STATE_BITS * (ROUNDS + 1)
    TOTAL_COLS = L_mat.shape[1]

    # ---- 染色：1 -> 2(masked) / 3(active) (numpy 加速) ----
    state = L_mat[:, :STATE_COLS]
    masked = np.zeros(STATE_COLS, dtype=bool)
    active = np.zeros(STATE_COLS, dtype=bool)
    for k in masked_bit_dic:
        j = int(k)
        if j < STATE_COLS:
            masked[j] = True
    for k in active_bit_dic:
        j = int(k)
        if j < STATE_COLS:
            active[j] = True
    ones = (state == 1)
    state[ones & active[None, :]] = 3
    state[ones & masked[None, :] & ~active[None, :]] = 2
    L_mat[:, :STATE_COLS] = state

    # ---- 抽 hidden constraint ----
    hidden = extract_hidden_constraints(L_original, active_bit_dic,
                                        masked_bit_dic, ROUNDS)
    if len(hidden) > 0:
        L_original = np.vstack((L_original, hidden))
        lm_h = hidden.copy()
        for i in range(len(hidden)):
            for j in range(STATE_COLS):
                if lm_h[i][j] == 1:
                    if str(j) in active_bit_dic:
                        lm_h[i][j] = 3
                    elif str(j) in masked_bit_dic:
                        lm_h[i][j] = 2
        L_mat = np.vstack((L_mat, lm_h))

    # ---- 选目标行 ----
    target_rows = []
    for r in range(L_mat.shape[0]):
        sp = L_mat[r, :STATE_COLS]
        if np.any(sp == 1):           # 仍有不该出现的 raw 1，跳过
            continue
        if np.all(L_mat[r] == 0):
            continue
        if np.any((sp == 2) | (sp == 3)):
            target_rows.append(r)

    # ---- 收集未知数（SK 模式下只有 masked state bits） ----
    unknown_vars = set()
    row_to_unknowns = {}
    for r in target_rows:
        un = [j for j in range(STATE_COLS) if L_mat[r][j] == 2]
        # SK: 没有 key 列要扫
        row_to_unknowns[r] = un
        unknown_vars.update(un)

    involved_sb = set()
    for v in unknown_vars:
        r_idx = v // FULL_STATE_BITS
        within = v % FULL_STATE_BITS
        cell_in_half = (within % HALF_STATE_BITS) // CELL_SIZE
        involved_sb.add((r_idx, cell_in_half))

    uf = UnionFind(list(unknown_vars))
    for r, un in row_to_unknowns.items():
        for o in un[1:]:
            uf.union(un[0], o)

    sb_to_un = {}
    for (rd, c) in involved_sb:
        sb_un = []
        for i in range(CELL_SIZE):
            xv = rd * FULL_STATE_BITS + c * CELL_SIZE + i
            yv = rd * FULL_STATE_BITS + HALF_STATE_BITS + c * CELL_SIZE + i
            if xv in unknown_vars: sb_un.append(xv)
            if yv in unknown_vars: sb_un.append(yv)
        sb_to_un[(rd, c)] = sb_un
        for o in sb_un[1:]:
            uf.union(sb_un[0], o)

    cluster_dict = {}
    for v in unknown_vars:
        cluster_dict.setdefault(uf.find(v), []).append(v)

    cons_str, Z_lst = [], []
    for root, members in cluster_dict.items():
        cset = set(members)
        c_rows = [r for r, un in row_to_unknowns.items() if cset & set(un)]
        c_sb   = [sb for sb, un in sb_to_un.items() if cset & set(un)]
        c_active_for_Z = set()

        l_tmp = ""
        if c_rows:
            l_tmp = show_L_equ(L_original[c_rows, :], active_bit_dic, ROUNDS)
            for r in c_rows:
                for j in range(STATE_COLS):
                    if L_mat[r][j] == 3:
                        c_active_for_Z.add(j)

        SB_CONS = ""
        for sb in c_sb:
            sb_con, var_S = generate_sb_equ(sb[0], sb[1], active_bit_dic, dic_x)
            SB_CONS += sb_con + "\n"
            c_active_for_Z.update(var_S)

        final = (l_tmp + "\n" + SB_CONS).strip()
        if final:
            cons_str.append(final)
            Z_lst.append(generate_Z(list(c_active_for_Z), [], active_bit_dic))
    return cons_str, Z_lst


def generate_Z(var_S, active_vars, active_bit_dic):
    Z = {}
    for v in var_S:
        if str(v) in active_bit_dic:
            rn = v // FULL_STATE_BITS
            ind = v % FULL_STATE_BITS
            label = (f"x_{rn}_{ind}" if ind < HALF_STATE_BITS
                     else f"y_{rn}_{ind - HALF_STATE_BITS}")
            Z[label] = {active_bit_dic[str(v)]}
    return Z


# ================ 7) self-test 入口 =========================================
if __name__ == "__main__":
    assert ADV_MODEL == "SK", \
        f"MASK_DIVIDER_SK 被在 ADV_MODEL={ADV_MODEL} 下调用，请检查 utils 配置。"
    print("MASK_DIVIDER_SK 已加载（库模式），请用 constraint_collector.py 驱动。")