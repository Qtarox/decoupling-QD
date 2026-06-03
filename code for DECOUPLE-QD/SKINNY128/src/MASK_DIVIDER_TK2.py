"""
MASK_DIVIDER_TK2.py  --  SKINNY TK2 模式 (2 层 tweakey)
========================================================

Key schedule (SKINNY 官方规范)
-------------------------------
TK2 下 master tweakey 有两层 TK1, TK2，各 16 个 cell。每轮 KS:
  1) 对两层都做 cell 级 PT 置换:  new[i] = old[PT[i]]
  2) 对 TK2 的 **top 2 rows** (cell 0..7) 的每个 cell 做内部 LFSR2
     —— 注意 LFSR 只作用在 top 2 rows！TK1 不做 LFSR.

⚠️ 关键修正 (相对于旧版本):
旧版本错误地以为 master TK2 的每个 cell 在 round r 处经历了 r 次 LFSR2,
事实上一个 master cell 沿 PT 轨道行走, 只有当它处于 top 2 rows 时才被
LFSR 作用一次. 因此在 round r 加到 state 的 round-key 上, master cell
经历的 LFSR2 次数为:
    a_r(rk_cell) = #{ k ∈ [0, r-1] : PT^k[rk_cell] < 8 }
其中 rk_cell ∈ [0, 8) 是 round-key cell 的位置, PT^r[rk_cell] = master cell.

由 PT 的结构, 对 SKINNY 而言, a_r(rk_cell) ≈ ⌈r/2⌉ (具体由轨道决定).

矩阵列布局
----------
    [ state_cols (16*CELL_SIZE * 2 * (R+1))
    | tk1_master (16 * CELL_SIZE)
    | tk2_master (16 * CELL_SIZE) ]
"""

import os
import re
import numpy as np

from utils import (
    NB_ROUNDS, MIN_CORR, THRESH, ADV_MODEL, SBOX_SIZE,
    extract_diff_trail_flat,
)
from GEN_MAT.GEN_LINEAR import Sbox, M_EQ


# --------- 配置常量 -------------------------------------------------------
CELL_SIZE       = SBOX_SIZE
HALF_STATE_BITS = 16 * CELL_SIZE
FULL_STATE_BITS = 2 * HALF_STATE_BITS
TK_LAYER_BITS   = 16 * CELL_SIZE           # 单层 master tweakey 的 bit 数
KEY_BITS        = 2 * TK_LAYER_BITS        # 两层加起来
SBOX_DOMAIN     = 1 << CELL_SIZE
PT              = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]


# ---------------- LFSR2 在 GF(2) 上的迭代矩阵 -----------------------------
def _lfsr2_step_matrix(n):
    """
    返回 n x n 的二元矩阵 M，使得 new = M @ old (mod 2)，向量按 LSB-first 排布
    （即 v[0] 是 LSB，v[n-1] 是 MSB）。

    SKINNY 官方论文 LFSR2 (MSB-first 表达，x0 是 LSB):
        4-bit:  (x3, x2, x1, x0) -> (x2, x1, x0, x3 ⊕ x2)
        8-bit:  (x7,..,x0)       -> (x6,..,x0, x7 ⊕ x5)

    换到 LSB-first:
        new[0]       = old[n-1] ⊕ old[FEED]
                       其中 FEED = n-2  (4-bit) 或 n-3 (8-bit)
        new[1..n-1]  = old[0..n-2]
    """
    if n == 4:
        feed = n - 2   # x3 ⊕ x2
    elif n == 8:
        feed = n - 3   # x7 ⊕ x5
    else:
        raise NotImplementedError(f"LFSR2 for CELL_SIZE={n} 尚未实现")

    M = np.zeros((n, n), dtype=np.uint8)
    M[0][n - 1] = 1
    M[0][feed]  = 1
    for i in range(1, n):
        M[i][i - 1] = 1
    return M


def _lfsr2_power(r):
    """返回 r 次 LFSR2 的累计映射矩阵 (n x n GF(2))."""
    n = CELL_SIZE
    M = np.eye(n, dtype=np.uint8)
    step = _lfsr2_step_matrix(n)
    for _ in range(r):
        M = (M @ step) % 2
    return M


# ---------------- ⚠️ 修正: 正确的 LFSR2 应用次数 -----------------------
def _lfsr2_applications(round_idx, rk_cell):
    """
    返回 master cell 在 round round_idx 处现身于 rk_cell (< 8) 之前
    经历的 LFSR2 实际应用次数.

    公式: a_r(i) = #{ k ∈ [0, r-1] : PT^k[i] < 8 }

    解释: master 在 round k 时所处的位置是 PT^{r-k}[i] (反向轨道),
    但用变量代换 j = r-k 即得 PT^j[i], j ∈ [0, r-1].
    只有当该位置 < 8 时 LFSR 才作用.
    """
    cnt = 0
    pos = rk_cell
    for _ in range(round_idx):
        if pos < 8:
            cnt += 1
        pos = PT[pos]
    return cnt


# 预计算: 对每 (round, rk_cell ∈ [0, 8)) 的累计 LFSR2 矩阵.
# round 范围 0..NB_ROUNDS-1, rk_cell 范围 0..7.
_LFSR2_FOR = [
    [_lfsr2_power(_lfsr2_applications(r, c)) for c in range(8)]
    for r in range(NB_ROUNDS)
]


def tk1_schedule(round_idx, cell_idx):
    """每轮 PT 一次, 应用 round_idx 次. 两层 tweakey 都共享同一套 PT 置换."""
    tmp = cell_idx
    for _ in range(round_idx):
        tmp = PT[tmp]
    return tmp


# ============== 工具: 把 0/1 矩阵的"行"做去重 ===============================
def _dedup_rows(mat):
    """去掉完全相同的行；保持首次出现顺序."""
    if mat.shape[0] == 0:
        return mat
    seen = set()
    keep_idx = []
    for i in range(mat.shape[0]):
        key = mat[i].tobytes()
        if key in seen:
            continue
        seen.add(key)
        keep_idx.append(i)
    return mat[keep_idx]


# ============== 0) 构造 bit-level 线性方程矩阵 =============================
def Global_mat_bit(round_num):
    """
    TK2 矩阵布局: [state | TK1 master | TK2 master]
    一个 round-key bit 同时收到两层 master 的贡献:
        RK_r[rk_cell][bit_idx]
          = TK1_master[mk_cell][bit_idx]                    (TK1: 无 LFSR)
          ⊕ (LFSR2^a · TK2_master[mk_cell])[bit_idx]        (TK2: a = a_r(rk_cell) 次 LFSR2)

    其中 mk_cell = PT^r[rk_cell], a = _lfsr2_applications(r, rk_cell).
    M_EQ 里 32..39 这 8 个位置代表 "该 cell 方程引入了 round-key", 我们把
    这一个 round-key bit 展开为上述 XOR 形式.
    """
    num_rows = HALF_STATE_BITS * round_num
    num_cols = FULL_STATE_BITS * (round_num + 1) + KEY_BITS
    res = np.zeros((num_rows, num_cols), dtype=int)
    state_base = FULL_STATE_BITS * (round_num + 1)
    tk1_base   = state_base
    tk2_base   = state_base + TK_LAYER_BITS

    for i in range(num_rows):
        equ_num_bit  = i % HALF_STATE_BITS
        rn           = i // HALF_STATE_BITS
        equ_num_cell = equ_num_bit // CELL_SIZE
        bit_idx      = equ_num_bit % CELL_SIZE

        for k in range(40):
            if M_EQ[equ_num_cell][k] != 1:
                continue
            if k < 16:                  # x_{r+1}
                res[i][(rn + 1) * FULL_STATE_BITS + k * CELL_SIZE + bit_idx] ^= 1
            elif k < 32:                # y_r
                res[i][rn * FULL_STATE_BITS + k * CELL_SIZE + bit_idx] ^= 1
            else:                       # round-key cell
                rk_cell = k - 32                        # 0..7
                mk_cell = tk1_schedule(rn, rk_cell)     # master cell index 0..15

                # ----- TK1 层: 直接拷贝 bit_idx (无 LFSR) -----
                res[i][tk1_base + mk_cell * CELL_SIZE + bit_idx] ^= 1

                # ----- TK2 层: 应用 a_r(rk_cell) 次 LFSR2 -----
                # bit_idx 位是 (LFSR2^a 的第 bit_idx 行) 与 master cell 的内积.
                lfsr_mat = _LFSR2_FOR[rn][rk_cell]
                lfsr_row = lfsr_mat[bit_idx]            # 长度 CELL_SIZE
                for src_bit in range(CELL_SIZE):
                    if lfsr_row[src_bit]:
                        res[i][tk2_base + mk_cell * CELL_SIZE + src_bit] ^= 1
                # 注意: M_EQ 中每个 equ 至多一个 k>=32 (round-key 不会混入多个),
                # 因此 break 是安全的.
                break
    return res


# ============== 1)-6) 抽取逻辑 ==============================================
def extract_hidden_constraints(L_original, active_bit_dic, masked_bit_dic, ROUNDS):
    """
    返回: 消去 "非 keep 状态列" 之后保留下来的有信息的方程行.
    支持两类输出 (合并在一个矩阵里):
      A) 仅含 (active + masked) 状态比特、key 全 0 的纯状态约束
      B) 含 (active + masked) 状态比特 + 任意 key 列的方程  (例如 [X]+[Y]+K=0)
    """
    STATE_COLS = FULL_STATE_BITS * (ROUNDS + 1)
    TOTAL_COLS = L_original.shape[1]
    KEY_COLS   = TOTAL_COLS - STATE_COLS

    keep_state, elim_state = [], []
    for j in range(STATE_COLS):
        if str(j) in masked_bit_dic or str(j) in active_bit_dic:
            keep_state.append(j)
        else:
            elim_state.append(j)
    key_cols_list = list(range(STATE_COLS, TOTAL_COLS))

    col_order = elim_state + keep_state + key_cols_list
    mat = L_original[:, col_order].astype(np.uint8).copy()
    rows = mat.shape[0]
    elim_count = len(elim_state)
    keep_count = len(keep_state)

    print(f"[*] GF(2) 消元: elim_state={elim_count} | "
          f"keep_state={keep_count} | key={KEY_COLS}")

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

    elim_zero = ~mat[:, :elim_count].any(axis=1)
    keep_seg  = mat[:, elim_count:elim_count + keep_count]
    key_seg   = mat[:, elim_count + keep_count:]

    has_keep  = keep_seg.any(axis=1)
    has_key   = key_seg.any(axis=1) if KEY_COLS > 0 else np.zeros(rows, dtype=bool)
    chosen = np.flatnonzero(elim_zero & (has_keep | has_key))

    pure = np.zeros((len(chosen), TOTAL_COLS), dtype=int)
    pure[:, keep_state] = keep_seg[chosen]
    if KEY_COLS > 0:
        pure[:, STATE_COLS:] = key_seg[chosen]

    # 自身去重
    before = len(pure)
    pure = _dedup_rows(pure)
    after = len(pure)

    if KEY_COLS > 0:
        cnt_with_key = int((pure[:, STATE_COLS:].any(axis=1)).sum())
    else:
        cnt_with_key = 0
    cnt_pure = after - cnt_with_key
    print(f"[*] hidden 候选 {before} 行, 自身去重后 {after} 行 "
          f"(纯状态={cnt_pure}, 含 key={cnt_with_key})")
    return pure


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


def _fmt_state_var(j):
    rn = j // FULL_STATE_BITS
    ind = j % FULL_STATE_BITS
    return (f"x_{rn}_{ind}" if ind < HALF_STATE_BITS
            else f"y_{rn}_{ind - HALF_STATE_BITS}")


def _fmt_key_var(rel_idx):
    """rel_idx 是从 STATE_COLS 起算的偏移;前 TK_LAYER_BITS 是 TK1, 后面是 TK2."""
    if rel_idx < TK_LAYER_BITS:
        return f"k1_{rel_idx}"
    return f"k2_{rel_idx - TK_LAYER_BITS}"


def show_L_equ(lmat, active_bit_dic, ROUNDS):
    STATE_COLS = FULL_STATE_BITS * (ROUNDS + 1)
    L = ""
    for i in range(lmat.shape[0]):
        l_tmp = ""
        for j in range(STATE_COLS):
            if lmat[i][j] == 1:
                var = _fmt_state_var(j)
                l_tmp += f" + [{var}]" if str(j) in active_bit_dic else f" + {var}"
        for k in range(KEY_BITS):
            if lmat[i][STATE_COLS + k] == 1:
                l_tmp += f" + {_fmt_key_var(k)}"
        l_tmp += " = 0"
        print(l_tmp)
        L += l_tmp + "\n"
    return L


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


def generate_constraints(L_mat, dic_x, active_bit_dic, masked_bit_dic,
                         ROUNDS, L_original):
    STATE_COLS = FULL_STATE_BITS * (ROUNDS + 1)
    TOTAL_COLS = L_mat.shape[1]

    # ---- 染色 (numpy 向量化) ----
    state = L_mat[:, :STATE_COLS]
    masked_mask = np.zeros(STATE_COLS, dtype=bool)
    active_mask = np.zeros(STATE_COLS, dtype=bool)
    for k in masked_bit_dic:
        j = int(k)
        if j < STATE_COLS:
            masked_mask[j] = True
    for k in active_bit_dic:
        j = int(k)
        if j < STATE_COLS:
            active_mask[j] = True
    ones = (state == 1)
    state[ones & active_mask[None, :]] = 3
    state[ones & masked_mask[None, :] & ~active_mask[None, :]] = 2
    L_mat[:, :STATE_COLS] = state

    # ---- 提 hidden 约束 ----
    hidden = extract_hidden_constraints(L_original, active_bit_dic,
                                        masked_bit_dic, ROUNDS)

    # ---- 把 hidden 里与 L_original 某行完全相同的去掉 ----
    if len(hidden) > 0:
        orig_set = set()
        for i in range(L_original.shape[0]):
            row = L_original[i]
            sp = row[:STATE_COLS]
            has_unkeep = False
            for j in np.flatnonzero(sp):
                if not (str(j) in masked_bit_dic or str(j) in active_bit_dic):
                    has_unkeep = True
                    break
            if has_unkeep:
                continue
            orig_set.add(row.tobytes())

        keep_mask = np.array(
            [hidden[i].tobytes() not in orig_set for i in range(hidden.shape[0])]
        )
        n_dup = int((~keep_mask).sum())
        hidden = hidden[keep_mask]
        print(f"[*] hidden 与 L_original 去重: 删除 {n_dup} 行, "
              f"剩 {hidden.shape[0]} 行")

    if len(hidden) > 0:
        L_original = np.vstack((L_original, hidden))
        lm_h = hidden.copy()
        h_state = lm_h[:, :STATE_COLS]
        h_ones = (h_state == 1)
        h_state[h_ones & active_mask[None, :]] = 3
        h_state[h_ones & masked_mask[None, :] & ~active_mask[None, :]] = 2
        lm_h[:, :STATE_COLS] = h_state
        L_mat = np.vstack((L_mat, lm_h))

    # ---- 选 target_rows ----
    target_rows = []
    n_skip_unclassified = 0
    for r in range(L_mat.shape[0]):
        sp = L_mat[r, :STATE_COLS]
        if np.any(sp == 1):
            n_skip_unclassified += 1
            continue
        if np.all(L_mat[r] == 0):
            continue
        has_state_info = np.any((sp == 2) | (sp == 3))
        has_key_info   = np.any(L_mat[r, STATE_COLS:] == 1) if TOTAL_COLS > STATE_COLS else False
        if has_state_info or has_key_info:
            target_rows.append(r)

    print(f"[*] target_rows={len(target_rows)}; 跳过未分类={n_skip_unclassified}")

    # ---- 收 unknown / active ----
    unknown_vars = set()
    row_unknowns = {}
    row_active   = {}

    for r in target_rows:
        un_m = [j for j in range(STATE_COLS) if L_mat[r][j] == 2]
        un_k = [j for j in range(STATE_COLS, TOTAL_COLS) if L_mat[r][j] == 1]
        ac   = [j for j in range(STATE_COLS) if L_mat[r][j] == 3]
        row_unknowns[r] = un_m + un_k
        row_active[r]   = ac
        unknown_vars.update(un_m + un_k)

    # ---- 涉及的 S-box ----
    involved_sb = set()
    for v in unknown_vars:
        if v < STATE_COLS:
            r_idx = v // FULL_STATE_BITS
            within = v % FULL_STATE_BITS
            cell_in_half = (within % HALF_STATE_BITS) // CELL_SIZE
            involved_sb.add((r_idx, cell_in_half))

    # ---- 并查集 ----
    uf = UnionFind(list(unknown_vars))
    for r, un in row_unknowns.items():
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

    # ---- cluster 收集 ----
    cluster_dict = {}
    for v in unknown_vars:
        cluster_dict.setdefault(uf.find(v), []).append(v)

    pure_active_rows = [r for r, un in row_unknowns.items()
                        if not un and row_active[r]]

    cons_str, Z_lst = [], []
    n_rows_used = 0

    for root, members in cluster_dict.items():
        cset = set(members)
        c_rows = [r for r, un in row_unknowns.items() if cset & set(un)]
        c_sb   = [sb for sb, un in sb_to_un.items() if cset & set(un)]
        c_active_for_Z = set()

        l_tmp = ""
        if c_rows:
            sub = L_original[c_rows, :]
            sub = _dedup_rows(sub)
            l_tmp = show_L_equ(sub, active_bit_dic, ROUNDS)
            n_rows_used += sub.shape[0]
            for r in c_rows:
                for j in row_active[r]:
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

    # 纯 active cluster
    if pure_active_rows:
        sub = L_original[pure_active_rows, :]
        sub = _dedup_rows(sub)
        l_tmp = show_L_equ(sub, active_bit_dic, ROUNDS)
        n_rows_used += sub.shape[0]
        active_for_Z = set()
        for r in pure_active_rows:
            for j in row_active[r]:
                active_for_Z.add(j)
        final = l_tmp.strip()
        if final:
            cons_str.append(final)
            Z_lst.append(generate_Z(list(active_for_Z), [], active_bit_dic))

    print(f"[*] 共 emit {len(cons_str)} 个 cluster, emit 行数 {n_rows_used}")
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


if __name__ == "__main__":
    assert ADV_MODEL == "TK2", \
        f"MASK_DIVIDER_TK2 在 ADV_MODEL={ADV_MODEL} 下被调用, 检查 utils 配置."
    print("MASK_DIVIDER_TK2 已加载 (库模式), 请用 constraint_collector.py 驱动.")

    # 简要 self-check
    print()
    print("=" * 60)
    print("LFSR2 应用次数自检 (NB_ROUNDS=%d, CELL_SIZE=%d):" % (NB_ROUNDS, CELL_SIZE))
    print("=" * 60)
    for c in range(8):
        cnts = [_lfsr2_applications(r, c) for r in range(NB_ROUNDS)]
        print(f"rk_cell={c}: a_r = {cnts}")