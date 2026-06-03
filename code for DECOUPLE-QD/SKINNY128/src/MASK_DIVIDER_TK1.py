import os
import re
import numpy as np
from utils import (
    NB_ROUNDS, MIN_CORR, THRESH, ADV_MODEL, SBOX_SIZE,
    extract_diff_trail_flat,
)
from GEN_MAT.GEN_LINEAR import Sbox, M_EQ
CELL_SIZE       = SBOX_SIZE
HALF_STATE_BITS = 16 * CELL_SIZE
FULL_STATE_BITS = 2 * HALF_STATE_BITS
KEY_BITS        = 16 * CELL_SIZE
SBOX_DOMAIN     = 1 << CELL_SIZE
PT              = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]
def tk1_schedule(round_idx, cell_idx):
    assert 0 <= cell_idx < 16, f"tk1_schedule: cell_idx 越界 {cell_idx}"
    tmp = cell_idx
    for _ in range(round_idx):
        tmp = PT[tmp]
    return tmp
def Global_mat_bit(round_num):
    num_rows = HALF_STATE_BITS * round_num
    num_cols = FULL_STATE_BITS * (round_num + 1) + KEY_BITS
    res = np.zeros((num_rows, num_cols), dtype=int)
    key_base = FULL_STATE_BITS * (round_num + 1)
    for i in range(num_rows):
        equ_num_bit  = i % HALF_STATE_BITS
        rn           = i // HALF_STATE_BITS
        equ_num_cell = equ_num_bit // CELL_SIZE
        bit_idx      = equ_num_bit % CELL_SIZE
        for k in range(40):
            if M_EQ[equ_num_cell][k] != 1:
                continue
            if k < 16:                  
                res[i][(rn + 1) * FULL_STATE_BITS + k * CELL_SIZE + bit_idx] = 1
            elif k < 32:                
                res[i][rn * FULL_STATE_BITS + k * CELL_SIZE + bit_idx] = 1
            else:                       
                rk_cell = k - 32
                assert rk_cell < 8, f"unexpected rk_cell={rk_cell}"
                mk_cell = tk1_schedule(rn, rk_cell)
                res[i][key_base + mk_cell * CELL_SIZE + bit_idx] = 1
                break
    return res
def _dedup_rows(mat):
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
def extract_hidden_constraints(L_original, active_bit_dic, masked_bit_dic, ROUNDS):
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
    print(f"[TK1] GF(2) 消元: elim_state={elim_count} | "
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
    has_key   = key_seg.any(axis=1)
    chosen = np.flatnonzero(elim_zero & (has_keep | has_key))
    pure = np.zeros((len(chosen), TOTAL_COLS), dtype=int)
    pure[:, keep_state] = keep_seg[chosen]
    pure[:, STATE_COLS:] = key_seg[chosen]
    before = len(pure)
    pure = _dedup_rows(pure)
    after = len(pure)
    cnt_with_key = int((pure[:, STATE_COLS:].any(axis=1)).sum())
    cnt_pure     = after - cnt_with_key
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
def show_L_equ(lmat, active_bit_dic, ROUNDS, verbose=False):
    STATE_COLS = FULL_STATE_BITS * (ROUNDS + 1)
    key_base = STATE_COLS
    L = ""
    for i in range(lmat.shape[0]):
        l_tmp = ""
        for j in range(STATE_COLS):
            if lmat[i][j] == 1:
                var = _fmt_state_var(j)
                l_tmp += f" + [{var}]" if str(j) in active_bit_dic else f" + {var}"
        for k in range(KEY_BITS):
            if lmat[i][key_base + k] == 1:
                l_tmp += f" + k_{k}"
        l_tmp += " = 0"
        if verbose:
            print(l_tmp)
        L += l_tmp + "\n"
    return L
def generate_sb_equ(r, sb_ind, active_bit_dic, dic_x, verbose=False):
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
    if verbose:
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
    hidden = extract_hidden_constraints(L_original, active_bit_dic,
                                        masked_bit_dic, ROUNDS)
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
        n_dup_with_orig = int((~keep_mask).sum())
        hidden = hidden[keep_mask]
    if len(hidden) > 0:
        L_original = np.vstack((L_original, hidden))
        lm_h = hidden.copy()
        h_state = lm_h[:, :STATE_COLS]
        h_ones = (h_state == 1)
        h_state[h_ones & active_mask[None, :]] = 3
        h_state[h_ones & masked_mask[None, :] & ~active_mask[None, :]] = 2
        lm_h[:, :STATE_COLS] = h_state
        L_mat = np.vstack((L_mat, lm_h))
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
        has_key_info   = np.any(L_mat[r, STATE_COLS:] == 1)
        if has_state_info or has_key_info:
            target_rows.append(r)
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
    involved_sb = set()
    for v in unknown_vars:
        if v < STATE_COLS:
            r_idx = v // FULL_STATE_BITS
            within = v % FULL_STATE_BITS
            cell_in_half = (within % HALF_STATE_BITS) // CELL_SIZE
            involved_sb.add((r_idx, cell_in_half))
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
    cluster_dict = {}
    for v in unknown_vars:
        cluster_dict.setdefault(uf.find(v), []).append(v)
    pure_active_rows = [r for r, un in row_unknowns.items()
                        if not un and row_active[r]]
    cons_str, Z_lst = [], []
    mask_lst = []
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
    pass