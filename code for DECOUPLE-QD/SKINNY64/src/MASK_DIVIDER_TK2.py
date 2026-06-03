import re
import os
import numpy as np
from GEN_MAT.GEN_LINEAR import (
    Sbox, M_EQ, key_schedule,
    Global_mat, Global_mat_bit,
    show_L_equ_GIFT, show_L_equ_GIFT_extract,
)
from utils import *  
CELL_SIZE        = 4
HALF_STATE_BITS  = 64
FULL_STATE_BITS  = 128
SBOX_DOMAIN      = 1 << CELL_SIZE   
N_CELLS          = HALF_STATE_BITS // CELL_SIZE  

if len(Sbox) != SBOX_DOMAIN:
    raise RuntimeError(
        f"GEN_LINEAR.Sbox is {len(Sbox)}."
    )
def extract_hidden_constraints(L_original, active_bit_dic, masked_bit_dic, ROUNDS):
    STATE_COLS = FULL_STATE_BITS * (ROUNDS + 1)
    TOTAL_COLS = np.shape(L_original)[1]
    keep_cols, elim_cols = [], []
    for j in range(STATE_COLS):
        if str(j) in masked_bit_dic or str(j) in active_bit_dic:
            keep_cols.append(j)
        else:
            elim_cols.append(j)
    for j in range(STATE_COLS, TOTAL_COLS):
        elim_cols.append(j)   
    col_order = elim_cols + keep_cols
    mat = L_original[:, col_order].copy()
    rows, _ = mat.shape
    elim_count = len(elim_cols)
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
    return np.array(pure_equations) if pure_equations else np.zeros((0, TOTAL_COLS), dtype=int)
def load_masks_from_file(file_path, round_num):
    mask_list = [[[0 for _ in range(HALF_STATE_BITS)] for _ in range(2)] for _ in range(round_num)]
    if not os.path.exists(file_path):
        print(f"no such file: {file_path} ")
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
        return mask_list
    except Exception as e:
        print(f"error loading: {e}")
        return None
def xddt_list(input_diff, output_diff):
    res = []
    if input_diff == 0:
        return res
    for x in range(SBOX_DOMAIN):
        if (Sbox[x] ^ Sbox[x ^ input_diff]) == output_diff:
            res.append(x)
    return res
def yddt_list(input_diff, output_diff):
    res = []
    if input_diff == 0:
        return res
    for x in range(SBOX_DOMAIN):
        if (Sbox[x] ^ Sbox[x ^ input_diff]) == output_diff:
            res.append(Sbox[x])
    return res
def creat_dic_GIFT(rd):
    x_dic, y_dic = {}, {}
    skipped = []
    print("running creat_dic_GIFT (SKINNY-64 TK2)!")
    for r in range(len(rd)):
        for cell_idx in range(N_CELLS):
            in_d  = rd[r][0][cell_idx]
            out_d = rd[r][1][cell_idx]
            if in_d == 0 and out_d == 0:
                continue
            if in_d == 0 or out_d == 0:
                skipped.append((r, cell_idx, in_d, out_d, "zero side"))
                continue
            xd = xddt_list(in_d, out_d)
            if not xd:
                skipped.append((r, cell_idx, in_d, out_d, "DDT=0"))
                continue
            yd = yddt_list(in_d, out_d)
            x_dic[f"x_{r}_{cell_idx}"] = xd[:]
            y_dic[f"y_{r}_{cell_idx}"] = yd[:]
    if skipped:
        print(f"[warn] creat_dic_SKINNY skipped {len(skipped)} illegal cell:")
        for r, c, a, b, why in skipped[:20]:
            print(f"       r={r} cell={c} {a:x}->{b:x}  ({why})")
        if len(skipped) > 20:
            print(f"       ... sklip {len(skipped)} cells")
    return x_dic, y_dic
def get_active_bit(x_dic, y_dic):
    res = {}
    pattern = r"(?:x|y)_(\d+)_(\d+)"
    for key, lst in x_dic.items():
        if not lst:
            continue
        m = re.match(pattern, key)
        rn = int(m.group(1))
        cell_idx = int(m.group(2))
        for i in range(CELL_SIZE):
            initial = (lst[0] >> i) & 1
            active_flag = all(((x >> i) & 1) == initial for x in lst)
            if active_flag:
                global_idx = rn * FULL_STATE_BITS + cell_idx * CELL_SIZE + i
                res[str(global_idx)] = initial
    for key, lst in y_dic.items():
        if not lst:
            continue
        m = re.match(pattern, key)
        rn = int(m.group(1))
        cell_idx = int(m.group(2))
        for i in range(CELL_SIZE):
            initial = (lst[0] >> i) & 1
            active_flag = all(((y >> i) & 1) == initial for y in lst)
            if active_flag:
                global_idx = rn * FULL_STATE_BITS + HALF_STATE_BITS + cell_idx * CELL_SIZE + i
                res[str(global_idx)] = initial
    return res
def get_var(L_mat, rounds):
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
    sb_lst = {}
    for v in var_lst:
        r = v // FULL_STATE_BITS
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
    print(f"[generate_constraints/TK2] STATE_COLS={STATE_COLS}, "
          f"TOTAL_COLS={TOTAL_COLS}, KEY_COLS={TOTAL_COLS - STATE_COLS}")
    for r in range(np.shape(L_mat)[0]):
        for j in range(STATE_COLS):
            if L_mat[r][j] == 1:
                if str(j) in masked_bit_dic:
                    if str(j) in active_bit_dic:
                        L_mat[r][j] = 3
                    else:
                        L_mat[r][j] = 2
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
    target_rows = []
    for r in range(np.shape(L_mat)[0]):
        state_part = L_mat[r, :STATE_COLS]
        if np.any(state_part == 1):
            continue
        if np.all(L_mat[r] == 0):
            continue
        if np.any((state_part == 2) | (state_part == 3)):
            target_rows.append(r)
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
    involved_sboxes = set()
    for v in unknown_vars:
        if v < STATE_COLS:
            r_idx = v // FULL_STATE_BITS
            within = v % FULL_STATE_BITS
            cell_in_half = (within % HALF_STATE_BITS) // CELL_SIZE
            involved_sboxes.add((r_idx, cell_in_half))
    uf = UnionFind(list(unknown_vars))
    for r, un_vars in row_to_unknowns.items():
        if len(un_vars) > 1:
            first = un_vars[0]
            for other in un_vars[1:]:
                uf.union(first, other)
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
    data = np.load(
        f'./freq_msk/masks_freq_{NB_ROUNDS}RD_CORR{MIN_CORR}_T{THRESH}.npy'
    ).tolist()
    print("Loaded mask data shape:", np.array(data).shape)
    print("================================")
    DIFF_TRAIL_FILE = f"../data/differential_trails/SKINNY{16 * SBOX_SIZE}_{ADV_MODEL}_R{NB_ROUNDS}.txt"
    diff_trail = extract_diff_trail_flat(DIFF_TRAIL_FILE, ROUNDS)
    print("trails ", diff_trail)
    dic_x, dic_y = creat_dic_GIFT(diff_trail)
    active_bit_dic = get_active_bit(dic_x, dic_y)
    print("active_bit_dic:", active_bit_dic)
    masked_bit_dic = active_bit_dic.copy()
    for r in range(ROUNDS):
        for s in range(2):
            for i in range(HALF_STATE_BITS):
                if data[r][s][i] == 1:
                    global_idx = r * FULL_STATE_BITS + s * HALF_STATE_BITS + i
                    masked_bit_dic[str(global_idx)] = 1
    print("masked_bit_dic after merging masks:", masked_bit_dic)
    L = Global_mat_bit(ROUNDS)
    L_mat = L.copy()
    cons_str, Z_lst = generate_constraints(
        L_mat, dic_x, active_bit_dic, masked_bit_dic, ROUNDS, L
    )
    out_file = f"./constraints/CONS_TK2_{NB_ROUNDS}R_{MIN_CORR}_TH{THRESH}.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    open(out_file, 'w', encoding='utf-8').close()

    cons_lst = "dic_cons={\n \n"
    for i in range(len(cons_str)):
        res_str = f'CONS{i}="""\n' + str(cons_str[i]) + '"""'
        res_str += f"\nZ{i}=" + str(Z_lst[i]) + "\n\n"
        print(f'CONS{i}="""\n', cons_str[i])
        print('"""')
        print(f"Z{i}=", Z_lst[i])
        cons_lst += f"'CONS{i}': (CONS{i},Z{i}),\n"
        with open(out_file, 'a', encoding='utf-8') as fh:
            fh.write(res_str)
    cons_lst += "}"
    with open(out_file, 'a', encoding='utf-8') as fh:
        fh.write(cons_lst)