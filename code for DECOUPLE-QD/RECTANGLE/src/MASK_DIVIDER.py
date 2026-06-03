import re
import os
import numpy as np
from diffs import *
from utils import (
    NB_ROUNDS, NB_ROWS, NB_COLS, RECTANGLE_SBOX as Sbox,
    ADV_MODEL, MIN_CORR, SBOX_SIZE, RECT_PERM,
)



def genLinear_RECT(rounds):
    STATE_COLS = 128 * (rounds + 1)
    KEY_COLS   = 64 * (rounds + 1)     
    TOTAL_COLS = STATE_COLS + KEY_COLS
    rows = []
    for r in range(rounds):
        for j in range(64):
            row = np.zeros(TOTAL_COLS, dtype=np.int8)
            
            row[r * 128 + 64 + j] = 1
            
            target_bit = RECT_PERM[j]
            
            row[(r + 1) * 128 + target_bit] = 1
            
            row[STATE_COLS + (r + 1) * 64 + target_bit] = 1
            rows.append(row)
    return np.array(rows, dtype=np.int8)



def load_masks_from_file(file_path, round_num):
    mask_list = [[[0 for _ in range(64)] for _ in range(2)] for _ in range(round_num)]
    if not os.path.exists(file_path):
        print(f"erroe: no {file_path}")
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
        return None



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
    res = {}
    
    for key, X_lst in x_dic.items():
        m = re.match(r"x_(\d+)_(\d+)", key)
        rn, col = int(m.group(1)), int(m.group(2))
        for i in range(NB_ROWS):                          
            ref = (X_lst[0] >> i) & 1
            if all(((x >> i) & 1) == ref for x in X_lst):
                
                res[str(rn * 128 + 4 * col + i)] = ref
    
    for key, Y_lst in y_dic.items():
        m = re.match(r"y_(\d+)_(\d+)", key)
        rn, col = int(m.group(1)), int(m.group(2))
        for i in range(NB_ROWS):
            ref = (Y_lst[0] >> i) & 1
            if all(((y >> i) & 1) == ref for y in Y_lst):
                res[str(rn * 128 + 64 + 4 * col + i)] = ref
    return res



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



def generate_sb_equ_RECT(r, col, active_bit_dic, dic_x):
    x_cons = ""
    if f'x_{r}_{col}' in dic_x:
        x_cons += "_["
        for x_v in dic_x[f'x_{r}_{col}']:
            x_cons += f'{x_v}_'
        x_cons = x_cons[:-1] + ']'
    l_tmp = f"S{x_cons}("
    var_lst = []
    
    for i in range(NB_ROWS):
        l_tmp += f"x_{r}_{4*col + i},"
        gidx = r * 128 + 4 * col + i
        if str(gidx) in active_bit_dic:
            var_lst.append(gidx)
    l_tmp = l_tmp[:-1] + ") = ("
    
    for i in range(NB_ROWS):
        l_tmp += f"y_{r}_{4*col + i},"
        gidx = r * 128 + 64 + 4 * col + i
        if str(gidx) in active_bit_dic:
            var_lst.append(gidx)
    l_tmp = l_tmp[:-1] + ')'
    return l_tmp, var_lst
def show_L_equ_RECT(cons_mat, active_bit_dic, rounds):
    STATE_COLS = 128 * (rounds + 1)
    lines = []
    for r_idx in range(cons_mat.shape[0]):
        row = cons_mat[r_idx]
        terms = []
        rhs = 0
        
        for j in range(STATE_COLS):
            if row[j] == 0:
                continue
            r = j // 128
            ind = j % 128
            var_name = f"x_{r}_{ind}" if ind < 64 else f"y_{r}_{ind - 64}"
            if str(j) in active_bit_dic:
                
                rhs ^= active_bit_dic[str(j)]
            else:
                terms.append(var_name)
        
        for j in range(STATE_COLS, cons_mat.shape[1]):
            if row[j] == 1:
                k_idx = j - STATE_COLS
                k_round = k_idx // 64
                k_bit = k_idx % 64
                terms.append(f"k_{k_round}_{k_bit}")
        if terms:
            lines.append(" + ".join(terms) + f" = {rhs}")
    return "\n".join(lines)



def generate_constraints_RECT(L_mat, dic_x, active_bit_dic, masked_bit_dic, ROUNDS, L_original):
    STATE_COLS = 128 * (ROUNDS + 1)
    TOTAL_COLS = np.shape(L_mat)[1]
    for r in range(np.shape(L_mat)[0]):
        for j in range(STATE_COLS):
            if L_mat[r][j] == 1:
                if str(j) in masked_bit_dic:
                    if str(j) in active_bit_dic:
                        L_mat[r][j] = 3
                    else:
                        L_mat[r][j] = 2
    
    target_rows = []
    independent_rows = []
    for r in range(np.shape(L_mat)[0]):
        state_part = L_mat[r, :STATE_COLS]
        if np.any(state_part == 1):     
            continue
        if np.all(L_mat[r] == 0):       
            continue
        if np.any(state_part == 2):     
            target_rows.append(r)
        elif np.any(state_part == 3):   
            independent_rows.append(r)
    
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
            r_idx = v // 128
            ind   = v % 128
            local = ind % 64           
            col   = local // 4         
            involved_sboxes.add((r_idx, col))
    
    uf = UnionFind(list(unknown_vars))
    
    for r, un_vars in row_to_unknowns.items():
        if len(un_vars) > 1:
            first = un_vars[0]
            for other in un_vars[1:]:
                uf.union(first, other)
    
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
        
        l_tmp = ""
        if c_rows:
            cons = L_original[c_rows, :]
            l_tmp = show_L_equ_RECT(cons, active_bit_dic, ROUNDS)
            for r in c_rows:
                for j in range(STATE_COLS):
                    if L_mat[r][j] == 3:
                        c_active_vars_for_Z.add(j)
        
        SB_CONS = ""
        for sb in c_sboxes:
            sb_con, var_S = generate_sb_equ_RECT(sb[0], sb[1], active_bit_dic, dic_x)
            SB_CONS += sb_con + "\n"
            c_active_vars_for_Z.update(var_S)
        
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
        
        MSK_LST.append((int(parts[1]), side_map[parts[0]], int(parts[2])))
    return MSK_LST
def extract_diff_trail_cell_RECT_from_diffs(diffs_list):
    nb_rounds = len(diffs_list) // 2
    diff_trail = [[[0] * NB_COLS for _ in range(2)] for _ in range(nb_rounds)]
    for r in range(nb_rounds):
        in_val, out_val = diffs_list[2 * r], diffs_list[2 * r + 1]
        for col in range(NB_COLS):
            diff_trail[r][0][col] = (in_val  >> (4 * col)) & 0xf
            diff_trail[r][1][col] = (out_val >> (4 * col)) & 0xf
    return diff_trail



if __name__ == "__main__":
    pass