import re
import os
import numpy as np
from GEN_MAT.GEN_LINEAR import *
from utils import *

NB_SBOXES_PER_ROUND = 32
BITS_PER_SBOX = 4
BLOCK_SIZE = 128
ROUND_BITS = 2 * BLOCK_SIZE   # = 256

def creat_dic_GIFT(rd):
    x_dic = {}
    y_dic = {}
    print("running!")
    for r in range(len(rd)):
        for x_index in range(NB_SBOXES_PER_ROUND):
            if rd[r][0][x_index] == 0:
                continue
            x_tmp = f'x_{r}_{x_index}'
            y_tmp = f'y_{r}_{x_index}'
            l_x = xddt_list(rd[r][0][x_index], rd[r][1][x_index])
            l_y = yddt_list(rd[r][0][x_index], rd[r][1][x_index])
            x_dic[x_tmp] = l_x.copy()
            y_dic[y_tmp] = l_y.copy()
    return x_dic, y_dic

def xddt_list(input_diff, output_diff):
    res = []
    for x in range(16):
        if (Sbox[x] ^ Sbox[x ^ input_diff] == output_diff) and input_diff != 0:
            res.append(x)
    return res

def yddt_list(input_diff, output_diff):
    res = []
    for x in range(16):
        if (Sbox[x] ^ Sbox[x ^ input_diff] == output_diff) and input_diff != 0:
            res.append(Sbox[x])
    return res

def get_active_bit(x_dic, y_dic):
    res = {}
    pattern = r"x_(\d+)_(\d+)"
    for key in x_dic:
        match = re.match(pattern, key)
        X_lst = x_dic[key]
        rn, ind = int(match.group(1)), int(match.group(2))
        for i in range(4):
            initial = X_lst[0] >> i & 1
            if all((x >> i & 1) == initial for x in X_lst):
                res[str(rn * ROUND_BITS + ind * 4 + i)] = initial
    
    pattern = r"y_(\d+)_(\d+)"
    for key in y_dic:
        match = re.match(pattern, key)
        Y_lst = y_dic[key]
        rn, ind = int(match.group(1)), int(match.group(2))
        for i in range(4):
            initial = Y_lst[0] >> i & 1
            if all((y >> i & 1) == initial for y in Y_lst):
                res[str(rn * ROUND_BITS + BLOCK_SIZE + ind * 4 + i)] = initial
    return res

class UnionFind:
    def __init__(self, elements):
        self.parent = {el: el for el in elements}
    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

def generate_sb_equ(r, sb_ind, active_bit_dic, dic_x):
    x_cons = ""
    if f'x_{r}_{sb_ind}' in dic_x:
        x_cons += "_[" + "_".join(str(v) for v in dic_x[f'x_{r}_{sb_ind}']) + "]"
    
    l_tmp = f"S{x_cons}("
    var_lst = []
    for i in range(4):
        l_tmp += f"x_{r}_{sb_ind*4 + i},"
        gidx = r * ROUND_BITS + sb_ind * 4 + i
        if str(gidx) in active_bit_dic: var_lst.append(gidx)
    l_tmp = l_tmp[:-1] + ") = ("
    for i in range(4):
        l_tmp += f"y_{r}_{sb_ind*4 + i},"
        gidx = r * ROUND_BITS + BLOCK_SIZE + sb_ind * 4 + i
        if str(gidx) in active_bit_dic: var_lst.append(gidx)
    l_tmp = l_tmp[:-1] + ')'
    return l_tmp, var_lst

def generate_constraints(L_mat, dic_x, active_bit_dic, masked_bit_dic, ROUNDS, L_original):
    STATE_COLS = 256 * (ROUNDS + 1)
    KEY_COLS_END = STATE_COLS + 128 
    
    def col_to_global(j):
        r = j // 256
        ind = j % 256
        return r * ROUND_BITS + ind if ind < 128 else r * ROUND_BITS + BLOCK_SIZE + (ind - 128)
    
    for r in range(np.shape(L_mat)[0]):
        for j in range(STATE_COLS):
            if L_mat[r][j] == 1:
                gidx = col_to_global(j)
                if str(gidx) in masked_bit_dic:
                    L_mat[r][j] = 3 if str(gidx) in active_bit_dic else 2
    
    target_rows = []
    independent_rows = []
    for r in range(np.shape(L_mat)[0]):
        state_part = L_mat[r, :STATE_COLS]
        if np.any(state_part == 1) or np.all(L_mat[r] == 0): continue
        if np.any(state_part == 2): target_rows.append(r)
        elif np.any(state_part == 3): independent_rows.append(r)
    
    unknown_vars = set()
    row_to_unknowns = {}
    for r in target_rows:
        un_vars = [j for j in range(STATE_COLS) if L_mat[r][j] == 2]
        # 严格控制边界
        un_vars += [j for j in range(STATE_COLS, KEY_COLS_END) if L_mat[r][j] == 1]
        row_to_unknowns[r] = un_vars
        unknown_vars.update(un_vars)
    
    involved_sboxes = set()
    for v in unknown_vars:
        if v < STATE_COLS:
            involved_sboxes.add((v // 256, ((v % 256) % 128) // 4))
    
    uf = UnionFind(list(unknown_vars))
    for r, un_vars in row_to_unknowns.items():
        if len(un_vars) > 1:
            for other in un_vars[1:]: uf.union(un_vars[0], other)
    
    sbox_to_unknowns = {}
    for (r_idx, ind) in involved_sboxes:
        sb_un_vars = []
        for i in range(4):
            x_var = r_idx * 256 + ind * 4 + i
            y_var = r_idx * 256 + 128 + ind * 4 + i
            if x_var in unknown_vars: sb_un_vars.append(x_var)
            if y_var in unknown_vars: sb_un_vars.append(y_var)
        sbox_to_unknowns[(r_idx, ind)] = sb_un_vars
        if len(sb_un_vars) > 1:
            for other in sb_un_vars[1:]: uf.union(sb_un_vars[0], other)
            
    cluster_dict = {}
    for v in unknown_vars:
        cluster_dict.setdefault(uf.find(v), []).append(v)
    
    cons_str, Z_lst, mask_lst = [], [], []
    for cluster_id, root in enumerate(cluster_dict.keys()):
        c_un_vars = set(cluster_dict[root])
        c_rows = [r for r, un_vars in row_to_unknowns.items() if c_un_vars.intersection(un_vars)]
        c_sboxes = [sb for sb, sb_un_vars in sbox_to_unknowns.items() if c_un_vars.intersection(sb_un_vars)]
        
        c_active_vars_for_Z = set()
        l_tmp = ""
        if c_rows:
            l_tmp = show_L_equ_GIFT(L_original[c_rows, :], active_bit_dic, ROUNDS)
            for r in c_rows:
                for j in range(STATE_COLS):
                    if L_mat[r][j] == 3: c_active_vars_for_Z.add(col_to_global(j))
        
        SB_CONS = ""
        for sb in c_sboxes:
            sb_con, var_S = generate_sb_equ(sb[0], sb[1], active_bit_dic, dic_x)
            SB_CONS += sb_con + "\n"
            c_active_vars_for_Z.update(var_S)
        
        c_mask_vars = set([col_to_global(v) for v in c_un_vars if v < STATE_COLS and str(col_to_global(v)) in masked_bit_dic])
        c_mask_vars.update([g for g in c_active_vars_for_Z if str(g) in masked_bit_dic])
        
        c_mask_formatted = []
        for v in sorted(list(c_mask_vars)):
            r_idx, ind = v // ROUND_BITS, v % ROUND_BITS
            c_mask_formatted.append(f"x_{r_idx}_{ind}" if ind < BLOCK_SIZE else f"y_{r_idx}_{ind - BLOCK_SIZE}")
        
        final_str = (l_tmp + "\n" + SB_CONS).strip()
        if final_str:
            cons_str.append(final_str)
            Z_lst.append(generate_Z(list(c_active_vars_for_Z), [], active_bit_dic))
            mask_lst.append(c_mask_formatted)
    
    if independent_rows:
        indep_active = set()
        indep_mask = set()
        indep_str = show_L_equ_GIFT(L_original[independent_rows, :], active_bit_dic, ROUNDS)
        for r_eq in independent_rows:
            for j in range(STATE_COLS):
                if L_original[r_eq][j] == 1:
                    gidx = col_to_global(j)
                    if str(gidx) in active_bit_dic: indep_active.add(gidx)
                    if str(gidx) in masked_bit_dic: indep_mask.add(gidx)
        
        indep_mask_formatted = []
        for v in sorted(list(indep_mask)):
            r_idx, ind = v // ROUND_BITS, v % ROUND_BITS
            indep_mask_formatted.append(f"x_{r_idx}_{ind}" if ind < BLOCK_SIZE else f"y_{r_idx}_{ind - BLOCK_SIZE}")
            
        if indep_str.strip():
            cons_str.append(indep_str.strip())
            Z_lst.append(generate_Z(list(indep_active), [], active_bit_dic))
            mask_lst.append(indep_mask_formatted)
    
    return cons_str, Z_lst, mask_lst

def generate_Z(var_S, active_vars, active_bit_dic):
    Z = {}
    for v in set(var_S).union(active_vars):
        if str(v) in active_bit_dic:
            r, ind = v // ROUND_BITS, v % ROUND_BITS
            Z[f"x_{r}_{ind}" if ind < BLOCK_SIZE else f"y_{r}_{ind - BLOCK_SIZE}"] = {active_bit_dic[str(v)]}
    return Z

def mask_trans(MASK):
    return [(int(m.split('_')[1]), {'x':0, 'y':1}[m.split('_')[0]], int(m.split('_')[2])) for m in MASK]