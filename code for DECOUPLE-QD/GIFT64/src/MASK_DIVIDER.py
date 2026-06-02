import re
import os
from GEN_MAT.GEN_LINEAR import *
from utils import *

def load_masks_from_file(file_path, round_num):

    
    mask_list = [[[0 for _ in range(64)] for _ in range(2)] for _ in range(round_num)]
    if not os.path.exists(file_path):

        print(f"error search: {file_path}")
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

                if s_idx > 1: continue 

               

                

                clean_bits = bit_raw.replace(" ", "").replace("\n", "").replace("\r", "").strip()

               

                

                

                for bit_pos, bit_val in enumerate(clean_bits):

                    if bit_pos < 64:

                        mask_list[r_idx][s_idx][bit_pos] = int(bit_val)

       

        print(f"loading success {len(round_blocks)} ")

        return mask_list



    except Exception as e:

        print(f"error load: {e}")

        return None

def creat_dic_GIFT(rd):

    x_dic={}

    y_dic={}

    print("running!")

    for r in range(len(rd)):

        for x_index in range(16):

            if(rd[r][0][x_index]==0):

                continue

            x_tmp='x_'+str(r)+'_'+str(x_index)

            y_tmp='y_'+str(r)+'_'+str(x_index)

            l_x=xddt_list(rd[r][0][x_index],rd[r][1][x_index])

            l_y=yddt_list(rd[r][0][x_index],rd[r][1][x_index])

            x_dic[x_tmp]=l_x.copy()

            y_dic[y_tmp]=l_y.copy()

    return x_dic,y_dic



def xddt_list(input,output):

    res=[]

   



    for x in range(16):

        x1=x

        x2=x^input

        y1=Sbox[x1]

        y2=Sbox[x2]

        if(y1^y2==output and input!=0):

            res.append(x)

            

    return res

   



def yddt_list(input,output):

    res=[]

    for x in range(16):

        x1=x

        x2=x^input

        y1=Sbox[x1]

        y2=Sbox[x2]

        if(y1^y2==output and input!=0):

            res.append(y1)

    return res





def get_active_bit(x_dic,y_dic):

    pattern = r"x_(\d+)_([\d]+)"

    res={}

    for key in x_dic:

        match = re.match(pattern, key)

        X_lst=x_dic[key]

        rn=int(match.group(1))

        ind=int(match.group(2))

        for i in range(4):

            

            activeFlg=True

            initial=X_lst[0]>>i&1

            for x in X_lst:

                curBit=x>>i&1

                if(initial!=curBit):

                    activeFlg=False

                    break

            if(activeFlg):    

                res[str(rn*128+ind*4+i)]=initial

    pattern = r"y_(\d+)_([\d]+)"

    for key in y_dic:

        match = re.match(pattern, key)

        Y_lst=y_dic[key]

        rn=int(match.group(1))

        ind=int(match.group(2))

        for i in range(4):

            activeFlg=True

            initial=Y_lst[0]>>i&1

            for y in Y_lst:

                curBit=y>>i&1

                if(initial!=curBit):

                    activeFlg=False

                    break

            if(activeFlg):    

                res[str(rn*128+64+ind*4+i)]=initial



    return res



def get_var(L_mat,rounds):

    var_lst=[]

    var_relation=[]

    for r in range(np.shape(L_mat)[0]):

        tmp_rela=[]

        for j in range(128*(rounds+1)):

            if(L_mat[r][j]==2):

                var_lst.append(j)

                tmp_rela.append(j)

        var_relation.append(tmp_rela)

    var_lst.sort()

    print("Variables corresponding to masked bits:", var_lst)

    sb_lst={}

    for v in var_lst:

        r=v//128

        ind=(v%64)//4

        if(r, ind) not in sb_lst:

            sb_lst[(r, ind)] = [v]

        else:

            sb_lst[(r, ind)].append(v)

    for sb in sb_lst:

        if(len(sb_lst[sb])>1):

            var_relation.append(sb_lst[sb])

    return var_lst,var_relation



import numpy as np



class UnionFind:

    def __init__(self, elements):

        self.parent = {el: el for el in elements}

       

    def find(self, i):

        if self.parent[i] == i:

            return i

        self.parent[i] = self.find(self.parent[i]) 

        return self.parent[i]



    def union(self, i, j):

        root_i = self.find(i)

        root_j = self.find(j)

        if root_i != root_j:

            self.parent[root_i] = root_j

def generate_sb_equ(r,sb_ind,active_bit_dic,dic_x):

    x_cons=""

    print("x_dictionary:\n",dic_x)



    if(f'x_{r}_{sb_ind}' in dic_x):

        x_cons+="_["

        for x_v in dic_x[f'x_{r}_{sb_ind}']:

            x_cons+=f'{x_v}_'

        x_cons=x_cons[:-1]+']'

    l_tmp=f"S{x_cons}("

    var_lst=[]

    for i in range(4):

        l_tmp=l_tmp+f"x_{r}_{sb_ind*4+i},"

        if(str(r*128+sb_ind*4+i) in active_bit_dic):

            var_lst.append(r*128+sb_ind*4+i)

    l_tmp=l_tmp[:-1]+") = ("

    for i in range(4):

        l_tmp=l_tmp+f"y_{r}_{sb_ind*4+i},"

        if(str(r*128+64+sb_ind*4+i) in active_bit_dic):

            var_lst.append(r*128+64+sb_ind*4+i)

    l_tmp=l_tmp[:-1]+')'

    print(l_tmp)

           

    return l_tmp, var_lst

def get_clusters(var_lst, var_relation):



   

    var_lst = list(set(var_lst)) 

    uf = UnionFind(var_lst)



    

    for r in var_relation:

        if len(r) > 1:

            first_var = r[0]

            for other_var in r[1:]:

                uf.union(first_var, other_var)



    

    clusters = {}

    for var in var_lst:

        root = uf.find(var)

        if root not in clusters:

            clusters[root] = []

        clusters[root].append(var)

       

    return list(clusters.values())

def generate_constraints(L_mat, dic_x, active_bit_dic, masked_bit_dic, ROUNDS, L_original):  

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

            ind = ((v % 128) % 64) // 4

            involved_sboxes.add((r_idx, ind))



    

    uf = UnionFind(list(unknown_vars))



    

    for r, un_vars in row_to_unknowns.items():

        if len(un_vars) > 1:

            first = un_vars[0]

            for other in un_vars[1:]:

                uf.union(first, other)



    

    sbox_to_unknowns = {}

    for (r_idx, ind) in involved_sboxes:

        sb_un_vars = []

        for i in range(4):

            x_var = r_idx * 128 + ind * 4 + i

            y_var = r_idx * 128 + 64 + ind * 4 + i

            if x_var in unknown_vars: sb_un_vars.append(x_var)

            if y_var in unknown_vars: sb_un_vars.append(y_var)

        sbox_to_unknowns[(r_idx, ind)] = sb_un_vars

       

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

            if ind < 64:

                c_mask_formatted.append(f"x_{r_idx}_{ind}")

            else:

                c_mask_formatted.append(f"y_{r_idx}_{ind-64}")



        

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

        indep_str = show_L_equ_GIFT(indep_cons_mat, active_bit_dic, ROUNDS)



        

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



        

        Z_indep = generate_Z(list(indep_active_vars_for_Z),[], active_bit_dic)



        

        if indep_str.strip():

            cons_str.append(indep_str.strip())

            Z_lst.append(Z_indep)

            mask_lst.append(indep_mask_formatted)

           



    return cons_str, Z_lst, mask_lst





def generate_Z(var_S, active_vars,active_bit_dic):

    Z={}

    for v in var_S:

        if(str(v) in active_bit_dic):

            r=v//128

            ind=v%128

            if(ind<64):

                Z[f"x_{r}_{ind}"] = {active_bit_dic[str(v)]}

            else:

                Z[f"y_{r}_{ind-64}"] = {active_bit_dic[str(v)]}

    for v in active_vars:

        r=v//128

        ind=v%128

        if(ind<64):

            Z[f"x_{r}_{ind}"] = {active_bit_dic[str(v)]}

        else:

            Z[f"y_{r}_{ind-64}"] = {active_bit_dic[str(v)]}

    print("Variables involved in the constraints (Z):", Z)

    return Z



def mask_trans(MASK):
    MSK_LST=[]
    side_map = {'x': 0, 'y': 1}
    for m_b in MASK:
        parts=m_b.split('_')
        r=parts[1]
        ind=parts[2]
        side=side_map[parts[0]]
        MSK_LST.append((int(r), side, int(ind)))
    return MSK_LST

       

if __name__ == "__main__":
    file_name= "../results/gift_cons/quasi_diff_trails_12_64_40.txt"
    file_name= "../results/gift_cons/quasi_diff_trails_18_64_6400_linear.txt"
    
    ROUNDS = NB_ROUNDS
    THRESH=1
    
    data =  (np.load(f'./freq_msk/masks_freq_{NB_ROUNDS}RD_CORR{MIN_CORR}_T{THRESH}.npy')).tolist()
    print("Loaded mask data:", data)
    print("Number of rounds loaded:", len(data))
    print("================================")
    DIFF_TRAIL_FILE = f"../data/differential_trails/GIFT{16 * SBOX_SIZE}_{ADV_MODEL}_R{NB_ROUNDS}.txt"
    diff_trail = extract_diff_trail_cell(DIFF_TRAIL_FILE, ROUNDS)
    print("trails ",diff_trail)
    dic_x,dic_y=creat_dic_GIFT(diff_trail)
    print("x_dic: ",dic_x)
    print("y_dic: ",dic_y)
    active_bit_dic=get_active_bit(dic_x,dic_y)
    print(active_bit_dic)
    masked_bit_dic=active_bit_dic.copy()

    for r in range(ROUNDS):
        for s in range(2):
            for i in range(64):
                if(data[r][s][i]==1):
                    
                    masked_bit_dic[str(r*128+s*64+(63-i))] = 1

    print("active_bit_dic after adding masked bits: ",masked_bit_dic)

    L = genLinear(ROUNDS)  
    L_mat = L.copy()  

    

    cons_str, Z_lst, mask_lst = generate_constraints(L_mat, dic_x, active_bit_dic, masked_bit_dic, ROUNDS, L)
    FULL_MSK=[]

    for i in range(len(cons_str)):
        print(f'CONS{i}="""\n', cons_str[i])
        print('"""')
        print(f'Z{i}=', Z_lst[i])
        print(f'MASK{i}=', mask_trans(mask_lst[i]))  
        FULL_MSK.append(mask_trans(mask_lst[i]))  




    print(FULL_MSK)