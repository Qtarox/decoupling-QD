# from GIFTMILP_CONSCOLLECTOR import *
import re
import os
from GEN_MAT.GEN_LINEAR import *
from utils import *
# from GIFTMILP_distri import *
import numpy as np

def extract_hidden_constraints(L_original, active_bit_dic, masked_bit_dic, ROUNDS):
    STATE_COLS = 128 * (ROUNDS + 1)
    TOTAL_COLS = np.shape(L_original)[1]
    
    keep_cols = []
    elim_cols = []
    
    # 1. 划分阵营
    for j in range(STATE_COLS):
        # 只要是 mask 或者 active 的，都是我们想保留的纯净状态
        if str(j) in masked_bit_dic or str(j) in active_bit_dic:
            keep_cols.append(j)
        else:
            elim_cols.append(j) # 没用的中间状态比特，消灭掉
            
    for j in range(STATE_COLS, TOTAL_COLS):
        elim_cols.append(j) # 所有的密钥比特，统统消灭掉
        
    # 2. 重排矩阵列，把 elim_cols 放在前面，强迫高斯消元优先消灭它们
    col_order = elim_cols + keep_cols
    mat = L_original[:, col_order].copy()
    
    rows, cols = mat.shape
    elim_count = len(elim_cols)
    
    print(f"[*] 开始 GF(2) 高斯消元: 试图消灭 {elim_count} 个冗余变量...")
    
    # 3. 在 GF(2) 上进行行阶梯化 (Row Reduction)
    r = 0
    for c in range(elim_count):
        if r >= rows:
            break
            
        # 寻找主元 (Pivot)
        pivot = r
        while pivot < rows and mat[pivot, c] == 0:
            pivot += 1
            
        if pivot == rows:
            continue # 这一列全是 0，跳过
            
        # 把有 1 的行交换上来
        mat[[r, pivot]] = mat[[pivot, r]]
        
        # 把其他行的这一列全部异或清零
        for i in range(rows):
            if i != r and mat[i, c] == 1:
                mat[i] = (mat[i] + mat[r]) % 2 # GF(2) 里的加法就是异或
        r += 1
        
    # 4. 提取战利品：找出那些 elim_cols 全被干掉，且 keep_cols 不为空的行
    pure_equations = []
    for i in range(rows):
        if np.all(mat[i, :elim_count] == 0): # 冗余变量全为 0
            if np.any(mat[i, elim_count:] == 1): # 且保留变量不全为 0
                # 把列顺序还原回最初的 L_original 格式
                pure_row = np.zeros(TOTAL_COLS, dtype=int)
                pure_row[keep_cols] = mat[i, elim_count:]
                pure_equations.append(pure_row)
                
    print(f"[+] 成功提取出 {len(pure_equations)} 个隐藏的纯状态约束！")
    return np.array(pure_equations)

def load_masks_from_file(file_path, round_num):
    """
    从文本文件读入并解析成 [ROUND_NUM][2][64] 的列表
    """
    # 初始化三维列表: [轮数][Side 0/1][64位比特]
    mask_list = [[[0 for _ in range(64)] for _ in range(2)] for _ in range(round_num)]
    
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. 按照 Round 分割块
        # 使用正则表达式匹配 --- Round X --- 及其之后的内容
        round_blocks = re.findall(r'--- Round (\d+) ---(.*?)(?=--- Round|$)', content, re.DOTALL)

        for r_idx_str, block_text in round_blocks:
            r_idx = int(r_idx_str)
            
            # 越界检查
            if r_idx >= round_num:
                continue
            
            # 2. 在每个 Round 块内匹配 Side 0 和 Side 1
            # 正则解释: 匹配 Side 数字, 忽略括号内文字, 捕获冒号后的 0 1 和空格
            side_matches = re.findall(r'Side (\d+) \(.*?\): ([\s01]+)', block_text)
            
            for s_idx_str, bit_raw in side_matches:
                s_idx = int(s_idx_str)
                if s_idx > 1: continue # 仅处理 Side 0 和 Side 1
                
                # 清洗数据: 去掉空格、换行符
                clean_bits = bit_raw.replace(" ", "").replace("\n", "").replace("\r", "").strip()
                
                # 3. 填充到 64 位
                # 注意: 如果文本中比特不足 64 位，后面保持为初始化的 0
                for bit_pos, bit_val in enumerate(clean_bits):
                    if bit_pos < 64:
                        mask_list[r_idx][s_idx][bit_pos] = int(bit_val)
        
        print(f"成功加载 {len(round_blocks)} 轮数据。")
        return mask_list

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
def creat_dic_GIFT(rd):# round
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
            #print("XDDT("+str(input)+", "+str(output)+")="+str(tmp))
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
        ind=int(match.group(2))#sb_ind
        for i in range(4):# check all 4 bits
            # print(key)
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
        for i in range(4):# check all 4 bits
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
        self.parent[i] = self.find(self.parent[i]) # 路径压缩
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
def generate_sb_equ(r,sb_ind,active_bit_dic,dic_x):
    x_cons=""

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

    
    var_lst = list(set(var_lst)) # 去重
    uf = UnionFind(var_lst)

    # 2. 建立联系：遍历矩阵，将同一行中的变量进行 Union 操作
    for r in var_relation:
        if len(r) > 1:
            first_var = r[0]
            for other_var in r[1:]:
                uf.union(first_var, other_var)

    # 3. 归类整理
    clusters = {}
    for var in var_lst:
        root = uf.find(var)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(var)
        
    return list(clusters.values())

# 使用示例
# clusters = get_clusters(L_mat, rounds)
# print("Found clusters:", clusters)
def generate_constraints(L_mat,dic_x, active_bit_dic, masked_bit_dic, ROUNDS, L_original):
    STATE_COLS = 128 * (ROUNDS + 1)
    TOTAL_COLS = np.shape(L_mat)[1]

    # ==========================================================
    # 第一步：原始状态标记（你之前不小心漏掉的核心逻辑！）
    # 必须先把原始矩阵里的 1 标记成 2 (masked) 或 3 (active)
    # ==========================================================
    for r in range(np.shape(L_mat)[0]):
        for j in range(STATE_COLS):
            if L_mat[r][j] == 1:
                if str(j) in masked_bit_dic:
                    if str(j) in active_bit_dic:
                        L_mat[r][j] = 3  # Active bit
                    else:
                        L_mat[r][j] = 2  # Masked inactive bit

    # ==========================================================
    # 【新增逻辑】：使用高斯消元提取所有隐藏的纯状态等式
    # ==========================================================
    hidden_eqs = extract_hidden_constraints(L_original, active_bit_dic, masked_bit_dic, ROUNDS)
    
    # 将提取出来的纯等式追加到 L_original 和 L_mat 的底部
    # 它们将作为“强力剪枝约束”协助 SAT 求解器瞬间跳出无效搜索树
    if len(hidden_eqs) > 0:
        L_original = np.vstack((L_original, hidden_eqs))
        
        # 为了适配后面的 L_mat 行筛选，给新加的纯等式上色 (2或3)
        L_mat_hidden = hidden_eqs.copy()
        for i in range(len(hidden_eqs)):
            for j in range(STATE_COLS):
                if L_mat_hidden[i][j] == 1:
                    if str(j) in active_bit_dic:
                        L_mat_hidden[i][j] = 3
                    elif str(j) in masked_bit_dic:
                        L_mat_hidden[i][j] = 2
        L_mat = np.vstack((L_mat, L_mat_hidden))

    # ====== 筛选目标行 ======
    target_rows = []
    for r in range(np.shape(L_mat)[0]):
        state_part = L_mat[r, :STATE_COLS]
        
        # 规则1：状态部分绝对不能包含未被 mask 且非 active 的比特 (即仍为 1)
        if np.any(state_part == 1):
            continue
        # 规则2：空等式不要
        if np.all(L_mat[r] == 0):
            continue
        # 规则3：必须至少包含一个 Masked (2) 或 Active (3) 状态比特
        if np.any((state_part == 2) | (state_part == 3)):
            target_rows.append(r)

    # 3. 收集【真正的未知数】，包括 Masked 状态位和 Key 密钥位
    unknown_vars = set()
    row_to_unknowns = {}
    for r in target_rows:
        un_vars = []
        # 3a. 收集 Masked State Bits
        for j in range(STATE_COLS):
            if L_mat[r][j] == 2:
                un_vars.append(j)
        # 3b. 收集 Key Bits (大于 STATE_COLS 且矩阵里值为1的列)
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

    # 4. 图聚类核心 (Union-Find)
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

    # 5. 分组生成最终结果
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
            sb_con, var_S = generate_sb_equ(sb[0], sb[1], active_bit_dic,dic_x)
            SB_CONS += sb_con + "\n"
            c_active_vars_for_Z.update(var_S)

        final_str = l_tmp + "\n" + SB_CONS
        if final_str.strip():
            cons_str.append(final_str.strip())
            Z = generate_Z(list(c_active_vars_for_Z), [], active_bit_dic)
            Z_lst.append(Z)

    return cons_str, Z_lst

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
if __name__ == "__main__":
    
    file_name= "../results/gift_cons/quasi_diff_trails_12_64_40.txt"
    file_name= "../results/gift_cons/quasi_diff_trails_18_64_6400_linear.txt"
    # file_name = f"../results/gift_cons/quasi_diff_trails_{3}_{16 * SBOX_SIZE}.txt" 
    ROUNDS = NB_ROUNDS
    # THRESH=1
    # data = load_masks_from_file(file_name, ROUNDS)
    data =  (np.load(f'./freq_msk/masks_freq_{NB_ROUNDS}RD_CORR{MIN_CORR}_T{THRESH}.npy')).tolist() #LOAD THE FREQ MASK
    print("Loaded mask data:", data)
    print("Number of rounds loaded:", len(data))
    print("================================")
    DIFF_TRAIL_FILE = f"../data/differential_trails/SKINNY{16 * SBOX_SIZE}_{ADV_MODEL}_R{NB_ROUNDS}.txt"
    
    diff_trail = extract_diff_trail_flat(DIFF_TRAIL_FILE, ROUNDS)
    print("trails ",diff_trail)
    
    dic_x,dic_y=creat_dic_GIFT(diff_trail)
    print("x_dic: ",dic_x)
    print("y_dic: ",dic_y)
    active_bit_dic=get_active_bit(dic_x,dic_y)
    print(active_bit_dic)
    
    masked_bit_dic=active_bit_dic.copy()
    # print("active_bit_dic: ",active_bit_dic)
    # transfer the data into linear mat
    # get the equation subsets for the trail
    for r in range(ROUNDS):
        for s in range(2):
            for i in range(64):
                if(data[r][s][i]==1):
                    # print(f"Round {r}, Side {s}, bit {i} is masked.")
                    masked_bit_dic[str(r*128+s*64+i)] = 1
    print("active_bit_dic after adding masked bits: ",masked_bit_dic)
    #####################################################################################
    L = Global_mat_bit(ROUNDS)  
    L_mat = L.copy()  
    
    # 注意这里多传了一个 L，用作 L_original
    cons_str, Z_lst = generate_constraints(L_mat, dic_x,active_bit_dic, masked_bit_dic, ROUNDS, L)
    res_str=""
    file_name = f"./constraints/CONS_{NB_ROUNDS}R_{MIN_CORR}_TH{THRESH}.txt"
    with open(file_name, 'w', encoding='utf-8') as file:
        pass
    print("\n#========= 最终分离的独立 Cluster =========")
    cons_lst="dic_cons={\n \n"
    for i in range(len(cons_str)):
        print(f"\n #[ Cluster {i} ]")
        res_str+=f'CONS{i}="""\n'+ str(cons_str[i])
        print(f'CONS{i}="""\n', cons_str[i])
        res_str+='"""'
        print('"""')
        res_str+=f'\nZ{i}='+ str(Z_lst[i])+'\n\n'
        print(f'Z{i}=', Z_lst[i])
        cons_lst+= f"'CONS{i}': (CONS{i},Z{i}),\n"
        print("#----------------------------------------")
        with open(file_name, 'a', encoding='utf-8') as file:
            file.write(res_str)
        res_str=""
    cons_lst+='}'
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write(cons_lst)




    