from GIFTMILP_msk import *
from MASK_DIVIDER import *
from utils import *
def run_milp_if_needed():
    mask_path = f'./freq_msk/masks_freq_GIFT128_{NB_ROUNDS}RD{NAME}_CORR{MIN_CORR}_T{THRESH}.npy'
    if os.path.exists(mask_path):
        print(f"[skip MILP] mask exist: {mask_path}")
    else:
        print(f"[run MILP] mask not exist, search trails...")
        GIFT_MILP_Quasi_Diff(NB_ROUNDS,THRESH)
    return mask_path
if __name__=="__main__":
    mask_path = run_milp_if_needed()
    THRES=THRESH
    data =  (np.load(mask_path)).tolist()
    print("Number of rounds loaded:", len(data))
    print("================================")
    
    DIFF_TRAIL_FILE = f"../data/differential_trails/GIFT{32 * SBOX_SIZE}_{ADV_MODEL}_R{NB_ROUNDS}{NAME}.txt"
    diff_trail = extract_diff_trail_cell(DIFF_TRAIL_FILE, NB_ROUNDS)
    dic_x, dic_y = creat_dic_GIFT(diff_trail)
    active_bit_dic = get_active_bit(dic_x, dic_y)
    masked_bit_dic = active_bit_dic.copy()

    for r in range(NB_ROUNDS):
        for s in range(2):
            
            for i in range(128):
                if(data[r][s][i]==1):
                    
                    masked_bit_dic[str(r*256 + s*128 + (127-i))] = 1
                    
    print("active_bit_dic after adding masked bits: ", masked_bit_dic)
    
    L = genLinear(NB_ROUNDS)  
    L_mat = L.copy()  
    
    cons_str, Z_lst, mask_lst = generate_constraints(L_mat, dic_x, active_bit_dic, masked_bit_dic, NB_ROUNDS, L)
    
    FULL_MSK=[]
    res_str=""
    file_prefix = f"./constraints/CONS_{NB_ROUNDS}R_{MIN_CORR}{NAME}_TH{THRES}"
    mask_py_file = f"{file_prefix}_FULL_MSK.py"
    file_name = f"./constraints/CONS_{NB_ROUNDS}R_{MIN_CORR}{NAME}_TH{THRES}.txt"
    
    with open(file_name, 'w', encoding='utf-8') as file:
        pass
        

    cons_lst="dic_cons={\n \n"
    for i in range(len(cons_str)):
        res_str+=f'CONS{i}="""\n'+ str(cons_str[i])
        print(f'CONS{i}="""\n', cons_str[i])
        res_str+='"""'
        print('"""\n')
        res_str+=f'\nZ{i}='+ str(Z_lst[i])+'\n\n'
        print(f'Z{i}=', Z_lst[i])
        cons_lst+= f"'CONS{i}': (CONS{i},Z{i}),\n"
        FULL_MSK.append(mask_trans(mask_lst[i]))
        
        with open(file_name, 'a', encoding='utf-8') as file:
            file.write(res_str)
        res_str=""
        
    cons_lst+='}'
    print("Full mask list for all clusters:", FULL_MSK)
    
    with open(mask_py_file, 'w', encoding='utf-8') as f:
        f.write(f'FULL_MSK = {FULL_MSK}\n')
        
    print(f"FULL_MSK save to: {mask_py_file}")
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write(cons_lst)