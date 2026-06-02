from GIFTMILP_msk import *
from MASK_DIVIDER import *

if __name__=="__main__":
    THRES=TH
    masks,sol_num,save_pth = GIFT_MILP_Quasi_Diff(NB_ROUNDS,THRES)
    print("mask's solution number: ",sol_num)
    # save_pth="./freq_msk/masks_freq_4RD_CORR50_T1.npy"
    #load mask and differential trails
    data =  (np.load(save_pth)).tolist()
    # print("Loaded mask data:", data)
    print("Number of rounds loaded:", len(data))
    print("================================")
    DIFF_TRAIL_FILE = f"../data/differential_trails/GIFT{16 * SBOX_SIZE}_{ADV_MODEL}_R{NB_ROUNDS}{NAME}.txt"
    diff_trail = extract_diff_trail_cell(DIFF_TRAIL_FILE, NB_ROUNDS)
    dic_x,dic_y=creat_dic_GIFT(diff_trail)
    active_bit_dic=get_active_bit(dic_x,dic_y)
    masked_bit_dic=active_bit_dic.copy()

    for r in range(NB_ROUNDS):
        for s in range(2):
            for i in range(64):
                if(data[r][s][i]==1):
                    # print(f"Round {r}, Side {s}, bit {63-i} is masked.")
                    masked_bit_dic[str(r*128+s*64+(63-i))] = 1
    print("active_bit_dic after adding masked bits: ",masked_bit_dic)
    ################################# generate constraint set ####################################################
    L = genLinear(NB_ROUNDS)  
    L_mat = L.copy()  

    cons_str, Z_lst, mask_lst = generate_constraints(L_mat, dic_x,active_bit_dic, masked_bit_dic, NB_ROUNDS, L)
    FULL_MSK=[]
    res_str=""
    file_prefix = f"./constraints/CONS_{NB_ROUNDS}R_{MIN_CORR}{NAME}_TH{THRES}"
    mask_py_file = f"{file_prefix}_FULL_MSK.py"
    file_name = f"./constraints/CONS_{NB_ROUNDS}R_{MIN_CORR}{NAME}_TH{THRES}.txt"
    with open(file_name, 'w', encoding='utf-8') as file:
        pass
    print("\n#========= Final Cluster =========")
    cons_lst="dic_cons={\n \n"
    for i in range(len(cons_str)):
        print(f"\n #[ Cluster {i} ]")
        res_str+=f'CONS{i}="""\n'+ str(cons_str[i])
        print(f'CONS{i}="""\n', cons_str[i])
        res_str+='"""'
        print('"""\n')
        res_str+=f'\nZ{i}='+ str(Z_lst[i])+'\n\n'
        print(f'Z{i}=', Z_lst[i])
        print("#----------------------------------------")
        cons_lst+= f"'CONS{i}': (CONS{i},Z{i}),\n"
        print(f'MASK{i}=', mask_lst[i]) 
        FULL_MSK.append(mask_lst[i])
        with open(file_name, 'a', encoding='utf-8') as file:
            file.write(res_str)
        res_str=""
    cons_lst+='}'
    print("Full mask list for all clusters:", FULL_MSK)
    with open(mask_py_file, 'w', encoding='utf-8') as f:
        f.write('# Auto-generated FULL_MSK file\n')
        f.write(f'# NB_ROUNDS = {NB_ROUNDS}, MIN_CORR = {MIN_CORR}, THRES = {THRES}\n')
        f.write(f'# Number of clusters: {len(FULL_MSK)}\n\n')
        f.write(f'FULL_MSK = {FULL_MSK}\n')
    print(f"FULL_MSK save to  Python file: {mask_py_file}")
    with open(file_name, 'a', encoding='utf-8') as file:
        file.write(cons_lst)
