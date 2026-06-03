from ORT_GENERAL_TK2 import *
from solution_plt import *
import itertools
import time
import matplotlib.pyplot as plt
def check_validity(dic):
    total_sum = 0
    cnt = 0
    for k, v in dic.items():
        total_sum += v * float(k) 
        cnt += v
    if cnt == 0 or (total_sum / cnt) == 0:
        return False
    return True
if __name__=="__main__":
    str_res = ""
    RES_DIC = {}
    dic_lst = []
    for i in range(len(dic_cons)):
        name = f"CONS{i}" 
        if name not in dic_cons: continue 
        cons_pair = dic_cons[name]
        cons_t = cons_pair[0]
        z = cons_pair[1]
        solu_txt = f"solve_results_gift_{name}.txt"
        t = time.time()
        dist = solve_with_ortools(cons_t, z, solu_txt)
        print("distribution:", dist)
        str_res += f"c{i} = {dist}\n"
        RES_DIC[f'c{i}'] = dist
        dic_lst.append(f'c{i}')
        print(f"Total time used for {name}:", time.time() - t)
    is_valid = True
    for key in dic_lst:
        if not check_validity(RES_DIC[key]):
            is_valid = False
            break
    if not is_valid:
        print("invalid trail!", dic_lst)
        normalized_dic_lst = [{'0': 1}]
    else:
        normalized_dic_lst = [dic_normalize(RES_DIC[key]) for key in dic_lst]
    print("normalized distribution:", normalized_dic_lst)
    joint_dist = {1.0: 1} 
    for d in normalized_dic_lst:
        new_dist = {}
        for val1, freq1 in joint_dist.items():
            for val2_str, freq2 in d.items():
                val2 = float(val2_str) 
                new_val = val1 * val2
                new_freq = freq1 * freq2
                new_dist[new_val] = new_dist.get(new_val, 0) + new_freq
        joint_dist = new_dist
    results = []
    for val, count in joint_dist.items():
        results.extend([val] * int(count))
    results.sort()
    plt.figure(figsize=(10, 6))
    plt.plot(results, linewidth=2) 
    plt.title("Sorted Products of Variable Combinations")
    plt.xlabel("Combination Index")
    plt.ylabel("Product Value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    try:
        plt.savefig(f"./distribut_prob_solution_{str_n}.png", dpi=300)
    except NameError:
        plt.savefig("./distribut_prob_solution_final.png", dpi=300)
    plt.show()