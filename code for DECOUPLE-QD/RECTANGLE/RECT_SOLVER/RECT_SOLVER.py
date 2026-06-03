from ORT_GENERAL import *
from solution_plt import *
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict

if __name__=="__main__":
    str_res=""
    RES_DIC={}
    dic_lst=[]
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
        str_res+= f"c{i} = {dist}\n"
        RES_DIC[f'c{i}']=dist
        dic_lst.append(f'c{i}')
        print("Total time used:", time.time() - t)
    
    print(RES_DIC)
    normalized_dic_lst=[dic_normalize(RES_DIC[key]) for key in dic_lst]
    print(normalized_dic_lst)
    distributions = normalized_dic_lst
    final_dist = defaultdict(int)
    final_dist[1.0] = 1  

    for current_dict in distributions:
        new_dist = defaultdict(int)
        
        for current_val, current_count in final_dist.items():
            for val, count in current_dict.items():
                new_dist[current_val * val] += current_count * count
        final_dist = new_dist

    
    
    sorted_items = sorted(final_dist.items())

    x_indices = []
    y_values = []
    current_index = 0

    
    for val, count in sorted_items:
        x_indices.append(current_index)      
        y_values.append(val)
        current_index += count               
        x_indices.append(current_index - 1)  
        y_values.append(val)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_indices, y_values, linewidth=2)
    plt.title("Sorted Products of Variable Combinations (Optimized)")
    plt.xlabel("Combination Index")
    plt.ylabel("Product Value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig("./distribut_prob_solution.png", dpi=300)
    plt.show()