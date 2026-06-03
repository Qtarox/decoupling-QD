from ORT_GENERAL import *
from solution_plt import *
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
    normalized_dic_lst=[dic_normalize(RES_DIC[key]) for key in dic_lst]
    print(normalized_dic_lst)
    distributions = normalized_dic_lst
    expanded_lists = []
    for d in distributions:
        current_list = []
        for val, count in d.items():
            current_list.extend([val] * count)
        expanded_lists.append(current_list)
    all_combinations = list(itertools.product(*expanded_lists))
    results = []
    for combo in all_combinations:
        product = 1.0
        for num in combo:
            product *= num
        results.append(product)
    results.sort()
    plt.figure(figsize=(10, 6))
    plt.plot(results, linewidth=2) 
    plt.title("Sorted Products of Variable Combinations")
    plt.xlabel("Combination Index")
    plt.ylabel("Product Value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    str1=f'_{9}R'
    plt.savefig(f"./distribut_prob_solution_{str_n}.png", dpi=300)
    plt.show()
