from ORT_GENERAL import *
from solution_plt import *


if __name__=="__main__":
    str_res=""
    RES_DIC={}
    dic_lst=[]
    for i in range(len(dic_cons)):
        # 注意这里获取字典 key 的方式适配了你的 'CONS0' 字符串格式
        name = f"CONS{i}" 
        if name not in dic_cons: continue # 容错
        
        cons_pair = dic_cons[name]
        cons_t = cons_pair[0]
        z = cons_pair[1]
        solu_txt = f"solve_results_gift_{name}.txt"
        
        t = time.time()
        dist = solve_with_ortools(cons_t, z, solu_txt)
        print("分布哈希 (数量: 出现次数):", dist)
        str_res+= f"c{i} = {dist}\n"
        RES_DIC[f'c{i}']=dist
        dic_lst.append(f'c{i}')
        print("Total time used:", time.time() - t)
    # print(str_res)
    # print(RES_DIC)
    normalized_dic_lst=[dic_normalize(RES_DIC[key]) for key in dic_lst]
    print(normalized_dic_lst)
    distributions = normalized_dic_lst

    # 1. 根据个数将字典展开成列表
    expanded_lists = []
    for d in distributions:
        current_list = []
        for val, count in d.items():
            # 将 val 重复 count 次并加入列表
            current_list.extend([val] * count)
        expanded_lists.append(current_list)

    # 2. 计算所有列表的笛卡尔积（获取所有可能的组合）
    all_combinations = list(itertools.product(*expanded_lists))

    # 3. 计算每一种组合中所有数字的乘积
    results = []
    for combo in all_combinations:
        product = 1.0
        for num in combo:
            product *= num
        results.append(product)

    # 4. 将计算结果从小到大排序，为了画出你图片中那样的上升曲线
    results.sort()

    # 5. 画图
    plt.figure(figsize=(10, 6))
    plt.plot(results, linewidth=2) # 画出折线图
    plt.title("Sorted Products of Variable Combinations")
    plt.xlabel("Combination Index")
    plt.ylabel("Product Value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    str1=f'_{9}R'
    plt.savefig(f"./distribut_prob_solution_{str_n}.png", dpi=300)
    plt.show()

    print(f"总共计算了 {len(results)} 种组合！")
