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
    print(RES_DIC)
    normalized_dic_lst=[dic_normalize(RES_DIC[key]) for key in dic_lst]
    print(normalized_dic_lst)
    distributions = normalized_dic_lst
    # distributions=[{0.75: 2, 1.25: 2}, {0.75: 16, 1.25: 16, 1.75: 16, 0.25: 16}, {0.0: 4096, 3.0: 1024, 1.0: 1024, 2.0: 2048}, {0.75: 2, 1.25: 2}, {0.75: 2, 1.25: 2}, {0.75: 2, 1.25: 2}, {0.75: 2, 1.25: 2}, {0.75: 2, 1.25: 2}, {0.75: 2, 1.25: 2}, {0.75: 2, 1.25: 2}, {0.75: 2, 1.25: 2}, {0.75: 2, 1.25: 2}]

    # 1. 根据个数将字典展开成列表
    # 1. 采用“分布合并”计算组合与乘积，避免暴力展开
    final_dist = defaultdict(int)
    final_dist[1.0] = 1  # 初始状态：乘积为1，组合数为1

    for current_dict in distributions:
        new_dist = defaultdict(int)
        # 将当前已累积的结果，与下一个字典进行交叉相乘
        for current_val, current_count in final_dist.items():
            for val, count in current_dict.items():
                new_dist[current_val * val] += current_count * count
        final_dist = new_dist

    # 2. 准备画图数据
    # 将最终的不同乘积结果从小到大排序
    sorted_items = sorted(final_dist.items())

    x_indices = []
    y_values = []
    current_index = 0

    # 为了在图上表现出 5500 亿个点排布的效果，我们通过记录区间的起点和终点来画图
    for val, count in sorted_items:
        x_indices.append(current_index)      # 区间起点
        y_values.append(val)
        current_index += count               # 加上这种乘积出现的次数
        x_indices.append(current_index - 1)  # 区间终点
        y_values.append(val)

    print(f"总共有 {len(sorted_items)} 种不同的乘积结果！")
    print(f"总共计算了 {current_index} 种组合！")

    # 3. 画图 (由于 X 轴刻度极大，matplotlib 会自动使用科学计数法)
    plt.figure(figsize=(10, 6))
    plt.plot(x_indices, y_values, linewidth=2)
    plt.title("Sorted Products of Variable Combinations (Optimized)")
    plt.xlabel("Combination Index")
    plt.ylabel("Product Value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig("./distribut_prob_solution.png", dpi=300)
    plt.show()