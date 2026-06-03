from ORT_GENERAL_TK2 import *
from solution_plt import *
import itertools
import time
import matplotlib.pyplot as plt

def check_validity(dic):
    total_sum = 0
    cnt = 0
    for k, v in dic.items():
        # 修复2：k 可能是字符串，必须转数值
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
        print("分布哈希 (数量: 出现次数):", dist)
        str_res += f"c{i} = {dist}\n"
        RES_DIC[f'c{i}'] = dist
        dic_lst.append(f'c{i}')
        print(f"Total time used for {name}:", time.time() - t)

    # =======================================================
    # 修复1：正确遍历验证每一个 Cluster 的有效性
    # =======================================================
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
        
    print("归一化后的分布列表:", normalized_dic_lst)

    # =======================================================
    # 修复3：通过频率字典计算笛卡尔积，降维打击避免内存爆炸
    # =======================================================
    # 初始状态：乘积为 1，频数为 1
    joint_dist = {1.0: 1} 
    
    for d in normalized_dic_lst:
        new_dist = {}
        for val1, freq1 in joint_dist.items():
            for val2_str, freq2 in d.items():
                val2 = float(val2_str) # 确保转换为数值
                
                new_val = val1 * val2
                new_freq = freq1 * freq2
                
                # 累加合并相同乘积的频数
                new_dist[new_val] = new_dist.get(new_val, 0) + new_freq
        joint_dist = new_dist

    # =======================================================
    # 最后阶段：将紧凑的频率字典展开，用于画出平滑上升曲线
    # =======================================================
    results = []
    for val, count in joint_dist.items():
        # count 已经是最终归一化后的频数，直接延展
        results.extend([val] * int(count))

    results.sort()

    # 画图
    plt.figure(figsize=(10, 6))
    plt.plot(results, linewidth=2) 
    plt.title("Sorted Products of Variable Combinations")
    plt.xlabel("Combination Index")
    plt.ylabel("Product Value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 注意：这里 str_n 需要您在前面提取模块名时定义，比如 str_n = args.module.split('.')[-1]
    # 此处假设已经存在，我用 try-except 包裹避免未定义报错
    try:
        plt.savefig(f"./distribut_prob_solution_{str_n}.png", dpi=300)
    except NameError:
        plt.savefig("./distribut_prob_solution_final.png", dpi=300)
        
    plt.show()

    print(f"总共计算了 {len(results)} 种有效组合点！")