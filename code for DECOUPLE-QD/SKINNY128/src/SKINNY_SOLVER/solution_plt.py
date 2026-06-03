import matplotlib.pyplot as plt
def dic_normalize(dic):
    sum=0
    cnt=0
    for k, v in dic.items():
        sum += v*int(k)
        cnt += v
    sum /= cnt
    dic_norm={}
    for k, v in dic.items():
        dic_norm[int(k)/sum]=v
    return dic_norm


import matplotlib.pyplot as plt
import itertools
if __name__=="__main__":
    dic_lst=[c0,c1,c2,c3,]
    normalized_dic_lst=[dic_normalize(dic) for dic in dic_lst]
    print(normalized_dic_lst)
    # 你的输入数据
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
    print(results)
    plt.title("Sorted Products of Variable Combinations")
    plt.xlabel("Combination Index")
    plt.ylabel("Product Value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print(f"总共计算了 {len(results)} 种组合！")