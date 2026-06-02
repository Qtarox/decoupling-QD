import matplotlib.pyplot as plt
def dic_normalize(dic):
    sum=0
    cnt=0
    for k, v in dic.items():
        sum += v*int(k)
        cnt += v
    sum /= cnt
    dic_norm={}
    if sum== 0:
        
        for k, v in dic.items():
            dic_norm[float(k)] = v
        return dic_norm
        
    for k, v in dic.items():
        dic_norm[int(k)/sum]=v
    return dic_norm


import matplotlib.pyplot as plt
import itertools
if __name__=="__main__":
    dic_lst=[c0,c1,c2,c3,]
    normalized_dic_lst=[dic_normalize(dic) for dic in dic_lst]
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
    print(results)
    plt.title("Sorted Products of Variable Combinations")
    plt.xlabel("Combination Index")
    plt.ylabel("Product Value")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
