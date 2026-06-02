import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from utils import *
def plot_quasi_distribut(trails, appendix='',avg_p=MIN_CORR, num_samples=2000):
    
    
    
    
    base_corr = (-1)*avg_p
    
    unique_keys = set()
    for t in trails:
        unique_keys.update(t['keys'])
    unique_keys = sorted(list(unique_keys))
    num_keys = len(unique_keys)

    print(f"involved {num_keys} keys:")
    print(unique_keys)
    print("-" * 50)

    key_to_idx = {k: i for i, k in enumerate(unique_keys)}

    
    
    
    total_space = 2 ** num_keys
    if total_space <= num_samples:
        X = np.array(list(itertools.product([0, 1], repeat=num_keys)))
        actual_samples = total_space
    else:
        X = np.random.randint(0, 2, size=(num_samples, num_keys))
        
        X[0, :] = 1
        actual_samples = num_samples

    
    
    
    total_sum = np.zeros(actual_samples)

    for t in trails:
        k_indices = [key_to_idx[k] for k in t['keys']]
        
        if len(k_indices) > 0:
            xor_sum = np.sum(X[:, k_indices], axis=1) % 2
        else:
            xor_sum = np.zeros(actual_samples)
            
        
        
        term_val = t['sign'] * (2**(t['corr'] - base_corr)) * ((-1)**xor_sum)
        total_sum += term_val
    
    
    sum_values=total_sum
    
    
    
    max_idx = np.argmax(sum_values)
    min_idx = np.argmin(sum_values)
    
    max_val = sum_values[max_idx]
    min_val = sum_values[min_idx]
    max_assignment = dict(zip(unique_keys, X[max_idx]))
    min_assignment = dict(zip(unique_keys, X[min_idx]))
    
    dist_lst=sum_values.copy()
    dist_lst.sort()
    
    
    
    def power2_fmt_with_coeff(x, pos=None):
        
        
        
        return f"${x:.2f} \\times 2^{{{base_corr:.3f}}}$"

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    sorted_values = np.sort(sum_values)
    x_coords = range(len(sorted_values))

    ax.plot(x_coords, sorted_values, marker='.', linestyle='--', color='b', markersize=4, linewidth=1)

    ax.set_title(f'Sorted Sum Values for {actual_samples} Key Combinations')
    ax.set_xlabel('Combination Index (Sorted)')
    ax.set_ylabel('Evaluated Sum (Probability Bias)')
    ax.grid(True, linestyle=':', alpha=0.6)

    
    ax.yaxis.set_major_formatter(FuncFormatter(power2_fmt_with_coeff))
    ax.axhline(0, color='black', linewidth=1) 

    plt.tight_layout()
    plt.savefig(f"./figs/distribut_prob_{appendix}.png", dpi=300)
    plt.show()
    print(dist_lst)
    cnt_ng=0
    nz_lst=[]
    z_cnt=0
    for e in dist_lst:
        if(e<0):
            cnt_ng+=1
        elif(e>0):
            nz_lst.append(e)
        else:
            z_cnt+=1
    print(nz_lst)
    if(cnt_ng>0):
        print(f"there are {cnt_ng} negative elements, wrong!")
    else:
        print("no negative probability")
    print(f"zeros:{z_cnt}")
    np.save(f'./quasi_distri/distri_{NB_ROUNDS}R{NAME}_{MIN_CORR}',dist_lst)

if __name__=="__main__":
    pass