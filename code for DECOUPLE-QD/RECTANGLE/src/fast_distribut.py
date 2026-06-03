import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from utils import *
def plot_quasi_distribut(trails, appendix='',avg_p=MIN_CORR, num_samples=2000):
    # =========================================================
    # 1. 提取基准概率和独立的密钥比特
    # =========================================================
    # 以第一条 trail (通常是最大的 DC 项) 为基准
    base_corr = (-1)*avg_p
    
    unique_keys = set()
    for t in trails:
        unique_keys.update(t['keys'])
    unique_keys = sorted(list(unique_keys))
    num_keys = len(unique_keys)

    print(f"涉及的密钥变量共 {num_keys} 个:")
    print(unique_keys)
    print("-" * 50)

    key_to_idx = {k: i for i, k in enumerate(unique_keys)}

    # =========================================================
    # 2. 智能判定：全空间遍历 vs 随机采样
    # =========================================================
    total_space = 2 ** num_keys
    if total_space <= num_samples:
        print(f"提示: 总组合数 ({total_space}) 小于等于设定的采样数 ({num_samples})。")
        print("为保证结果精确，将执行【全空间遍历】。")
        X = np.array(list(itertools.product([0, 1], repeat=num_keys)))
        actual_samples = total_space
    else:
        print(f"提示: 总组合数 ({total_space}) 较大。")
        print(f"为提高速度，将执行【随机采样】，共随机生成 {num_samples} 个密钥组合。")
        X = np.random.randint(0, 2, size=(num_samples, num_keys))
        # 可选：强行塞入全 0 密钥作为基准对照
        X[0, :] = 1
        actual_samples = num_samples

    # =========================================================
    # 3. 使用 NumPy 矩阵化计算 (提前提取基准系数)
    # =========================================================
    total_sum = np.zeros(actual_samples)

    for t in trails:
        k_indices = [key_to_idx[k] for k in t['keys']]
        
        if len(k_indices) > 0:
            xor_sum = np.sum(X[:, k_indices], axis=1) % 2
        else:
            xor_sum = np.zeros(actual_samples)
            
        # 你的正确思路：指数变为 t['corr'] - base_corr
        # 此时算出来的 term_val 直接就是相对于 base_corr 的系数！
        term_val = t['sign'] * (2**(t['corr'] - base_corr)) * ((-1)**xor_sum)
        total_sum += term_val
    
    # sum_values = np.where(np.abs(total_sum) < 1e-10, 0.0, total_sum)
    sum_values=total_sum
    print("【未过滤的原始最大误差】:", np.max(np.abs(sum_values)))
    # =========================================================
    # 4. 打印极值信息
    # =========================================================
    max_idx = np.argmax(sum_values)
    min_idx = np.argmin(sum_values)
    
    max_val = sum_values[max_idx]
    min_val = sum_values[min_idx]
    max_assignment = dict(zip(unique_keys, X[max_idx]))
    min_assignment = dict(zip(unique_keys, X[min_idx]))

    print(f"最大概率偏差值: {max_val:g} * 2^{base_corr:.3f}")
    print(f"对应的密钥组合: {max_assignment}")
    print(f"最小概率偏差值: {min_val:g} * 2^{base_corr:.3f}")
    print(f"对应的密钥组合: {min_assignment}")

    # =========================================================
    # 5. 可视化 (直接使用求出的系数)
    # =========================================================
    def power2_fmt_with_coeff(x, pos=None):
        # x 现在已经是纯系数了，直接格式化，不要再除以 base_val
        # if abs(x) < 1e-10:
        #     return "$0$"
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

    # 应用自定义坐标轴格式
    ax.yaxis.set_major_formatter(FuncFormatter(power2_fmt_with_coeff))
    ax.axhline(0, color='black', linewidth=1) 

    plt.tight_layout()
    plt.savefig(f"./figs/distribut_prob_{appendix}.png", dpi=300)
    plt.show()

if __name__=="__main__":
    pass