from tst_dic import dic1
from fast_distribut import *
from utils import *
from SKINNYMILP_msk import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 你的原始准差分轨迹数据
def plot_corr(data):
# 1. 计算所有轨迹的平均相关性 corr_{avg}
    corrs = [d['corr'] for d in data]
    avg_corr = corrs[0]
    print(f"平均相关性 (avg_corr): {avg_corr:.4f}\n")

    # 2. 提取所有出现过的独立密钥比特
    all_keys = set()
    for d in data:
        all_keys.update(d['keys'])

    # 3. 基于公式 \sum 2**(corr - avg_corr) 计算每个 key bit 的累积能量
    key_energies = {}
    for k in all_keys:
        # 只把那些包含了当前密钥比特 k 的轨迹的能量加起来
        energy = sum(2 ** (d['corr'] - avg_corr) for d in data if k in d['keys'])
        key_energies[k] = energy

    # 4. 根据阈值 1 筛选强弱比特
    threshold = 1.0
    weak_bits = [k for k, e in key_energies.items() if e < threshold]
    strong_bits = [k for k, e in key_energies.items() if e >= threshold]

    print(f"被判定为不重要的弱比特 (能量 < {threshold}):")
    print(weak_bits, "\n")
    print(f"被判定为重要的强比特 (能量 >= {threshold}):")
    print(strong_bits, "\n")

    # 5. 根据弱比特筛选出含有这些比特的弱轨迹，以及完全不含弱比特的强轨迹
    # 只要轨迹的 keys 列表里包含了哪怕一个 weak_bit，整条轨迹就被归为弱轨迹
    weak_trails = [d for d in data if any(k in weak_bits for k in d['keys'])]
    # 只有完全由 strong_bits 组成的轨迹才被保留
    strong_trails = [d for d in data if not any(k in weak_bits for k in d['keys'])]

    print(f"被剔除的弱轨迹数量: {len(weak_trails)}")
    print(f"保留下来的强轨迹数量: {len(strong_trails)}")

    # # 如果需要保存为 CSV
    # pd.DataFrame(weak_trails).to_csv('weak_trails.csv', index=False)
    # pd.DataFrame(strong_trails).to_csv('strong_trails.csv', index=False)

    # 6. 画图展示 (可选)
    plt.figure(figsize=(12, 5))

    # 图1: 各个 Key Bit 的能量条形图
    plt.subplot(1, 2, 1)
    # 按照能量大小排序以便作图
    sorted_keys = sorted(key_energies.keys(), key=lambda x: key_energies[x], reverse=True)
    energies = [key_energies[k] for k in sorted_keys]
    # 强比特用蓝色，弱比特用红色
    colors = ['#1f77b4' if e >= threshold else '#d62728' for e in energies]

    bars = plt.bar(sorted_keys, energies, color=colors)
    plt.axhline(y=threshold, color='black', linestyle='--', label=f'Threshold = {threshold}')
    plt.yscale('log')
    plt.ylabel('Energy ($\sum 2^{corr - avg\_corr}$)')
    plt.xlabel('Key Bits')
    plt.title('Energy of Key Bits (Log Scale)')
    plt.xticks(rotation=45)
    plt.legend()

    # 图2: 轨迹相关性分布 (保留的强轨迹 vs 被删除的弱轨迹)
    plt.subplot(1, 2, 2)
    if strong_trails:
        plt.hist([d['corr'] for d in strong_trails], bins=10, alpha=0.7, label='Retained (Strong) Trails', color='#1f77b4')
    if weak_trails:
        plt.hist([d['corr'] for d in weak_trails], bins=10, alpha=0.7, label='Discarded (Weak) Trails', color='#d62728')
    plt.xlabel('Correlation ($log_2$)')
    plt.ylabel('Frequency (Count)')
    plt.title('Distribution of Quasi-differential Trails')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'weak_trails_distribution_{MIN_CORR}.png', dpi=300, bbox_inches='tight')
    plt.show()
    return weak_trails
def filter_weak_bit(dic):
    pass

if __name__=="__main__":
    LST_RES=SKINNY_MILP_Quasi_Diff(NB_ROUNDS)
    weak_trails=plot_corr(LST_RES)
    plot_quasi_distribut(weak_trails,appendix=f'_weak_set_{MIN_CORR}', num_samples=20000)