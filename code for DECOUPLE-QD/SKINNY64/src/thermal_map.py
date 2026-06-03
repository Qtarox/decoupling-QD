import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

# 1. 生成一个模拟的 mask 数据，形状为 (4, 2, 64)
# 这里用随机数代替，实际使用时替换为你的真实变量

# 设置统一的绘图风格
sns.set_theme(style="white")

def plot_subplots_heatmap(data, name=""):
    # 1. 动态计算画布高度：宽度固定为 20，高度根据轮数动态调整 (每轮分配约 1.2 英寸)
    fig, axes = plt.subplots(nrows=NB_ROUNDS, ncols=1, figsize=(20, 1.2 * NB_ROUNDS))
    
    # 如果 NB_ROUNDS=1，axes 不是一个列表，这里做一下兼容处理
    if NB_ROUNDS == 1:
        axes = [axes]
        
    for i in range(NB_ROUNDS):
        slice_data = data[i]
        
        # 2. 核心参数：square=True 强制每个数据点画成正方形，linewidths=0.5 增加网格线
        sns.heatmap(
            slice_data, 
            ax=axes[i], 
            cmap="viridis", 
            cbar=False,         # 关闭单个子图的 colorbar，后面统一画
            square=True,        # 让 2x64 的每一个 bit 都变成正方形格子
            linewidths=0.5,     # 增加极细的网格线，彻底分清每一个 bit
            linecolor='gray'    # 网格线颜色
        )
        
        axes[i].set_title(f"Round {i} Mask (Shape: 2 x 64)", fontsize=12, pad=8)
        axes[i].set_ylabel("Dim 2", fontsize=10)
        
        # 3. 隐藏上方子图的 X 轴标签，只保留最后一张图的 X 轴
        if i < NB_ROUNDS - 1:
            axes[i].set_xticks([]) # 清空 x 轴刻度
            axes[i].set_xlabel("")
        else:
            axes[i].set_xlabel("64-bit State Index", fontsize=12)
            # 旋转 X 轴的 0~63 数字，防止它们挤在一起
            axes[i].tick_params(axis='x', rotation=45, labelsize=9)

    # 4. 解决所有图对齐问题：在整个图的右侧添加一个全局 Colorbar
    plt.tight_layout(rect=[0, 0, 0.92, 1]) # 给右侧留出 8% 的空间画 Colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7]) # [左, 下, 宽, 高]
    fig.colorbar(axes[0].collections[0], cax=cbar_ax)

    # 5. 保存高清图
    plt.savefig(f'thermal_{name}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__=="__main__":
    mask =  (np.load(f'./freq_msk/masks_freq_{NB_ROUNDS}RD_CORR{MIN_CORR}.npy'))
    print("original mask:",mask)
    mask2= np.load(f'./freq_msk/masks_freq_{NB_ROUNDS}RD_CORR{MIN_CORR}_T{THRESH}.npy')
    mask_min = np.min(mask)
    mask_max = np.max(mask)

    # 避免最大值和最小值相等导致除以 0 的报错
    if mask_max > mask_min:
        normalized_mask = (mask - mask_min) / (mask_max - mask_min)
    else:
        normalized_mask = np.zeros_like(mask)
    # 执行绘图
    print(normalized_mask)
    # print("展示方式一：合并维度的单张热力图")
    # plot_single_heatmap(normalized_mask)

    print("展示方式二：拆分维度的多子图热力图")
    plot_subplots_heatmap(normalized_mask)
    plot_subplots_heatmap(mask2,name=f'{THRESH}')