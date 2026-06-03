import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

def multiply_distributions(distributions):
    """
    将多个分布字典相乘，算出最终的乘积分布。
    """
    final_dist = defaultdict(int)
    final_dist[1.0] = 1  # 初始状态

    for current_dict in distributions:
        new_dist = defaultdict(int)
        for current_val, current_count in final_dist.items():
            for val, count in current_dict.items():
                new_dist[current_val * val] += current_count * count
        final_dist = new_dist
        
    return final_dist

def sample_proportionally(distribution_dict, target_samples=2000):
    """
    使用最大余额法 (Largest Remainder Method) 从分布中按严格比例抽取 target_samples 个样本。
    """
    total_combinations = sum(distribution_dict.values())
    
    # 记录每个 value 应该分到的精确浮点数样本量
    exact_allocations = {}
    for val, count in distribution_dict.items():
        exact_allocations[val] = target_samples * (count / total_combinations)
        
    # 第一轮分配：向下取整
    allocated_samples = {}
    remainders = []
    
    for val, exact_val in exact_allocations.items():
        floor_val = int(np.floor(exact_val))
        allocated_samples[val] = floor_val
        remainders.append((exact_val - floor_val, val))
        
    # 计算还差几个样本才到 target_samples
    current_total = sum(allocated_samples.values())
    missing_samples = target_samples - current_total
    
    # 第二轮分配：按小数部分从大到小排序
    remainders.sort(reverse=True, key=lambda x: x[0])
    
    for i in range(missing_samples):
        val_to_increment = remainders[i][1]
        allocated_samples[val_to_increment] += 1
        
    # 展开成最终的 target_samples 个元素的 list
    final_array = []
    for val, count in allocated_samples.items():
        final_array.extend([val] * count)
        
    # 打乱顺序
    np.random.shuffle(final_array)
    
    return final_array, allocated_samples

def plot_step_check(theoretical_dict, sampled_array, title, filename):
    """
    使用百分比阶梯图对比理论分布与采样分布。
    为了画出完美的理论基准曲线，我们生成一个 10000 点的超精细数组。
    """
    # 1. 强制排序
    arr2 = np.sort(np.array(sampled_array))
    
    # 生成一个 10000 点的理论分布数组作为平滑基准线
    theory_array, _ = sample_proportionally(theoretical_dict, target_samples=10000)
    arr1 = np.sort(np.array(theory_array))

    # 2. 独立生成 0~100 的百分比 X 轴
    x1 = np.linspace(0, 100, len(arr1)) if len(arr1) > 0 else []
    x2 = np.linspace(0, 100, len(arr2)) if len(arr2) > 0 else []

    plt.figure(figsize=(10, 6)) 

    if len(x1) > 0:
        plt.plot(x1, arr1, label='Original Theoretical', color='blue', alpha=1, linewidth=2)
        
    custom_red = (230/255, 0/255, 100/255)
    if len(x2) > 0:
        plt.plot(x2, arr2, label=f'Sampled Array (N={len(sampled_array)})', color=custom_red, alpha=0.8, linewidth=1.5)

    # 格式化 X 轴
    plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

    # 图表装饰
    plt.title(title)
    plt.xlabel('Percentage of Keys (Sorted by Probability)', fontsize=12)                               
    plt.ylabel('Normalized Probability ($P\ /\ P_{avg}$)', fontsize=12)                               
    plt.legend(loc='upper left', fontsize=12) 
    plt.grid(True, linestyle=':', alpha=0.7) 

    plt.tight_layout() 
    plt.savefig(filename, dpi=300)
    plt.show()

if __name__ == "__main__":
    # 18r
    distributions=[{0.8: 2, 1.2: 2}, {0.0: 2, 2.0: 2}, {0.0: 4, 0.5: 8, 3.0: 4}, {0.0: 4, 3.0: 4, 0.5: 8}, {0.0: 8, 0.875: 4, 0.75: 2, 3.25: 2, 1.125: 4, 2.25: 2, 1.75: 2, 1.625: 4, 0.375: 4}, {0.0: 72, 2.0: 48, 4.0: 8}, {1.75: 2, 0.25: 2}, {0.0: 2, 2.0: 2}, {0.0: 2, 2.0: 2}, {1.853323167615505: 1, 1.0359055566341824: 1, 2.0094572474998547: 1, 2.1589206863717925: 1, 3.1679774019809455: 1, 2.614470693927614: 1, 1.2217054051265148: 1, 0.0: 47, 1.364879661929566: 1, 2.499504836948309: 1, 2.1453740033773787: 1, 1.0711562754153134: 1, 2.5534177009665093: 1, 0.6545052991004905: 1, 0.8535388386335749: 1, 1.3491126922638659: 1, 0.4358804194679039: 1, 1.0662266521451442: 1, 1.1494140448292507: 1, 2.4218926134387986: 1, 2.1615517749822595: 1, 3.150635691548386: 1, 0.811891346799486: 1, 3.2735339383533004: 1, 4.207228060119886: 1, 2.3321617327234554: 1, 1.0540982139407593: 1, 0.7938747474668436: 1, 1.7063538833743075: 1, 1.9915091151570754: 1, 0.6386796434434787: 1, 1.4184501909984493: 1, 1.2301561878753764: 1, 2.8414876703177505: 1, 2.5550902517188883: 1, 1.5828589956577257: 1, 2.8064423525062097: 1, 1.9061209977987863: 1, 1.6341994570567704: 1, 1.0060050440609531: 1, 4.14123566289006: 1, 1.192127665505499: 1, 1.0239336144066284: 1, 1.7295544119394692: 1, 1.9436409082439634: 1, 1.362845214230766: 1, 0.9546743636604604: 1, 2.385409488840125: 1, 1.613620236103524: 1, 2.012049212116114: 1, 0.9942580648000935: 1, 2.790469981870919: 1, 3.8911942159086963: 1, 3.2359944659110194: 1}]
    
    final_distribution = multiply_distributions(distributions)
    print(f"理论最终分布共有 {len(final_distribution)} 种不同的乘积结果。")

    # ==========================================================
    # 任务 1: 全局 2000 点采样
    # ==========================================================
    sampled_array_2000, _ = sample_proportionally(final_distribution, target_samples=2000)
    
    save_filename_2000 = "TRACE_18R_OPT.npy"
    np.save(save_filename_2000, np.array(sampled_array_2000))
    print(f"✅ 全局2000个采样数据已保存至: {save_filename_2000}")

    plot_step_check(
        theoretical_dict=final_distribution,
        sampled_array=sampled_array_2000,
        title="Global Distribution Shape Preserved Check",
        filename="1_global.png"
    )

    # ==========================================================
    # 任务 2: 非零数据段 200 点独立采样
    # ==========================================================
    # 过滤出非零的数据字典
    nonzero_distribution = {k: v for k, v in final_distribution.items() if abs(k) > 1e-10}

    if not nonzero_distribution:
        print("⚠️ 警告: 最终分布中全部为 0，无法进行非零采样！")
    else:
        # 对非零分布抽取 200 个样本
        sampled_array_200, _ = sample_proportionally(nonzero_distribution, target_samples=200)
        
        save_filename_200 = "TRACE_18R_OPT_nonzero.npy"
        np.save(save_filename_200, np.array(sampled_array_200))
        print(f"✅ 非零段200个采样数据已保存至: {save_filename_200}")

        plot_step_check(
            theoretical_dict=nonzero_distribution,
            sampled_array=sampled_array_200,
            title="Non-Zero Distribution Shape Preserved Check (N=200)",
            filename="2_nonzero.png"
        )