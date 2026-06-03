import numpy as np
import itertools

SBOX=[1,10, 4, 12, 6, 15, 3, 9, 2, 13, 11, 7, 5, 0, 8, 14]

SBOX=[12, 6, 9, 0, 1, 10, 2, 11, 3, 8, 5, 13, 4, 14, 7, 15]
def xddt_list(input,output):
    if(input==0 and output==0):
        return [i for i in range(16)]
    res=[]
    for x in range(16):
        x1=x
        x2=x^input
        y1=SBOX[x1]
        y2=SBOX[x2]
        if(y1^y2==output and input!=0):
            res.append(x)
    return res

def inner(c,d):
    return bin(c & d).count('1') % 2

def get_QDTM(a,b,X_lst):
    res=np.zeros((16,16),dtype=int)
    if(X_lst is None or X_lst==[]):
        x_lst=xddt_list(a,b)
    else:
        x_lst=X_lst
    non_zero=0
    for u in range(16):
        for v in range(16):
            sum=0
            for x in x_lst:
                sum+=(-1)**(inner(u,x)^inner(v,SBOX[x]))
            res[u][v]=sum
            if(sum!=0):
                non_zero+=1
    return res

def analyze_mask_combinations(qdtm, bit_list):
    """
    根据选定的死引脚 (bit_list)，过滤并输出局部 LAT
    约定：0-3 对应 u 的第 0-3 位， 4-7 对应 v 的第 0-3 位。
    """
    k = len(bit_list)
    
    # === 新增：初始化一个全0的局部LAT矩阵 ===
    local_lat = np.zeros((16, 16), dtype=int)
    
    if k == 0:
        print("未选择任何死引脚，局部 LAT 即为原 LAT。")
        return 1.0, qdtm # 概率为1，矩阵不变

    stats = {}
    for bits_val in itertools.product([0, 1], repeat=k):
        stats[bits_val] = {'total': 0, 'nonzero': 0}

    # 遍历整个 16x16 矩阵
    for u in range(16):
        for v in range(16):
            current_pattern = []
            for pos in bit_list:
                if pos < 4:
                    bit_val = (u >> pos) & 1
                else:
                    bit_val = (v >> (pos - 4)) & 1
                current_pattern.append(bit_val)
            
            key = tuple(current_pattern)
            stats[key]['total'] += 1
            
            if qdtm[u][v] != 0:
                stats[key]['nonzero'] += 1
                # === 新增：只有当死引脚全部为0时，才保留该掩码对 ===
                if key == tuple([0] * k):
                    local_lat[u][v] = qdtm[u][v]

    # 构建死引脚必须为 0 的目标键
    target_key = tuple([0] * k)
    total = stats[target_key]['total']
    nonzero = stats[target_key]['nonzero']
    d_p = nonzero / total if total > 0 else 0
    
    print(f"\n--- 分析完毕: 约束引脚 {bit_list} ---")
    print(f'Dead pin all-zero prob (rho): {d_p}')
    
    # === 新增：直观打印局部 LAT 存活的掩码路径 ===
    print("=== 局部 LAT 存活路径 (非零项) ===")
    survived_count = 0
    for u in range(16):
        for v in range(16):
            if local_lat[u][v] != 0:
                # 打印二进制格式，方便核对位是否正确被限制
                print(f"  u={u:2d} ({u:04b}) -> v={v:2d} ({v:04b}) | Corr Value: {local_lat[u][v]:3d}")
                survived_count += 1
    if survived_count == 0:
        print("  无存活路径！(该死引脚约束导致局部轨迹彻底湮灭)")
        
    # 返回概率 d_p 以及 局部 LAT 矩阵
    return d_p, local_lat


def LAT_DIS(LAT):
    DIC={}
    for u in range(16):
        for v in range(16):
            if LAT[u][v] != 0:
                x=LAT[u][v]
                if abs(x) not in DIC:
                    DIC[abs(x)] = 0
                DIC[abs(x)] += 1
    return DIC

if __name__ == "__main__":
    a = 0
    b = 0
    qdtm_matrix = get_QDTM(a, b, None)
    
    print("=== 原始 LAT 矩阵 ===")
    print(qdtm_matrix)
    
    # 测试：选定死引脚并获取局部LAT
    print("\n\n>>> 施加局部死引脚约束: [0, 1, 4, 6, 7] <<<")
    rho, local_lat = analyze_mask_combinations(qdtm_matrix, [0, 1, 4, 6, 7])
    print(f"LAT: {local_lat}")
    print(f"LAT Distribution: {LAT_DIS(local_lat)}")
    
    # 您现在可以直接拿着 local_lat 去进行进一步的路径拼接和解耦分析了！