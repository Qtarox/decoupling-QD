import numpy as np
from utils import RECT_PERM, NB_ROWS, NB_COLS

def genLinear_RECT(rounds):
    """
    生成 RECTANGLE 线性层的约束矩阵 L。
    
    每轮的状态变量：
      x_{r}_{j} : 第 r 轮 before-sbox（即 AddRoundKey 输出 = SubColumn 输入），j = 0..63
      y_{r}_{j} : 第 r 轮 after-sbox（即 ShiftRow 输入），j = 0..63
      k_{r}_{j} : 第 r 轮注入的轮密钥，j = 0..63
    
    线性约束（融合了 ShiftRow 和 AddRoundKey）：
      y_{r}_{j} ⊕ x_{r+1}_{RECT_PERM[j]} ⊕ k_{r+1}_{RECT_PERM[j]} = 0
    
    L 的列分布：
      0 ~ 128*(rounds+1)-1: 状态变量（每轮 x[64] + y[64]）
      128*(rounds+1) ~ : 密钥变量 k（分配 (rounds+1)*64 的空间以对其索引）
    """
    STATE_COLS = 128 * (rounds + 1)
    
    # 密钥列起始于 STATE_COLS。
    # 为了让 k_r_j 的偏移量刚好等于 STATE_COLS + r * 64 + j，我们分配 (rounds + 1) * 64 的密钥空间。
    KEY_COLS = 64 * (rounds + 1)  
    TOTAL_COLS = STATE_COLS + KEY_COLS
    
    rows = []
    
    # ── 融合 ShiftRow 与 AddRoundKey 约束 ──────
    for r in range(rounds):
        for j in range(64):
            row = np.zeros(TOTAL_COLS, dtype=np.int8)
            
            # 1. 对应 y_{r}_{j} (ShiftRow 的输入)
            row[r * 128 + 64 + j] = 1
            
            if r + 1 <= rounds:
                # 经过 ShiftRow 后，位 j 去到了目标位置 target_bit
                target_bit = RECT_PERM[j]
                
                # 2. 对应 x_{r+1}_{target_bit} (AddRoundKey 的输出，S盒输入)
                row[(r + 1) * 128 + target_bit] = 1
                
                # 3. 对应 k_{r+1}_{target_bit} (在进入 S盒前异或的轮密钥)
                # 全局索引 = 状态列总数 + 轮次偏移量 + 比特偏移量
                row[STATE_COLS + (r + 1) * 64 + target_bit] = 1
                
            rows.append(row)
            
    L = np.array(rows, dtype=np.int8)
    return L