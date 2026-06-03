import numpy as np

# 按 SKINNY 论文 row-major 编号重建 M_EQ
M_EQ_new = np.zeros((16, 40), dtype=int)

# (row, col) → row-major cell index
def cell(r, c):
    return r * 4 + c

# SKINNY MC: out[i,j] = ... (考虑 SR 反向 shift)
# row 0: out = y[0,j] ⊕ y[2,(j-2)%4] ⊕ y[3,(j-3)%4] + TK[0,j]
# row 1: out = y[0,j]                                + TK[1,j]
# row 2: out = y[1,(j-1)%4] ⊕ y[2,(j-2)%4]
# row 3: out = y[0,j]       ⊕ y[2,(j-2)%4]

for c in range(4):
    # row 0
    i = cell(0, c)
    M_EQ_new[i][i] = 1                                    # x_{r+1} cell i
    M_EQ_new[i][16 + cell(0, c)] = 1                       # y[0, c]
    M_EQ_new[i][16 + cell(2, (c-2) % 4)] = 1               # y[2, (c-2)%4]
    M_EQ_new[i][16 + cell(3, (c-3) % 4)] = 1               # y[3, (c-3)%4]
    M_EQ_new[i][32 + cell(0, c)] = 1                       # TK row 0
    
    # row 1
    i = cell(1, c)
    M_EQ_new[i][i] = 1
    M_EQ_new[i][16 + cell(0, c)] = 1
    M_EQ_new[i][32 + cell(1, c)] = 1                       # TK row 1
    
    # row 2
    i = cell(2, c)
    M_EQ_new[i][i] = 1
    M_EQ_new[i][16 + cell(1, (c-1) % 4)] = 1
    M_EQ_new[i][16 + cell(2, (c-2) % 4)] = 1
    
    # row 3
    i = cell(3, c)
    M_EQ_new[i][i] = 1
    M_EQ_new[i][16 + cell(0, c)] = 1
    M_EQ_new[i][16 + cell(2, (c-2) % 4)] = 1

# 但这里 TK 只在 row 0, 1 —— 即 cell 0..7
# 关键：M_EQ key 段 32..39 只有 8 项，对应 row 0, 1 的 8 个 cell

# 验证：
for i in range(16):
    row = M_EQ_new[i]
    n_y = sum(row[16:32])
    n_k = sum(row[32:40])
    expected_y = {0:3, 1:1, 2:2, 3:2}[i // 4]
    expected_k = 1 if i < 8 else 0
    print(f"cell {i}: y={n_y} (期望 {expected_y}), k={n_k} (期望 {expected_k})")

np.save("M_EQ_SKINNY_correct.npy", M_EQ_new)