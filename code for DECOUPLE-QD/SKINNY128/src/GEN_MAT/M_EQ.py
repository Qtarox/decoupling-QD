import numpy as np
M_EQ_new = np.zeros((16, 40), dtype=int)
def cell(r, c):
    return r * 4 + c
for c in range(4):
    i = cell(0, c)
    M_EQ_new[i][i] = 1                                    
    M_EQ_new[i][16 + cell(0, c)] = 1                       
    M_EQ_new[i][16 + cell(2, (c-2) % 4)] = 1               
    M_EQ_new[i][16 + cell(3, (c-3) % 4)] = 1               
    M_EQ_new[i][32 + cell(0, c)] = 1                       
    i = cell(1, c)
    M_EQ_new[i][i] = 1
    M_EQ_new[i][16 + cell(0, c)] = 1
    M_EQ_new[i][32 + cell(1, c)] = 1                       
    i = cell(2, c)
    M_EQ_new[i][i] = 1
    M_EQ_new[i][16 + cell(1, (c-1) % 4)] = 1
    M_EQ_new[i][16 + cell(2, (c-2) % 4)] = 1
    i = cell(3, c)
    M_EQ_new[i][i] = 1
    M_EQ_new[i][16 + cell(0, c)] = 1
    M_EQ_new[i][16 + cell(2, (c-2) % 4)] = 1
for i in range(16):
    row = M_EQ_new[i]
    n_y = sum(row[16:32])
    n_k = sum(row[32:40])
    expected_y = {0:3, 1:1, 2:2, 3:2}[i // 4]
    expected_k = 1 if i < 8 else 0
np.save("M_EQ_SKINNY_correct.npy", M_EQ_new)