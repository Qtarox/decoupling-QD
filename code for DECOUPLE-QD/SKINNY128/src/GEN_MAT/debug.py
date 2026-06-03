import numpy as np
M_EQ = np.load("./M_EQ.npy")
print("M_EQ shape:", M_EQ.shape)
print("\n每一行（=每个 x_{r+1} cell 的来源）:")
for i in range(16):
    row = M_EQ[i]
    n_x = sum(row[0:16])
    n_y = sum(row[16:32])
    n_k = sum(row[32:40])
    print(f"cell {i}: x_next={n_x}, y_r 来源数={n_y}, key 数={n_k}")