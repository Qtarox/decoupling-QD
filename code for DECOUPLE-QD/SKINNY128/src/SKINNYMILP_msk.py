"""
SKINNY MILP for quasi-differential trail search
================================================
- 用 Gurobi solution pool 一次性枚举所有满足 |corr| >= 2^-MIN_CORR 的 trail
- 直接用 model.PoolObjVal 拿 log2|corr|，跳过 QDTM 矩阵累乘（省内存）
- 输出频率 mask 到 ./freq_msk/masks_freq_*.npy 供下游 MASK_DIVIDER 使用

支持 SKINNY-64 (SBOX_SIZE=4) 和 SKINNY-128 (SBOX_SIZE=8)，
所有维度依赖于 utils.SBOX_SIZE / BIT_RANGE / CORR_RANGE。
"""
import os
import numpy as np
from gurobipy import Model, GRB, quicksum
from tqdm import tqdm

from utils import (
    NB_ROUNDS, MIN_CORR, THRESH, SBOX_SIZE,
    STATE_RANGE, BIT_RANGE, CORR_RANGE,
    DIFF_TRAIL_FILE, SBOX_INEQUALITIES_DIR,
    bin_to_int, extract_diff_trail, extract_inequalities_by_corr,
)


# ---------- MILP 辅助：XOR ---------------------------------------------
def add_xor_constraints(model, x1, x2, y):
    """y = x1 XOR x2 (binary)"""
    model.addConstr(-x1 + x2 + y >= 0)
    model.addConstr( x1 - x2 + y >= 0)
    model.addConstr( x1 + x2 - y >= 0)
    model.addConstr(-x1 - x2 - y >= -2)


def add_xor_constraints2(model, x1, x2, x3, y):
    """y = x1 XOR x2 XOR x3 (binary)"""
    model.addConstr(-x1 + x2 + x3 + y >= 0)
    model.addConstr( x1 - x2 + x3 + y >= 0)
    model.addConstr( x1 + x2 - x3 + y >= 0)
    model.addConstr( x1 + x2 + x3 - y >= 0)
    model.addConstr( x1 - x2 - x3 - y >= -2)
    model.addConstr(-x1 + x2 - x3 - y >= -2)
    model.addConstr(-x1 - x2 + x3 - y >= -2)
    model.addConstr(-x1 - x2 - x3 + y >= -2)


# ---------- 主入口 -----------------------------------------------------
def SKINNY_MILP_Quasi_Diff(nb_rounds, save_pth=None):
    diff_trail = extract_diff_trail(DIFF_TRAIL_FILE, nb_rounds)
    sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)

    model = Model("SKINNY_SK_Quasi_Diff_MILP")

    # ===== 变量 =========================================================
    # u[r, side, i, j, l]: 第 r 轮 side ∈ {before-SBox=0, after-SBox=1} 的
    #                     state[i][j] 的第 l 个 bit 上的 mask 值
    # Q[r, i, j, corr]:   该 SBox 上选择 |corr| 值对应的 one-hot
    u = model.addVars(nb_rounds, 2, 4, 4, SBOX_SIZE, vtype=GRB.BINARY)
    Q = model.addVars(nb_rounds, 4, 4, CORR_RANGE, vtype=GRB.BINARY)

    # ===== 头尾零 mask 边界条件 =========================================
    for i in STATE_RANGE:
        for j in STATE_RANGE:
            model.addConstrs(u[0, 0, i, j, l] == 0 for l in BIT_RANGE)
            model.addConstrs(u[nb_rounds - 1, 1, i, j, l] == 0 for l in BIT_RANGE)

    # ===== 非线性层（S-box mask 传播） =================================
    for r in tqdm(range(nb_rounds), desc="building SBox constraints"):
        for i in STATE_RANGE:
            for j in STATE_RANGE:
                b = bin_to_int(
                    [diff_trail[r][1][i][j][l] for l in BIT_RANGE], SBOX_SIZE
                )
                a = bin_to_int(
                    [diff_trail[r][0][i][j][l] for l in BIT_RANGE], SBOX_SIZE
                )
                model.addConstr(quicksum(Q[r, i, j, c] for c in CORR_RANGE) == 1)
                for corr in CORR_RANGE:
                    if sbox_inequalities[b][a][corr] == []:
                        model.addConstr(Q[r, i, j, corr] == 0)
                        continue
                    for ineq in sbox_inequalities[b][a][corr]:
                        model.addConstr(
                            quicksum(ineq[2 * SBOX_SIZE - l - 1] * u[r, 1, i, j, l]
                                     for l in BIT_RANGE) +
                            quicksum(ineq[1 * SBOX_SIZE - l - 1] * u[r, 0, i, j, l]
                                     for l in BIT_RANGE) -
                            ineq[2 * SBOX_SIZE]
                            + 50000 * (1 - Q[r, i, j, corr]) >= 0
                        )

    # ===== 线性层（MixColumns + ShiftRows 复合） ========================
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:
            # row 0
            model.addConstrs(u[r, 1, 3, (j - 3) % 4, l] == u[r + 1, 0, 0, j, l]
                             for l in BIT_RANGE)
            # row 1
            for l in BIT_RANGE:
                add_xor_constraints2(
                    model,
                    u[r, 1, 0, j, l],
                    u[r, 1, 1, (j - 1) % 4, l],
                    u[r, 1, 2, (j - 2) % 4, l],
                    u[r + 1, 0, 1, j, l],
                )
            # row 2
            model.addConstrs(u[r, 1, 1, (j - 1) % 4, l] == u[r + 1, 0, 2, j, l]
                             for l in BIT_RANGE)
            # row 3
            for l in BIT_RANGE:
                add_xor_constraints2(
                    model,
                    u[r, 1, 1, (j - 1) % 4, l],
                    u[r, 1, 2, (j - 2) % 4, l],
                    u[r, 1, 3, (j - 3) % 4, l],
                    u[r + 1, 0, 3, j, l],
                )

    # ===== 目标 / correlation 下界 ======================================
    total_corr = quicksum(Q[r, i, j, c] * c
                          for r in range(nb_rounds)
                          for i in STATE_RANGE for j in STATE_RANGE
                          for c in CORR_RANGE)
    model.addConstr(total_corr >= -MIN_CORR)
    model.setObjective(total_corr, GRB.MAXIMIZE)

    print("Searching for quasi-differential trails...\n")
    model.params.PoolSearchMode = 2
    model.params.PoolSolutions = 2_000_000
    model.optimize()

    n_sol = model.SolCount
    print(f"Found {n_sol} trails")
    if n_sol == 0:
        print("no masks identified!")
        return

    # ===== 收集每个解的 mask + correlation ==============================
    print("Collecting masks and correlations from solution pool...")
    masks = []
    correlations = []
    for m in tqdm(range(n_sol), desc="reading pool"):
        model.params.SolutionNumber = m
        # 每个 cell 内 bit 顺序：LSB first（与下游 MASK_DIVIDER.get_active_bit 一致）
        mt = [[[[[round(u[r, side, i, j, l].Xn) for l in reversed(BIT_RANGE)]
                  for j in STATE_RANGE] for i in STATE_RANGE]
                  for side in range(2)] for r in range(nb_rounds)]
        arr = np.array(mt)
        # shape: (R, 2, 4, 4, SBOX_SIZE) -> (R, 2, 16 * SBOX_SIZE)
        masks.append(arr.reshape(*arr.shape[:2], 16 * SBOX_SIZE))
        correlations.append(model.PoolObjVal)

    # ===== 平均概率：取最大目标值（最不负的 log2|corr|），即全零 mask =====
    avg_prob = max(correlations)
    # sanity check：第 0 个解理论上就应当是全零 mask
    if not np.all(masks[0] == 0):
        print(f"⚠ 警告: pool[0] 不是全零 mask，max corr 来自其他解。"
              f" correlations[0] = {correlations[0]}, max = {avg_prob}")

    # ===== 频率 mask 累加 ===============================================
    res = np.zeros_like(masks[0], dtype=np.float64)
    for m in range(n_sol):
        w = 2 ** (correlations[m] - avg_prob)
        res += masks[m] * w

    # ===== 阈值化 =======================================================
    MSK = (res >= THRESH).astype(np.int8)

    # ===== 调试输出 =====================================================
    for r in range(nb_rounds):
        print(f'round {r}')
        for side in range(2):
            row = ','.join(f'{v:.3g}' for v in res[r][side])
            print(f'[{row}]')
    print('=' * 50)

    # ===== 保存 =========================================================
    if save_pth is None:
        save_pth = f'./freq_msk/masks_freq_{NB_ROUNDS}RD_CORR{MIN_CORR}'
    os.makedirs(os.path.dirname(save_pth) or '.', exist_ok=True)
    np.save(save_pth + f'_T{THRESH}.npy', MSK)
    np.save(save_pth + '.npy', res)
    print(f"thresholded MSK saved at: {save_pth}_T{THRESH}.npy")
    print(f"raw freq saved at:        {save_pth}.npy")
    print("+++++ MSK +++++")
    print(MSK)


if __name__ == "__main__":
    SKINNY_MILP_Quasi_Diff(NB_ROUNDS)