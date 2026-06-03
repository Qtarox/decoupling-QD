from gurobipy import *
from utils import *
from tqdm import tqdm
import time
import numpy as np
from math import log2
import sys
np.set_printoptions(threshold=np.inf, linewidth=sys.maxsize)

# ─── 差分路径（直接嵌入整数列表）──────────────────────────────────
from diffs import *

diffs= DIF[TRAIL_ID]

# ─── 预加载 ──────────────────────────────────────────────────────
diff_trail        = extract_diff_trail_from_list(diffs)
blocks            = get_transitions_from_list(diffs)
QDTM_RECT         = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)
sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)


def print_mask(mask_trail, nb_rounds):
    """打印一条 mask trail，按 j = 4*col + row 排列"""
    for n in range(nb_rounds):
        for side in range(2):
            for row in range(NB_ROWS):
                for col in range(NB_COLS):
                    print(mask_trail[n][side][4*col + row], end='')
                    if col % 4 == 3:
                        print(' ', end='')
                print()
        print()


def compute_correlation(diff_trail, mask_trail, nb_rounds):
    """
    在选项 A（RC 仅作用于 key state）下计算一条 trail 的 correlation。

    cipher-state 侧的轮变换是
        F_i(x) = ShiftRow ∘ SubColumn ∘ AddRoundKey_{K_i}(x)
    其中：
      - AddRoundKey 只引入对 K_i 的线性贡献，通过 mask_trail[k][0] 在
        AddRoundKey 上的 bit 体现 —— 这就是 trail 给出的密钥条件位置；
      - SubColumn 的 QDT 贡献来自 QDTM_RECT[b][a][v][u]；
      - ShiftRow 的贡献是 ±1，由 MILP 约束已强制满足（mask 自洽则 +1）。
    RC 不进 cipher-state 这一侧的乘积。
    """
    conditions = [[[] for _ in range(64)] for _ in range(nb_rounds)]
    corr = 1

    # ── S 盒贡献 ──────────────────────────────────────────────
    for k in range(nb_rounds):
        for col in range(NB_COLS):
            # MSB-first：row3 = MSB
            a_bits = [diff_trail[k][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            b_bits = [diff_trail[k][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            u_bits = [mask_trail[k][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            v_bits = [mask_trail[k][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]

            a = bin_to_int(a_bits, SBOX_SIZE)
            b = bin_to_int(b_bits, SBOX_SIZE)
            u = bin_to_int(u_bits, SBOX_SIZE)
            v = bin_to_int(v_bits, SBOX_SIZE)

            entry = QDTM_RECT[b][a][v][u]
            if entry == 0:
                print(f"[Error] Zero QDTM at r={k} col={col}: a={a} b={b} u={u} v={v}")
                return "Error"
            corr *= entry

    # ── 轮常数贡献（选项 A：不存在）─────────────────────────────
    # RECTANGLE 的 RC 仅在 key schedule 中 XOR 进 key state；不直接作用于
    # cipher state，因此对作用于 state 的 trail correlation 无贡献。
    # 若以后做 master-key 条件反推，RC 会在那里作为方程右端常数出现。

    # ── 密钥条件位置 ───────────────────────────────────────────
    # mask_trail[k][0][j] = 1  ⇔  trail 给出对子密钥位 K_{k,j} 的线性条件。
    for k in range(nb_rounds):
        for j in range(64):
            if mask_trail[k][0][j] != 0:
                conditions[k][j] = mask_trail[k][0][j]

    if corr > 0:
        return  1, log2( corr), conditions
    elif corr < 0:
        return -1, log2(-corr), conditions
    return 1, 0, []


# ─── 主函数 ──────────────────────────────────────────────────────
def RECTANGLE_MILP_Quasi_Diff(nb_rounds, THRESH=1, save_pth=None):

    model = Model("RECTANGLE_Quasi_Diff_MILP")

    # ── 变量 ──────────────────────────────────────────────────
    u = model.addVars(nb_rounds, 2, 64, vtype=GRB.BINARY, name="m")
    Q = model.addVars(nb_rounds, NB_COLS, CORR_RANGE, vtype=GRB.BINARY, name="c")

    # ── 边界约束 ───────────────────────────────────────────────
    model.addConstrs(u[0,          0, j] == 0 for j in range(64))
    model.addConstrs(u[nb_rounds-1, 1, j] == 0 for j in range(64))

    # ── S 盒约束 ───────────────────────────────────────────────
    for r in tqdm(range(nb_rounds), desc="S-box constraints"):
        for col in range(NB_COLS):
            a_bits = [diff_trail[r][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            b_bits = [diff_trail[r][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            a = bin_to_int(a_bits, SBOX_SIZE)
            b = bin_to_int(b_bits, SBOX_SIZE)

            model.addConstr(quicksum(Q[r, col, corr] for corr in CORR_RANGE) == 1)

            for corr in CORR_RANGE:
                if sbox_inequalities[b][a][corr] == []:
                    model.addConstr(Q[r, col, corr] == 0)
                    continue
                for ineq in sbox_inequalities[b][a][corr]:
                    model.addConstr(
                        quicksum(ineq[l]           * u[r, 1, 4*col + (SBOX_SIZE-1-l)]
                                 for l in BIT_RANGE) +
                        quicksum(ineq[SBOX_SIZE+l] * u[r, 0, 4*col + (SBOX_SIZE-1-l)]
                                 for l in BIT_RANGE) +
                        ineq[2*SBOX_SIZE] + 50000*(1 - Q[r, col, corr]) >= 0
                    )

    # ── 线性层约束（ShiftRow 在 mask 上的正向传播）────────────
    for r in range(nb_rounds - 1):
        model.addConstrs(
            u[r, 1, j] == u[r+1, 0, RECT_PERM[j]]
            for j in range(64)
        )

    model.write("rectangle_quasi_diff.lp")

    # ── 目标函数 ───────────────────────────────────────────────
    total_corr = quicksum(
        Q[r, col, corr] * corr
        for r in range(nb_rounds)
        for col in range(NB_COLS)
        for corr in CORR_RANGE
    )
    model.addConstr(total_corr >= -MIN_CORR)
    model.setObjective(total_corr, GRB.MAXIMIZE)

    # ── Gurobi 参数 ─────────────────────────────────────────────
    model.params.PoolSearchMode = 2
    model.params.PoolSolutions  = 2000000

    t1 = time.time()
    model.optimize()
    t = time.time() - t1
    print(f"Time used: {t:.2f}s")
    print(f"Found {model.SolCount} trails")
    sol_num = model.SolCount

    # ── 提取每条 trail 的 mask、sign、相关度 ─────────────────────
    print("Computing sign and conditions for each trail...")
    signs = []
    correlations = []
    trails_conditions = []
    corr_dict = {}
    MASK = []
    ref_corr = None     # 第一条（即最佳 |corr|）trail 的 log2|c|，用作权重基准

    for m in tqdm(range(model.SolCount)):
        model.params.SolutionNumber = m
        mask_trail = [
            [
                [round(u[r, side, j].Xn) for j in range(64)]
                for side in range(2)
            ]
            for r in range(nb_rounds)
        ]

        result = compute_correlation(diff_trail, mask_trail, nb_rounds)
        if result == "Error":
            continue
        sign, corr, conditions = result

        if ref_corr is None:
            ref_corr = corr

        corr_dict[corr] = corr_dict.get(corr, 0) + 1
        signs.append(sign)
        correlations.append(corr)
        trails_conditions.append(conditions)
        MASK.append(mask_trail)

    print(f"Total valid masks: {len(MASK)}")

    # ── 加权累加 mask（按相关度大小）─────────────────────────────
    if len(MASK) == 0:
        print("no masks identified!")
        return None, sol_num, None

    # 修正：初始化为零，避免把 MASK[0] 重复累计。
    res = np.zeros_like(np.array(MASK[0], dtype=np.float64))
    for m in range(len(MASK)):
        # 权重 = 2^(corr_m - ref_corr)，|corr| 越大权重越大
        weight = 2 ** (correlations[m] - ref_corr)
        res += np.array(MASK[m]) * weight

    # ── 阈值过滤生成二进制重要性 mask ───────────────────────────
    MSK = res.copy()
    for r in range(nb_rounds):
        for i in range(2):
            for j in range(64):
                if MSK[r][i][j] < THRESH:
                    MSK[r][i][j] = 0
                else:
                    MSK[r][i][j] = 1

    # ── 保存 ───────────────────────────────────────────────────
    if save_pth is None:
        save_pth = f'./freq_msk/masks_freq_RECT_{nb_rounds}RD_CORR{MIN_CORR}'

    np.save(save_pth + f'_T{THRESH}.npy', np.array(MSK))
    print(f"Filtered mask saved at: {save_pth}_T{THRESH}.npy")
    np.save(save_pth + '.npy', np.array(res))
    print(f"Original (weighted) mask saved at: {save_pth}.npy")

    return MSK, sol_num, save_pth + f'_T{THRESH}.npy'


if __name__ == "__main__":
    masks = RECTANGLE_MILP_Quasi_Diff(NB_ROUNDS)