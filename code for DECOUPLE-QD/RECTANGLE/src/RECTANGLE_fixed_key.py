from gurobipy import *
from utils import *
from tqdm import tqdm
import time
import numpy as np
from math import log2

# ─── 差分路径（直接嵌入，无需文件）──────────────────────────────
diffs = [
    0x0020000600000000,
    0x0060000200000000,
    0x0200006000000000,
    0x0600002000000000,
    0x2000060000000000,
    0x6000020000000000,
    0x0000600000000002,
    0x0000200000000006,
    0x0006000000000020,
    0x0002000000000060,
    0x0060000000000200,
    0x0020000000000600,
    0x0600000000002000,
    0x0200000000006000,
    0x6000000000020000,
    0x2000000000060000,
    0x0000000000200006,
    0x0000000000600002,
    0x0000000002000060,
    0x000000000c000020,
    0x0000000000008600,
    0x0000000000001200,
    0x0000000000003000,
    0x0000000000008000,
    0x0000000000000008,
    0x0000000000000001,
    0x0000000000000001,
    0x0000000000000006,
    0x0004000000000020,
]
#diffs = [
#    0x0020000600000000,
#    0x0060000200000000,
#    0x0200006000000000,
#    0x0600002000000000,
#    0x2000060000000000,
#    0x6000020000000000,
#    0x0000600000000002,
#    0x0000200000000006,
#    0x0006000000000020,
#    0x0002000000000060,
#    0x0060000000000200,
#    0x0020000000000600,
#    0x0600000000002000,
#    0x0200000000006000,
#    0x6000000000020000,
#    0x2000000000060000,
#    0x0000000000200006,
#    0x0000000000600002,
#    0x0000000002000060,
#    0x000000000c000020,
#    0x0000000000008600,
#    0x0000000000009200,
#    0x0000000000003008,
#    0x0000000000008001,
#    0x0000000000000009,
#    0x0000000000000001,
#    0x0000000000000001,
#    0x0000000000000006,
#    0x0004000000000020
#]

# ─── 全局约定说明 ─────────────────────────────────────────────────
# 索引 j = 4*col + row，与Tim Beyne原始代码完全一致：
#   col = j // 4,  row = j % 4
#   state[row][col] = (val >> j) & 1
#   第col列nibble = (val >> (4*col)) & 0xf，LSB=row0，MSB=row3
#
# RECT_PERM[j]：行移位等价置换，由utils.py预计算

# ─── 预加载数据 ───────────────────────────────────────────────────
diff_trail        = extract_diff_trail_from_list(diffs)   # utils新增函数
blocks            = get_transitions_from_list(diffs)       # utils新增函数
QDTM_RECT         = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)
sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)

# ─── 打印工具 ─────────────────────────────────────────────────────
def print_mask(mask_trail, nb_rounds):
    """
    以4行×16列格式打印，索引约定 j=4*col+row。
    每行打印一个row，按col从0到15排列。
    """
    for n in range(nb_rounds):
        for side, label in enumerate(['in ', 'out']):
            print(f"  r{n} {label}:", end='')
            for row in range(NB_ROWS):
                s = ''.join(
                    str(mask_trail[n][side][4 * col + row])
                    for col in range(NB_COLS)
                )
                print(f"  row{row}=[{s}]", end='')
            print()
        print()

# ─── 相关度计算 ───────────────────────────────────────────────────
def compute_correlation(diff_trail, mask_trail, nb_rounds):
    """
    计算一条quasi-differential trail的总相关度。
    索引约定：j = 4*col + row，与Tim Beyne原始代码一致。
    
    对应原始代码 compute_sign 的逻辑：
      a[j] = (diffs[2*i] >> (4*j)) & 0xf  →  第j列输入差分nibble
      u[j] = (trail[2*i] >> (4*j)) & 0xf  →  第j列输入mask nibble
    """
    conditions = [[[] for _ in range(64)] for _ in range(nb_rounds)]
    corr = 1

    # ── S盒贡献 ────────────────────────────────────────────────────
    for k in range(nb_rounds):
        for col in range(NB_COLS):
            # 第col列：索引 4*col+row，row=0..3
            # bin_to_int用MSB-first，row3=MSB，row0=LSB
            # 与原始代码 (val >> (4*col)) & 0xf 等价（LSB=row0）
            a_bits = [diff_trail[k][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            b_bits = [diff_trail[k][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            u_bits = [mask_trail[k][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            v_bits = [mask_trail[k][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]

            a = bin_to_int(a_bits, SBOX_SIZE)
            b = bin_to_int(b_bits, SBOX_SIZE)
            u = bin_to_int(u_bits, SBOX_SIZE)
            v = bin_to_int(v_bits, SBOX_SIZE)
            print(f'position in mat [{a}][{b}][{v}][{u}]: ')
            # print(f"第一维长度: {len(QDTM_RECT)}")
            # print(f"第二维长度: {len(QDTM_RECT[a])}")
            # print(f"第一个元素的子项长度: {len(QDTM_RECT[a][b])}")
            # print(f"第一个元素的子项长度2: {len(QDTM_RECT[a][b][v])}")
            entry = QDTM_RECT[b][a][v][u]
            if entry == 0:
                print(f"  [Error] Zero QDTM at r={k} col={col}: a={a} b={b} u={u} v={v}")
                return "Error"
            corr *= entry

    # ── 轮常数贡献 ─────────────────────────────────────────────────
    # rect_rc_corr_factor 在utils.py中定义，使用 mask_trail[k][0]（before-sbox）
    for k in range(nb_rounds):
        corr *= rect_rc_corr_factor(mask_trail[k][0], RECTANGLE_RC[k])

    # ── 密钥条件 ───────────────────────────────────────────────────
    for k in range(nb_rounds):
        for j in range(64):
            if mask_trail[k][1][j] != 0:
                conditions[k][j] = mask_trail[k][1][j]

    if corr > 0:
        return  1, log2( corr), conditions
    elif corr < 0:
        return -1, log2(-corr), conditions
    return 1, 0, []

# ─── 主函数 ───────────────────────────────────────────────────────
def RECTANGLE_MILP_Quasi_Diff(nb_rounds):

    model = Model("RECTANGLE_Quasi_Diff_MILP")

    # ── 变量 ──────────────────────────────────────────────────────
    # u[r, side, j]：side=0 before-sbox，side=1 after-sbox
    # j = 4*col + row，与diff_trail展平约定完全一致
    u = model.addVars(nb_rounds, 2, 64, vtype=GRB.BINARY, name="m")
    Q = model.addVars(nb_rounds, NB_COLS, CORR_RANGE, vtype=GRB.BINARY, name="c")

    # ── 边界约束 ───────────────────────────────────────────────────
    model.addConstrs(u[0,          0, j] == 0 for j in range(64))
    model.addConstrs(u[nb_rounds-1, 1, j] == 0 for j in range(64))

    # ── S盒约束 ────────────────────────────────────────────────────
    # 第col列：bit索引 [4*col+0, 4*col+1, 4*col+2, 4*col+3]
    #   row0=LSB=索引偏移0，row3=MSB=索引偏移3
    # bin_to_int要求MSB-first，所以取bits时反转：row3,row2,row1,row0
    for r in tqdm(range(nb_rounds), desc="S-box constraints"):
        for col in range(NB_COLS):
            # 统一用 row3(MSB) → row0(LSB) 的顺序
            a_bits = [diff_trail[r][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            b_bits = [diff_trail[r][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            a = bin_to_int(a_bits, SBOX_SIZE)
            b = bin_to_int(b_bits, SBOX_SIZE)

            # 每列恰好选一个相关度
            model.addConstr(quicksum(Q[r, col, corr] for corr in CORR_RANGE) == 1)

            for corr in CORR_RANGE:
                if sbox_inequalities[b][a][corr] == []:
                    model.addConstr(Q[r, col, corr] == 0)
                    continue
                for ineq in sbox_inequalities[b][a][corr]:
                    # ineq布局：[out_coeffs(4) | in_coeffs(4) | const]
                    # out=after-sbox(side=1)，in=before-sbox(side=0)
                    # ineq[0]=row3(MSB), ineq[3]=row0(LSB)，与bin_to_int MSB-first一致
                    model.addConstr(
                        quicksum(ineq[l]          * u[r, 1, 4*col + (SBOX_SIZE-1-l)]
                                 for l in BIT_RANGE) +
                        quicksum(ineq[SBOX_SIZE+l] * u[r, 0, 4*col + (SBOX_SIZE-1-l)]
                                 for l in BIT_RANGE) +
                        ineq[2*SBOX_SIZE] + 50000*(1 - Q[r, col, corr]) >= 0
                    )

    # ── 线性层约束（行移位，使用utils预计算的RECT_PERM）──────────
    # RECT_PERM[j]：after-sbox第j位 → next round before-sbox的目标位
    # 验证：j=4*col+row → 4*((col+ROW_SHIFT[row])%16)+row
    for r in range(nb_rounds - 1):
        model.addConstrs(
            u[r, 1, j] == u[r+1, 0, RECT_PERM[j]]
            for j in range(64)
        )

    model.write("rectangle_quasi_diff.lp")

    # ── 目标函数 ───────────────────────────────────────────────────
    total_corr = quicksum(
        Q[r, col, corr] * corr
        for r in range(nb_rounds)
        for col in range(NB_COLS)
        for corr in CORR_RANGE
    )
    model.addConstr(total_corr >= -MIN_CORR)
    model.setObjective(total_corr, GRB.MAXIMIZE)

    # ── Gurobi参数 ─────────────────────────────────────────────────
    model.params.PoolSearchMode = 2
    model.params.PoolSolutions  = 2000000

    # ── 求解 ───────────────────────────────────────────────────────
    t1 = time.time()
    model.optimize()
    print(f"\nTime used: {time.time()-t1:.2f}s")
    print(f"Found {model.SolCount} trails")

    # ── 后处理 ─────────────────────────────────────────────────────
    signs, correlations, trails_conditions = [], [], []
    corr_dict = {}
    MSKS = []

    for m in tqdm(range(model.SolCount), desc="Computing correlations"):
        model.params.SolutionNumber = m

        # 读取mask trail，索引约定与u变量完全一致：j=4*col+row
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

        corr_dict[corr] = corr_dict.get(corr, 0) + 1
        signs.append(sign)
        correlations.append(corr)
        trails_conditions.append(conditions)
        MSKS.append(mask_trail)

    # ── 打印所有trail ──────────────────────────────────────────────
    print("=" * 60)
    for cnt, (mk, corr_val) in enumerate(zip(MSKS, correlations)):
        print(f"Trail {cnt}  |  log2(|corr|) = {corr_val:.4f}")
        print_mask(mk, nb_rounds)

    # ── 因子分析 ───────────────────────────────────────────────────
    if corr_dict:
        factor_dict = {
            corr: int(2 ** (corr - min(corr_dict)))
            for corr in sorted(corr_dict)
        }
        trail_indic(trails_conditions, correlations, signs, factor_dict, nb_rounds)

    # ── 导出mask文件 ───────────────────────────────────────────────
    out_mask = ""
    for m in tqdm(range(model.SolCount), desc="Exporting"):
        model.params.SolutionNumber = m
        for r in range(nb_rounds):
            for side in range(2):
                for col in range(NB_COLS):
                    # 还原nibble（MSB=row3，LSB=row0）
                    nibble = bin_to_int(
                        [round(u[r, side, 4*col + row].Xn)
                         for row in range(NB_ROWS-1, -1, -1)],
                        SBOX_SIZE
                    )
                    out_mask += str(nibble) + " "
        if m != model.SolCount - 1:
            out_mask += "\n"

    with open(RESULTS_FILE + "_mask", 'w') as f:
        f.write(out_mask)
    solutions_to_readable(RESULTS_FILE + "_mask")

    return model.SolCount


if __name__ == "__main__":
    nb_sol = RECTANGLE_MILP_Quasi_Diff(NB_ROUNDS)
    print(f"number of trails: {nb_sol}")