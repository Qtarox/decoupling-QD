"""
GEN_LINEAR.py  --  SKINNY-64 版本 (TK1 / TK2 / TK3 统一)
========================================================

修正点（相对旧版纯 TK1 实现）：
  1. 旧版 KEY_BITS = HALF_STATE_BITS，只有一条 64-bit 主密钥列 -> 永远是 TK1。
     现按 ADV_MODEL 选寄存器数 N_TK (SK/TK1=1, TK2=2, TK3=3)，
     KEY_BITS = N_TK * HALF_STATE_BITS。
  2. 旧版 key_schedule 只有置换 P_T，没有 LFSR；TK2/TK3 在置换后还要对
     前两行 (cell 0..7) 的寄存器 cell 套 LFSR。LFSR 是 GF(2) 线性映射，
     一个轮密钥 bit 会展开成若干主密钥 bit 的异或。
  3. 旧版注入一个 key 列后 break -> 只能表达单寄存器；现对每个激活寄存器
     各注入一份 (TK1 ⊕ TK2 ⊕ TK3)。TK1 时 LFSR = 单位阵，退化为旧行为。
  4. 旧版 show_L_equ_GIFT(_extract) 每命中一个 key 就补 "= 0"，多 key 时会
     打出 "+ k_3 = 0 + k_67 = 0"；现统一在末尾补一个 "= 0"。

state / 列布局 (与 bit 级 divider 完全一致)：
    CELL_SIZE        = 4
    HALF_STATE_BITS  = 64                      (16 cell * 4 bit)
    FULL_STATE_BITS  = 128                     (x 半态 + y 半态)
    每轮状态列        = FULL_STATE_BITS
    总状态列          = (round_num + 1) * FULL_STATE_BITS
    key 列            = N_TK * HALF_STATE_BITS  (依次 TK1 | TK2 | TK3)

key 扁平标号约定：
    TK1 -> k_0   .. k_63
    TK2 -> k_64  .. k_127
    TK3 -> k_128 .. k_191
即 col(z, master_cell, bit) = key_base + z*HALF_STATE_BITS + master_cell*CELL_SIZE + bit
"""

import numpy as np

# ============================================================
# ADV_MODEL -> 寄存器数
# ============================================================
try:
    from utils import ADV_MODEL
except Exception:
    ADV_MODEL = "TK1"   # standalone 测试时的回退

_NTK_BY_MODE = {"SK": 1, "TK1": 1, "TK2": 2, "TK3": 3}
if ADV_MODEL not in _NTK_BY_MODE:
    raise ValueError(f"未知 ADV_MODEL={ADV_MODEL}; 支持 {sorted(_NTK_BY_MODE)}")
N_TK = _NTK_BY_MODE[ADV_MODEL]

# ============================================================
# SKINNY-64 常量
# ============================================================
CELL_SIZE        = 4                          # 每个 cell 的 bit 数 (s = 4)
HALF_STATE_BITS  = 16 * CELL_SIZE             # 半态 = 16 cell = 64 bit
FULL_STATE_BITS  = 2 * HALF_STATE_BITS        # 完整状态 (x + y) = 128 bit
KEY_BITS         = N_TK * HALF_STATE_BITS     # 全部寄存器的主密钥总宽度

# 4-bit S-box (SKINNY-64) —— 论文官方值
Sbox = [0xc, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb,
        0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf]

# cell 级方程矩阵 (16 x 40): [0:16]=x_{r+1}, [16:32]=y_r, [32:40]=round-tweakey(cell 0..7)
M_EQ = np.load("./M_EQ.npy")

# ============================================================
# tweakey schedule: 置换 + LFSR
# ============================================================
# SKINNY tweakey 置换 P_T (SKINNY-64/128 相同): new[i] = old[P_T[i]]
P_T = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]

# s=4 的 LFSR, 作用在列向量 [b0,b1,b2,b3]^T (b0=LSB), new = L @ old (mod 2)
# TK2: (x3,x2,x1,x0)->(x2,x1,x0,x3^x2)
_L2 = np.array([[0, 0, 1, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]], dtype=int)
# TK3: (x3,x2,x1,x0)->(x0^x3,x3,x2,x1)   (= L2 的逆)
_L3 = np.array([[0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 1]], dtype=int)
_I4 = np.eye(CELL_SIZE, dtype=int)

if N_TK >= 2 and CELL_SIZE != 4:
    raise NotImplementedError("此文件的 TK2/TK3 LFSR 仅为 s=4 (SKINNY-64) 定义")


def key_schedule(round=0, key_index=0):
    """保留旧 API：返回 round 轮后落在 round-tweakey 位置 key_index 的主密钥 cell 号 = P_T^round[key_index]."""
    tmp = key_index
    for _ in range(round):
        tmp = P_T[tmp]
    return tmp


def _build_tweakey_schedule(round_num):
    """
    符号推进生成每轮、每个寄存器、前两行 (位置 0..7) 的轮密钥 cell。

    返回 sched[z][r][j] = (master_cell_index, B)   z=0..N_TK-1, r=0..round_num-1, j=0..7
        B 为 CELL_SIZE x CELL_SIZE 的 GF(2) 矩阵: 该 round-tweakey bit = B @ (master cell 的 4 个 bit)
    TK1 (z=0) 的 B 恒为单位阵 -> 退化为单点映射。
    """
    lfsrs = [_I4, _L2, _L3][:N_TK]   # z=0:TK1(单位) 1:TK2 2:TK3
    sched = []
    for z in range(N_TK):
        L = lfsrs[z]
        # 寄存器初始状态: 16 个 cell, 第 i 个 = (主密钥 cell i, 单位阵)
        state = [(i, _I4.copy()) for i in range(16)]
        per_round = []
        for r in range(round_num):
            # 读出本轮前两行 (位置 0..7) 的轮密钥, 须在更新之前
            per_round.append([(state[j][0], state[j][1].copy()) for j in range(8)])
            # 更新到下一轮: 先置换, 再对位置 0..7 套 LFSR
            newstate = [(state[P_T[i]][0], state[P_T[i]][1].copy()) for i in range(16)]
            for i in range(8):
                m, B = newstate[i]
                newstate[i] = (m, (L @ B) % 2)   # 累积一次 LFSR (current = B@master -> L@current = (L@B)@master)
            state = newstate
        sched.append(per_round)
    return sched


# ============================================================
# 矩阵生成
# ============================================================
def Global_mat(res, M_EQ, round_num):
    """
    Cell 级版本 (legacy, bit 级 divider 不用它)。
    多寄存器时, 在 cell 粒度上注入 TK1 ⊕ TK2 ⊕ TK3 各一个 cell 列;
    注意 LFSR 是 cell 内 bit 混合, 在 cell 粒度无法表达, 故此函数只对 TK1 严格正确,
    TK2/TK3 仅作 cell 级近似 (忽略 LFSR)。调用方须按 32*(round_num+1) + N_TK*16 列分配 res。
    """
    key_base_cell = 32 * (round_num + 1)
    for i in range(np.shape(res)[0]):
        equ_num = i % 16
        rn = i // 16
        rn_k_ind = None
        for k in range(40):
            if M_EQ[equ_num][k] == 1:
                if k < 16:
                    res[i][(rn + 1) * 32 + k] = 1
                elif k < 32:
                    res[i][rn * 32 + k] = 1
                else:
                    rn_k_ind = k - 32
                    break
        if rn_k_ind is not None:
            k_ind = key_schedule(rn, rn_k_ind)
            for z in range(N_TK):
                res[i][key_base_cell + z * 16 + k_ind] = 1
    return res


def Global_mat_bit(round_num):
    """
    Bit 级版本: 每个 cell 展开成 CELL_SIZE 条 bit 方程。
    tweakey 注入按 TK1 ⊕ TK2 ⊕ TK3, 每个寄存器按其累积 LFSR 展开成若干主密钥 bit。
    """
    num_rows = HALF_STATE_BITS * round_num
    num_cols = FULL_STATE_BITS * (round_num + 1) + KEY_BITS
    res = np.zeros((num_rows, num_cols), dtype=int)

    key_base = FULL_STATE_BITS * (round_num + 1)
    sched = _build_tweakey_schedule(round_num)

    for i in range(num_rows):
        equ_num_bit = i % HALF_STATE_BITS       # 轮内第几条 bit 方程
        rn = i // HALF_STATE_BITS               # 第几轮
        equ_num_cell = equ_num_bit // CELL_SIZE # 对应 M_EQ 的 cell 方程 (0..15)
        bit_idx = equ_num_bit % CELL_SIZE       # cell 内 bit 偏移 (0..CELL_SIZE-1)

        for k in range(40):
            if M_EQ[equ_num_cell][k] != 1:
                continue

            if k < 16:
                # x_{r+1}_k -> 下一轮 x 半态 [0, HALF)
                res[i][(rn + 1) * FULL_STATE_BITS + k * CELL_SIZE + bit_idx] = 1
            elif k < 32:
                # y_r_k -> 当前轮 y 半态 [HALF, FULL); k>=16 时 k*CELL_SIZE 恰好落在 y 半态
                res[i][rn * FULL_STATE_BITS + k * CELL_SIZE + bit_idx] = 1
            else:
                # round-tweakey cell j = k - 32 (j in 0..7): 注入所有激活寄存器
                j = k - 32
                for z in range(N_TK):
                    m, B = sched[z][rn][j]
                    for bp in range(CELL_SIZE):
                        if B[bit_idx][bp] == 1:
                            col = key_base + z * HALF_STATE_BITS + m * CELL_SIZE + bp
                            res[i][col] = 1
                break   # M_EQ 每行至多一个 round-tweakey cell
    return res


# ============================================================
# 打印
# ============================================================
def _fmt_state_var(j):
    """bit 索引 j -> 可读的 x_r_b / y_r_b."""
    rn = j // FULL_STATE_BITS
    ind = j % FULL_STATE_BITS
    if ind < HALF_STATE_BITS:
        return f"x_{rn}_{ind}"
    return f"y_{rn}_{ind - HALF_STATE_BITS}"


def show_L_equ_GIFT(lmat, active_bit_dic, round_num, FILE_FLG=False, file=None):
    """按 GF(2) 方程打印 lmat; active_bit_dic 中的状态变量用 [..] 强调。返回整段字符串。"""
    F = file if file else "output.txt"
    L = ""
    out_handle = open(F, "w") if FILE_FLG else None

    state_cols = FULL_STATE_BITS * (round_num + 1)
    key_base = (round_num + 1) * FULL_STATE_BITS
    try:
        for i in range(np.shape(lmat)[0]):
            l_tmp = ""
            for j in range(state_cols):
                if lmat[i][j] == 1:
                    var_name = _fmt_state_var(j)
                    if str(j) in active_bit_dic:
                        l_tmp += f" + [{var_name}]"
                    else:
                        l_tmp += f" + {var_name}"

            # key 段: 先拼所有 key 项, 最后统一补一个 "= 0"
            key_terms = ""
            for k in range(KEY_BITS):
                if lmat[i][key_base + k] == 1:
                    key_terms += f" + k_{k}"
            if key_terms:
                l_tmp += key_terms + " = 0"
            else:
                l_tmp += "= 0   "

            # 兼容: 若矩阵尾部还有额外列 (旧版 SBOX 标志)
            if np.shape(lmat)[1] > key_base + KEY_BITS:
                l_tmp += "  SBOX: " + str(lmat[i][-1])

            print(l_tmp)
            if out_handle:
                print(l_tmp, file=out_handle)
            L += l_tmp + "\n"
    finally:
        if out_handle:
            out_handle.close()
    return L


def show_L_equ_GIFT_extract(lmat, round_num, FILE_FLG=False, file=None):
    """同上, 但用矩阵中的数值 2 标记 active/masked 变量 (而非外部字典)。"""
    F = file if file else "output.txt"
    out_handle = open(F, "w") if FILE_FLG else None
    state_cols = FULL_STATE_BITS * (round_num + 1)
    key_base = (round_num + 1) * FULL_STATE_BITS
    try:
        for i in range(np.shape(lmat)[0]):
            l_tmp = ""
            for j in range(state_cols):
                if lmat[i][j] == 2:
                    l_tmp += f" + [{_fmt_state_var(j)}]"
                elif lmat[i][j] == 1:
                    l_tmp += f" + {_fmt_state_var(j)}"

            key_terms = ""
            for k in range(KEY_BITS):
                if lmat[i][key_base + k] == 1:
                    key_terms += f" + k_{k}"
            if key_terms:
                l_tmp += key_terms + " = 0"
            else:
                l_tmp += "= 0   "

            print(l_tmp)
            if out_handle:
                print(l_tmp, file=out_handle)
    finally:
        if out_handle:
            out_handle.close()


# ============================================================
# standalone 自检
# ============================================================
if __name__ == "__main__":
    print(f"ADV_MODEL={ADV_MODEL}  N_TK={N_TK}  "
          f"CELL_SIZE={CELL_SIZE} HALF={HALF_STATE_BITS} FULL={FULL_STATE_BITS} KEY_BITS={KEY_BITS}")

    # LFSR 互逆性检查 (L3 应为 L2 的逆)
    assert np.array_equal((_L2 @ _L3) % 2, _I4), "L2 @ L3 != I"
    print("[ok] L2 @ L3 == I (TK3 LFSR 为 TK2 的逆)")

    R = 4
    res = Global_mat_bit(R)
    print(f"Global_mat_bit({R}) shape = {res.shape}  "
          f"(期望 cols = {FULL_STATE_BITS*(R+1)} state + {KEY_BITS} key = "
          f"{FULL_STATE_BITS*(R+1)+KEY_BITS})")

    # 打印每行非零 key 列数的分布: TK1 应全为 <=1; TK2/TK3 可见 >=2
    key_base = (R + 1) * FULL_STATE_BITS
    per_row_keys = [int(np.sum(res[i, key_base:key_base + KEY_BITS])) for i in range(res.shape[0])]
    nz = [c for c in per_row_keys if c > 0]
    if nz:
        print(f"含 key 的方程数 = {len(nz)}; 每行 key 项 max = {max(nz)}, "
              f"涉及寄存器 -> 至少一行有 {max(nz)} 个 key 项")
    show_L_equ_GIFT_extract(res, R)