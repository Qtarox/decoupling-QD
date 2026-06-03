"""
GEN_LINEAR.py  --  SKINNY-128 版本
================================================

与原 SKINNY-64 版本相比，主要把 cell 宽度从 4-bit 换成 8-bit：
    CELL_SIZE        : 4 -> 8
    HALF_STATE_BITS  : 64 -> 128         (16 cells * cell_size)
    FULL_STATE_BITS  : 128 -> 256        (x 半态 + y 半态)
    KEY_BITS         : 64 -> 128         (TK1 时 16 cells * 8 bit)
方程矩阵 M_EQ 是 cell 级的 16x40，本身与 cell 宽度无关，可直接复用。

所有形如 r*128+ind*4+i / r*128+64+ind*4+i 的 bit 寻址，统一改为
    r * FULL_STATE_BITS + ind * CELL_SIZE + i
    r * FULL_STATE_BITS + HALF_STATE_BITS + ind * CELL_SIZE + i
保留原 API：Global_mat / Global_mat_bit / show_L_equ_GIFT(_extract) 的对外行为一致，
只是内部宽度切换为 SKINNY-128。
"""

import numpy as np

# ===== SKINNY-128 参数（如需切回 SKINNY-64，把这三个改回 4/64/128 即可） =====
CELL_SIZE = 8                              # 每个 cell 的 bit 数
HALF_STATE_BITS = 16 * CELL_SIZE           # 半态 = 16 个 cell = 128 bit
FULL_STATE_BITS = 2 * HALF_STATE_BITS      # 完整状态 (x 和 y 拼起来) = 256 bit
KEY_BITS = HALF_STATE_BITS                 # TK1 模型下的密钥/tweakey 宽度 = 128 bit

# 8-bit S-box (SKINNY-128)  —— 论文表 5 的官方值
Sbox = [
    0x65, 0x4c, 0x6a, 0x42, 0x4b, 0x63, 0x43, 0x6b, 0x55, 0x75, 0x5a, 0x7a, 0x53, 0x73, 0x5b, 0x7b,
    0x35, 0x8c, 0x3a, 0x81, 0x89, 0x33, 0x80, 0x3b, 0x95, 0x25, 0x98, 0x2a, 0x90, 0x23, 0x99, 0x2b,
    0xe5, 0xcc, 0xe8, 0xc1, 0xc9, 0xe0, 0xc0, 0xe9, 0xd5, 0xf5, 0xd8, 0xf8, 0xd0, 0xf0, 0xd9, 0xf9,
    0xa5, 0x1c, 0xa8, 0x12, 0x1b, 0xa0, 0x13, 0xa9, 0x05, 0xb5, 0x0a, 0xb8, 0x03, 0xb0, 0x0b, 0xb9,
    0x32, 0x88, 0x3c, 0x85, 0x8d, 0x34, 0x84, 0x3d, 0x91, 0x22, 0x9c, 0x2c, 0x94, 0x24, 0x9d, 0x2d,
    0x62, 0x4a, 0x6c, 0x45, 0x4d, 0x64, 0x44, 0x6d, 0x52, 0x72, 0x5c, 0x7c, 0x54, 0x74, 0x5d, 0x7d,
    0xa1, 0x1a, 0xac, 0x15, 0x1d, 0xa4, 0x14, 0xad, 0x02, 0xb1, 0x0c, 0xbc, 0x04, 0xb4, 0x0d, 0xbd,
    0xe1, 0xc8, 0xec, 0xc5, 0xcd, 0xe4, 0xc4, 0xed, 0xd1, 0xf1, 0xdc, 0xfc, 0xd4, 0xf4, 0xdd, 0xfd,
    0x36, 0x8e, 0x38, 0x82, 0x8b, 0x30, 0x83, 0x39, 0x96, 0x26, 0x9a, 0x28, 0x93, 0x20, 0x9b, 0x29,
    0x66, 0x4e, 0x68, 0x41, 0x49, 0x60, 0x40, 0x69, 0x56, 0x76, 0x58, 0x78, 0x50, 0x70, 0x59, 0x79,
    0xa6, 0x1e, 0xaa, 0x11, 0x19, 0xa3, 0x10, 0xab, 0x06, 0xb6, 0x08, 0xba, 0x00, 0xb3, 0x09, 0xbb,
    0xe6, 0xce, 0xea, 0xc2, 0xcb, 0xe3, 0xc3, 0xeb, 0xd6, 0xf6, 0xda, 0xfa, 0xd3, 0xf3, 0xdb, 0xfb,
    0x31, 0x8a, 0x3e, 0x86, 0x8f, 0x37, 0x87, 0x3f, 0x92, 0x21, 0x9e, 0x2e, 0x97, 0x27, 0x9f, 0x2f,
    0x61, 0x48, 0x6e, 0x46, 0x4f, 0x67, 0x47, 0x6f, 0x51, 0x71, 0x5e, 0x7e, 0x57, 0x77, 0x5f, 0x7f,
    0xa2, 0x18, 0xae, 0x16, 0x1f, 0xa7, 0x17, 0xaf, 0x01, 0xb2, 0x0e, 0xbe, 0x07, 0xb7, 0x0f, 0xbf,
    0xe2, 0xca, 0xee, 0xc6, 0xcf, 0xe7, 0xc7, 0xef, 0xd2, 0xf2, 0xde, 0xfe, 0xd7, 0xf7, 0xdf, 0xff,
]

# 与 SKINNY-64 版本完全相同的 cell 级方程矩阵 (16 x 40)
M_EQ = np.load("./M_EQ.npy")


def key_schedule(round=0, key_index=0):
    """SKINNY 的 tweakey 置换 PT —— 在 SKINNY-64/128 里相同。"""
    key_permu = [9, 15, 8, 13, 10, 14, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7]
    tmp = key_index
    for _ in range(round):
        tmp = key_permu[tmp]
    return tmp


def Global_mat(res, M_EQ, round_num):
    """
    Cell 级版本：保留原签名与行为。每轮 16 个 cell 方程，每个 cell 占 1 列。
    每轮状态占 32 个 cell (x 半态 16 + y 半态 16)，密钥 16 个 cell。
    此函数与 cell 宽度无关，原样保留。
    """
    for i in range(np.shape(res)[0]):
        equ_num = i % 16
        rn = i // 16
        rn_k_ind = None
        for k in range(40):
            if M_EQ[equ_num][k] == 1:
                if k < 16:                       # x_{r+1}_k
                    res[i][(rn + 1) * 32 + k] = 1
                elif k < 32:                     # y_r_k
                    res[i][rn * 32 + k] = 1
                else:                            # tweakey
                    rn_k_ind = k - 32
                    break
        if rn_k_ind is not None:
            k_ind = key_schedule(rn, rn_k_ind)
            res[i][32 * (round_num + 1) + k_ind] = 1
    return res


def Global_mat_bit(round_num):
    """
    Bit 级版本：每个 cell 展开成 CELL_SIZE 个 bit，每个 bit 对应一条独立方程。
    每轮方程数 = 16 cell * CELL_SIZE = HALF_STATE_BITS。
    每轮状态列数 = FULL_STATE_BITS；总状态列 = (round_num+1) * FULL_STATE_BITS；
    末尾再加 KEY_BITS 列密钥变量。
    """
    num_rows = HALF_STATE_BITS * round_num
    num_cols = FULL_STATE_BITS * (round_num + 1) + KEY_BITS
    res = np.zeros((num_rows, num_cols), dtype=int)

    for i in range(num_rows):
        equ_num_bit = i % HALF_STATE_BITS           # 当前轮内第几条 bit 方程
        rn = i // HALF_STATE_BITS                   # 当前是第几轮

        equ_num_cell = equ_num_bit // CELL_SIZE     # 对应 M_EQ 里的 cell 方程 (0..15)
        bit_idx = equ_num_bit % CELL_SIZE           # 该 cell 内部 bit 偏移 (0..CELL_SIZE-1)

        for k in range(40):
            if M_EQ[equ_num_cell][k] != 1:
                continue

            if k < 16:
                # x_{r+1}_k -> 下一轮 x 半态
                res[i][(rn + 1) * FULL_STATE_BITS + k * CELL_SIZE + bit_idx] = 1
            elif k < 32:
                # y_r_k -> 当前轮 y 半态。k - 16 是 y 半态内的 cell 索引；
                # 但为了与原版语义一致（原版用 `rn*128 + k*4 + bit_idx`，k 仍在 16..31），
                # 这里也保留 k 直接乘 CELL_SIZE 的写法 —— 这样 y 半态的 cell 0..15 自然
                # 落在每轮 [HALF_STATE_BITS, FULL_STATE_BITS) 的区间里。
                res[i][rn * FULL_STATE_BITS + k * CELL_SIZE + bit_idx] = 1
            else:
                # tweakey
                rn_k_ind = k - 32
                k_ind = key_schedule(rn, rn_k_ind)
                res[i][FULL_STATE_BITS * (round_num + 1) + k_ind * CELL_SIZE + bit_idx] = 1
                break
    return res


def _fmt_state_var(j):
    """把 bit 索引 j 翻译成可读的 x_r_b / y_r_b 形式。"""
    rn = j // FULL_STATE_BITS
    ind = j % FULL_STATE_BITS
    if ind < HALF_STATE_BITS:
        return f"x_{rn}_{ind}"
    else:
        return f"y_{rn}_{ind - HALF_STATE_BITS}"


def show_L_equ_GIFT(lmat, active_bit_dic, round_num, FILE_FLG=False, file=None):
    """
    把矩阵 lmat 按 GF(2) 方程的形式打印出来。
    active_bit_dic 中的变量用方括号 [..] 强调。
    返回拼接后的整段字符串。
    """
    F = file if file else "output.txt"
    L = ""
    out_handle = open(F, "w") if FILE_FLG else None

    state_cols = FULL_STATE_BITS * (round_num + 1)
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

            has_key = False
            # key 段紧跟在最后一轮 state 之后，共 KEY_BITS 列
            key_base = round_num * FULL_STATE_BITS + FULL_STATE_BITS  # = (R+1)*FULL_STATE_BITS
            for k in range(KEY_BITS):
                if lmat[i][key_base + k] == 1:
                    has_key = True
                    l_tmp += f" + k_{k} = 0"
            if not has_key:
                l_tmp += "= 0   "

            # 兼容旧版：如果矩阵有额外的最后一列（SBOX 标志），打印它
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
    """
    与 show_L_equ_GIFT 类似，但用矩阵中数值 2 来标记 "active/masked" 变量
    （而非通过 active_bit_dic 外部传入）。
    """
    F = file if file else "output.txt"
    out_handle = open(F, "w") if FILE_FLG else None
    state_cols = FULL_STATE_BITS * (round_num + 1)
    key_base = (round_num + 1) * FULL_STATE_BITS
    try:
        for i in range(np.shape(lmat)[0]):
            l_tmp = ""
            for j in range(state_cols):
                if lmat[i][j] == 2:
                    var_name = _fmt_state_var(j)
                    l_tmp += f" + [{var_name}]"
                elif lmat[i][j] == 1:
                    var_name = _fmt_state_var(j)
                    l_tmp += f" + {var_name}"

            has_key = False
            for k in range(KEY_BITS):
                if lmat[i][key_base + k] == 1:
                    has_key = True
                    l_tmp += f" + k_{k} = 0"
            if not has_key:
                l_tmp += "= 0   "

            print(l_tmp)
            if out_handle:
                print(l_tmp, file=out_handle)
    finally:
        if out_handle:
            out_handle.close()


if __name__ == "__main__":
    res = Global_mat_bit(2)
    show_L_equ_GIFT_extract(res, 2)