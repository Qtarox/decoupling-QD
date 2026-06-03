import multiprocessing
import os
import re

# ── 配置参数 ──────────────────────────────────────────────────────
NB_ROUNDS = 14
FILE_NB_ROUNDS = NB_ROUNDS
NAME = "_RECT"
ADV_MODEL = "SK"
SBOX_SIZE = 4
MIN_CORR = 68
IGNORE_KEY_SCHEDULE = False   # 在 trail 搜索阶段未使用；保留供后处理参考
THRESH = 1
TRAIL_ID="d1"

# ── RECTANGLE 新增参数 ────────────────────────────────────────────
NB_ROWS = 4
NB_COLS = 16

# RECTANGLE S-box
RECTANGLE_SBOX = [6, 5, 0xC, 0xA, 1, 0xE, 7, 9, 0xB, 0, 3, 0xD, 8, 0xF, 4, 2]

# 行移位量：行 row 循环左移 ROW_SHIFT[row] 位（在 16 列内）
ROW_SHIFT = [0, 1, 12, 13]

# RECTANGLE 轮常数（25 轮）。
# ★ 重要：这些常数 ONLY 出现在 RECTANGLE 的 key schedule 中（XOR 进 key state 的 row 0
#   的 col 0..4），不直接作用于 cipher state。
#   因此在 SK 模型的 quasidifferential trail 搜索阶段（即本仓库的当前用途）这些常数
#   完全不进入 cipher-state 侧的 correlation 计算。它们的影响在后处理阶段（把对
#   子密钥 K_i 的线性条件反推到 master key 的线性条件）才出现，作为方程右端常数项。
RECTANGLE_RC = [
    0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16,
    0x0C, 0x19, 0x13, 0x07, 0x0F, 0x1F, 0x1E, 0x1C,
    0x18, 0x11, 0x03, 0x06, 0x0D, 0x1B, 0x17, 0x0E, 0x1D
]

# ── 文件路径 ──────────────────────────────────────────────────────
MATRIX_FILE           = f"../data/quasi_differential_matrix"
SBOX_INEQUALITIES_DIR = f"../data/inequalities/"
DIFF_TRAIL_FILE       = f"../data/differential_trails/RECTANGLE_{ADV_MODEL}_R{FILE_NB_ROUNDS}{NAME}.txt"
RESULTS_FILE          = f"../results/rectangle_characteristics/quasi_diff_trails_64.txt"
CONS_FILE             = f"../results/rectangle_cons/quasi_diff_trails_{FILE_NB_ROUNDS}{NAME}_64_{MIN_CORR}.txt"

# ── 范围常量 ───────────────────────────────────────────────────────
STATE_RANGE = range(4)
BIT_RANGE   = range(SBOX_SIZE)
if SBOX_SIZE == 4:
    CORR_RANGE = [0.0, -3.0, -2.0, -1.41504, -1.0]
if SBOX_SIZE == 8:
    CORR_RANGE = [0.0, -5.0, -2.0, -1.0, -3.41504, -3.0, -4.0, -2.41504, -1.83007,
                  -4.41504, -1.09311, -2.19265, -1.41504, -1.29956, -1.54057,
                  -3.67807, -2.67807, -6.0, -3.19265, -7.00009, -5.41501]

Z = 1
if ADV_MODEL != "SK":
    Z = int(ADV_MODEL[2])

# ══════════════════════════════════════════════════════════════════
# 索引约定（必须与 SubColumn 的 nibble 解释、QDTM 索引、不等式索引、
#   MASK_DIVIDER 的 (r*128 + s*64 + j) 编址完全一致）：
#
#   diff_trail[r][side]    — 长度 64 的展平比特列表，索引 j = 4*col + row
#   val 64 位整数          — bit j = (val >> j) & 1 = a[row][col]
#   row 0 = LSB，row 3 = MSB
#   每个 nibble Col(c) = a[3][c]<<3 | a[2][c]<<2 | a[1][c]<<1 | a[0][c]
# ══════════════════════════════════════════════════════════════════

def rect_val_to_flat(val):
    """64 位整数 → 展平 64-bit 列表，索引 j = 4*col + row"""
    return [(val >> j) & 1 for j in range(64)]


def rect_flat_to_val(flat):
    """展平列表 → 64 位整数"""
    val = 0
    for j in range(64):
        val |= flat[j] << j
    return val


# ── 线性层等价比特置换 ────────────────────────────────────────────
def _build_rect_perm():
    """
    构造 RECTANGLE 行移位的等价比特置换表。
    索引约定：j = 4*col + row
    行 row 循环左移 ROW_SHIFT[row]：
      after[row][col] ← before[row][(col - ROW_SHIFT[row]) % 16]
    等价地，「前进方向」是：bit 在位置 j 经 ShiftRow 移到位置 RECT_PERM[j]：
      RECT_PERM[4*col + row] = 4 * ((col + ROW_SHIFT[row]) % 16) + row
    """
    perm = [0] * 64
    for col in range(NB_COLS):
        for row in range(NB_ROWS):
            j   = 4 * col + row
            dst = 4 * ((col + ROW_SHIFT[row]) % NB_COLS) + row
            perm[j] = dst
    return perm


RECT_PERM = _build_rect_perm()


# ── 轮常数对 cipher-state 侧 trail correlation 的贡献 ──────────────
def rect_rc_corr_factor(mask_before_sbox, rc):
    """
    选项 A：严格按 RECTANGLE 规范。

    RC[i] 在 RECTANGLE 中只 XOR 进 key state（key schedule 内部），不直接作用于
    cipher state。因此在 SK quasidifferential 框架下，对作用于 cipher state 的
    trail correlation 没有任何贡献。

    保留此函数签名以兼容现有调用点，永远返回 1。
    （若以后要做 master-key 条件反推，RC 会在那里作为方程右端常数出现。）
    """
    return 1


# ── 从整数列表构建 diff_trail（list 路径，与上面约定一致）───────
def extract_diff_trail_from_list(diffs_list):
    """
    diffs_list[2*r]   = 第 r 轮 SubColumn 输入差分
    diffs_list[2*r+1] = 第 r 轮 SubColumn 输出差分
    返回 diff_trail[r][0/1] = 展平 64-bit 列表，索引 j = 4*col + row
    """
    nb_rounds = len(diffs_list) // 2
    diff_trail = [[[], []] for _ in range(nb_rounds)]
    for r in range(nb_rounds):
        diff_trail[r][0] = rect_val_to_flat(diffs_list[2 * r])
        diff_trail[r][1] = rect_val_to_flat(diffs_list[2 * r + 1])
    return diff_trail


def get_transitions_from_list(diffs_list):
    diff_trail = extract_diff_trail_from_list(diffs_list)
    nb_rounds  = len(diffs_list) // 2
    T = []
    for k in range(nb_rounds):
        for col in range(NB_COLS):
            # MSB-first：row3 是高位
            b_bits = [diff_trail[k][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            a_bits = [diff_trail[k][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            b = bin_to_int(b_bits, SBOX_SIZE)
            a = bin_to_int(a_bits, SBOX_SIZE)
            T.append((b, a))
    return list(set(T))


# ── 打印工具 ────────────────────────────────────────────────────
def print_mask_rectangle(mask_trail, nb_rounds):
    """以 4 行 × 16 列格式打印 mask trail。索引 j = 4*col + row"""
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


# ══════════════════════════════════════════════════════════════════
# 通用工具
# ══════════════════════════════════════════════════════════════════
def trail_indic(trails_conditions, correlations, signs, factor_dict, nb_rounds):
    """
    打印每条 trail 的相关度统计 + 密钥条件位置统计。
    （这里 condition[k][j] 在被赋值时为 int 0/1，未赋值时为 []。）
    """
    print("\n" + "="*60)
    print("Trail Analysis Summary")
    print("="*60)

    corr_count = {}
    for corr in correlations:
        corr_count[corr] = corr_count.get(corr, 0) + 1

    print(f"\nCorrelation distribution:")
    for corr in sorted(corr_count):
        factor = factor_dict.get(corr, 1)
        print(f"  log2|corr| = {corr:.5f}: {corr_count[corr]} trails (factor = {factor})")

    all_key_positions = set()
    for conditions in trails_conditions:
        for k in range(nb_rounds):
            for j in range(64):
                # 修正：原代码 `!= [] and != 0` 在 Python 中对 [] 永远为 True，
                #   导致未赋值的 [] 也被计入。这里改为「确为 int 且非零」。
                if isinstance(conditions[k][j], int) and conditions[k][j] != 0:
                    all_key_positions.add((k, j))

    print(f"\nKey bit positions with conditions: {len(all_key_positions)} bits")
    if all_key_positions:
        for (k, j) in sorted(all_key_positions):
            col = j // 4
            row = j % 4
            print(f"  Round {k}, col {col}, row {row} (flat index {j})")

    print("="*60)


def int_to_bin(x, n):
    return [(x >> i) & 1 for i in range(n - 1, -1, -1)]


def bin_to_int(X, n):
    x = 0
    for i in range(n):
        x |= (X[n - i - 1] << i)
    return x


def get_formatted_row(row):
    row_str = ""
    for el in row:
        if el == "0":
            row_str += '{:>4}'.format(el) + " "
        else:
            row_str += '{:>4}'.format(hex(int(el))) + " "
    return row_str


def print_solution(solution):
    solution = solution[:-1].split(" ")
    size = len(solution) // 16
    line0, line1, line2, line3 = "", "", "", ""
    for k in range(size):
        line0 += get_formatted_row(solution[16 * k     : 16 * k + 4 ]) + " | "
        line1 += get_formatted_row(solution[16 * k + 4 : 16 * k + 8 ]) + " | "
        line2 += get_formatted_row(solution[16 * k + 8 : 16 * k + 12]) + " | "
        line3 += get_formatted_row(solution[16 * k + 12: 16 * k + 16]) + " | "
    return (line0 + "\n" + line1 + "\n" + line2 + "\n" + line3)


def solutions_to_readable(fileName):
    fileIn = open(fileName, "r")
    solutions = []
    if fileIn.mode == "r":
        contents = fileIn.read()
        solutions = contents.split("\n")
        fileIn.close()
    readable = ""
    for solution in solutions:
        readable += print_solution(solution) + "\n------------------------\n"
    fileOut = open(fileName + "_readable.txt", "w+")
    fileOut.write(readable)
    fileOut.close()


# ────────────────────────────────────────────────────────────────
# 注意：以下两个 file-path 函数与 list-path 的 nibble 索引约定不兼容
#  （file-path 把 MSB 放在 4*col+0，list-path 把 LSB 放在 4*col+0）。
# 当前 RECTANGLE_msk.py 走 list-path，这两个保留只为兼容旧文件读取，
# 不要和 list-path 在同一流程中混用。
# ────────────────────────────────────────────────────────────────
def get_transitions(diff_trail_file):
    diff_trail = extract_diff_trail(diff_trail_file, NB_ROUNDS)
    T = []
    for k in range(NB_ROUNDS):
        for col in range(NB_COLS):
            b_bits = [diff_trail[k][1][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            a_bits = [diff_trail[k][0][4*col + row] for row in range(NB_ROWS-1, -1, -1)]
            b = bin_to_int(b_bits, SBOX_SIZE)
            a = bin_to_int(a_bits, SBOX_SIZE)
            T.append((b, a))
    return list(set(T))


def extract_diff_trail(trail_file, nb_rounds):
    diff_trail = [[[] for _ in range(2)] for _ in range(nb_rounds)]
    f = open(trail_file, "r")
    lines = f.readlines()
    lines = [line for line in lines if line != '\n']
    k = 0
    while k // 2 < nb_rounds:
        input_line  = lines[k]
        output_line = lines[k + 1]
        input_diff  = [int(x, 0) for x in input_line.split()]
        output_diff = [int(x, 0) for x in output_line.split()]
        for i in range(16):
            diff_trail[k // 2][0] += int_to_bin(input_diff[i],  SBOX_SIZE)
            diff_trail[k // 2][1] += int_to_bin(output_diff[i], SBOX_SIZE)
        k += 2
    return diff_trail


def extract_diff_trail_cell(trail_file, nb_rounds):
    diff_trail = [[[] for _ in range(2)] for _ in range(nb_rounds)]
    f = open(trail_file, "r")
    lines = f.readlines()
    lines = [line for line in lines if line != '\n']
    k = 0
    while k // 2 < nb_rounds:
        input_line  = lines[k]
        output_line = lines[k + 1]
        input_diff  = [int(x, 0) for x in input_line.split()]
        output_diff = [int(x, 0) for x in output_line.split()]
        for i in range(16):
            diff_trail[k // 2][0].append(input_diff[15 - i])
            diff_trail[k // 2][1].append(output_diff[15 - i])
        k += 2
    return diff_trail


def extract_inequalities_by_corr(inequalities_dir, n):
    list_file = os.listdir(inequalities_dir)
    I = [[{corr: [] for corr in CORR_RANGE} for j in range(2**n)] for i in range(2**n)]
    for filename in list_file:
        tmp = []
        f = open(os.path.join(inequalities_dir, filename), 'r')
        for line in f:
            tmp.append([int(x) for x in line.split()])
        f.close()
        fsplited = filename.split("_")
        a, b = int(fsplited[1]), int(fsplited[2])
        corr = float(fsplited[3][:-4])
        I[b][a][corr] = tmp
    return I


def extract_block(file, i, j, n):
    f = open(file, 'r')
    block = [[] for j in range(2**n)]
    for idx, line in enumerate(f):
        if idx // 2**n != i:
            continue
        elif idx // 2**n > i:
            break
        fline = [float(x) for x in line.split()]
        block[idx % 2**n] = fline[2**n * j:2**n * (j + 1)].copy()
    f.close()
    return block


def extract_quasi_diff_matrix(matrix_file, blocks, n):
    M = [[None for j in range(2**n)] for i in range(2**n)]
    with multiprocessing.Pool(16) as P:
        T = P.starmap(extract_block,
                      [(matrix_file, blocks[i][0], blocks[i][1], n)
                       for i in range(len(blocks))])
    for i in range(len(blocks)):
        M[blocks[i][0]][blocks[i][1]] = T[i]
    return M


def character(x, y, n):  # (-1)**(x . y)
    res = 0
    for i in range(n):
        res += x[i] * y[i]
    return (-1)**res