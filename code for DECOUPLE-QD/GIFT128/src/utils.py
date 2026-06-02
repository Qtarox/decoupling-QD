import multiprocessing
import os
import re

NB_ROUNDS = 20
FILE_NB_ROUNDS = NB_ROUNDS
NAME = "_4"
ADV_MODEL = "SK"
SBOX_SIZE = 4
MIN_CORR = 133
THRESH=1
IGNORE_KEY_SCHEDULE = False
BLOCK_SIZE = 128
NB_SBOXES = BLOCK_SIZE // 4   




MATRIX_FILE = f"../data/quasi_differential_matrix"
SBOX_INEQUALITIES_DIR = f"../data/inequalities/"

Z = 1
if ADV_MODEL != "SK":
    Z = int(ADV_MODEL[2])

DIFF_TRAIL_FILE = f"../data/differential_trails/GIFT128_{ADV_MODEL}_R{FILE_NB_ROUNDS}{NAME}.txt"
RESULTS_FILE = f"../results/gift_characteristics/quasi_diff_trails_128.txt"
CONS_FILE = f"../results/gift_cons/quasi_diff_trails_{FILE_NB_ROUNDS}{NAME}_128_{MIN_CORR}.txt"

STATE_RANGE = range(4)
BIT_RANGE = range(SBOX_SIZE)

if SBOX_SIZE == 4:
    CORR_RANGE = [0.0, -3.0, -2.0, -1.41504, -1.0]

if SBOX_SIZE == 8:
    CORR_RANGE = [0.0, -5.0, -2.0, -1.0, -3.41504, -3.0, -4.0, -2.41504,
                  -1.83007, -4.41504, -1.09311, -2.19265, -1.41504, -1.29956,
                  -1.54057, -3.67807, -2.67807, -6.0, -3.19265, -7.00009, -5.41501]


BIT_PERM = [
    0,  33, 66, 99, 96,  1, 34, 67, 64, 97,  2, 35, 32, 65, 98,  3,
    4,  37, 70,103,100,  5, 38, 71, 68,101,  6, 39, 36, 69,102,  7,
    8,  41, 74,107,104,  9, 42, 75, 72,105, 10, 43, 40, 73,106, 11,
    12, 45, 78,111,108, 13, 46, 79, 76,109, 14, 47, 44, 77,110, 15,
    16, 49, 82,115,112, 17, 50, 83, 80,113, 18, 51, 48, 81,114, 19,
    20, 53, 86,119,116, 21, 54, 87, 84,117, 22, 55, 52, 85,118, 23,
    24, 57, 90,123,120, 25, 58, 91, 88,121, 26, 59, 56, 89,122, 27,
    28, 61, 94,127,124, 29, 62, 95, 92,125, 30, 63, 60, 93,126, 31,
]





def key_schedule(rn):
    rotr2_idx  = [(i - 2)  % 16 for i in range(16)]
    rotr12_idx = [(i - 12) % 16 for i in range(16)] 
    Tmp_Key = list(range(127, -1, -1))
    
    for _ in range(rn):
        W7 = Tmp_Key[  0: 16]
        W6 = Tmp_Key[ 16: 32]
        W5 = Tmp_Key[ 32: 48]
        W4 = Tmp_Key[ 48: 64]
        W3 = Tmp_Key[ 64: 80]
        W2 = Tmp_Key[ 80: 96]
        W1 = Tmp_Key[ 96:112]
        W0 = Tmp_Key[112:128]
        
        new_W0 = [W0[rotr12_idx[i]] for i in range(16)]
        new_W1 = [W1[rotr2_idx[i]]  for i in range(16)]
        
        
        Tmp_Key = new_W1 + new_W0 + W7 + W6 + W5 + W4 + W3 + W2
    
    U = Tmp_Key[32:64]   
    V = Tmp_Key[96:128]  
    return U, V


def get_master_key_index(round_idx, subkey_index):
    U, V = key_schedule(round_idx)
    if subkey_index % 2 == 0:
        return V[31 - (subkey_index // 2)]
    else:
        return U[31 - (subkey_index // 2)]
def int_to_bin(x, n):
    return [(x >> i) & 1 for i in range(n - 1, -1, -1)]


def bin_to_int(X, n):
    x = 0
    for i in range(n):
        x |= (X[n - i - 1] << i)
    return x


def character(x, y, n):
    """(-1)**(x . y)"""
    res = 0
    for i in range(n):
        res += x[i] * y[i]
    return (-1) ** res

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
    size = len(solution) // NB_SBOXES
    lines = ["" for _ in range(4)]
    for k in range(size):
        for row in range(4):
            start = NB_SBOXES * k + 8 * row
            lines[row] += get_formatted_row(solution[start:start + 8]) + " | "
    return "\n".join(lines)


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





def extract_diff_trail(trail_file, nb_rounds):
    diff_trail = [[[] for _ in range(2)] for _ in range(nb_rounds)]
    f = open(trail_file, "r")
    lines = f.readlines()
    lines = [line for line in lines if line != '\n']
    
    k = 0
    while k // 2 < nb_rounds:
        input_line = lines[k]
        output_line = lines[k + 1]
        input_diff = [int(x, 0) for x in input_line.split()]
        output_diff = [int(x, 0) for x in output_line.split()]
        for i in range(NB_SBOXES):
            diff_trail[k // 2][0] += int_to_bin(input_diff[i], SBOX_SIZE)
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
        input_line = lines[k]
        output_line = lines[k + 1]
        input_diff = [int(x, 0) for x in input_line.split()]
        output_diff = [int(x, 0) for x in output_line.split()]
        for i in range(NB_SBOXES):
            diff_trail[k // 2][0].append(input_diff[NB_SBOXES - 1 - i])
            diff_trail[k // 2][1].append(output_diff[NB_SBOXES - 1 - i])
        k += 2
    return diff_trail


def get_transitions(diff_trail_file):
    diff_trail = extract_diff_trail(diff_trail_file, NB_ROUNDS)
    T = []
    for k in range(NB_ROUNDS):
        for i in range(NB_SBOXES):
            b = bin_to_int([diff_trail[k][1][4*i+l] for l in BIT_RANGE], SBOX_SIZE)
            a = bin_to_int([diff_trail[k][0][4*i+l] for l in BIT_RANGE], SBOX_SIZE)
            T.append((b, a))
    return list(set(T))





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





def lfsr_ac(current_lfsr_state):
    tmp = current_lfsr_state[0] ^ current_lfsr_state[1]
    return current_lfsr_state[1:] + [tmp ^ 1]


def ac_state(current_state, sbox_size):
    if sbox_size == 4:
        c0 = current_state[2:]
        c1 = [0, 0] + current_state[:2]
        c2 = int_to_bin(2, SBOX_SIZE)
    if sbox_size == 8:
        c0 = [0 for _ in range(4)] + current_state[2:]
        c1 = [0 for _ in range(6)] + current_state[:2]
        c2 = int_to_bin(2, SBOX_SIZE)
    return [c0, c1, c2]


def compute_ac_states(nb_rounds, sbox_size):
    lfsr_state = [0, 0, 0, 0, 0, 0]
    states = []
    for k in range(nb_rounds):
        lfsr_state = lfsr_ac(lfsr_state)
        states.append(ac_state(lfsr_state, sbox_size))
    return states