import multiprocessing
import os
import re


NB_ROUNDS = 12 
FILE_NB_ROUNDS = NB_ROUNDS   
NAME= "_ZDY" 
ADV_MODEL = "SK"        
SBOX_SIZE = 4            
MIN_CORR =68
IGNORE_KEY_SCHEDULE = False
TH=0.5
"""
For the MIN_CORR constant, it means that all found quasidifferential trails will
have an absolute correlation higher than 2^-MIN_CORR. Thus, if this value is 
higher than the assumed probability the model will be infeasible
The lower this bound is the longer the model will run and for some characteristics
in fixed-key the formula will return negative values. Those characteristics are:
 - ../data/differential_trails/SKINNY64_SK_R7.txt
 - ../data/differential_trails/SKINNY128_TK2_R16.txt
 - ../data/differential_trails/SKINNY128_SK_R13.txt
"""

MATRIX_FILE = f"../data/quasi_differential_matrix"


SBOX_INEQUALITIES_DIR = f"../data/inequalities/" 

Z = 1
if ADV_MODEL != "SK":
    Z = int(ADV_MODEL[2])

DIFF_TRAIL_FILE = f"../data/differential_trails/GIFT{16 * SBOX_SIZE}_{ADV_MODEL}_R{FILE_NB_ROUNDS}{NAME}.txt"
RESULTS_FILE = f"../results/gift_characteristics/quasi_diff_trails_{16 * SBOX_SIZE}.txt"
CONS_FILE = f"../results/gift_cons/quasi_diff_trails_{FILE_NB_ROUNDS}{NAME}_{16 * SBOX_SIZE}_{MIN_CORR}.txt"


STATE_RANGE = range(4)
BIT_RANGE = range(SBOX_SIZE)
if SBOX_SIZE == 4:
    CORR_RANGE = [0.0, -3.0, -2.0, -1.41504, -1.0]

if SBOX_SIZE == 8:
    CORR_RANGE = [0.0, -5.0, -2.0, -1.0, -3.41504, -3.0, -4.0, -2.41504, -1.83007, -4.41504, -1.09311, -2.19265, -1.41504, -1.29956, -1.54057, -3.67807, -2.67807, -6.0, -3.19265, -7.00009, -5.41501]




BIT_PERM = [0,17,34,51,48,1,18,35,32,49,2,19,16,33,50,3,4,21,38,55,52,5,22,39,36,53,6,23,20,37,54,7,8,25,42,59,56,9,26,43,40,57,10,27,24,41,58,11,12,29,46,63,60,13,30,47,44,61,14,31,28,45,62,15]

def key_schedule(rn):
    l6=[12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11]
    l7=[18,19,20,21,22,23,24,25,26,27,28,29,30,31,16,17]
    MAP_key=[]
    for i in range(96):
        MAP_key.append(32+i)
    for i in range(16):
        MAP_key.append(l6[i])
    for i in range(16):
        MAP_key.append(l7[i])
    
    Tmp_Key=list(range(128))
    Tmp_Key2=list(range(128))
    for i in range(rn):
        Tmp_Key2=Tmp_Key.copy()
        for j in range(128):
            Tmp_Key2[j]=Tmp_Key[MAP_key[j]]
        Tmp_Key=Tmp_Key2.copy()
    
    U=Tmp_Key2[16:32]
    V=Tmp_Key2[0:16]
    return U,V
def get_master_key_index(round_idx, subkey_index):
    
    
    U, V = key_schedule(round_idx)
    
    
    if subkey_index % 2 == 0:
        master_bit = V[subkey_index // 2]
    else:
        master_bit = U[subkey_index // 2]
        
    return master_bit
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
                row_str += '{:>4}'.format(hex(int(el)))+ " "
    return row_str








def print_solution(solution):
    solution = solution[:-1].split(" ")
    size = len(solution) // 16
    line0, line1, line2, line3 = "", "", "", ""
    for k in range(size):
        line0 += get_formatted_row(solution[16 * k : 16 * k + 4]) + " | "
        line1 += get_formatted_row(solution[16 * k + 4 : 16 * k + 8]) + " | "
        line2 += get_formatted_row(solution[16 * k + 8: 16 * k + 12]) + " | "
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

    
    fileOut = open(fileName+ "_readable.txt", "w+")
    fileOut.write(readable)
    fileOut.close()

def get_transitions(diff_trail_file):
    diff_trail = extract_diff_trail(diff_trail_file, NB_ROUNDS)
    T = []
    for k in range(NB_ROUNDS):
        for i in range(16):
            b = bin_to_int([diff_trail[k][1][4*i+l] for l in BIT_RANGE], SBOX_SIZE)
            a = bin_to_int([diff_trail[k][0][4*i+l] for l in BIT_RANGE], SBOX_SIZE)
            T.append((b, a))
    return list(set(T))


def extract_diff_trail(trail_file, nb_rounds):
    diff_trail = [[[] for _ in range(2)]
              for _ in range(nb_rounds)]
    
    f = open(trail_file, "r")
    lines = f.readlines()
    lines = [line for line in lines if line != '\n']
    
    k = 0
    while k // 2 < nb_rounds:
        input_line = lines[k]
        output_line = lines[k+1]
        input_diff = [int(x, 0) for x in input_line.split()]
        output_diff = [int(x, 0) for x in output_line.split()]
        for i in range(16):
            diff_trail[k//2][0] += int_to_bin(input_diff[i], SBOX_SIZE)
            diff_trail[k//2][1] += int_to_bin(output_diff[i], SBOX_SIZE)
        k += 2
    return diff_trail

def extract_diff_trail_cell(trail_file, nb_rounds):
    diff_trail = [[[] for _ in range(2)]
              for _ in range(nb_rounds)]
    
    f = open(trail_file, "r")
    lines = f.readlines()
    lines = [line for line in lines if line != '\n']
    
    k = 0
    while k // 2 < nb_rounds:
        input_line = lines[k]
        output_line = lines[k+1]
        input_diff = [int(x, 0) for x in input_line.split()]
        output_diff = [int(x, 0) for x in output_line.split()]
        for i in range(16):
            diff_trail[k//2][0].append(input_diff[15-i])
            diff_trail[k//2][1].append(output_diff[15-i])
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
        a,b = int(fsplited[1]), int(fsplited[2])
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
        T = P.starmap(extract_block, [(matrix_file, blocks[i][0], blocks[i][1], n) for i in range(len(blocks))])

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
        c2 =  int_to_bin(2, SBOX_SIZE)
    if sbox_size == 8:
        c0 = [0 for _ in range(4)] + current_state[2:]
        c1 = [0 for _ in range(6)] + current_state[:2]
        c2 =  int_to_bin(2, SBOX_SIZE)
    return [c0, c1, c2]

def compute_ac_states(nb_rounds, sbox_size):
    lfsr_state = [0, 0, 0, 0, 0, 0]
    states = []
    for k in range(nb_rounds):
        lfsr_state = lfsr_ac(lfsr_state)
        states.append(ac_state(lfsr_state, sbox_size))
    return states

def character(x, y, n): 
    res = 0
    for i in range(n):
        res += x[i] * y[i]
    return (-1)**res

