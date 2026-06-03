import multiprocessing
import os
import re


THRESH=0.5
NB_ROUNDS = 15          
FILE_NB_ROUNDS = NB_ROUNDS      
ADV_MODEL = "TK3"        
SBOX_SIZE = 4           
MIN_CORR = 60
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
PT =[9,15,8,13,10,14,12,11,0,1,2,3,4,5,6,7]
MATRIX_FILE = f"../data/quasi_differential_matrix"


SBOX_INEQUALITIES_DIR = f"../data/trails_inequalities/SKINNY{16 * SBOX_SIZE}_{ADV_MODEL}_R{FILE_NB_ROUNDS}_ineq/" 





Z = 1
if ADV_MODEL != "SK":
    Z = int(ADV_MODEL[2])

DIFF_TRAIL_FILE = f"../data/differential_trails/SKINNY{16 * SBOX_SIZE}_{ADV_MODEL}_R{FILE_NB_ROUNDS}.txt"
RESULTS_FILE = f"../results/skinny_characteristics/quasi_diff_trails_{16 * SBOX_SIZE}.txt"

res_path = f"../results/skinny_characteristics/quasi_diff_trails_{16 * SBOX_SIZE}"
STATE_RANGE = range(4)
BIT_RANGE = range(SBOX_SIZE)
if SBOX_SIZE == 4:
    CORR_RANGE = [0.0, -3.0, -2.0, -1.0]

if SBOX_SIZE == 8:
    CORR_RANGE = [0.0, -5.0, -2.0, -1.0, -3.41504, -3.0, -4.0, -2.41504, -1.83007, -4.41504, -1.09311, -2.19265, -1.41504, -1.29956, -1.54057, -3.67807, -2.67807, -6.0, -3.19265, -7.00009, -5.41501]


KS = [[ 9, 15,  8, 13], 
      [10, 14, 12, 11],
      [ 0,  1,  2,  3],
      [ 4,  5,  6,  7]]


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
        for i in STATE_RANGE:
            for j in STATE_RANGE:
                b = bin_to_int([diff_trail[k][1][i][j][l] for l in BIT_RANGE], SBOX_SIZE)
                a = bin_to_int([diff_trail[k][0][i][j][l] for l in BIT_RANGE], SBOX_SIZE)
                T.append((b, a))
    return list(set(T))

def extract_diff_trail_flat(trail_file, nb_rounds):
    diff_trail = [[[0 for _ in range(16)] for _ in range(2)] for _ in range(nb_rounds)]
    
    with open(trail_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip() != '']
    
    k = 0
    while k // 4 < nb_rounds:
        line = lines[k]
        round_idx = k // 4
        i = k % 4
        diff = [int(x, 0) for x in line.split()]
        
        for j in range(4):
            cell_idx = i * 4 + j
            
            diff_trail[round_idx][0][cell_idx] = diff[j]
            diff_trail[round_idx][1][cell_idx] = diff[4 + j]
            
        k += 1
    return diff_trail
def extract_diff_trail(trail_file, nb_rounds):
    diff_trail = [[[[0 for j in STATE_RANGE] for i in STATE_RANGE] for _ in range(2)]
              for k in range(nb_rounds)]
    

    f = open(trail_file, "r")
    lines = f.readlines()
    lines = [line for line in lines if line != '\n']
    k = 0
    while k // 4 < nb_rounds:
        line = lines[k]
        i = k % 4
        diff = [int(x, 0) for x in line.split()]
        for j in STATE_RANGE:
            diff_trail[k // 4][0][i][j] = int_to_bin(diff[j], SBOX_SIZE)
            diff_trail[k // 4][1][i][j] = int_to_bin(diff[4 + j], SBOX_SIZE)
            
        k += 1
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

