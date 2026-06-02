from gurobipy import *
from utils import *
from tqdm import tqdm
import time
import numpy as np
from math import log2
from key_master import *
from fast_distribut import *

blocks = get_transitions(DIFF_TRAIL_FILE)
QDTM_GIFT_SBOX = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)

AC_STATES = [0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,0x37,0x2F,0x1E,0x3C,0x39,0x33,0x27,0x0E,
0x1D,0x3A,0x35,0x2B,0x16,0x2C,0x18,0x30,0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38,
0x31,0x23,0x06,0x0D,0x1B,0x36,0x2D,0x1A,0x34,0x29,0x12,0x24,0x08,0x11,0x22,0x04]

def print_mask(mask_trail,nb_rounds):
    for n in range(nb_rounds):
        for i in range(64):
            print(mask_trail[n][0][i],end='')
            if i % 4 == 3: print(' ',end='')
        print()
        for i in range(64):
            print(mask_trail[n][1][i],end='')
            if i % 4 == 3: print(' ',end='')
        print()
    print()

def get_equations(diff_trail,mask_trail,nb_rounds):
    for n in range(nb_rounds):
        
        for i in range(16):
            if mask_trail[n][0][4*i:4*i+4] != [0,0,0,0] or mask_trail[n][1][4*i:4*i+4] != [0,0,0,0]:
                a, b = diff_trail[n][0][4*i:4*i+4], diff_trail[n][1][4*i:4*i+4]
                u, v = mask_trail[n][0][4*i:4*i+4], mask_trail[n][1][4*i:4*i+4]

                a, b = bin_to_int(a, SBOX_SIZE), bin_to_int(b, SBOX_SIZE)
                u, v = bin_to_int(u, SBOX_SIZE), bin_to_int(v, SBOX_SIZE)
                if a != 0 and b != 0:
                    x_bs = ['x_%s_%s' % (n,63-4*i-j) for j,bit in enumerate(mask_trail[n][0][4*i:4*i+4]) if bit == 1]
                    x_bits = ' + '.join(x_bs)
                    y_bs = ['y_%s_%s' % (n,63-4*i-j) for j,bit in enumerate(mask_trail[n][1][4*i:4*i+4]) if bit == 1]
                    y_bits = ' + '.join(y_bs)
                    
                    if QDTM_GIFT_SBOX[b][a][v][u] < 0:
                        if x_bits == '':
                            s = y_bits + ' = 1'
                        elif y_bits == '':
                            s = x_bits + ' = 1'
                        else:
                            s = x_bits + ' + ' + y_bits + ' = 1'
                    else:
                        if x_bits == '':
                            s = y_bits + ' = 0'
                        elif y_bits == '':
                            s = x_bits + ' = 0'
                        else:
                            s = x_bits + ' + ' + y_bits + ' = 0'
                    print(s)
                else:
                    print('Nonlinear constraints involving: ',end='')
                    x_bits = ['x_%s_%s' % (n,63-4*i-j) for j,bit in enumerate(mask_trail[n][0][4*i:4*i+4]) if bit == 1]
                    x_bits = ','.join(x_bits)
                    y_bits = ['y_%s_%s' % (n,63-4*i-j) for j,bit in enumerate(mask_trail[n][1][4*i:4*i+4]) if bit == 1]
                    y_bits = ','.join(y_bits)
                    s = x_bits + ',' + y_bits
                    print(s)
                for bit in y_bs:
                    index = int(bit.split('_')[2])
                    print('linear layer constraint: %s + x_%s_%s = 0' % (bit,n+1,BIT_PERM[index]))



def compute_correlation(diff_trail, mask_trail, nb_rounds):
    
    
    
    
    conditions = [[[] for i in range(64)] for k in range(nb_rounds)]
    corr = 1
    for k in range(nb_rounds):
        for i in range(16):
            a, b = diff_trail[k][0][4*i:4*i+4], diff_trail[k][1][4*i:4*i+4]
            u, v = mask_trail[k][0][4*i:4*i+4], mask_trail[k][1][4*i:4*i+4]

            a, b = bin_to_int(a, SBOX_SIZE), bin_to_int(b, SBOX_SIZE)
            u, v = bin_to_int(u, SBOX_SIZE), bin_to_int(v, SBOX_SIZE)

            corr *= QDTM_GIFT_SBOX[b][a][v][u]
            if QDTM_GIFT_SBOX[b][a][v][u] == 0:
                
                
                return "Error"    

                        
    
    
    for k in range(nb_rounds):

        corr *= character([1], [mask_trail[k][1][48]], 1)
        
        corr *= character([(AC_STATES[k] >> 5) & 1], [mask_trail[k][1][36]], 1)
        
        corr *= character([(AC_STATES[k] >> 4) & 1], [mask_trail[k][1][52]], 1)
        
        corr *= character([(AC_STATES[k] >> 3) & 1], [mask_trail[k][1][60]], 1)
        
        corr *= character([(AC_STATES[k] >> 2) & 1], [mask_trail[k][1][20]], 1)
        
        corr *= character([(AC_STATES[k] >> 1) & 1], [mask_trail[k][1][32]], 1)
        
        corr *= character([(AC_STATES[k] >> 0) & 1], [mask_trail[k][1][56]], 1)
       
    
    for k in range(nb_rounds):
        for j in range(64):
            if mask_trail[k][0][j] == 0:
                continue
            elif j in [4*m+2 for m in range(16)] + [4*m+3 for m in range(16)]:
                conditions[k][j] = mask_trail[k][0][j]
                
            else:
                
                continue
    
    
    
    
    if corr > 0:
        return  1, log2( corr), conditions
    elif corr < 0:
        return -1, log2(-corr), conditions
    return 1, 0, []


def add_xor_constraints(model, x1, x2, y):
    model.addConstr(-x1 + x2 + y >=  0)
    model.addConstr( x1 - x2 + y >=  0)
    model.addConstr( x1 + x2 - y >=  0)
    model.addConstr(-x1 - x2 - y >= -2)
   

def add_xor_constraints2(model, x1, x2, x3, y):
    model.addConstr(-x1 + x2 + x3 + y >=  0)
    model.addConstr( x1 - x2 + x3 + y >=  0)
    model.addConstr( x1 + x2 - x3 + y >=  0)
    model.addConstr( x1 + x2 + x3 - y >=  0)
    model.addConstr( x1 - x2 - x3 - y >= -2)
    model.addConstr(-x1 + x2 - x3 - y >= -2)
    model.addConstr(-x1 - x2 + x3 - y >= -2)    
    model.addConstr(-x1 - x2 - x3 + y >= -2)
   
def GIFT_MILP_Quasi_Diff(nb_rounds,all=True):
    diff_trail = extract_diff_trail(DIFF_TRAIL_FILE, nb_rounds)
    sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE) 
    model = Model("GIFT_SK_Quasi_Diff_MILP")

                        
    
    
    u = model.addVars(nb_rounds, 2, 64, vtype=GRB.BINARY, name="m") 
    Q = model.addVars(nb_rounds, 16, CORR_RANGE, vtype=GRB.BINARY, name="c") 
   
                        

                        
    model.addConstrs(u[0, 0, l] == 0 for l in range(64))
    model.addConstrs(u[nb_rounds - 1, 1, l] == 0 for l in range(64))
           

                        
    for r in tqdm(range(nb_rounds)):
        for i in range(16):
            b = bin_to_int([diff_trail[r][1][4*i+l] for l in range(4)], SBOX_SIZE) 
            a = bin_to_int([diff_trail[r][0][4*i+l] for l in range(4)], SBOX_SIZE) 

            model.addConstr(quicksum(Q[r, i, corr] for corr in CORR_RANGE) == 1) 
            for corr in CORR_RANGE:
                if sbox_inequalities[b][a][corr] == []:
                    model.addConstr(Q[r, i, corr] == 0) 
                    continue
                for ineq in sbox_inequalities[b][a][corr]: 
                    
                    model.addConstr(quicksum(ineq[0 * SBOX_SIZE + l] * u[r, 1, 4*i + l] for l in BIT_RANGE) + \
                                    quicksum(ineq[1 * SBOX_SIZE + l] * u[r, 0, 4*i + l] for l in BIT_RANGE) + \
                                    ineq[2 * SBOX_SIZE] + 500000 * (1 - Q[r, i, corr]) >= 0)  
                       
                        
    
    for r in range(nb_rounds-1):
        
        model.addConstrs(u[r, 1, l] == u[r + 1, 0, 63-BIT_PERM[63-l]] for l in range(64))

    model.write("gift_fixed_key.lp")

    
    model.addConstr(quicksum(Q[r, i, corr] * corr for r in range(NB_ROUNDS) for i in range(16) for corr in CORR_RANGE) >= -MIN_CORR)
    if(not all):
        model.addConstr(quicksum(Q[r, i, corr] * corr for r in range(NB_ROUNDS) for i in range(16) for corr in CORR_RANGE) <= -(MIN_CORR-1.0000001))
    model.setObjective(quicksum(Q[r, i, corr] * corr for r in range(NB_ROUNDS) for i in range(16) for corr in CORR_RANGE), GRB.MAXIMIZE)
    
    
                        
    
    model.params.PoolSearchMode = 2
    model.params.PoolSolutions = 2000000
                        

    t1=time.time()
    model.optimize()    
    t=time.time()-t1
    print("Time used:", t, "s")
    print("Found ", model.SolCount, "trails")

    
    

                        
    print("Computing sign and conditions for each trail...  ")
    signs = []
    correlations = []
    trails_conditions = []
   
    corr_dict = {}
    for m in tqdm(range(model.SolCount)):
        model.params.SolutionNumber = m
        mask_trail = [[[round(u[r, before, l].Xn) for l in range(64)]
                            for before in range(2)] for r in range(nb_rounds)]
        sign, corr, conditions = compute_correlation(diff_trail, mask_trail, nb_rounds)
        if m == 0:
            avg_prob = corr
       
        if corr not in corr_dict:
            corr_dict[corr] = 0

        corr_dict[corr] += 1
       
        signs.append(sign)
        correlations.append(corr)
        trails_conditions.append(conditions)
    print("solution numbers:",model.SolCount)
    key_map = [[0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31], [32, 48, 33, 49, 34, 50, 35, 51, 36, 52, 37, 53, 38, 54, 39, 55, 40, 56, 41, 57, 42, 58, 43, 59, 44, 60, 45, 61, 46, 62, 47, 63], [64, 80, 65, 81, 66, 82, 67, 83, 68, 84, 69, 85, 70, 86, 71, 87, 72, 88, 73, 89, 74, 90, 75, 91, 76, 92, 77, 93, 78, 94, 79, 95], [96, 112, 97, 113, 98, 114, 99, 115, 100, 116, 101, 117, 102, 118, 103, 119, 104, 120, 105, 121, 106, 122, 107, 123, 108, 124, 109, 125, 110, 126, 111, 127], [12, 18, 13, 19, 14, 20, 15, 21, 0, 22, 1, 23, 2, 24, 3, 25, 4, 26, 5, 27, 6, 28, 7, 29, 8, 30, 9, 31, 10, 16, 11, 17], [44, 50, 45, 51, 46, 52, 47, 53, 32, 54, 33, 55, 34, 56, 35, 57, 36, 58, 37, 59, 38, 60, 39, 61, 40, 62, 41, 63, 42, 48, 43, 49], [76, 82, 77, 83, 78, 84, 79, 85, 64, 86, 65, 87, 66, 88, 67, 89, 68, 90, 69, 91, 70, 92, 71, 93, 72, 94, 73, 95, 74, 80, 75, 81], [108, 114, 109, 115, 110, 116, 111, 117, 96, 118, 97, 119, 98, 120, 99, 121, 100, 122, 101, 123, 102, 124, 103, 125, 104, 126, 105, 127, 106, 112, 107, 113], [8, 20, 9, 21, 10, 22, 11, 23, 12, 24, 13, 25, 14, 26, 15, 27, 0, 28, 1, 29, 2, 30, 3, 31, 4, 16, 5, 17, 6, 18, 7, 19], [40, 52, 41, 53, 42, 54, 43, 55, 44, 56, 45, 57, 46, 58, 47, 59, 32, 60, 33, 61, 34, 62, 35, 63, 36, 48, 37, 49, 38, 50, 39, 51], [72, 84, 73, 85, 74, 86, 75, 87, 76, 88, 77, 89, 78, 90, 79, 91, 64, 92, 65, 93, 66, 94, 67, 95, 68, 80, 69, 81, 70, 82, 71, 83], [104, 116, 105, 117, 106, 118, 107, 119, 108, 120, 109, 121, 110, 122, 111, 123, 96, 124, 97, 125, 98, 126, 99, 127, 100, 112, 101, 113, 102, 114, 103, 115], [4, 22, 5, 23, 6, 24, 7, 25, 8, 26, 9, 27, 10, 28, 11, 29, 12, 30, 13, 31, 14, 16, 15, 17, 0, 18, 1, 19, 2, 20, 3, 21], [36, 54, 37, 55, 38, 56, 39, 57, 40, 58, 41, 59, 42, 60, 43, 61, 44, 62, 45, 63, 46, 48, 47, 49, 32, 50, 33, 51, 34, 52, 35, 53], [68, 86, 69, 87, 70, 88, 71, 89, 72, 90, 73, 91, 74, 92, 75, 93, 76, 94, 77, 95, 78, 80, 79, 81, 64, 82, 65, 83, 66, 84, 67, 85], [100, 118, 101, 119, 102, 120, 103, 121, 104, 122, 105, 123, 106, 124, 107, 125, 108, 126, 109, 127, 110, 112, 111, 113, 96, 114, 97, 115, 98, 116, 99, 117], [0, 24, 1, 25, 2, 26, 3, 27, 4, 28, 5, 29, 6, 30, 7, 31, 8, 16, 9, 17, 10, 18, 11, 19, 12, 20, 13, 21, 14, 22, 15, 23], [32, 56, 33, 57, 34, 58, 35, 59, 36, 60, 37, 61, 38, 62, 39, 63, 40, 48, 41, 49, 42, 50, 43, 51, 44, 52, 45, 53, 46, 54, 47, 55], [64, 88, 65, 89, 66, 90, 67, 91, 68, 92, 69, 93, 70, 94, 71, 95, 72, 80, 73, 81, 74, 82, 75, 83, 76, 84, 77, 85, 78, 86, 79, 87], [96, 120, 97, 121, 98, 122, 99, 123, 100, 124, 101, 125, 102, 126, 103, 127, 104, 112, 105, 113, 106, 114, 107, 115, 108, 116, 109, 117, 110, 118, 111, 119]]
    print(len(key_map))
    T = []
    sum_c2=0
    sig=signs
    for i in range(len(sig)):
        k_lst = []
        for r in range(1,len(trails_conditions[i])):
            for j in range(len(trails_conditions[i][r])):
                if trails_conditions[i][r][j] != []:
                    
                
                    if((63-j)%4<2):
                        ind=(63-j)//2+(63-j)%2
                    else:
                        ind=-1
                        print(f"condition at [{63-j}] is unreasonable")
                    
                    k_lst.append(f'k_{key_map[r-1][ind]}')
                    
        
        T.append({'sign': sig[i],  'corr': correlations[i], 'keys': (k_lst)})
        
    
    return T, avg_prob
if __name__=="__main__":
    ALL=True
    T, avg_prob = GIFT_MILP_Quasi_Diff(NB_ROUNDS,all=ALL) 
    print(T)
    
    appendix=f'_{NB_ROUNDS}RD_{MIN_CORR}_{ALL}{NAME}'
    plot_quasi_distribut(T,appendix,-1*avg_prob)
    
