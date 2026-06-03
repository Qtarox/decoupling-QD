from utils import *

# Sage imports
from sage.all import var
from sage.rings.finite_rings.finite_field_constructor import GF
from sage.symbolic.ring import SR
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.integer_ring import ZZ
from sage.ext.fast_callable import fast_callable
from sage.matrix.constructor import matrix, Matrix
from sage.modules.free_module_element import vector
from sage.misc.misc_c import prod


import random, sys, numpy as np
from tqdm      import tqdm
from math      import log2
from itertools import product

sys.setrecursionlimit(10000)

# Keyschedule permutation
KS = [[ 9, 15,  8, 13], 
      [10, 14, 12, 11],
      [ 0,  1,  2,  3],
      [ 4,  5,  6,  7]]

def TK2_lfsr(cell):
    if SBOX_SIZE == 4:
        tmp = cell[0]
        for l in BIT_RANGE[:-1]:
            cell[l] = cell[l + 1]

        cell[SBOX_SIZE - 1] = tmp + cell[0] #xor is bit addition
        return cell
    elif SBOX_SIZE == 8:
        tmp = cell[0]
        for l in BIT_RANGE[:-1]:
            cell[l] = cell[l + 1]

        cell[SBOX_SIZE - 1] = tmp + cell[1] #xor is bit addition
        return cell

def TK3_lfsr(cell):
    if SBOX_SIZE == 4:
        tmp = cell[SBOX_SIZE - 1]
        for l in range(SBOX_SIZE - 1, 0, -1):
            cell[l] = cell[l - 1]
        
        cell[0] = tmp + cell[1]
        return cell
    
    elif SBOX_SIZE == 8:
        tmp = cell[SBOX_SIZE - 1]
        for l in range(SBOX_SIZE - 1, 0, -1):
            cell[l] = cell[l - 1]

        cell[0] = tmp + cell[2] # xor is bit addition
        return cell

def key_schedule_perm(s):
    new_s = [[None for j in STATE_RANGE] for i in STATE_RANGE]
    for i in STATE_RANGE:
        for j in STATE_RANGE:
            new_s[i][j] = s[KS[i][j] // 4][KS[i][j] % 4].copy()
    return new_s

def key_schedule(nb_rounds):    
    P = PolynomialRing(GF(2), [f"tk_{key_idx}_{4 * i + j}_{SBOX_SIZE - l - 1}" 
                               for l in BIT_RANGE for i in STATE_RANGE 
                               for j in STATE_RANGE for key_idx in range(Z)])
    
    S = [[[[[P(f"tk_{key_idx}_{4 * i + j}_{SBOX_SIZE - l - 1}") for l in BIT_RANGE] for j in STATE_RANGE] for i in STATE_RANGE] for key_idx in range(Z)]]
    for _ in range(nb_rounds):
        tmp = []
        for key_idx in range(Z):
            tmp.append(key_schedule_perm(S[-1][key_idx]))

        if Z >= 2:
            for i in range(2):
                for j in STATE_RANGE:
                    TK2_lfsr(tmp[1][i][j])
            
        if Z >= 3:
            for i in range(2):
                for j in STATE_RANGE:
                    TK3_lfsr(tmp[2][i][j])

        S.append(tmp)
    return S

def build_full_sum(mask_trails_conditions, nb_rounds, simplified_ring, full_ring, K):
    full_bits = []
    monomials = []
    monomials_by_bits = {}
    for m, conditions in enumerate(tqdm(mask_trails_conditions)):
        monomial = simplified_ring(1)
        for k in range(nb_rounds):
            for i in STATE_RANGE:
                for j in STATE_RANGE:
                    if conditions[k][i][j] == []:
                        continue

                    for l in BIT_RANGE:
                        if conditions[k][i][j][l] == 0:
                            continue
                        
                        for key_idx in range(Z):
                            for bit in full_ring(K[k][key_idx][i][j][l]).variables():
                                if bit not in full_bits:
                                    full_bits.append(bit)
                        
                        monomial *= simplified_ring(f"tk_{k}_{4 * i + j}_{SBOX_SIZE - l - 1}")
        monomials.append(monomial.change_ring(ZZ))
    
    for m, monomial in enumerate(monomials):
        for bit in monomial.variables():
            if bit not in monomials_by_bits:
                monomials_by_bits[bit] = []
            monomials_by_bits[bit].append(m) # which monomial contains the bit (inverse mapping)
    return full_bits, monomials, monomials_by_bits

def key_space_estimation(correlations, factor_dict, simplified_bits, full_bits, full_sum, ff, full_ring, T, K, probabilistic=False):
    if probabilistic:
        # Trying 10**6 different combination to estimate the key space & fixed_key 
        # probability distribution
        valid = {}
        ctr = 0
        for _ in tqdm(range(10**6)): # Magic number 
            el = tuple([random.choice([-1, 1]) for _ in range(len(full_sum.variables()))])
            val = ff(*el) 
            if val < 0:
                return None
            
            if val != 0:
                valid[el] = val / factor_dict[correlations[0]]
            
                
    else:
        valid = {}
        for el in tqdm(product([-1,1], repeat=len(full_sum.variables())), total=2**len(full_sum.variables())):
            val = ff(*el)
            if val < 0:
                return None
            
            if val != 0:
                valid[el] = val / factor_dict[correlations[0]]
    print('valid',valid)
    # Taking the key schedule in consideration                
    M = matrix(GF(2), len(simplified_bits), len(full_bits))
    F = GF(2)
    for m, bit in enumerate(simplified_bits):
        tmp = str(bit).split("_")
        
        k = int(tmp[1]) # which tk
        i,j = int(tmp[2]) // 4, int(tmp[2]) % 4 # which row/col
        l = int(tmp[3]) # bit position
        full_bit = full_ring(0)
        for key_idx in range(Z):
            full_bit += K[k][key_idx][i][j][SBOX_SIZE - l - 1]

        for b in full_bit.variables():
            M[m, full_bits.index(b)] += F(1)
        print(full_bit)
    
    print(M,full_bits)
    kernel_cardinality = 2**(len(M.right_kernel().basis()))



    ctr = 0
    avg_prob = correlations[0]
    distrib = {}
    idx_bits = {bit: simplified_bits.index(bit) for bit in simplified_bits}
    for simplified_sol in tqdm(valid, total=len(valid)):
        tmp = [0 for _ in range(len(simplified_bits))]
        for i in range(len(full_sum.variables())):
            tmp[idx_bits[full_sum.variables()[i]]] = simplified_sol[i]

        for rel in T:
            if rel == -1 or rel == 1:
                for bit in T[rel]:
                    tmp[idx_bits[bit]] = rel
            else:
                for bit in T[rel]:
                    tmp[idx_bits[bit]] *= rel.subs({el: simplified_sol[full_sum.variables().index(el)] for el in rel.variables()})


        tmp = [(1 - el) // 2 for el in tmp]
        Y = vector(GF(2), tmp)
        try:
            X = M.solve_right(Y)
            p = log2(valid[simplified_sol] * 2**avg_prob)
            ctr += kernel_cardinality
            
            if p not in distrib:
                distrib[p] = 0
            
            distrib[p] += 1
        except:
            continue
    
    for p in distrib:
        distrib[p] /= len(valid)
        distrib[p] *= 100
        distrib[p] = distrib[p]
    key_space = log2(ctr) - len(full_bits)

    return distrib, key_space

def trail_indic (mask_trails_conditions, correlations, signs, factor_dict, nb_rounds):
    K = key_schedule(nb_rounds)
    # Bits of round keys when seeing the tweakey as TK1 + TK2 + TK3
    simplified_ring = PolynomialRing(GF(2), [f"tk_{k}_{4 * i + j}_{SBOX_SIZE - l - 1}" 
                               for l in BIT_RANGE for i in STATE_RANGE 
                               for j in STATE_RANGE for k in range(NB_ROUNDS)])
    
    # Bit of round keys when making a distinction between TK1/TK2/TK3
    full_ring = PolynomialRing(GF(2), [f"tk_{key_idx}_{4 * i + j}_{SBOX_SIZE - l - 1}" 
                               for l in BIT_RANGE for i in STATE_RANGE 
                               for j in STATE_RANGE for key_idx in range(Z)])
    
    full_bits, monomials, monomials_by_bits = build_full_sum(mask_trails_conditions, 
                                                             nb_rounds, simplified_ring, 
                                                             full_ring, K)
    
    simplified_bits = list(monomials_by_bits.keys())  
    nb_bits = len(simplified_bits)
    print(mask_trails_conditions)
    print('length:',len(mask_trails_conditions))
    print(monomials)
    print(monomials_by_bits)
    optimal_monomials = []
    optimal_signs = []
    optimal_bits = []
    print('correlations')
    print(correlations)
    for m in tqdm(range(len(correlations))): # 36 trails 
        if correlations[m] != correlations[0]: # look at the trails that are optimal
            continue
        
        for bit in monomials[m].variables():
            if bit not in optimal_bits:
                optimal_bits.append(bit)
        optimal_monomials.append(monomials[m]) 
        optimal_signs.append(signs[m])
    
    # optimal_bits involved all the bits existing in the optimal trails.
    ret = bits_system(optimal_bits, optimal_monomials, optimal_signs)
    # try to solve the system

    if ret == False:
        print("Characteristic is impossible")
        return False

    linear_relations = list(ret[0])
    constant_term = list(ret[1])
    
    linear_relations.reverse()
    constant_term.reverse()
    optimal_bits.reverse()

    T = {} # contains the mask effect of (-1)^(x0+x1) k0k1
    for l, rel in enumerate(linear_relations):
        rel = list(rel)
        rel.reverse()
        print(rel)
        rel_bits = []
        for i in range(len(rel)):
            if rel[i] != 0:
                rel_bits.append(i) # putting the index of the bits that are non-zero in here
        if rel_bits == []:
            continue
        print('optimal bits')
        print(optimal_bits)
        print(constant_term)
        tmp = prod([optimal_bits[rel_bits[idx]] for idx in range(len(rel_bits) - 1)]) * (-1)**int(constant_term[l])
        print('this is the tmp') # 1
        print(tmp)
        if tmp not in T:
            T[tmp] = []
        print('rel_bits =', rel_bits)
        T[tmp].append(optimal_bits[rel_bits[-1]])
        print(T)
        print('before',monomials[m])

        for m in monomials_by_bits[optimal_bits[rel_bits[-1]]]:
            monomials[m] = monomials[m].subs({optimal_bits[rel_bits[-1]]:tmp})
        print('after',monomials[m])

    full_sum = 0
    for m, monomial in enumerate(tqdm(monomials)):
        print(signs[m],factor_dict[correlations[m]] * monomial.change_ring(ZZ))
        if signs[m] == -1:
            full_sum -= factor_dict[correlations[m]] * monomial.change_ring(ZZ)
        else:
            full_sum += factor_dict[correlations[m]] * monomial.change_ring(ZZ)
    print('full_sum',full_sum)  
    if full_sum == 0:
        print("Characteristic is impossible")
        return False
    
    new_ring = PolynomialRing(ZZ, full_sum.variables())
    full_sum = new_ring(full_sum)
    ff = fast_callable(full_sum)
    probabilistic_estimation = len(full_sum.variables()) >= 20
    distrib, key_space = key_space_estimation(correlations, factor_dict, 
                                              simplified_bits, full_bits, 
                                              full_sum, ff, full_ring, T, K,
                                              probabilistic_estimation)

    new_dict = {}
    for key in sorted(distrib.keys()):
        new_dict[-key] = distrib[key]
        
    print(new_dict)
    print(f"Characteristic is possible for 2^{key_space} of the keys")

    return

def bits_system(bits, monomials, signs):
    # optimal bits contain all the possible bits in the involved monomial
    # optimal monomial are all the involved monomials
    # construct a matrix with columns as all the bits, rows with all possible monomial
    M = Matrix(GF(2), [[0 for _ in range(len(bits))] for i in range(len(monomials))])
    Y = vector(GF(2), [0 for _ in range(len(monomials))])
    index_bits = {bit: bits.index(bit) for bit in bits} # getting the index of the elements in bits
    for i, monomial in enumerate(tqdm(monomials)):
        for bit in monomial.variables():
            M[i, index_bits[bit]] += 1 
        if signs[i] == -1:
            Y[i] += 1
    '''
    constructing a matrix where
    -x0x1x3
    x1x2x3
    x2
    becomes
    1 1 0 1 | 1
    0 1 1 1 | 0
    0 1 0 0 | 0
    '''

    M = M.change_ring(ZZ)
    N, R = M.echelon_form(transformation=True)
    N, R, M = N.change_ring(GF(2)), R.change_ring(GF(2)), M.change_ring(GF(2))
    try:
        X = M.solve_right(Y)
        # solving this system of linear equations
        # N is the REF, R is the row operations permformed
        # So, MX = Y --> R(MX) = RY --> NX = RY
        return N, R*Y
    except:
        return False
    
def key_sched_on_diff(init_diff, nb_rounds):
    for i in STATE_RANGE:
        for j in STATE_RANGE:
            init_diff[i][j] = int_to_bin(init_diff[i][j], SBOX_SIZE)
    tmp = init_diff
    print(np.array([[bin_to_int(tmp[i][j], SBOX_SIZE) for j in STATE_RANGE] for i in STATE_RANGE]))
    for r in range(nb_rounds):
        tmp = key_schedule_perm(tmp)
        for i in STATE_RANGE:
            for j in STATE_RANGE:
                tmp[i][j] = TK2_lfsr(tmp[i][j])
        print(np.array([[bin_to_int(tmp[i][j], SBOX_SIZE) for j in STATE_RANGE] for i in STATE_RANGE]))
    return

