from gurobipy import *
from utils import *
import numpy as np
from math import log2
from tqdm import tqdm

def compute_correlation(qdtm, constants, diff_trail, mask_trail, nb_rounds):
    corr = 1

    for k in range(nb_rounds):
        for i in STATE_RANGE:
            for j in STATE_RANGE:
                a, b = diff_trail[k][0][i][j], diff_trail[k][1][i][j]
                u, v = mask_trail[k][0][i][j], mask_trail[k][1][i][j]

                a, b = bin_to_int(a, SBOX_SIZE), bin_to_int(b, SBOX_SIZE)
                u, v = bin_to_int(u, SBOX_SIZE), bin_to_int(v, SBOX_SIZE)

                corr *= qdtm[b][a][v][u]
                if qdtm[b][a][v][u] == 0:
                    print(a, b)
                    print(u, v)
                    return "Error"    
                        # Linear layer
    # AddConstants
    for k in range(nb_rounds):
        corr *= character(constants[k][0],  mask_trail[k][1][0][0], SBOX_SIZE)
        corr *= character(constants[k][1],  mask_trail[k][1][1][0], SBOX_SIZE)
        corr *= character(constants[k][2],  mask_trail[k][1][2][0], SBOX_SIZE)
    

    if corr > 0:
        return  1, log2( corr)
    elif corr < 0:
        return -1, log2(-corr)
    
    return 1, 0, []

# Add constraints to the model so that y = x1 ^ x2
def add_xor_constraints(model, x1, x2, y):
    model.addConstr(-x1 + x2 + y >=  0)
    model.addConstr( x1 - x2 + y >=  0)
    model.addConstr( x1 + x2 - y >=  0)
    model.addConstr(-x1 - x2 - y >= -2)
    
# Add constraints to the model so that y = x1 ^ x2 ^ x3
def add_xor_constraints2(model, x1, x2, x3, y):
    model.addConstr(-x1 + x2 + x3 + y >=  0)
    model.addConstr( x1 - x2 + x3 + y >=  0)
    model.addConstr( x1 + x2 - x3 + y >=  0)
    model.addConstr( x1 + x2 + x3 - y >=  0)
    model.addConstr( x1 - x2 - x3 - y >= -2)
    model.addConstr(-x1 + x2 - x3 - y >= -2)
    model.addConstr(-x1 - x2 + x3 - y >= -2)    
    model.addConstr(-x1 - x2 - x3 + y >= -2)
    
def SKINNY_MILP_Quasi_Diff(nb_rounds, diff_trail=[], sbox_inequalities=[], qdtm=[], constants=[], verbose=True):
    if diff_trail == []:
        diff_trail = extract_diff_trail(DIFF_TRAIL_FILE, nb_rounds)

    if sbox_inequalities == []:
        sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)
    
    if qdtm == []:
        blocks = get_transitions(DIFF_TRAIL_FILE)
        print(blocks)
        qdtm = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)

    if constants == []:
        constants = compute_ac_states(nb_rounds, SBOX_SIZE)

    env = Env(empty=True)
    if not verbose:
        env.setParam("OutputFlag", 0)
    env.start()
    model = Model(env=env,name="SKINNY_Quasi_Diff_MILP")
    

                        # Definition of Variables
    # State variables
    u = model.addVars(nb_rounds, 2, 4, 4, SBOX_SIZE, vtype=GRB.BINARY)
    Q = model.addVars(nb_rounds, 4, 4, CORR_RANGE, vtype=GRB.BINARY)

    # Key variables
    tk1 = model.addVars(nb_rounds, 2, 4, 4, SBOX_SIZE, vtype=GRB.BINARY)
    if Z >= 2:
        tk2 = model.addVars(nb_rounds, 2, 4, 4, SBOX_SIZE, vtype=GRB.BINARY)

    if Z == 3:
        tk3 = model.addVars(nb_rounds, 2, 4, 4, SBOX_SIZE, vtype=GRB.BINARY)

                        # Adding Constraints

                        # Starting/Ending with zero masks contraints
    for i in STATE_RANGE:
        for j in STATE_RANGE:
            model.addConstrs(u[0, 0, i, j, l] == 0 for l in BIT_RANGE)
            model.addConstrs(u[nb_rounds - 1, 1, i, j, l] == 0 for l in BIT_RANGE)
            model.addConstrs(tk1[0, 0, i, j, l] == 0 for l in BIT_RANGE)
            model.addConstrs(tk1[nb_rounds - 1, 1, i, j, l] == 0 for l in BIT_RANGE)
            if Z >= 2:
                model.addConstrs(tk2[0, 0, i, j, l] == 0 for l in BIT_RANGE)
                model.addConstrs(tk2[nb_rounds - 1, 1, i, j, l] == 0 for l in BIT_RANGE)

            if Z == 3:
                model.addConstrs(tk3[0, 0, i, j, l] == 0 for l in BIT_RANGE)
                model.addConstrs(tk3[nb_rounds - 1, 1, i, j, l] == 0 for l in BIT_RANGE)

            
    

                        # Non-linear layer constraints
    for r in range(nb_rounds):
        for i in STATE_RANGE:
            for j in STATE_RANGE:                
                b = bin_to_int([diff_trail[r][1][i][j][l] for l in BIT_RANGE], SBOX_SIZE) 
                a = bin_to_int([diff_trail[r][0][i][j][l] for l in BIT_RANGE], SBOX_SIZE)

                model.addConstr(quicksum(Q[r, i, j, corr] for corr in CORR_RANGE) == 1)
                for corr in CORR_RANGE:
                    if sbox_inequalities[b][a][corr] == []:
                        model.addConstr(Q[r, i, j, corr] == 0)
                        continue
                    for ineq in sbox_inequalities[b][a][corr]:
                        model.addConstr(quicksum(ineq[2 * SBOX_SIZE - l - 1] * u[r, 1, i, j, l] for l in BIT_RANGE) + \
                                        quicksum(ineq[1 * SBOX_SIZE - l - 1] * u[r, 0, i, j, l] for l in BIT_RANGE) - \
                                        ineq[2 * SBOX_SIZE] + 500 * (1 - Q[r, i, j, corr]) >= 0)  # M = 500

                        # Linear layer constraints
    # Permutation of TK1
    for r in range(nb_rounds - 1):
        for i in STATE_RANGE:
            for j in STATE_RANGE:
                model.addConstrs(tk1[r + 1, 0, i, j, l] == tk1[r, 1, KS[i][j] // 4, KS[i][j] % 4, l] for l in BIT_RANGE)

    # Permutation & LFSR of TK2
    if Z >= 2:
        if SBOX_SIZE == 4:
            for r in range(nb_rounds - 1):
                for i in range(2):
                    for j in STATE_RANGE:
                        new_i, new_j = KS[i][j] // 4, KS[i][j] % 4
                        model.addConstrs(tk2[r + 1, 0, i, j, l] == tk2[r, 1, new_i, new_j, l + 1] for l in range(SBOX_SIZE - 1))
                        add_xor_constraints(model, tk2[r + 1, 0, i, j, SBOX_SIZE - 1], tk2[r, 1, new_i, new_j, 0], tk2[r, 1, new_i, new_j, 1])
                
        else:
            for r in range(nb_rounds - 1):
                for i in range(2, 4):
                    for j in STATE_RANGE:
                        new_i, new_j = KS[i][j] // 4, KS[i][j] % 4
                        model.addConstrs(tk2[r + 1, 0, i, j, l] == tk2[r, 1, new_i, new_j, l + 1] for l in range(SBOX_SIZE - 1))
                        add_xor_constraints(model, tk2[r + 1, 0, i, j, SBOX_SIZE - 1], tk2[r, 1, new_i, new_j, 0], tk2[r, 1, new_i, new_j, 2])

        for r in range(nb_rounds - 1):
            for i in range(2, 4):
                for j in STATE_RANGE:
                    model.addConstrs(tk2[r + 1, 0, i, j, l] == tk2[r, 1, KS[i][j] // 4, KS[i][j] % 4, l] for l in BIT_RANGE)

    # Permutation & LFSR of TK3
    if Z == 3:
        if SBOX_SIZE == 4:
            for r in range(nb_rounds - 1):
                for i in range(2):
                    for j in STATE_RANGE:
                        new_i, new_j = KS[i][j] // 4, KS[i][j] % 4
                        model.addConstrs(tk3[r + 1, 0, i, j, l] == tk3[r, 1, new_i, new_j, l - 1] for l in range(1, SBOX_SIZE))
                        add_xor_constraints(model, tk3[r + 1, 0, i, j, 0], tk3[r, 1, new_i, new_j, 0], tk3[r, 1, new_i, new_j, SBOX_SIZE - 1])
        else:
            for r in range(nb_rounds - 1):
                for i in range(2):
                    for j in STATE_RANGE:
                        new_i, new_j = KS[i][j] // 4, KS[i][j] % 4
                        model.addConstrs(tk3[r + 1, 0, i, j, l] == tk3[r, 1, new_i, new_j, l - 1] for l in range(1, SBOX_SIZE))
                        add_xor_constraints(model, tk3[r + 1, 0, i, j, 0], tk3[r, 1, new_i, new_j, SBOX_SIZE - 1], tk3[r, 1, new_i, new_j, 1])
        
        for r in range(nb_rounds - 1):
            for i in range(2, 4):
                for j in STATE_RANGE:
                    model.addConstrs(tk3[r + 1, 0, i, j, l] == tk3[r, 1, KS[i][j] // 4, KS[i][j] % 4, l] for l in BIT_RANGE)

    # MixColumns
    # First row 
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:     
                model.addConstrs(u[r, 1, 3, (j - 3) % 4, l] == u[r + 1, 0, 0, j, l] for l in BIT_RANGE)
            
    
    # Second row
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:
            for l in BIT_RANGE:
                add_xor_constraints2(model, u[r, 1, 0, j, l],
                                            u[r, 1, 1, (j - 1) % 4, l], 
                                            u[r, 1, 2, (j - 2) % 4, l], 
                                            u[r + 1, 0, 1, j, l]) 

    # Third row
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:
            model.addConstrs(u[r, 1, 1, (j - 1) % 4, l] == u[r + 1, 0, 2, j, l] for l in BIT_RANGE)

    # Fourth row
    for r in range(nb_rounds - 1):
        for j in STATE_RANGE:
            for l in BIT_RANGE:                
                add_xor_constraints2(model, u[r, 1, 1, (j - 1) % 4, l], 
                                            u[r, 1, 2, (j - 2) % 4, l], 
                                            u[r, 1, 3, (j - 3) % 4, l], 
                                            u[r + 1, 0, 3, j, l])

                        # Key addition constraint
    for r in range(nb_rounds):
        for i in range(2):
            for j in STATE_RANGE:
                for l in BIT_RANGE:
                    add_xor_constraints(model, tk1[r, 1, i, j, l], tk1[r, 0, i, j, l], u[r, 1, i, j, l])
                    if Z >= 2:
                        add_xor_constraints(model, tk2[r, 1, i, j, l], tk2[r, 0, i, j, l], u[r, 1, i, j, l])
                    if Z == 3:
                        add_xor_constraints(model, tk3[r, 1, i, j, l], tk3[r, 0, i, j, l], u[r, 1, i, j, l])

        for i in range(2, 4):
            for j in STATE_RANGE:
                model.addConstrs(tk1[r, 1, i, j, l] == tk1[r, 0, i, j, l] for l in BIT_RANGE)
                if Z >= 2:
                    model.addConstrs(tk2[r, 1, i, j, l] == tk2[r, 0, i, j, l] for l in BIT_RANGE)
                if Z == 3:
                    model.addConstrs(tk3[r, 1, i, j, l] == tk3[r, 0, i, j, l] for l in BIT_RANGE)

    
    # Correlation constraints
    model.addConstr(quicksum(Q[r, i, j, corr] * corr for r in range(nb_rounds) for i in STATE_RANGE for j in STATE_RANGE for corr in CORR_RANGE) >= -MIN_CORR)
    model.setObjective(quicksum(Q[r, i, j, corr] * corr for r in range(nb_rounds) for i in STATE_RANGE for j in STATE_RANGE for corr in CORR_RANGE), GRB.MAXIMIZE)

                        # Gurobi options
    model.params.PoolSearchMode = 2
    model.params.PoolSolutions = 2000000
                        # Resolution of the model
    model.optimize()    

                        # Computation of correlation

    avg_proba = 0
    for m in range(model.SolCount):
        model.params.SolutionNumber = m
        mask_trail = [[[[[round(u[r, before, i, j, l].Xn) for l in BIT_RANGE]
                            for j in STATE_RANGE] for i in STATE_RANGE] 
                            for before in range(2)] for r in range(nb_rounds)]
        
        sign, corr = compute_correlation(qdtm, constants, diff_trail, mask_trail, nb_rounds)

        avg_proba += (2**corr) * sign

    if verbose:
        if avg_proba != 0:
            print("AVG PROBA =", np.log2(avg_proba))
        else:
            print("Characteristic is impossible...")
        

        # Exportation of trails
        print("Printing trails...")
        outputfile_name = RESULTS_FILE
        file_mask = open(outputfile_name + "_mask", "w")
        file_tk1 = open(outputfile_name + "_tk1", "w")
        file_tk2 = open(outputfile_name + "_tk2", "w")
        file_tk3 = open(outputfile_name + "_tk3", "w")

        out_mask = ""
        out_tk1 = ""
        out_tk2 = ""
        out_tk3 = ""
        for m in tqdm(range(model.SolCount)):
            model.params.SolutionNumber = m              
            for r in range(nb_rounds):
                for before in range(2):
                    for i in STATE_RANGE:
                        for j in STATE_RANGE:
                            
                            tmp_mask = bin_to_int([round(u[r, before, i, j, l].Xn)
                                            for l in BIT_RANGE], SBOX_SIZE)
                            tmp_tk1 = bin_to_int([round(tk1[r, before, i, j, l].Xn)
                                            for l in BIT_RANGE], SBOX_SIZE)
                            
                            if Z >= 2:
                                tmp_tk2 = bin_to_int([round(tk2[r, before, i, j, l].Xn)
                                                for l in BIT_RANGE], SBOX_SIZE)
                                out_tk2 += str(tmp_tk2) + " "

                            if Z >= 3:
                                tmp_tk3 = bin_to_int([round(tk3[r, before, i, j, l].Xn)
                                                for l in BIT_RANGE], SBOX_SIZE)
                                out_tk3 += str(tmp_tk3) + " "
                            
                            out_mask += str(tmp_mask) + " "
                            out_tk1 += str(tmp_tk1) + " "
                            

            if m != model.SolCount - 1:
                out_mask += "\n"
                out_tk1 += "\n"
                out_tk2 += "\n"
                out_tk3 += "\n"

        file_mask.write(out_mask)
        file_tk1.write(out_tk1)
        file_tk2.write(out_tk2)
        file_tk3.write(out_tk3)

        file_mask.close()
        file_tk1.close()
        file_tk2.close()
        file_tk3.close()

        solutions_to_readable(outputfile_name + "_mask")
        solutions_to_readable(outputfile_name + "_tk1")
        solutions_to_readable(outputfile_name + "_tk2")
        solutions_to_readable(outputfile_name + "_tk3")

    return avg_proba

if __name__ == "__main__":
    nb_sol = SKINNY_MILP_Quasi_Diff(NB_ROUNDS)
