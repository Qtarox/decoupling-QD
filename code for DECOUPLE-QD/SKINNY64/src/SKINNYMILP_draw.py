from gurobipy import *
from utils import *
from analyse_trails import *

blocks = get_transitions(DIFF_TRAIL_FILE)
QDTM_SKINNY_SBOX = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)
AC_STATES = compute_ac_states(NB_ROUNDS, SBOX_SIZE)

# blocks = [(15,15)]
# QDTM_SKINNY_SBOX = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)
# for i in range(16):
#     for j in range(16):
#         print(QDTM_SKINNY_SBOX[15][15][i][j],end='\t')
#     print()

# assert False

def compute_correlation(diff_trail, mask_trail, nb_rounds):
    conditions = [[[[] for j in range(4)] for i in range(4)] for k in range(nb_rounds)]
    corr = 1

    for k in range(nb_rounds):
        for i in STATE_RANGE:
            for j in STATE_RANGE:
                a, b = diff_trail[k][0][i][j], diff_trail[k][1][i][j]
                u, v = mask_trail[k][0][i][j], mask_trail[k][1][i][j]

                a, b = bin_to_int(a, SBOX_SIZE), bin_to_int(b, SBOX_SIZE)
                u, v = bin_to_int(u, SBOX_SIZE), bin_to_int(v, SBOX_SIZE)

                corr *= QDTM_SKINNY_SBOX[b][a][v][u]
                if QDTM_SKINNY_SBOX[b][a][v][u] == 0:
                    print(a, b)
                    print(u, v)
                    return "Error"    

                        # Linear layer
    # AddConstants
    for k in range(nb_rounds):
        corr *= character(AC_STATES[k][0],  mask_trail[k][1][0][0], SBOX_SIZE)
        corr *= character(AC_STATES[k][1],  mask_trail[k][1][1][0], SBOX_SIZE)
        corr *= character(AC_STATES[k][2],  mask_trail[k][1][2][0], SBOX_SIZE)
        
    # AddRoundKey
    for k in range(nb_rounds):
        for i in range(2):
            for j in STATE_RANGE:
                if mask_trail[k][1][i][j] == [0 for _ in BIT_RANGE]:
                    continue
                
                conditions[k][i][j] = mask_trail[k][1][i][j]

    if corr > 0:
        return  1, log2( corr), conditions
    elif corr < 0:
        return -1, log2(-corr), conditions
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
    
def SKINNY_MILP_Quasi_Diff(nb_rounds):
    diff_trail = extract_diff_trail(DIFF_TRAIL_FILE, nb_rounds)
    sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)

    model = Model("SKINNY_SK_Quasi_Diff_MILP")

                        # Definition of Variables
    # State variables
    u = model.addVars(nb_rounds, 2, 4, 4, SBOX_SIZE, vtype=GRB.BINARY)
    Q = model.addVars(nb_rounds, 4, 4, CORR_RANGE, vtype=GRB.BINARY)
    
                        # Adding Constraints

                        # Starting/Ending with zero masks contraints"""
    for i in STATE_RANGE:
        for j in STATE_RANGE:
            model.addConstrs(u[0, 0, i, j, l] == 0 for l in BIT_RANGE)
            model.addConstrs(u[nb_rounds - 1, 1, i, j, l] == 0 for l in BIT_RANGE)
            
    

                        # Non-linear layer constraints
    for r in tqdm(range(nb_rounds)):
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


    # Correlation constraints
    model.addConstr(quicksum(Q[r, i, j, corr] * corr for r in range(NB_ROUNDS) for i in STATE_RANGE for j in STATE_RANGE for corr in CORR_RANGE) >= -MIN_CORR)
    model.setObjective(quicksum(Q[r, i, j, corr] * corr for r in range(NB_ROUNDS) for i in STATE_RANGE for j in STATE_RANGE for corr in CORR_RANGE), GRB.MAXIMIZE)

                        # Gurobi options
    print("Searching for quasi-differential trails...\n")
    model.params.PoolSearchMode = 2
    model.params.PoolSolutions = 2000000
                        # Resolution of the model
    model.optimize()    

    print("Found ", model.SolCount, "trails")
                        # Computation of correlation
    print("Computing sign and conditions for each trail...  ")
    signs = []
    correlations = []
    trails_conditions = []
    
    corr_dict = {}
    for m in tqdm(range(model.SolCount)):
        model.params.SolutionNumber = m
        mask_trail = [[[[[round(u[r, before, i, j, l].Xn) for l in BIT_RANGE]
                            for j in STATE_RANGE] for i in STATE_RANGE] 
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


    # Extracting conditions on the key and taking the key schedule in 
    # consideration to detect some impossibilities

    factor_dict = {}
    for corr in sorted(corr_dict):
        factor_dict[corr] = (int)(2**(corr - min(corr_dict)))
    trail_indic(trails_conditions, correlations, signs, factor_dict, NB_ROUNDS)

    # Exportation of trails
    print("Printing trails...")
    outputfile_name = res_path
    file_mask = open(outputfile_name + f"{NB_ROUNDS}_mask.txt", "w")
   

    out_mask = ""
    for m in tqdm(range(model.SolCount)):
        model.params.SolutionNumber = m              
        for r in range(nb_rounds):
            for before in range(2):
                for i in STATE_RANGE:
                    for j in STATE_RANGE:
                        tmp_mask = bin_to_int([round(u[r, before, i, j, l].Xn)
                                          for l in BIT_RANGE], SBOX_SIZE)
                        
                        out_mask += str(tmp_mask) + " "
                        

        if m != model.SolCount - 1:
            out_mask += "\n"

    print(out_mask)
    file_mask.write(out_mask)
    file_mask.close()

    solutions_to_readable(outputfile_name + f"{NB_ROUNDS}_mask.txt")


    return model.SolCount

nb_sol = SKINNY_MILP_Quasi_Diff(NB_ROUNDS)
