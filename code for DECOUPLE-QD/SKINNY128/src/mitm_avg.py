from utils import *
from SKINNYMILP_avg import *
from multiprocessing import Pool


blocks = [(4, 208), (84, 192), (48, 9), (4, 80), (68, 80), (6, 4), (1, 5), 
          (33, 6), (5, 196), (212, 64), (196, 80), (128, 160), (128, 41), 
          (128, 32), (5, 68), (8, 2), (4, 192), (4, 64), (80, 16), (68, 64), 
          (32, 33), (208, 16), (196, 64), (5, 4), (41, 33), (160, 33), (48, 8), 
          (144, 40), (16, 9), (64, 16), (33, 5), (192, 16), (128, 40), (80, 48), 
          (32, 1), (40, 33), (2, 144), (208, 48), (1, 4), (5, 212), (84, 208), 
          (0, 0), (16, 8), (212, 80), (9, 2), (144, 160), (144, 41), (64, 48), 
          (144, 32), (2, 128), (192, 48), (5, 84)]

prob = 131
NUM_PROCESSES = 128
qdtm = extract_quasi_diff_matrix(MATRIX_FILE, blocks, SBOX_SIZE)
constants = compute_ac_states(NB_ROUNDS, SBOX_SIZE)
sbox_inequalities = extract_inequalities_by_corr(SBOX_INEQUALITIES_DIR, SBOX_SIZE)

def test_diff_mitm_trails():
    differential_prob = 0
    ctr = 0
    valid_trails = []
    invalid_trails = []
    with open(f"../data/MITM_trails/spaced_trails_{prob}.trails", "r") as f:
        diff_trails = []
        lines = f.readlines()
        for line in lines:
            line = line.split()
            line = [int(x) for x in line]
            diff_trail = [[[[0 for j in range(4)] for i in range(4)] for before in range(2)] for r in range(NB_ROUNDS)]
            for i, el in enumerate(line):
                r = i // 32
                before = (i % 8) // 4
                row = (i % 32)  // 8
                col = i % 4
                diff_trail[r][before][row][col] = int_to_bin(el, SBOX_SIZE)

            diff_trails.append(diff_trail)

        func = SKINNY_MILP_Quasi_Diff
        argument_list = [(NB_ROUNDS, diff_trail, sbox_inequalities, qdtm, constants, False) for diff_trail in diff_trails]
        pool = Pool(processes=NUM_PROCESSES)

        jobs = [pool.apply_async(func=func, args=(*argument,)) if isinstance(argument, tuple) else pool.apply_async(func=func, args=(argument,)) for argument in argument_list]
        pool.close()
        result_list_tqdm = []
        for job in tqdm(jobs):
            result_list_tqdm.append(job.get())
        
        for i, p in enumerate(result_list_tqdm):
            if p == 0:
                ctr += 1
                invalid_trails.append(diff_trails[i])
            else:
                differential_prob += p
                valid_trails.append((p, diff_trails[i]))
                
        
    file_imp_mask_name = f"../results/mitm/impossible_trails_{prob}"
    file_pos_mask_name = f"../results/mitm/possible_trails_{prob}"
    file_imp_mask = open(file_imp_mask_name, "w")
    file_pos_mask = open(file_pos_mask_name, "w")
   
    out_mask = ""
    for diff_trail in invalid_trails:
        for r in range(14):
            for before in range(2):
                for i in STATE_RANGE:
                    for j in STATE_RANGE:
                        tmp_mask = bin_to_int(diff_trail[r][before][i][j], SBOX_SIZE)
                        out_mask += str(tmp_mask) + " "
        out_mask += "\n"
    file_imp_mask.write(out_mask)

    out_mask = ""
    for (p, diff_trail) in valid_trails:
        for r in range(14):
            for before in range(2):
                for i in STATE_RANGE:
                    for j in STATE_RANGE:
                        tmp_mask = bin_to_int(diff_trail[r][before][i][j], SBOX_SIZE)
                        out_mask += str(tmp_mask) + " "
        out_mask += "\n"
    file_pos_mask.write(out_mask)


    file_imp_mask.close()
    file_pos_mask.close()

    
    

    print("Differential proba =", np.log2(differential_prob))
    print("Number of impossible characteristics = ", ctr)

if __name__ == "__main__":
    test_diff_mitm_trails()