import numpy as np
from utils import RECT_PERM, NB_ROWS, NB_COLS

def genLinear_RECT(rounds):
    STATE_COLS = 128 * (rounds + 1) 
    KEY_COLS = 64 * (rounds + 1)  
    TOTAL_COLS = STATE_COLS + KEY_COLS
    
    rows = []
    for r in range(rounds):
        for j in range(64):
            row = np.zeros(TOTAL_COLS, dtype=np.int8)
            
            
            row[r * 128 + 64 + j] = 1
            
            if r + 1 <= rounds:
                
                target_bit = RECT_PERM[j]
                row[(r + 1) * 128 + target_bit] = 1
                
                row[STATE_COLS + (r + 1) * 64 + target_bit] = 1
                
            rows.append(row)
            
    L = np.array(rows, dtype=np.int8)
    return L