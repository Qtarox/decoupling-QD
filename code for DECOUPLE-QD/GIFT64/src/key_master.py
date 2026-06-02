import itertools
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

def get_gift64_key_mapping(max_rounds):
   
    mk_state = list(range(128))
    
    round_key_map = {}
    
    for r in range(max_rounds):
        round_key_map[r] = {}
        
        
        V_indices = mk_state[0:16]   
        U_indices = mk_state[16:32]  
        
        
        for i in range(16):
            state_pos_v = 4 * i
            state_pos_u = 4 * i + 1
            
            round_key_map[r][state_pos_v] = V_indices[i]
            round_key_map[r][state_pos_u] = U_indices[i]
            
        
        
        
        old_k0 = mk_state[0:16]
        old_k1 = mk_state[16:32]
        old_k2_to_k7 = mk_state[32:128] 
        
        
        new_k6 = old_k0[12:] + old_k0[:12] 
        new_k7 = old_k1[2:] + old_k1[:2]   
        
        
        mk_state = old_k2_to_k7 + new_k6 + new_k7
    LST=[]
    for r in round_key_map:
        T_LST=[]
        for key in round_key_map[r]:
            T_LST.append(round_key_map[r][key])
        LST.append(T_LST.copy())
    return LST

if __name__=="__main__":
    mapping = get_gift64_key_mapping(20)
    
    print(f"Round 0, State Bit 0 maps to Master Key Bit: {mapping}")
    