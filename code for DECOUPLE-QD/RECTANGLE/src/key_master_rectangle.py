from utils import RECTANGLE_RC
def xor_sym(a, b):
    return frozenset(a) ^ frozenset(b)
def xor_many(*sets):
    result = frozenset()
    for s in sets:
        result = result ^ frozenset(s)
    return result
NB_ROWS_80 = 5
NB_COLS = 16
def init_key_state_80():
    state = [[None] * NB_COLS for _ in range(NB_ROWS_80)]
    for row in range(NB_ROWS_80):
        for col in range(NB_COLS):
            mk_idx = row * NB_COLS + col
            state[row][col] = frozenset({mk_idx})
    return state
def key_update_80(state, round_idx, nl_log):
    new_state = [[None] * NB_COLS for _ in range(NB_ROWS_80)]    
    for col in range(4):
        in_bits = [state[r][col] for r in range(4)]
        for row in range(4):
            nl_marker = ('NL_KS80', round_idx, row, col)
            new_state[row][col] = frozenset({nl_marker})
            nl_log.append({
                'round': round_idx,
                'position': (row, col),
                'inputs': in_bits,  
                'marker': nl_marker,
            })
    for col in range(4):
        new_state[4][col] = state[4][col]
    for col in range(4, NB_COLS):
        for row in range(NB_ROWS_80):
            new_state[row][col] = state[row][col]
    rotated = [[None] * NB_COLS for _ in range(NB_ROWS_80)]
    shifts_80 = [8, 0, 0, 0, 0]   
    for row in range(NB_ROWS_80):
        sh = shifts_80[row]
        for col in range(NB_COLS):
            src_col = (col - sh) % NB_COLS
            rotated[row][col] = new_state[row][src_col]
    final_state = [[None] * NB_COLS for _ in range(NB_ROWS_80)]
    for col in range(NB_COLS):
        final_state[0][col] = xor_sym(rotated[0][col], new_state[1][col])
        final_state[1][col] = new_state[2][col]
        final_state[2][col] = xor_sym(rotated[2][col], new_state[3][col])
        final_state[3][col] = new_state[4][col]
        final_state[4][col] = xor_sym(rotated[4][col], new_state[0][col])
    return final_state
def build_round_keys_80(nb_rounds):
    state = init_key_state_80()
    round_keys = []
    nl_log = []
    for r in range(nb_rounds):
        rk = []
        for row in range(4):
            for col in range(NB_COLS):
                rk.append(state[row][col])
        round_keys.append(rk)
        if r < nb_rounds - 1:
            state = key_update_80(state, r, nl_log)
    return round_keys, nl_log
NB_ROWS_128 = 4
NB_COLS_128 = 32
def init_key_state_128():
    state = [[None] * NB_COLS_128 for _ in range(NB_ROWS_128)]
    for row in range(NB_ROWS_128):
        for col in range(NB_COLS_128):
            mk_idx = row * NB_COLS_128 + col
            state[row][col] = frozenset({mk_idx})
    return state
def key_update_128(state, round_idx, nl_log):
    new_state = [[None] * NB_COLS_128 for _ in range(NB_ROWS_128)]
    for col in range(8):
        in_bits = [state[r][col] for r in range(4)]
        for row in range(4):
            nl_marker = ('NL_KS128', round_idx, row, col)
            new_state[row][col] = frozenset({nl_marker})
            nl_log.append({
                'round': round_idx,
                'position': (row, col),
                'inputs': in_bits,
                'marker': nl_marker,
            })
    for col in range(8, NB_COLS_128):
        for row in range(4):
            new_state[row][col] = state[row][col]
    shifts_128 = [8, 16, 24, 0]
    final_state = [[None] * NB_COLS_128 for _ in range(NB_ROWS_128)]
    for row in range(4):
        sh = shifts_128[row]
        for col in range(NB_COLS_128):
            src_col = (col - sh) % NB_COLS_128
            final_state[row][col] = new_state[row][src_col]
    return final_state
def build_round_keys_128(nb_rounds):
    state = init_key_state_128()
    round_keys = []
    nl_log = []
    for r in range(nb_rounds):
        rk = []
        for row in range(4):
            for col in range(16):  
                rk.append(state[row][col])
        round_keys.append(rk)
        if r < nb_rounds - 1:
            state = key_update_128(state, r, nl_log)
    return round_keys, nl_log
def extract_master_key_conditions(mask_trail, round_keys, nb_rounds):
    accumulated = frozenset()
    for k in range(nb_rounds):
        for j in range(64):
            if mask_trail[k][0][j] == 1:  
                accumulated = accumulated ^ round_keys[k][j]
    keys_list = []
    for sym in sorted(accumulated, key=lambda x: (str(type(x)), x if isinstance(x, int) else str(x))):
        if isinstance(sym, int):
            keys_list.append(f'k_{sym}')
        else:
            keys_list.append(f'NL_{sym[1]}_{sym[2]}_{sym[3]}')
    return keys_list