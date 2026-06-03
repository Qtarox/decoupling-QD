import re





RECTANGLE_SBOX = [0x6, 0x5, 0xC, 0xA, 0x1, 0xE, 0x7, 0x9,
                  0xB, 0x0, 0x3, 0xD, 0x8, 0xF, 0x4, 0x2]

RECTANGLE_RC = [0x01, 0x02, 0x04, 0x09, 0x12, 0x05, 0x0B, 0x16,
                0x0C, 0x19, 0x13, 0x07, 0x0F, 0x1F, 0x1E, 0x1C,
                0x18, 0x11, 0x03, 0x06, 0x0D, 0x1B, 0x17, 0x0E, 0x1D]






_NL_COUNTER = [0]
def _new_nl_tag(r, row, col):
    _NL_COUNTER[0] += 1
    return (r, row, col, _NL_COUNTER[0])


def _make_mk(i):
    return {'mk': frozenset({i}), 'nl': frozenset(), 'const': 0}


def _make_const(c):
    return {'mk': frozenset(), 'nl': frozenset(), 'const': c & 1}


def _make_nl(tag):
    return {'mk': frozenset(), 'nl': frozenset({tag}), 'const': 0}


def _xor(a, b):
    return {
        'mk': a['mk'] ^ b['mk'],
        'nl': a['nl'] ^ b['nl'],
        'const': a['const'] ^ b['const'],
    }


def _format_symbol(sym):
    parts = []
    for i in sorted(sym['mk']):
        parts.append(f'mk_{i}')
    for tag in sorted(sym['nl'], key=lambda t: (t[0], t[1], t[2], t[3])):
        r, row, col, ctr = tag
        parts.append(f'nl_{r}_{row}_{col}_{ctr}')
    if sym['const']:
        parts.append('1')
    if not parts:
        return '0'
    return ' + '.join(parts)






def symbolic_key_schedule_80(nb_rounds):
    
    K = [[_make_mk(row * 16 + col) for col in range(16)] for row in range(5)]
    round_keys = []
    
    for r in range(nb_rounds):
        
        rk = [None] * 64
        for col in range(16):
            for row in range(4):
                rk[4 * col + row] = K[row][col]
        round_keys.append(rk)
        
        if r >= nb_rounds - 1:
            break
        
        
        new_K = [row[:] for row in K]
        for col in range(4):
            for row in range(4):
                tag = _new_nl_tag(r, row, col)
                new_K[row][col] = _make_nl(tag)
        
        
        shifts = [8, 0, 0, 0, 0]
        rotated = [[None]*16 for _ in range(5)]
        for row in range(5):
            sh = shifts[row]
            for col in range(16):
                rotated[row][col] = new_K[row][(col - sh) % 16]
        
        
        K0_old = new_K[0][:]
        K = [[None]*16 for _ in range(5)]
        for col in range(16):
            K[0][col] = _xor(rotated[0][col], new_K[1][col])
            K[1][col] = new_K[2][col]
            K[2][col] = new_K[3][col]
            K[3][col] = new_K[4][col]
            K[4][col] = _xor(rotated[4][col], K0_old[col])
        
        
        rc = RECTANGLE_RC[r]
        for i in range(5):
            bit = (rc >> i) & 1
            if bit:
                K[0][i] = _xor(K[0][i], _make_const(1))
    
    return round_keys






def symbolic_key_schedule_128(nb_rounds):
    K = [[_make_mk(row * 32 + col) for col in range(32)] for row in range(4)]
    round_keys = []
    
    for r in range(nb_rounds):
        rk = [None] * 64
        for col in range(16):
            for row in range(4):
                rk[4 * col + row] = K[row][col]
        round_keys.append(rk)
        
        if r >= nb_rounds - 1:
            break
        
        
        new_K = [row[:] for row in K]
        for col in range(8):
            for row in range(4):
                tag = _new_nl_tag(r, row, col)
                new_K[row][col] = _make_nl(tag)
        
        
        shifts = [8, 16, 24, 0]
        K = [[None]*32 for _ in range(4)]
        for row in range(4):
            sh = shifts[row]
            for col in range(32):
                K[row][col] = new_K[row][(col - sh) % 32]
        
        
        rc = RECTANGLE_RC[r]
        for i in range(5):
            if (rc >> i) & 1:
                K[0][i] = _xor(K[0][i], _make_const(1))
    
    return round_keys






def expand_constraints(cons_text, round_keys_sym):
    lines_out = []
    
    for line in cons_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        
        
        sbox_match = re.match(r'S(?:_\[[\d_]+\])?\(.*?\)\s*=\s*\(.*?\)', line)
        if sbox_match:
            lines_out.append(line)
            continue
        
        
        
        rhs = 0
        line_body = re.sub(r'[\[\]]', '', line)
        m_rhs = re.search(r'=\s*(\d+)\s*$', line_body)
        if m_rhs:
            rhs = int(m_rhs.group(1)) & 1
            line_body = line_body[:m_rhs.start()].strip()
        
        
        tokens = re.findall(r'[xy]_\d+_\d+|k_\d+_\d+', line_body)
        
        
        state_terms = []
        accumulated = {'mk': frozenset(), 'nl': frozenset(), 'const': 0}
        
        for tok in tokens:
            if tok.startswith('k_'):
                parts = tok.split('_')
                r_k, j_k = int(parts[1]), int(parts[2])
                if r_k < len(round_keys_sym) and j_k < len(round_keys_sym[r_k]):
                    sym = round_keys_sym[r_k][j_k]
                    accumulated = _xor(accumulated, sym)
                else:
                    
                    state_terms.append(tok)
            else:
                state_terms.append(tok)
        
        
        rhs ^= accumulated['const']
        
        
        left_parts = list(state_terms)
        for i in sorted(accumulated['mk']):
            left_parts.append(f'mk_{i}')
        for tag in sorted(accumulated['nl'], key=lambda t: (t[0], t[1], t[2], t[3])):
            r_tag, row, col, ctr = tag
            left_parts.append(f'nl_{r_tag}_{row}_{col}_{ctr}')
        
        if not left_parts:
            pass
        else:
            lines_out.append(' + '.join(left_parts) + f' = {rhs}')
    
    return '\n'.join(lines_out)






if __name__ == "__main__":
    
    CONS1 = """
x_8_7 + k_8_7 = 0
y_8_4 + k_9_4 = 1
y_8_5 + x_9_9 + k_9_9 = 0
x_9_11 + k_9_11 = 0
y_9_8 + k_10_8 = 1
y_9_9 + k_10_13 = 0
S(x_9_8,x_9_9,x_9_10,x_9_11) = (y_9_8,y_9_9,y_9_10,y_9_11)
S(x_8_4,x_8_5,x_8_6,x_8_7) = (y_8_4,y_8_5,y_8_6,y_8_7)
"""
    
    
    KEY_SIZE = 80
    NB_ROUNDS = 14
    
    print(f"=== 生成 RECTANGLE-{KEY_SIZE} 的符号密钥扩展（{NB_ROUNDS} 轮）===\n")
    
    if KEY_SIZE == 80:
        rk_sym = symbolic_key_schedule_80(NB_ROUNDS)
    else:
        rk_sym = symbolic_key_schedule_128(NB_ROUNDS)
    
    
    for (r, j) in [(0, 0), (1, 0), (5, 55), (8, 7), (10, 13)]:
        if r < len(rk_sym) and j < len(rk_sym[r]):
            print(f"  k_{r}_{j} = {_format_symbol(rk_sym[r][j])}")
    print()
    
    print("origin cons")
    print(CONS1.strip())
    print("\n expanded")
    expanded = expand_constraints(CONS1, rk_sym)
    print(expanded)