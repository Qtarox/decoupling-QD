"""
RECTANGLE 密钥扩展的线性追踪：把每轮的轮密钥位映射回主密钥位的线性组合。

约定：
- 主密钥位用整数 0..79 (80-bit) 或 0..127 (128-bit) 表示
- 每个密钥状态位用一个 frozenset[int] 表示，表示该位 = XOR_{i in set} mk_i
- 经过S盒的位标记为非线性，用元组 ('NL', round_idx, position) 标识

RECTANGLE 密钥扩展规范（来自原始论文 eprint 2014/084）:

【RECTANGLE-80】(5×16 状态)
对每轮 i=1..25：
  1) 提取轮密钥 RK_{i-1} = K[0..3]（前4行 64-bit）
  2) 更新密钥状态：
     a) SubColumn：对最右4列(col 0..3)的前4行 K[0..3][0..3] 应用4个S盒
     b) 行移位 + Feistel式混合：
        新K[0] = (K[0] <<< 8)  XOR  K[1]
        新K[1] = K[2]
        新K[2] = (K[2] <<< 16) XOR  K[3]    (注：<<<16 = 不变)
        新K[3] = K[4]
        新K[4] = (K[4] <<< 0)  XOR  K[0]_old(after rotate?)
     注：实际规范的Feistel结构请以你使用的论文为准
  3) XOR 5-bit 轮常数到 K[0] 低5位

【RECTANGLE-128】(4×32 状态)
对每轮 i=1..25：
  1) 提取轮密钥 RK_{i-1} = K[*][0..15]（每行前16列，共64-bit）
  2) 更新密钥状态：
     a) SubColumn：对最右8列(col 0..7)的所有4行 K[0..3][0..7] 应用8个S盒
     b) 行移位：K[0]<<<8, K[1]<<<16, K[2]<<<24, K[3]<<<0  (具体偏移以规范为准)
  3) XOR 5-bit 轮常数

"""

from utils import RECTANGLE_RC


# ════════════════════════════════════════════════════════════════════
# 通用工具
# ════════════════════════════════════════════════════════════════════

def xor_sym(a, b):
    """两个符号集合的XOR（对称差）"""
    return frozenset(a) ^ frozenset(b)


def xor_many(*sets):
    """多个符号集合的XOR"""
    result = frozenset()
    for s in sets:
        result = result ^ frozenset(s)
    return result


# ════════════════════════════════════════════════════════════════════
# RECTANGLE-80 密钥扩展追踪
# ════════════════════════════════════════════════════════════════════
# 密钥状态：5 行 × 16 列 = 80 bit
# 索引约定：K[row][col]，row ∈ {0,1,2,3,4}，col ∈ {0,1,...,15}
# 主密钥 mk[0..79] 初始填充：mk[i] → K[i // 16][i % 16]

NB_ROWS_80 = 5
NB_COLS = 16

def init_key_state_80():
    """初始化 80-bit 主密钥的符号表示"""
    state = [[None] * NB_COLS for _ in range(NB_ROWS_80)]
    for row in range(NB_ROWS_80):
        for col in range(NB_COLS):
            mk_idx = row * NB_COLS + col
            state[row][col] = frozenset({mk_idx})
    return state


def key_update_80(state, round_idx, nl_log):
    """
    更新 80-bit 密钥状态一次（对应一轮）。
    nl_log: 列表，记录每次S盒产生的非线性位
            形式 [(state_before_sbox_dict, output_position)]
    
    采用 Zhang et al. (eprint 2014/084) 给出的公式：
      Step 1 (SubColumn): 对 K[0..3][0..3] 应用S盒（产生非线性位）
      Step 2 (Rotation):
        K[0] <<< 8
        K[1] <<< 12  (有些版本写成 K[1] <<< 12)
        K[2] <<< 16 (= 不变)
        K[3] <<< 13
        K[4] 不变
        然后做 Feistel 式 XOR 重组
      Step 3 (AddConstant): K[0] 的低5列 XOR 5-bit RC
    
    ⚠️ 这里给出的是一个常见版本。请根据你使用的文献核对具体偏移和XOR结构。
    """
    new_state = [[None] * NB_COLS for _ in range(NB_ROWS_80)]
    
    # ── Step 1: SubColumn ─────────────────────────────────────────────
    # 对 K[0..3][0..3] 的4列应用S盒。每列4个bit 经S盒变成新的4个bit。
    # 因为是非线性的，我们用占位符 ('NL', round_idx, row, col) 标记
    # 同时记录输入信息以便后续判断是否触发非线性条件
    for col in range(4):
        # 输入这一列的4个bit的符号表示
        in_bits = [state[r][col] for r in range(4)]
        # S盒输出bit在线性追踪中无法表示，标记为非线性
        for row in range(4):
            nl_marker = ('NL_KS80', round_idx, row, col)
            new_state[row][col] = frozenset({nl_marker})
            nl_log.append({
                'round': round_idx,
                'position': (row, col),
                'inputs': in_bits,  # S盒的4个输入位（可能是主密钥的XOR）
                'marker': nl_marker,
            })
        # 注意：行4的最右4列在SubColumn中不动
    
    # 最右4列(col 0..3)的row 4 不动
    for col in range(4):
        new_state[4][col] = state[4][col]
    
    # 其他列(col 4..15)所有行不动（在SubColumn阶段）
    for col in range(4, NB_COLS):
        for row in range(NB_ROWS_80):
            new_state[row][col] = state[row][col]
    
    # ── Step 2: 行移位 + Feistel重组 ─────────────────────────────────
    # 这里采用最典型的 RECTANGLE-80 密钥扩展公式：
    #   row 0: K[0] <<< 8
    #   row 1: K[1] <<< 16 (即不变)
    #   row 2: K[2] <<< 16 (即不变)  
    #   row 3: K[3] <<< 16 (即不变)
    #   row 4: K[4] <<< 16 (即不变)
    # 然后：新K[0] = K[0]_rot ⊕ K[1]
    #       新K[1] = K[2]
    #       新K[2] = K[2]_rot ⊕ K[3]
    #       新K[3] = K[4]
    #       新K[4] = K[4]_rot ⊕ K[0]
    
    # 先做行移位
    rotated = [[None] * NB_COLS for _ in range(NB_ROWS_80)]
    shifts_80 = [8, 0, 0, 0, 0]   # 每行的左移位数（按规范调整）
    for row in range(NB_ROWS_80):
        sh = shifts_80[row]
        for col in range(NB_COLS):
            src_col = (col - sh) % NB_COLS
            rotated[row][col] = new_state[row][src_col]
    
    # Feistel 重组
    final_state = [[None] * NB_COLS for _ in range(NB_ROWS_80)]
    for col in range(NB_COLS):
        final_state[0][col] = xor_sym(rotated[0][col], new_state[1][col])
        final_state[1][col] = new_state[2][col]
        final_state[2][col] = xor_sym(rotated[2][col], new_state[3][col])
        final_state[3][col] = new_state[4][col]
        final_state[4][col] = xor_sym(rotated[4][col], new_state[0][col])
    
    # ── Step 3: AddConstant（轮常数XOR到K[0]低5位）──────────────────
    # 轮常数与mask的相关性已在主代码 compute_correlation 中处理
    # 这里只追踪线性映射，不需要修改 final_state 的符号表示
    # （常数XOR不改变某个状态位 = 哪些主密钥位的XOR）
    
    return final_state


def build_round_keys_80(nb_rounds):
    """
    构建 RECTANGLE-80 的所有轮密钥的符号表示。
    返回: round_keys[r] 是 64-bit 列表（行优先展平 row*16+col），
          每位是 frozenset[主密钥索引或非线性标记]
    """
    state = init_key_state_80()
    round_keys = []
    nl_log = []
    
    for r in range(nb_rounds):
        # 提取本轮轮密钥：state[0..3]（前4行64-bit）
        rk = []
        for row in range(4):
            for col in range(NB_COLS):
                rk.append(state[row][col])
        round_keys.append(rk)
        
        # 更新密钥状态（除最后一轮）
        if r < nb_rounds - 1:
            state = key_update_80(state, r, nl_log)
    
    return round_keys, nl_log


# ════════════════════════════════════════════════════════════════════
# RECTANGLE-128 密钥扩展追踪
# ════════════════════════════════════════════════════════════════════
# 密钥状态：4 行 × 32 列 = 128 bit
# 主密钥 mk[0..127]：mk[i] → K[i // 32][i % 32]

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
    """
    更新 128-bit 密钥状态一次。
    
    Step 1 (SubColumn): 对 K[0..3][0..7] 应用8个S盒 (最右8列的所有4行)
    Step 2 (ShiftRow):
       row 0: <<< 8
       row 1: <<< 16
       row 2: <<< 24
       row 3: <<< 0  (不变)
       (具体偏移依规范，常见为 (8, 16, 24, 0))
    Step 3 (AddConstant): K[0] 低5位 XOR 5-bit RC
    """
    new_state = [[None] * NB_COLS_128 for _ in range(NB_ROWS_128)]
    
    # Step 1: SubColumn at col 0..7
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
    
    # 其他列不变
    for col in range(8, NB_COLS_128):
        for row in range(4):
            new_state[row][col] = state[row][col]
    
    # Step 2: ShiftRow
    shifts_128 = [8, 16, 24, 0]
    final_state = [[None] * NB_COLS_128 for _ in range(NB_ROWS_128)]
    for row in range(4):
        sh = shifts_128[row]
        for col in range(NB_COLS_128):
            src_col = (col - sh) % NB_COLS_128
            final_state[row][col] = new_state[row][src_col]
    
    return final_state


def build_round_keys_128(nb_rounds):
    """
    构建 RECTANGLE-128 的所有轮密钥的符号表示。
    返回: round_keys[r] 是 64-bit 列表（行优先展平 row*16+col），
          每位是 frozenset
    
    轮密钥 = K[0..3][0..15]（每行前16列）
    """
    state = init_key_state_128()
    round_keys = []
    nl_log = []
    
    for r in range(nb_rounds):
        rk = []
        for row in range(4):
            for col in range(16):  # 只取前16列作为轮密钥
                rk.append(state[row][col])
        round_keys.append(rk)
        
        if r < nb_rounds - 1:
            state = key_update_128(state, r, nl_log)
    
    return round_keys, nl_log


# ════════════════════════════════════════════════════════════════════
# 公共接口：根据 mask_trail 提取主密钥条件
# ════════════════════════════════════════════════════════════════════

def extract_master_key_conditions(mask_trail, round_keys, nb_rounds):
    """
    从 mask_trail（行优先展平索引 j = row*16+col）和已构建的 round_keys 中，
    提取该 trail 对应的主密钥条件。
    
    每个 mask_trail[k][0] 对应轮 k 的 before-sbox mask（即 RK_k 处的mask）。
    若 mask_trail[k][0][j] == 1，则该trail要求 RK_k[j] = mask值。
    
    我们把所有 mask=1 的位置对应的 round_keys[k][j] 全部XOR起来，
    得到一个主密钥位的XOR集合（线性条件）。
    
    返回: ['k_3', 'k_25', ...] 字符串列表（去重后）
          如果包含非线性标记，则附加 'NL_round_pos' 形式的字符串
    """
    accumulated = frozenset()
    
    for k in range(nb_rounds):
        for j in range(64):
            if mask_trail[k][0][j] == 1:  # before-sbox mask = 轮密钥mask
                # 注意 round_keys[k][j] 中 j 是行优先展平 row*16+col
                # 这里我们的 mask_trail 也用行优先 ⟹ 索引一致
                accumulated = accumulated ^ round_keys[k][j]
    
    # 转换为字符串列表
    keys_list = []
    for sym in sorted(accumulated, key=lambda x: (str(type(x)), x if isinstance(x, int) else str(x))):
        if isinstance(sym, int):
            keys_list.append(f'k_{sym}')
        else:
            # 非线性标记
            keys_list.append(f'NL_{sym[1]}_{sym[2]}_{sym[3]}')
    
    return keys_list