"""
constraint_collector.py
=======================
顶层驱动脚本。根据 utils.ADV_MODEL 自动选择 MASK_DIVIDER 的实现：

    ADV_MODEL == "SK"   ->  MASK_DIVIDER_SK
    ADV_MODEL == "TK1"  ->  MASK_DIVIDER_TK1
    ADV_MODEL == "TK2"  ->  MASK_DIVIDER_TK2

工作流程：
    1) 跑 MILP 搜 quasi-differential trails，保存频率 mask (.npy)
       —— 若 mask 已存在则跳过。
    2) 加载 freq mask；从差分轨迹派生 active_bit_dic。
    3) 把 freq mask 合并进 masked_bit_dic。
    4) 调对应 MASK_DIVIDER 的 generate_constraints，输出 cluster 约束。
    5) 收集每个 cluster 的 mask，落盘为 FULL_MSK.npy。
"""
import os
import re
import importlib
import numpy as np

from utils import (
    NB_ROUNDS, SBOX_SIZE, MIN_CORR, THRESH, ADV_MODEL,
    extract_diff_trail_flat,
)
from SKINNYMILP_msk import SKINNY_MILP_Quasi_Diff


# --------- 从约束串里抽 mask 变量 ------------------------------------------
# 匹配形如 x_数字_数字 / y_数字_数字 的变量名。
_VAR_RE = re.compile(r'\b[xy]_\d+_\d+')


def _extract_mask_vars(cons_text):
    """返回 cons_text 中出现的所有 x_*_* / y_*_* 变量，去重并保持首次出现顺序。"""
    seen = dict.fromkeys(_VAR_RE.findall(cons_text))
    return list(seen)



# --------- 按 ADV_MODEL 选 MASK_DIVIDER 模块 -------------------------------
_MASK_DIVIDER_BY_MODE = {
    "SK":  "MASK_DIVIDER_TK1",
    "TK1": "MASK_DIVIDER_TK1",
    "TK2": "MASK_DIVIDER_TK1",
}


def _load_mask_divider():
    if ADV_MODEL not in _MASK_DIVIDER_BY_MODE:
        raise ValueError(
            f"不支持的 ADV_MODEL={ADV_MODEL}。"
            f"已实现的模式: {sorted(_MASK_DIVIDER_BY_MODE)}"
        )
    name = _MASK_DIVIDER_BY_MODE[ADV_MODEL]
    print(f"[loader] ADV_MODEL={ADV_MODEL}  ->  import {name}")
    return importlib.import_module(name)


# --------- 主流程 ----------------------------------a------------------------
def run_milp_if_needed():
    mask_path = f'./freq_msk/masks_freq_{NB_ROUNDS}RD_CORR{MIN_CORR}_T{THRESH}.npy'
    if os.path.exists(mask_path):
        print(f"[skip MILP] mask 已存在: {mask_path}")
    else:
        print(f"[run MILP] mask 不存在，启动 Gurobi 搜索 ...")
        SKINNY_MILP_Quasi_Diff(NB_ROUNDS)
    return mask_path


def main():
    ROUNDS = NB_ROUNDS

    # ---- 0) 选 MASK_DIVIDER -------------------------------------------
    MD = _load_mask_divider()

    # ---- 1) MILP（必要时） --------------------------------------------
    mask_path = run_milp_if_needed()

    # ---- 2) 加载 freq mask ---------------------------------------------
    data = np.load(mask_path)
    expected = (ROUNDS, 2, MD.HALF_STATE_BITS)
    assert data.shape == expected, (
        f"freq mask 形状 {data.shape} 与 SKINNY-{16*SBOX_SIZE} / {ROUNDS} 轮 "
        f"预期 {expected} 不一致。检查 utils.SBOX_SIZE / NB_ROUNDS。"
    )
    data = data.tolist()
    print(f"[load] {mask_path}  shape={expected}")

    # ---- 3) 差分轨迹 + active_bit_dic ----------------------------------
    diff_file = (
        f"../data/differential_trails/SKINNY{16 * SBOX_SIZE}_{ADV_MODEL}_R{ROUNDS}.txt"
    )
    diff_trail = extract_diff_trail_flat(diff_file, ROUNDS)
    dic_x, dic_y = MD.creat_dic_GIFT(diff_trail)
    active_bit_dic = MD.get_active_bit(dic_x, dic_y)
    print(f"[derive] #active bits = {len(active_bit_dic)}")

    # ---- 4) 合并 freq mask 到 masked_bit_dic ---------------------------
    masked_bit_dic = active_bit_dic.copy()
    for r in range(ROUNDS):
        for s in range(2):
            for i in range(MD.HALF_STATE_BITS):
                if data[r][s][i] == 1:
                    g = r * MD.FULL_STATE_BITS + s * MD.HALF_STATE_BITS + i
                    masked_bit_dic[str(g)] = 1
    print(f"[merge] #masked bits (含 active) = {len(masked_bit_dic)}")

    # ---- 5) 构造线性矩阵 + 抽 cluster ----------------------------------
    L = MD.Global_mat_bit(ROUNDS)
    L_mat = L.copy()
    cons_str, Z_lst = MD.generate_constraints(
        L_mat, dic_x, active_bit_dic, masked_bit_dic, ROUNDS, L
    )

    # ---- 6) 输出 -------------------------------------------------------
    out_file = (
        f"./constraints/CONS_{ADV_MODEL}_{ROUNDS}R_{MIN_CORR}_TH{THRESH}.txt"
    )
    msk_file = (
        f"./constraints/CONS_{ADV_MODEL}_{ROUNDS}R_{MIN_CORR}_TH{THRESH}_FULL_MSK.npy"
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    FULL_MSK = []
    print(f"\n#========= 最终分离的独立 Cluster ({ADV_MODEL}) =========")
    cons_lst_str = "dic_cons={\n \n"
    with open(out_file, 'w', encoding='utf-8') as fh:
        for i, (cstr, Z) in enumerate(zip(cons_str, Z_lst)):
            block = f'CONS{i}="""\n{cstr}"""\nZ{i}={Z}\n\n'
            fh.write(block)
            print(f"\n#[ Cluster {i} ]\n{block}")
            msk = _extract_mask_vars(cstr)
            print(f"MASK{i}= {msk}")
            FULL_MSK.append(msk)
            cons_lst_str += f"'CONS{i}': (CONS{i},Z{i}),\n"
        cons_lst_str += "}"
        fh.write(cons_lst_str)

    # ---- 7) 落盘 FULL_MSK.npy ------------------------------------------
    # FULL_MSK[i] 是 cluster i 中出现的 x_*_* / y_*_* 变量名列表；
    # 各 cluster 长度不一（ragged），用 object dtype 保存。
    # 读取：np.load(msk_file, allow_pickle=True)
    np.save(msk_file, np.array(FULL_MSK, dtype=object), allow_pickle=True)
    print(FULL_MSK)
    print(f"\nConstraints written: {out_file}")
    print(f"FULL_MSK written:    {msk_file}  (#clusters={len(FULL_MSK)})")
    print(f"#clusters = {len(cons_str)}")


if __name__ == "__main__":
    main()