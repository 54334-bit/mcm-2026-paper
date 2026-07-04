# -*- coding: utf-8 -*-
"""
Task 4: DASS 动态自适应评分系统 (Dynamic Adaptive Scoring System)
-----------------------------------------------------------------
设计一个新的公平评分系统，融合:
  1. 熵权法 (Entropy Weight Method): 自适应调整评委分和粉丝票的权重
  2. TOPSIS: 多准则决策，向量化归一消除量纲差异
  3. 量化评委救人机制: 基于争议指数 ΔR 触发

核心优势:
  - 当评委意见分歧大时，自动增加评委权重（专业判断更重要）
  - 当评委意见一致时，自动增加粉丝权重（观众参与感更重要）
  - 争议阈值触发机制，防止极端不公平结果
"""

import numpy as np
from collections import defaultdict

from config import RESULT_DIR, CONTROVERSY_THRESHOLD, RANDOM_SEED
from data_loader import save_json

np.random.seed(RANDOM_SEED)


# DASS 可调参数（用于参数扫描）
# 经_diag_dass_scan.py三轮参数扫描（30+配置）得到的最优配置：
#   "无boost+高jw" 方案 → SPI≈92.4%（达到≥92%目标）, FII≈43.2%, save_rate≈16.3%
# 核心权衡: K1=11.5使粉丝票更集中→熵权法偏向粉丝→需提高评委权重下限以保护技能
DASS_PARAMS = {
    'log_smooth_factor': 1.0,       # 对数平滑因子，越大对粉丝票的压缩越强
    'entropy_floor': 0.0,           # 熵值下限，防止权重极端化
    'min_judge_weight': 0.80,       # 评委权重下限（提高以增强技能保护，匹配SPI≥92%目标）
    'max_judge_weight': 0.90,       # 评委权重上限
    'fan_boost_for_low_scores': 0.0,  # 低评委分选手的粉丝票补偿（关闭以提升SPI）
    'use_entropy': True,            # 是否使用熵权法；False则使用固定评委权重
    'fixed_judge_weight': 0.85,     # 固定评委权重（当use_entropy=False时使用）
    'save_judge_weight': 0.98,      # 救人决策中评委分权重（接近1=主要看评委分）
}


def entropy_weight_method(judge_scores, fan_votes, params=None):
    """
    熵权法计算自适应权重
    
    原理:
      当评委分差异大（熵值低）→ 评委权重高（专业判断更重要）
      当评委分差异小（熵值高）→ 粉丝权重高（观众参与感更重要）
    
    步骤:
      1. 计算每个指标的比例 p_ij = x_ij / Σx_ij
      2. 计算熵值 E_j = -1/ln(n) * Σp_ij * ln(p_ij)
      3. 计算权重 ω_j = (1 - E_j) / Σ(1 - E_j)
    
    参数:
        judge_scores: dict {name: score}
        fan_votes: dict {name: vote_share}
        params: 可选参数配置
    
    返回:
        tuple: (judge_weight, fan_weight)
    """
    if params is None:
        params = DASS_PARAMS
    
    names = list(judge_scores.keys())
    n = len(names)
    
    if n <= 1:
        return 0.5, 0.5
    
    # 构建决策矩阵
    j_scores = np.array([judge_scores[n] for n in names], dtype=float)
    f_votes = np.array([fan_votes.get(n, 0) for n in names], dtype=float)
    
    # 归一化（避免零值）
    epsilon = 1e-10
    j_scores = j_scores + epsilon
    f_votes = f_votes + epsilon
    
    # 计算比例
    p_j = j_scores / j_scores.sum()
    p_f = f_votes / f_votes.sum()
    
    # 计算熵值，加入下限防止极端权重
    def calc_entropy(p):
        # E_j = -1/ln(n) * Σ p * ln(p)
        entropy = -np.sum(p * np.log(p)) / np.log(n)
        return max(entropy, params.get('entropy_floor', 0.05))
    
    E_j = calc_entropy(p_j)
    E_f = calc_entropy(p_f)
    
    # 计算权重
    d_j = 1 - E_j  # 评委分差异度
    d_f = 1 - E_f  # 粉丝票差异度
    
    total_d = d_j + d_f
    if total_d == 0:
        return 0.5, 0.5
    
    w_j = d_j / total_d
    w_f = d_f / total_d
    
    # 限制权重范围，避免完全偏向某一方
    min_wj = params.get('min_judge_weight', 0.15)
    max_wj = params.get('max_judge_weight', 0.85)
    w_j = np.clip(w_j, min_wj, max_wj)
    w_f = 1 - w_j
    
    return float(w_j), float(w_f)


def get_dass_weights(judge_scores, fan_votes, params=None):
    """
    获取DASS权重（熵权法或固定权重）
    
    返回:
        tuple: (judge_weight, fan_weight)
    """
    if params is None:
        params = DASS_PARAMS
    
    if params.get('use_entropy', True):
        return entropy_weight_method(judge_scores, fan_votes, params)
    else:
        w_j = params.get('fixed_judge_weight', 0.85)
        w_j = np.clip(w_j, params.get('min_judge_weight', 0.15), params.get('max_judge_weight', 0.85))
        w_f = 1 - w_j
        return float(w_j), float(w_f)


def topsis_scoring(judge_scores, fan_votes, params=None):
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    
    步骤:
      1. 数据标准化（向量归一化）
      2. 熵权法确定权重
      3. 构建加权标准化矩阵
      4. 确定正理想解和负理想解
      5. 计算各方案到理想解的距离
      6. 计算贴近度 C_i
    
    参数:
        judge_scores: dict {name: score}
        fan_votes: dict {name: vote_share}
        params: 可选参数配置
    
    返回:
        dict: {name: closeness_score} (0-1, 越大越好)
    """
    if params is None:
        params = DASS_PARAMS
    
    names = list(judge_scores.keys())
    n = len(names)
    
    if n <= 1:
        return {names[0]: 1.0} if names else {}
    
    # 构建原始矩阵 X (n×2)
    j_scores = np.array([judge_scores[n] for n in names], dtype=float)
    f_votes = np.array([fan_votes.get(n, 0) for n in names], dtype=float)
    
    # 对粉丝票做对数平滑，防止超级明星的票数碾压
    factor = params.get('log_smooth_factor', 1.0)
    f_votes_smoothed = np.log1p(f_votes * 1000 * factor)
    
    # 对低评委分选手的粉丝票做小幅补偿（增强逆袭可能性）
    fan_boost = params.get('fan_boost_for_low_scores', 0.05)
    if fan_boost > 0:
        j_mean = j_scores.mean()
        for i in range(n):
            if j_scores[i] < j_mean:
                f_votes_smoothed[i] *= (1 + fan_boost)
    
    X = np.column_stack([j_scores, f_votes_smoothed])
    
    # 步骤1: 向量归一化 z_ij = x_ij / sqrt(Σx_ij²)
    Z = X / np.sqrt((X ** 2).sum(axis=0))
    
    # 步骤2: 确定权重（熵权法或固定权重）
    w_j, w_f = get_dass_weights(judge_scores, fan_votes, params)
    weights = np.array([w_j, w_f])
    
    # 步骤3: 加权标准化矩阵 v_ij = ω_j * z_ij
    V = Z * weights
    
    # 步骤4: 确定正负理想解
    V_plus = V.max(axis=0)   # 正理想解（最大值）
    V_minus = V.min(axis=0)  # 负理想解（最小值）
    
    # 步骤5: 计算欧氏距离
    D_plus = np.sqrt(((V - V_plus) ** 2).sum(axis=1))
    D_minus = np.sqrt(((V - V_minus) ** 2).sum(axis=1))
    
    # 步骤6: 计算贴近度 C_i = D_i- / (D_i- + D_i+)
    C = D_minus / (D_minus + D_plus + 1e-10)
    
    return {name: float(C[i]) for i, name in enumerate(names)}


def compute_controversy_index(judge_scores, fan_votes):
    """
    计算争议指数 ΔR
    
    ΔR = |Rank_Judge - Rank_Fan|
    
    当 ΔR >= CONTROVERSY_THRESHOLD (3) 时，触发争议事件
    
    参数:
        judge_scores: dict {name: score}
        fan_votes: dict {name: vote_share}
    
    返回:
        dict: {name: delta_R}
    """
    names = list(judge_scores.keys())
    
    # 计算排名
    j_scores = np.array([judge_scores[n] for n in names])
    f_votes = np.array([fan_votes.get(n, 0) for n in names])
    
    j_ranks = len(names) - np.argsort(np.argsort(j_scores))
    f_ranks = len(names) - np.argsort(np.argsort(f_votes))
    
    delta_R = {}
    for i, name in enumerate(names):
        delta_R[name] = abs(int(j_ranks[i]) - int(f_ranks[i]))
    
    return delta_R


def judges_save_mechanism(judge_scores, fan_votes, closeness_scores, params=None):
    """
    量化评委救人机制
    
    规则:
      1. 找到TOPSIS得分最低的两名选手（即"最终得分排名"倒数两名）
      2. 计算他们的争议指数 ΔR = Rank_Judge - Rank_Fan
         （论文定义: ΔR = Rank_Judge - Rank_Fan，非绝对值）
      3. 如果 max(ΔR1, ΔR2) >= 3，触发争议事件
         （意味着至少有一人粉丝排名远高于评委排名，即"人气远大于实力"）
      4. 触发后: 用 save_judge_weight 加权评委分和粉丝票，保留综合分高的选手
      5. 否则，正常淘汰TOPSIS得分最低的选手
    
    参数:
        judge_scores: dict {name: score}
        fan_votes: dict {name: vote_share}
        closeness_scores: dict {name: TOPSIS score}
        params: 可选参数配置
    
    返回:
        str: 被淘汰的选手姓名
        bool: 是否触发了评委救人
        dict: 详细信息
    """
    if params is None:
        params = DASS_PARAMS
    
    names = list(closeness_scores.keys())
    
    if len(names) <= 1:
        return names[0] if names else None, False, {}
    
    # 按TOPSIS得分排序（升序，找最低分=最差）
    sorted_names = sorted(closeness_scores, key=closeness_scores.get)
    bottom2 = sorted_names[:2]
    
    # 计算评委排名和粉丝排名
    # 排名1=最高分/最高票，使用降序排名
    j_scores = np.array([judge_scores[n] for n in names])
    f_votes = np.array([fan_votes.get(n, 0) for n in names])
    
    j_ranks = len(names) - np.argsort(np.argsort(j_scores))  # 1=最高分
    f_ranks = len(names) - np.argsort(np.argsort(f_votes))   # 1=最高票
    
    j_ranks_dict = {names[i]: int(j_ranks[i]) for i in range(len(names))}
    f_ranks_dict = {names[i]: int(f_ranks[i]) for i in range(len(names))}
    
    # 论文定义: ΔR = Rank_Judge - Rank_Fan（非绝对值）
    # 当 ΔR > 0 且大时，表示粉丝排名远优于评委排名（人气>实力）
    delta_R1 = j_ranks_dict.get(bottom2[0], 0) - f_ranks_dict.get(bottom2[0], 0)
    delta_R2 = j_ranks_dict.get(bottom2[1], 0) - f_ranks_dict.get(bottom2[1], 0)
    
    # max(ΔR1, ΔR2) >= 3 表示至少一人的人气远超实力
    max_delta = max(delta_R1, delta_R2)
    
    if max_delta >= CONTROVERSY_THRESHOLD:
        # 触发评委救人
        # 用 save_judge_weight 加权评委分和粉丝票，保留综合分高的选手
        save_wj = params.get('save_judge_weight', 0.6)
        save_wf = 1.0 - save_wj
        
        # 归一化评委分和粉丝票（在bottom2内部归一化）
        b2_j = np.array([judge_scores.get(n, 0) for n in bottom2], dtype=float)
        b2_f = np.array([fan_votes.get(n, 0) for n in bottom2], dtype=float)
        
        j_max = b2_j.max() if b2_j.max() > 0 else 1.0
        f_max = b2_f.max() if b2_f.max() > 0 else 1.0
        b2_j_norm = b2_j / j_max
        b2_f_norm = b2_f / f_max
        
        combined = save_wj * b2_j_norm + save_wf * b2_f_norm
        
        if combined[0] <= combined[1]:
            eliminated, saved = bottom2[0], bottom2[1]
        else:
            eliminated, saved = bottom2[1], bottom2[0]
        
        return eliminated, True, {
            'triggered': True,
            'max_delta_R': int(max_delta),
            'bottom2': bottom2,
            'eliminated': eliminated,
            'saved': saved,
            'delta_R1': int(delta_R1),
            'delta_R2': int(delta_R2)
        }
    else:
        # 正常淘汰
        eliminated = bottom2[0]
        return eliminated, False, {
            'triggered': False,
            'max_delta_R': int(max_delta),
            'bottom2': bottom2,
            'eliminated': eliminated,
            'saved': bottom2[1],
            'delta_R1': int(delta_R1),
            'delta_R2': int(delta_R2)
        }


def run_dass_simulation(fan_votes_results):
    """
    运行DASS系统对所有赛季的模拟
    
    对每周:
      1. 用TOPSIS计算得分
      2. 用熵权法自适应加权
      3. 应用评委救人机制
      4. 计算SPI和FII
    
    返回:
        dict: DASS结果
    """
    print("=" * 60)
    print("Task 4: DASS 动态自适应评分系统")
    print("=" * 60)
    
    from data_loader import (
        load_raw_data, extract_weekly_judge_scores, get_weekly_contestants
    )
    
    df = load_raw_data()
    total_scores, avg_scores = extract_weekly_judge_scores(df)
    weekly_contestants = get_weekly_contestants(df, total_scores)
    
    # 存储结果
    weekly_results = {}
    save_triggered_count = 0
    total_weeks = 0
    
    # SPI/FII 累计
    spi_fii = {
        'n_total': 0,
        'n_skill_protected': 0,
        'n_fan_influenced': 0,
        'n_save_triggered': 0
    }
    
    weight_history = []  # 记录每周的权重变化
    
    for week_key, week_data in fan_votes_results['weekly_estimates'].items():
        season = week_data['season']
        week = week_data['week']
        contestants = week_data['contestants']
        fan_votes = week_data['estimated_fan_votes']
        
        if len(contestants) <= 1:
            continue
        
        # 获取评委分
        judge_scores = {}
        for name in contestants:
            score = total_scores.get((season, name, week), 0)
            judge_scores[name] = score
        
        if all(v == 0 for v in judge_scores.values()):
            continue
        
        total_weeks += 1
        
        # 计算DASS权重（熵权法或固定权重）
        w_j, w_f = get_dass_weights(judge_scores, fan_votes)
        weight_history.append({
            'week_key': week_key,
            'judge_weight': w_j,
            'fan_weight': w_f
        })
        
        # TOPSIS计算得分
        closeness = topsis_scoring(judge_scores, fan_votes)
        
        # 评委救人机制
        eliminated, triggered, save_detail = judges_save_mechanism(judge_scores, fan_votes, closeness, DASS_PARAMS)
        
        if triggered:
            save_triggered_count += 1
        
        # 找评委分最低和粉丝票最低的选手
        min_judge = min(judge_scores, key=judge_scores.get)
        min_fan = min(fan_votes, key=fan_votes.get)
        
        spi_fii['n_total'] += 1
        if eliminated == min_judge:
            spi_fii['n_skill_protected'] += 1
        if eliminated == min_fan:
            spi_fii['n_fan_influenced'] += 1
        if triggered:
            spi_fii['n_save_triggered'] += 1
        
        # 存储每周结果
        weekly_results[week_key] = {
            'season': season,
            'week': week,
            'contestants': contestants,
            'judge_weight': w_j,
            'fan_weight': w_f,
            'topsis_scores': closeness,
            'eliminated': eliminated,
            'save_triggered': triggered,
            'save_detail': save_detail,
            'controversy_index': compute_controversy_index(judge_scores, fan_votes)
        }
    
    # 汇总
    n = spi_fii['n_total']
    results = {
        'weekly_results': weekly_results,
        'weight_history': weight_history,
        'spi': round(spi_fii['n_skill_protected'] / n * 100, 1) if n > 0 else 0,
        'fii': round(spi_fii['n_fan_influenced'] / n * 100, 1) if n > 0 else 0,
        'save_trigger_rate': round(spi_fii['n_save_triggered'] / n * 100, 1) if n > 0 else 0,
        'save_triggered_count': spi_fii['n_save_triggered'],
        'total_weeks': total_weeks,
        'summary': {
            'SPI': round(spi_fii['n_skill_protected'] / n * 100, 1) if n > 0 else 0,
            'FII': round(spi_fii['n_fan_influenced'] / n * 100, 1) if n > 0 else 0,
            'Save_Trigger_Rate': f"{round(spi_fii['n_save_triggered'] / n * 100, 1)}%" if n > 0 else "0%",
            'In_Symbiotic_Interval': (
                'Yes' if (48 <= spi_fii['n_fan_influenced'] / n * 100 <= 51 and
                          spi_fii['n_skill_protected'] / n * 100 >= 92)
                else 'No'
            ) if n > 0 else 'N/A'
        }
    }
    
    print(f"\nDASS 系统评估:")
    print(f"  SPI (技能保护指数): {results['spi']}%")
    print(f"  FII (粉丝影响指数): {results['fii']}%")
    print(f"  评委救人触发率: {results['save_trigger_rate']}%")
    print(f"  总模拟周数: {total_weeks}")
    print(f"  是否处于共生区间 (SPI≥92, FII∈[48,51]): "
          f"{results['summary']['In_Symbiotic_Interval']}")
    
    print(f"\n{'=' * 60}")
    print("Task 4 完成!")
    print(f"{'=' * 60}")
    
    return results


def run_task4(fan_votes_results):
    """运行Task 4"""
    return run_dass_simulation(fan_votes_results)


if __name__ == '__main__':
    import json
    with open(f'{RESULT_DIR}/task1_fan_vote_reconstruction.json', 'r', encoding='utf-8') as f:
        task1_results = json.load(f)
    results = run_task4(task1_results)
    save_json(results, "task4_dass_system.json")