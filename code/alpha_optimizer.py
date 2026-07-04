# -*- coding: utf-8 -*-
"""
Dirichlet α 参数迭代优化器
---------------------------
目标: 自动搜索最优的 K1（评委分权重系数）和 K2（人气权重系数），
      使得模型预测与实际淘汰结果最匹配。

优化指标:
  1. 平均匹配率 (Match Rate): 模拟淘汰与实际淘汰一致的比例
  2. 全局 CV (不确定性): 越低越好，但不能以牺牲匹配率为代价
  3. 综合评分 = 匹配率 - λ * CV  (λ=0.1，避免过拟合)

搜索策略:
  - 粗网格搜索 + 精细局部搜索
  - 考虑到计算成本，使用 1000 次模拟进行快速评估
"""

import numpy as np
from itertools import product
import time

from config import RESULT_DIR, RANDOM_SEED
from data_loader import (
    load_raw_data, extract_weekly_judge_scores, get_weekly_contestants,
    get_elimination_info, get_week_numbers_for_season,
    build_celebrity_features, compute_industry_popularity,
    compute_region_popularity, compute_age_advantage, save_json
)
from task1_fan_vote_reconstruction import (
    get_stage, compute_alpha_vector, monte_carlo_simulation_week
)

np.random.seed(RANDOM_SEED)


def evaluate_k1_k2(k1, k2, data_bundle, n_sim=1000):
    """
    评估一组 K1/K2 参数的表现
    
    参数:
        k1, k2: 待评估的参数
        data_bundle: 预加载的数据字典
        n_sim: 每次模拟的次数（快速评估使用1000次）
    
    返回:
        dict: 评估指标
    """
    df = data_bundle['df']
    total_scores = data_bundle['total_scores']
    weekly_contestants = data_bundle['weekly_contestants']
    eliminations = data_bundle['eliminations']
    features_df = data_bundle['features_df']
    industry_pop = data_bundle['industry_pop']
    region_pop = data_bundle['region_pop']
    age_scores = data_bundle['age_scores']
    
    match_rates = []
    all_cv = []
    
    # 每周评估
    for (season, week), contestants in weekly_contestants.items():
        if len(contestants) <= 1:
            continue
        
        actual_eliminated = eliminations.get((season, week), None)
        if actual_eliminated is None:
            continue
        
        judge_scores_week = {name: total_scores.get((season, name, week), 0) 
                             for name in contestants}
        
        if all(v == 0 for v in judge_scores_week.values()):
            continue
        
        stage = get_stage(season)
        
        # 构建 α 向量
        alpha = compute_alpha_vector(
            contestants, judge_scores_week, season, week,
            features_df, industry_pop, region_pop, age_scores,
            k1=k1, k2=k2
        )
        
        # 临时修改全局 N_SIMULATIONS
        import task1_fan_vote_reconstruction as t1
        old_n = t1.N_SIMULATIONS
        t1.N_SIMULATIONS = n_sim
        
        sim_result = monte_carlo_simulation_week(
            contestants, judge_scores_week, alpha, stage, actual_eliminated
        )
        
        t1.N_SIMULATIONS = old_n
        
        match_rates.append(sim_result['match_rate'])
        
        # 计算本周CV（从匹配样本）
        if sim_result['matched_fan_votes']:
            matched = np.array(sim_result['matched_fan_votes'])
            means = matched.mean(axis=0)
            stds = matched.std(axis=0)
            cvs = [s / m if m > 0 else 0 for s, m in zip(stds, means)]
            all_cv.append(np.mean(cvs))
        elif sim_result['all_fan_votes']:
            all_samples = np.array(sim_result['all_fan_votes'])
            means = all_samples.mean(axis=0)
            stds = all_samples.std(axis=0)
            cvs = [s / m if m > 0 else 0 for s, m in zip(stds, means)]
            all_cv.append(np.mean(cvs))
    
    mean_match = np.mean(match_rates) if match_rates else 0
    mean_cv = np.mean(all_cv) if all_cv else 1
    median_cv = np.median(all_cv) if all_cv else 1
    
    # 综合评分：匹配率 - 0.1 * CV
    score = mean_match - 0.1 * mean_cv
    
    return {
        'k1': k1,
        'k2': k2,
        'mean_match_rate': float(mean_match),
        'mean_cv': float(mean_cv),
        'median_cv': float(median_cv),
        'score': float(score)
    }


def load_data_bundle():
    """预加载所有需要的数据"""
    df = load_raw_data()
    total_scores, avg_scores = extract_weekly_judge_scores(df)
    weekly_contestants = get_weekly_contestants(df, total_scores)
    eliminations, placements = get_elimination_info(df)
    features_df, industry_map, partner_experience = build_celebrity_features(df)
    
    industry_pop = compute_industry_popularity(df, features_df)
    region_pop = compute_region_popularity(features_df)
    age_scores = compute_age_advantage(features_df)
    
    return {
        'df': df,
        'total_scores': total_scores,
        'weekly_contestants': weekly_contestants,
        'eliminations': eliminations,
        'features_df': features_df,
        'industry_pop': industry_pop,
        'region_pop': region_pop,
        'age_scores': age_scores
    }


def grid_search_k1_k2(data_bundle, k1_range, k2_range, n_sim=1000):
    """
    网格搜索 K1/K2
    
    参数:
        data_bundle: 数据包
        k1_range: K1 候选值列表
        k2_range: K2 候选值列表
        n_sim: 快速评估模拟次数
    
    返回:
        list: 所有参数组合的评估结果
        dict: 最优参数
    """
    results = []
    best_score = -np.inf
    best_params = None
    
    total = len(k1_range) * len(k2_range)
    start = time.time()
    
    for idx, (k1, k2) in enumerate(product(k1_range, k2_range)):
        print(f"  [{idx+1}/{total}] 评估 K1={k1:.2f}, K2={k2:.2f}...", end=' ')
        res = evaluate_k1_k2(k1, k2, data_bundle, n_sim=n_sim)
        results.append(res)
        print(f"Match={res['mean_match_rate']:.3f}, CV={res['mean_cv']:.3f}, Score={res['score']:.3f}")
        
        if res['score'] > best_score:
            best_score = res['score']
            best_params = res
    
    elapsed = time.time() - start
    print(f"\n网格搜索完成，耗时 {elapsed:.1f}秒")
    print(f"最优参数: K1={best_params['k1']:.2f}, K2={best_params['k2']:.2f}")
    print(f"  匹配率={best_params['mean_match_rate']:.3f}, CV={best_params['mean_cv']:.3f}")
    
    return results, best_params


def fine_search(data_bundle, center_k1, center_k2, radius=1.0, step=0.2, n_sim=2000):
    """
    在中心点附近进行精细搜索
    """
    k1_values = np.arange(max(0.1, center_k1 - radius), center_k1 + radius + step, step)
    k2_values = np.arange(max(0.1, center_k2 - radius), center_k2 + radius + step, step)
    
    results, best = grid_search_k1_k2(data_bundle, k1_values, k2_values, n_sim=n_sim)
    return results, best


def run_alpha_optimization():
    """
    运行 α 参数迭代优化
    
    两个阶段:
      1. 粗网格搜索：K1 ∈ [1, 10], K2 ∈ [0.5, 8], 步长1.0
      2. 精细搜索：在最优点附近 ±1.0, 步长0.2
    
    返回:
        dict: 优化结果，包含最优 K1/K2
    """
    print("=" * 60)
    print("Dirichlet α 参数迭代优化")
    print("=" * 60)
    
    data_bundle = load_data_bundle()
    
    # 第一阶段：粗网格搜索
    print("\n[阶段1/2] 粗网格搜索...")
    k1_range = np.arange(1.0, 10.5, 1.5)
    k2_range = np.arange(0.5, 8.5, 1.5)
    coarse_results, coarse_best = grid_search_k1_k2(
        data_bundle, k1_range, k2_range, n_sim=1000
    )
    
    # 第二阶段：精细搜索
    print("\n[阶段2/2] 精细搜索...")
    fine_results, fine_best = fine_search(
        data_bundle, coarse_best['k1'], coarse_best['k2'],
        radius=1.5, step=0.3, n_sim=2000
    )
    
    results = {
        'coarse_search': coarse_results,
        'coarse_best': coarse_best,
        'fine_search': fine_results,
        'fine_best': fine_best,
        'recommended_k1': float(fine_best['k1']),
        'recommended_k2': float(fine_best['k2']),
        'recommended_mean_match_rate': float(fine_best['mean_match_rate']),
        'recommended_mean_cv': float(fine_best['mean_cv'])
    }
    
    print("\n" + "=" * 60)
    print(f"最终推荐参数: K1={results['recommended_k1']:.2f}, K2={results['recommended_k2']:.2f}")
    print(f"预期匹配率: {results['recommended_mean_match_rate']:.4f}")
    print(f"预期平均CV: {results['recommended_mean_cv']:.4f}")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    results = run_alpha_optimization()
    save_json(results, "alpha_optimization.json")