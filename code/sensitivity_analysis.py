# -*- coding: utf-8 -*-
"""
敏感性分析模块（优化版）
------------------------
1. Dirichlet α 参数鲁棒性分析: 真实蒙特卡洛重跑，观测CV变化
2. 数据噪声鲁棒性分析: 对粉丝投票添加0%-30%高斯噪声, 观察排名稳定性
3. DASS权重敏感性分析: 对粉丝权重进行OAT单因素分析, 寻找"稳定平台"

优化：
  - α分析不再使用理论近似，而是真实重跑蒙特卡洛模拟
  - 噪声分析对每个机制分别计算真实排名变化
"""

import numpy as np
from scipy.stats import spearmanr

from config import (
    RESULT_DIR, RANDOM_SEED, ALPHA_RANGE, NOISE_RANGE, WEIGHT_RANGE, N_SIMULATIONS
)
from data_loader import (
    load_raw_data, extract_weekly_judge_scores, get_weekly_contestants,
    get_elimination_info, save_json
)
from task1_fan_vote_reconstruction import (
    get_stage, compute_alpha_vector, monte_carlo_simulation_week
)
from task3_mechanism_comparison import (
    compute_rank_method_result, compute_percentage_method_result,
    get_ranking_from_scores
)
from task4_dass_system import topsis_scoring

np.random.seed(RANDOM_SEED)


def analyze_alpha_robustness(fan_votes_results, sample_weeks=20):
    """
    Dirichlet α 参数鲁棒性分析（真实蒙特卡洛重跑）
    
    测试 α 缩放因子 ∈ [0.9, 1.1] 范围内（即±10%微调），全局中位CV的变化
    
    论文结论: ΔCV ≈ 0.005，模型对α参数极为鲁棒
    
    注意：这里对基准alpha做缩放，观察CV变化
    """
    print("=" * 60)
    print("敏感性分析 1: Dirichlet α 参数鲁棒性")
    print("=" * 60)
    
    df = load_raw_data()
    total_scores, _ = extract_weekly_judge_scores(df)
    weekly_contestants = get_weekly_contestants(df, total_scores)
    eliminations, _ = get_elimination_info(df)
    
    from data_loader import (
        build_celebrity_features, compute_industry_popularity,
        compute_region_popularity, compute_age_advantage
    )
    features_df, _, _ = build_celebrity_features(df)
    industry_pop = compute_industry_popularity(df, features_df)
    region_pop = compute_region_popularity(features_df)
    age_scores = compute_age_advantage(features_df)
    
    # 选择 sample_weeks 个代表性周
    all_weeks = list(fan_votes_results['weekly_estimates'].keys())
    selected_weeks = np.random.choice(all_weeks, size=min(sample_weeks, len(all_weeks)), replace=False)
    
    alpha_multipliers = np.linspace(ALPHA_RANGE[0], ALPHA_RANGE[1], 10)
    cv_results = []
    
    import task1_fan_vote_reconstruction as t1
    old_n = t1.N_SIMULATIONS
    t1.N_SIMULATIONS = 2000  # 敏感性分析使用较少模拟次数
    
    for mult in alpha_multipliers:
        week_cvs = []
        
        for week_key in selected_weeks:
            week_data = fan_votes_results['weekly_estimates'][week_key]
            season = week_data['season']
            week = week_data['week']
            contestants = week_data['contestants']
            actual_eliminated = eliminations.get((season, week), None)
            
            if actual_eliminated is None or len(contestants) <= 1:
                continue
            
            judge_scores_week = {name: total_scores.get((season, name, week), 0) for name in contestants}
            
            # 基准 alpha
            base_alpha = compute_alpha_vector(
                contestants, judge_scores_week, season, week,
                features_df, industry_pop, region_pop, age_scores
            )
            # 缩放 alpha
            scaled_alpha = [max(0.1, a * mult) for a in base_alpha]
            
            stage = get_stage(season)
            sim_result = monte_carlo_simulation_week(
                contestants, judge_scores_week, scaled_alpha, stage, actual_eliminated
            )
            
            if sim_result['matched_fan_votes']:
                matched = np.array(sim_result['matched_fan_votes'])
                means = matched.mean(axis=0)
                stds = matched.std(axis=0)
                cvs = [s / m if m > 0 else 0 for s, m in zip(stds, means)]
                week_cvs.append(np.median(cvs))
        
        median_cv = np.median(week_cvs) if week_cvs else 0
        cv_results.append({
            'alpha_multiplier': round(float(mult), 2),
            'median_cv': round(float(median_cv), 6),
            'n_weeks': len(week_cvs)
        })
    
    t1.N_SIMULATIONS = old_n
    
    cv_values = [r['median_cv'] for r in cv_results]
    delta_cv = max(cv_values) - min(cv_values)
    
    results = {
        'alpha_range': list(ALPHA_RANGE),
        'cv_results': cv_results,
        'delta_cv': round(float(delta_cv), 6),
        'baseline_cv': round(float(cv_results[len(cv_results)//2]['median_cv']), 6),
        'conclusion': f'ΔCV ≈ {delta_cv:.4f}, {"模型对α参数极为鲁棒" if delta_cv < 0.05 else "α参数对CV影响较显著"}'
    }
    
    print(f"  α缩放范围: [{ALPHA_RANGE[0]}, {ALPHA_RANGE[1]}]")
    print(f"  CV变化范围: {min(cv_values):.4f} → {max(cv_values):.4f}")
    print(f"  ΔCV: {delta_cv:.6f}")
    print(f"  结论: {results['conclusion']}")
    
    return results


def compute_ranking_for_mechanism(mechanism, judge_scores, fan_votes):
    """根据机制计算排名"""
    if mechanism == 'rank':
        score = compute_rank_method_result(judge_scores, fan_votes)
        ranking = get_ranking_from_scores(score, ascending=True)
    elif mechanism == 'percentage':
        score = compute_percentage_method_result(judge_scores, fan_votes)
        ranking = get_ranking_from_scores(score, ascending=False)
    elif mechanism == 'dass':
        closeness = topsis_scoring(judge_scores, fan_votes)
        ranking = get_ranking_from_scores(closeness, ascending=False)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    
    return ranking


def analyze_noise_robustness(fan_votes_results):
    """
    数据噪声鲁棒性分析（真实机制排名）
    
    对粉丝投票向量添加0%-30%的加性高斯噪声，
    比较Rank、Percentage和DASS三种机制的排名稳定性
    
    排名稳定性 = 噪声前后排名的Spearman相关系数
    """
    print("\n" + "=" * 60)
    print("敏感性分析 2: 数据噪声鲁棒性")
    print("=" * 60)
    
    df = load_raw_data()
    total_scores, _ = extract_weekly_judge_scores(df)
    
    noise_levels = np.linspace(NOISE_RANGE[0], NOISE_RANGE[1], 7)
    mechanisms = ['rank', 'percentage', 'dass']
    
    # 选取代表性周
    sample_weeks = []
    for week_key, week_data in fan_votes_results['weekly_estimates'].items():
        if len(week_data['contestants']) >= 5:
            sample_weeks.append(week_key)
            if len(sample_weeks) >= 15:
                break
    
    noise_results = {mech: {level: [] for level in noise_levels} for mech in mechanisms}
    
    for week_key in sample_weeks:
        week_data = fan_votes_results['weekly_estimates'][week_key]
        season = week_data['season']
        week = week_data['week']
        contestants = week_data['contestants']
        original_fan_votes = week_data['estimated_fan_votes']
        
        judge_scores = {name: total_scores.get((season, name, week), 0) for name in contestants}
        
        # 原始各机制排名
        original_rankings = {}
        for mech in mechanisms:
            original_rankings[mech] = compute_ranking_for_mechanism(mech, judge_scores, original_fan_votes)
        
        for noise_level in noise_levels:
            if noise_level == 0:
                for mech in mechanisms:
                    noise_results[mech][noise_level].append(1.0)
                continue
            
            # 添加噪声并归一化
            noisy_fan_votes = {}
            for name in contestants:
                original = original_fan_votes.get(name, 0)
                noise = np.random.normal(0, noise_level * original + 1e-6)
                noisy_fan_votes[name] = max(0, original + noise)
            
            total = sum(noisy_fan_votes.values())
            if total > 0:
                for name in noisy_fan_votes:
                    noisy_fan_votes[name] /= total
            
            # 噪声后各机制排名
            for mech in mechanisms:
                noisy_ranking = compute_ranking_for_mechanism(mech, judge_scores, noisy_fan_votes)
                
                orig_arr = np.array([original_rankings[mech][n] for n in contestants])
                noisy_arr = np.array([noisy_ranking[n] for n in contestants])
                
                if len(contestants) >= 3:
                    rho, _ = spearmanr(orig_arr, noisy_arr)
                    noise_results[mech][noise_level].append(rho)
    
    # 汇总
    summary = {
        'noise_levels': [float(n) for n in noise_levels]
    }
    for mech in mechanisms:
        summary[f'{mech}_stability'] = []
        for noise_level in noise_levels:
            if noise_results[mech][noise_level]:
                mean_rho = float(np.mean(noise_results[mech][noise_level]))
                summary[f'{mech}_stability'].append({
                    'noise_level': round(float(noise_level), 2),
                    'mean_rho': round(mean_rho, 4)
                })
    
    print(f"  噪声水平: 0% - 30%")
    for mech in mechanisms:
        last_rho = summary[f'{mech}_stability'][-1]['mean_rho']
        print(f"  {mech.capitalize():12s} 30%噪声下ρ: {last_rho:.4f}")
    
    return summary


def analyze_weight_sensitivity(fan_votes_results):
    """
    DASS权重敏感性分析 (OAT方法)
    
    寻找"稳定平台"区间: w_fan ∈ [0.3, 0.7] 内排名不变
    """
    print("\n" + "=" * 60)
    print("敏感性分析 3: DASS权重敏感性")
    print("=" * 60)
    
    df = load_raw_data()
    total_scores, _ = extract_weekly_judge_scores(df)
    
    weight_values = np.linspace(WEIGHT_RANGE[0], WEIGHT_RANGE[1], 11)
    
    # 选择多个代表性周
    sample_weeks = []
    for week_key, week_data in fan_votes_results['weekly_estimates'].items():
        if len(week_data['contestants']) >= 5:
            sample_weeks.append(week_key)
            if len(sample_weeks) >= 5:
                break
    
    all_stable_intervals = []
    
    for sample_week in sample_weeks:
        week_data = fan_votes_results['weekly_estimates'][sample_week]
        season = week_data['season']
        week = week_data['week']
        contestants = week_data['contestants']
        fan_votes = week_data['estimated_fan_votes']
        
        judge_scores = {name: total_scores.get((season, name, week), 0) for name in contestants}
        
        j_arr = np.array([judge_scores[n] for n in contestants], dtype=float)
        f_arr = np.array([fan_votes.get(n, 0) for n in contestants], dtype=float)
        
        j_norm = j_arr / (j_arr.max() + 1e-10)
        f_norm = f_arr / (f_arr.max() + 1e-10)
        
        weight_rankings = []
        for w_fan in weight_values:
            w_judge = 1 - w_fan
            combined = w_judge * j_norm + w_fan * f_norm
            ranking = tuple([contestants[i] for i in np.argsort(-combined)])
            weight_rankings.append(ranking)
        
        # 找稳定区间
        stable_intervals = []
        for i in range(len(weight_rankings) - 1):
            if weight_rankings[i] == weight_rankings[i+1]:
                stable_intervals.append((weight_values[i], weight_values[i+1]))
        
        # 合并
        if stable_intervals:
            merged = []
            current_start, current_end = stable_intervals[0]
            for start, end in stable_intervals[1:]:
                if start == current_end:
                    current_end = end
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = start, end
            merged.append((current_start, current_end))
            
            longest = max(merged, key=lambda x: x[1] - x[0])
            all_stable_intervals.append(longest)
    
    # 取所有样本周稳定区间的交集
    if all_stable_intervals:
        overall_start = max([s for s, _ in all_stable_intervals])
        overall_end = min([e for _, e in all_stable_intervals])
        if overall_start < overall_end:
            stable_platform = (overall_start, overall_end)
        else:
            stable_platform = None
    else:
        stable_platform = None
    
    results = {
        'sample_weeks': sample_weeks,
        'stable_platform': {
            'start': round(stable_platform[0], 2) if stable_platform else None,
            'end': round(stable_platform[1], 2) if stable_platform else None,
            'width': round(stable_platform[1] - stable_platform[0], 2) if stable_platform else 0
        },
        'conclusion': (f'稳定平台区间: [{stable_platform[0]:.2f}, {stable_platform[1]:.2f}]' 
                       if stable_platform else '未发现稳定平台')
    }
    
    print(f"  样本周数: {len(sample_weeks)}")
    print(f"  权重搜索范围: [{WEIGHT_RANGE[0]}, {WEIGHT_RANGE[1]}]")
    print(f"  稳定平台: {results['conclusion']}")
    
    return results


def run_sensitivity_analysis(fan_votes_results):
    """运行所有敏感性分析"""
    print("\n" + "=" * 60)
    print("敏感性分析")
    print("=" * 60)
    
    # 1. α鲁棒性（真实重跑）
    alpha_results = analyze_alpha_robustness(fan_votes_results)
    
    # 2. 噪声鲁棒性（真实机制排名）
    noise_results = analyze_noise_robustness(fan_votes_results)
    
    # 3. 权重敏感性
    weight_results = analyze_weight_sensitivity(fan_votes_results)
    
    results = {
        'alpha_robustness': alpha_results,
        'noise_robustness': noise_results,
        'weight_sensitivity': weight_results
    }
    
    print(f"\n{'=' * 60}")
    print("敏感性分析完成!")
    print(f"{'=' * 60}")
    
    return results


if __name__ == '__main__':
    import json
    with open(f'{RESULT_DIR}/task1_fan_vote_reconstruction.json', 'r', encoding='utf-8') as f:
        task1_results = json.load(f)
    results = run_sensitivity_analysis(task1_results)
    save_json(results, "sensitivity_analysis.json")