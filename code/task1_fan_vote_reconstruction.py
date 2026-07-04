# -*- coding: utf-8 -*-
"""
Task 1: 粉丝投票逆向估算模型
------------------------------
三阶段逆向推断模型:
  阶段1: 构建Dirichlet先验分布
  阶段2: 蒙特卡洛随机模拟
  阶段3: 贝叶斯后验过滤

核心思想:
  粉丝投票是"黑箱"数据（不可观测），但每周淘汰结果是已知的。
  我们利用淘汰结果作为约束条件，通过大规模蒙特卡洛模拟，
  反推出粉丝投票的后验分布。

赛制阶段:
  Stage 1 (Season 1-2): 排名制
  Stage 2 (Season 3-27): 百分比制
  Stage 3 (Season 28-34): 排名制 + 评委救人机制
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from config import (
    N_SIMULATIONS, RANDOM_SEED, OMEGA_1, OMEGA_2, OMEGA_3, K1, K2,
    STAGE1_SEASONS, STAGE2_SEASONS, STAGE3_SEASONS, RESULT_DIR
)
from data_loader import (
    load_raw_data, extract_weekly_judge_scores, get_weekly_contestants,
    get_elimination_info, get_week_numbers_for_season,
    build_celebrity_features, compute_industry_popularity,
    compute_region_popularity, compute_age_advantage, save_json
)

np.random.seed(RANDOM_SEED)


def get_stage(season):
    """
    根据赛季确定赛制阶段
    
    返回:
        int: 1=排名制, 2=百分比制, 3=排名制+评委救人
    """
    if STAGE1_SEASONS[0] <= season <= STAGE1_SEASONS[1]:
        return 1
    elif STAGE2_SEASONS[0] <= season <= STAGE2_SEASONS[1]:
        return 2
    else:
        return 3


def compute_popularity_score(contestant_name, industry, homestate, age,
                              industry_pop, region_pop, age_scores):
    """
    计算人气分数 S_pop
    
    公式: S_pop = ω1 * S_ind + ω2 * S_loc + ω3 * S_age
    
    参数:
        contestant_name: 选手姓名
        industry: 行业
        homestate: 家乡
        age: 年龄
        industry_pop: 行业人气字典
        region_pop: 地区人气字典
        age_scores: 年龄优势字典
    
    返回:
        float: 人气分数
    """
    s_ind = industry_pop.get(industry, 0.5)
    s_loc = region_pop.get(homestate, 0.0)
    s_age = age_scores.get(contestant_name, 0.5)
    
    return OMEGA_1 * s_ind + OMEGA_2 * s_loc + OMEGA_3 * s_age


def compute_alpha_vector(contestants, judge_scores_week, season, week,
                          features_df, industry_pop, region_pop, age_scores,
                          k1=None, k2=None):
    """
    计算Dirichlet分布的参数向量 α
    
    公式: α_i = 1 + K1 * j_norm_i + K2 * S_pop_i
    
    参数:
        contestants: 本周活跃选手列表
        judge_scores_week: 本周评委分字典 {name: score}
        season: 赛季
        week: 周数
        features_df: 特征DataFrame
        industry_pop: 行业人气
        region_pop: 地区人气
        age_scores: 年龄优势
        k1, k2: 可选的局部K1/K2参数（用于参数扫描）
    
    返回:
        list: α向量
    """
    # 使用传入的k1/k2或全局默认值
    local_k1 = k1 if k1 is not None else K1
    local_k2 = k2 if k2 is not None else K2
    
    n = len(contestants)
    alpha = []
    
    # 获取评委分
    j_scores = np.array([judge_scores_week.get(name, 0) for name in contestants])
    j_max = j_scores.max() if j_scores.max() > 0 else 1.0
    j_norm = j_scores / j_max
    
    # 获取人气分数
    for i, name in enumerate(contestants):
        # 从features_df中查找该选手的特征
        feat_row = features_df[(features_df['celebrity_name'] == name) & 
                               (features_df['season'] == season)]
        if len(feat_row) > 0:
            row = feat_row.iloc[0]
            industry = row['celebrity_industry']
            homestate = row['celebrity_homestate']
        else:
            industry = 'Actor/Actress'
            homestate = 'Unknown'
        
        s_pop = compute_popularity_score(
            name, industry, homestate, None, 
            industry_pop, region_pop, age_scores
        )
        
        # α_i = 1 + K1 * j_norm_i + K2 * S_pop_i
        alpha_i = 1.0 + local_k1 * j_norm[i] + local_k2 * s_pop
        alpha.append(max(alpha_i, 0.1))  # 确保α > 0
    
    return alpha


def rank_method_score(judge_ranks, fan_ranks):
    """
    排名制计算综合得分
    
    公式: S_rank = R_Judge + R_Fan
    排名越小越好（1=最好），综合得分越小越好
    
    参数:
        judge_ranks: 评委排名（1=最高分）
        fan_ranks: 粉丝排名（1=最高票）
    
    返回:
        array: 综合得分
    """
    return np.array(judge_ranks) + np.array(fan_ranks)


def percentage_method_score(judge_scores, fan_votes):
    """
    百分比制计算综合得分
    
    公式: S_percent = J_i / ΣJ_k + V_i / ΣV_k
    百分比越大越好
    
    参数:
        judge_scores: 评委分数组
        fan_votes: 粉丝投票比例数组
    
    返回:
        array: 综合百分比
    """
    j_sum = np.sum(judge_scores)
    if j_sum == 0:
        j_sum = 1
    j_pct = np.array(judge_scores) / j_sum
    
    v_sum = np.sum(fan_votes)
    if v_sum == 0:
        v_sum = 1
    v_pct = np.array(fan_votes) / v_sum
    
    return j_pct + v_pct


def monte_carlo_simulation_week(contestants, judge_scores_week, alpha,
                                 stage, actual_eliminated):
    """
    对某一周执行蒙特卡洛模拟
    
    流程:
      1. 从Dirichlet(α)分布采样N_SIMULATIONS次，得到模拟粉丝投票比例
      2. 根据stage对应的计分规则计算综合得分
      3. 找出模拟淘汰者，与实际淘汰者比对
      4. 保留匹配的模拟结果（贝叶斯后验过滤）
    
    参数:
        contestants: 本周活跃选手列表
        judge_scores_week: 本周评委分 {name: score}
        alpha: Dirichlet参数向量
        stage: 赛制阶段 (1/2/3)
        actual_eliminated: 实际淘汰者姓名
    
    返回:
        dict: {
            'matched_fan_votes': 匹配的粉丝投票比例列表,
            'match_rate': 匹配率,
            'n_matched': 匹配次数,
            'n_total': 总模拟次数,
            'all_fan_votes': 所有模拟结果（用于不确定性分析）
        }
    """
    n = len(contestants)
    
    # 获取评委分数组
    j_scores = np.array([judge_scores_week.get(name, 0) for name in contestants])
    
    # 从Dirichlet分布采样
    # 使用 Gamma 分布生成 Dirichlet 样本
    fan_votes_samples = np.random.dirichlet(alpha, size=N_SIMULATIONS)
    
    # 存储匹配的模拟结果
    matched_fan_votes = []
    n_matched = 0
    
    for sim_idx in range(N_SIMULATIONS):
        fan_votes = fan_votes_samples[sim_idx]
        
        # 计算评委排名
        # 排名1=最高分，使用降序排名
        judge_order = np.argsort(np.argsort(-j_scores)) + 1
        
        # 计算粉丝排名
        fan_order = np.argsort(np.argsort(-fan_votes)) + 1
        
        # 根据赛制计算综合得分
        if stage == 1 or stage == 3:
            # 排名制: 得分越小越好
            combined = rank_method_score(judge_order, fan_order)
            # 找出综合得分最高的（排名最差）= 淘汰者
            eliminated_idx = np.argmax(combined)
        else:
            # 百分比制: 百分比越大越好
            combined = percentage_method_score(j_scores, fan_votes)
            # 找出综合百分比最低的 = 淘汰者
            eliminated_idx = np.argmin(combined)
        
        simulated_eliminated = contestants[eliminated_idx]
        
        # Stage 3 有评委救人机制
        # 论文中将其建模为"引入淘汰概率而非直接淘汰"
        # 这里简化处理：如果Stage 3，模拟排出倒数两名，随机选择
        if stage == 3:
            if stage == 1 or stage == 3:
                # 排名制下找倒数两名（得分最高）
                bottom2_idx = np.argsort(combined)[-2:]
            else:
                # 百分比制下找倒数两名（得分最低）
                bottom2_idx = np.argsort(combined)[:2]
            
            # 评委救人：评委分低的那个被淘汰
            bottom2_scores = j_scores[bottom2_idx]
            actual_bottom_idx = bottom2_idx[np.argmin(bottom2_scores)]
            simulated_eliminated = contestants[actual_bottom_idx]
        
        # 贝叶斯过滤：检查是否匹配实际淘汰结果
        if simulated_eliminated == actual_eliminated:
            matched_fan_votes.append(fan_votes.tolist())
            n_matched += 1
    
    match_rate = n_matched / N_SIMULATIONS if N_SIMULATIONS > 0 else 0
    
    return {
        'matched_fan_votes': matched_fan_votes,
        'match_rate': match_rate,
        'n_matched': n_matched,
        'n_total': N_SIMULATIONS,
        'all_fan_votes': fan_votes_samples.tolist()
    }


def reconstruct_fan_votes():
    """
    主函数：对所有赛季所有周执行粉丝投票重建
    
    返回:
        dict: 完整的重建结果
    """
    print("=" * 60)
    print("Task 1: 粉丝投票逆向估算模型")
    print("=" * 60)
    
    # 加载数据
    df = load_raw_data()
    total_scores, avg_scores = extract_weekly_judge_scores(df)
    weekly_contestants = get_weekly_contestants(df, total_scores)
    eliminations, placements = get_elimination_info(df)
    features_df, industry_map, partner_experience = build_celebrity_features(df)
    
    # 计算人气相关分数
    industry_pop = compute_industry_popularity(df, features_df)
    region_pop = compute_region_popularity(features_df)
    age_scores = compute_age_advantage(features_df)
    
    # 存储结果
    results = {
        'weekly_estimates': {},       # 每周估计的粉丝投票
        'match_rates': {},            # 每周匹配率
        'uncertainty_metrics': {},    # 不确定性指标
        'season_summary': {},         # 赛季汇总
        'global_metrics': {}          # 全局指标
    }
    
    all_cv_values = []  # 用于计算全局CV
    
    seasons = sorted(df['season'].unique())
    
    for season in seasons:
        season = int(season)
        stage = get_stage(season)
        max_week = get_week_numbers_for_season(df, season)
        
        print(f"\n处理 Season {season} (Stage {stage}, {max_week}周)...")
        
        season_cvs = []
        
        for week in range(1, max_week + 1):
            key = (season, week)
            
            if key not in weekly_contestants:
                continue
            
            contestants = weekly_contestants[key]
            n = len(contestants)
            
            if n <= 1:
                continue  # 决赛周，不淘汰
            
            # 获取本周评委分
            judge_scores_week = {}
            for name in contestants:
                score = total_scores.get((season, name, week), 0)
                judge_scores_week[name] = score
            
            # 获取实际淘汰者
            actual_eliminated = eliminations.get(key, None)
            if actual_eliminated is None:
                continue
            
            # 计算Dirichlet α参数
            alpha = compute_alpha_vector(
                contestants, judge_scores_week, season, week,
                features_df, industry_pop, region_pop, age_scores
            )
            
            # 执行蒙特卡洛模拟
            sim_result = monte_carlo_simulation_week(
                contestants, judge_scores_week, alpha, stage, actual_eliminated
            )
            
            # 计算估计的粉丝投票（取匹配样本的均值）
            if sim_result['matched_fan_votes']:
                matched = np.array(sim_result['matched_fan_votes'])
                estimated_fan_votes = matched.mean(axis=0)
                fan_votes_std = matched.std(axis=0)
            else:
                # 如果没匹配上，使用所有样本的均值
                all_samples = np.array(sim_result['all_fan_votes'])
                estimated_fan_votes = all_samples.mean(axis=0)
                fan_votes_std = all_samples.std(axis=0)
            
            # 计算不确定性指标 CV (Coefficient of Variation)
            # CV = std / mean
            cv_values = []
            for i in range(n):
                mean_val = estimated_fan_votes[i]
                std_val = fan_votes_std[i]
                if mean_val > 0:
                    cv = std_val / mean_val
                else:
                    cv = 0
                cv_values.append(cv)
            
            cv_mean = np.mean(cv_values)
            cv_median = np.median(cv_values)
            season_cvs.append(cv_mean)
            all_cv_values.append(cv_mean)
            
            # 存储每周结果
            week_key = f"S{season}_W{week}"
            results['weekly_estimates'][week_key] = {
                'season': season,
                'week': week,
                'contestants': contestants,
                'estimated_fan_votes': {name: float(estimated_fan_votes[i]) 
                                         for i, name in enumerate(contestants)},
                'fan_votes_std': {name: float(fan_votes_std[i]) 
                                   for i, name in enumerate(contestants)},
                'cv_per_contestant': {name: float(cv_values[i]) 
                                       for i, name in enumerate(contestants)},
                'match_rate': sim_result['match_rate'],
                'n_matched': sim_result['n_matched'],
                'n_total': sim_result['n_total'],
                'alpha': [float(a) for a in alpha],
                'stage': stage,
                'actual_eliminated': actual_eliminated
            }
            
            results['match_rates'][week_key] = sim_result['match_rate']
        
        # 赛季汇总
        if season_cvs:
            results['season_summary'][f"S{season}"] = {
                'season': season,
                'stage': stage,
                'mean_cv': float(np.mean(season_cvs)),
                'median_cv': float(np.median(season_cvs)),
                'mean_match_rate': float(np.mean([
                    results['match_rates'].get(f"S{season}_W{w}", 0) 
                    for w in range(1, max_week + 1)
                ]))
            }
    
    # 全局指标
    results['global_metrics'] = {
        'global_median_cv': float(np.median(all_cv_values)),
        'global_mean_cv': float(np.mean(all_cv_values)),
        'global_std_cv': float(np.std(all_cv_values)),
        'cv_25th_percentile': float(np.percentile(all_cv_values, 25)),
        'cv_75th_percentile': float(np.percentile(all_cv_values, 75)),
        'total_weeks_simulated': len(all_cv_values),
        'n_simulations_per_week': N_SIMULATIONS
    }
    
    print(f"\n{'=' * 60}")
    print(f"Task 1 完成!")
    print(f"  全局中位CV: {results['global_metrics']['global_median_cv']:.4f}")
    print(f"  全局平均CV: {results['global_metrics']['global_mean_cv']:.4f}")
    print(f"  模拟总周数: {results['global_metrics']['total_weeks_simulated']}")
    print(f"{'=' * 60}")
    
    return results


def run_task1():
    """运行Task 1并保存结果"""
    results = reconstruct_fan_votes()
    
    # 保存完整结果
    save_json(results, "task1_fan_vote_reconstruction.json")
    
    # 保存简化版结果（供后续任务使用）
    fan_votes_simple = {}
    for week_key, week_data in results['weekly_estimates'].items():
        fan_votes_simple[week_key] = {
            'season': week_data['season'],
            'week': week_data['week'],
            'fan_votes': week_data['estimated_fan_votes'],
            'match_rate': week_data['match_rate']
        }
    save_json(fan_votes_simple, "task1_fan_votes_simple.json")
    
    return results


if __name__ == '__main__':
    run_task1()