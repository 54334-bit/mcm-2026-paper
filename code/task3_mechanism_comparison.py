# -*- coding: utf-8 -*-
"""
Task 3: 计分机制比较与反事实模拟
-----------------------------------
比较两种历史计分机制:
  1. 排名制 (Rank Method): S = R_Judge + R_Fan
  2. 百分比制 (Percentage Method): S = J/ΣJ + V/ΣV

分析内容:
  1. Spearman秩相关分析: 量化每种机制对粉丝/评委的偏向性
  2. 方差差异分析: 解释百分比制为何偏向粉丝
  3. 争议案例反事实模拟: 对Jerry Rice, Bobby Bones等案例重建
  4. SPI和FII指标: 量化公平性
  5. 评委救人机制影响分析
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict
import json
import os

from config import RESULT_DIR, CONTROVERSIAL_CASES, RANDOM_SEED
from data_loader import (
    load_raw_data, extract_weekly_judge_scores, get_weekly_contestants,
    get_elimination_info, save_json
)

np.random.seed(RANDOM_SEED)


def compute_rank_method_result(judge_scores, fan_votes):
    """
    排名制计算
    
    参数:
        judge_scores: dict {name: score}
        fan_votes: dict {name: vote_share}
    
    返回:
        dict: {name: combined_rank} (值越小排名越好)
    """
    names = list(judge_scores.keys())
    j_scores = np.array([judge_scores[n] for n in names])
    f_votes = np.array([fan_votes.get(n, 0) for n in names])
    
    # 排名: 1=最高分/最高票，使用降序排名
    j_ranks = len(names) - np.argsort(np.argsort(j_scores))
    f_ranks = len(names) - np.argsort(np.argsort(f_votes))
    
    combined = j_ranks + f_ranks
    
    return {name: int(combined[i]) for i, name in enumerate(names)}


def compute_percentage_method_result(judge_scores, fan_votes):
    """
    百分比制计算
    
    参数:
        judge_scores: dict {name: score}
        fan_votes: dict {name: vote_share}
    
    返回:
        dict: {name: combined_percentage} (值越大排名越好)
    """
    names = list(judge_scores.keys())
    j_scores = np.array([judge_scores[n] for n in names])
    f_votes = np.array([fan_votes.get(n, 0) for n in names])
    
    j_sum = j_scores.sum()
    v_sum = f_votes.sum()
    
    if j_sum == 0:
        j_sum = 1
    if v_sum == 0:
        v_sum = 1
    
    j_pct = j_scores / j_sum
    v_pct = f_votes / v_sum
    
    combined = j_pct + v_pct
    
    return {name: float(combined[i]) for i, name in enumerate(names)}


def get_ranking_from_scores(scores_dict, ascending=True):
    """
    从得分字典转换为排名
    
    参数:
        scores_dict: {name: score}
        ascending: True=分越低排名越好(排名制), False=分越高排名越好(百分比制)
    
    返回:
        dict: {name: rank}
    """
    names = list(scores_dict.keys())
    values = np.array([scores_dict[n] for n in names])
    
    if ascending:
        ranks = np.argsort(np.argsort(values)) + 1
    else:
        ranks = np.argsort(np.argsort(-values)) + 1
    
    return {name: int(ranks[i]) for i, name in enumerate(names)}


def compute_spearman_correlations(fan_votes_results):
    """
    计算Spearman秩相关系数
    
    对每周的数据:
      - 排名制结果 vs 粉丝排名: ρ1
      - 排名制结果 vs 评委排名: ρ3
      - 百分比制结果 vs 粉丝排名: ρ2
      - 百分比制结果 vs 评委排名: ρ4
    
    返回:
        dict: 相关系数结果
    """
    all_rho1 = []  # Rank vs Fan
    all_rho2 = []  # Percent vs Fan
    all_rho3 = []  # Rank vs Judge
    all_rho4 = []  # Percent vs Judge
    
    weekly_details = {}
    
    for week_key, week_data in fan_votes_results['weekly_estimates'].items():
        season = week_data['season']
        week = week_data['week']
        contestants = week_data['contestants']
        fan_votes = week_data['estimated_fan_votes']
        
        if len(contestants) <= 2:
            continue
        
        # 获取评委分
        judge_scores = {}
        for name in contestants:
            # 从数据中获取评委分
            judge_scores[name] = week_data.get('alpha', [0])[0]  # 临时占位
        
        # 实际上需要从原始数据中重新获取评委分
        
        # 计算排名
        fan_ranks = get_ranking_from_scores(fan_votes, ascending=False)
        # judge_ranks 需要从原始数据获取
        
        # 这里简化处理，使用fan_votes中的已有信息
        # 实际上需要完整的评委分数据
    
    # 由于需要评委分，我们从Task1结果中重新计算
    # 重新设计这个函数...
    
    return {
        'rho1_rank_vs_fan': None,
        'rho2_percent_vs_fan': None,
        'rho3_rank_vs_judge': None,
        'rho4_percent_vs_judge': None,
    }


def run_full_comparison(fan_votes_results):
    """
    运行完整的两机制比较
    
    使用Task1重建的粉丝投票数据，对每周同时应用两种计分方式，
    比较结果差异
    """
    print("=" * 60)
    print("Task 3: 计分机制比较与反事实模拟")
    print("=" * 60)
    
    # 加载原始数据获取评委分
    df = load_raw_data()
    total_scores, avg_scores = extract_weekly_judge_scores(df)
    weekly_contestants = get_weekly_contestants(df, total_scores)
    eliminations, placements = get_elimination_info(df)
    
    # 存储结果
    rank_results = {}
    percent_results = {}
    all_rho = {'rho1': [], 'rho2': [], 'rho3': [], 'rho4': []}
    reversal_weeks = []
    
    for week_key, week_data in fan_votes_results['weekly_estimates'].items():
        season = week_data['season']
        week = week_data['week']
        contestants = week_data['contestants']
        fan_votes = week_data['estimated_fan_votes']
        
        if len(contestants) <= 2:
            continue
        
        # 获取评委分
        judge_scores = {}
        for name in contestants:
            score = total_scores.get((season, name, week), 0)
            judge_scores[name] = score
        
        if all(v == 0 for v in judge_scores.values()):
            continue
        
        # 计算排名制结果
        rank_score = compute_rank_method_result(judge_scores, fan_votes)
        rank_ranking = get_ranking_from_scores(rank_score, ascending=True)
        
        # 计算百分比制结果
        percent_score = compute_percentage_method_result(judge_scores, fan_votes)
        percent_ranking = get_ranking_from_scores(percent_score, ascending=False)
        
        # 计算Spearman相关
        fan_ranks = get_ranking_from_scores(fan_votes, ascending=False)
        judge_ranks = get_ranking_from_scores(judge_scores, ascending=False)
        
        names_list = list(contestants)
        r_fan = np.array([fan_ranks[n] for n in names_list])
        r_judge = np.array([judge_ranks[n] for n in names_list])
        r_rank = np.array([rank_ranking[n] for n in names_list])
        r_percent = np.array([percent_ranking[n] for n in names_list])
        
        if len(names_list) >= 3:
            rho1, _ = spearmanr(r_rank, r_fan)
            rho2, _ = spearmanr(r_percent, r_fan)
            rho3, _ = spearmanr(r_rank, r_judge)
            rho4, _ = spearmanr(r_percent, r_judge)
            
            all_rho['rho1'].append(rho1)
            all_rho['rho2'].append(rho2)
            all_rho['rho3'].append(rho3)
            all_rho['rho4'].append(rho4)
        
        # 检查两种机制是否产生不同的淘汰结果
        rank_eliminated = max(rank_ranking, key=rank_ranking.get)
        percent_eliminated = min(percent_ranking, key=percent_ranking.get)
        
        rank_results[week_key] = {
            'season': season, 'week': week,
            'ranking': rank_ranking,
            'eliminated': rank_eliminated
        }
        percent_results[week_key] = {
            'season': season, 'week': week,
            'ranking': percent_ranking,
            'eliminated': percent_eliminated
        }
        
        if rank_eliminated != percent_eliminated:
            reversal_weeks.append({
                'week_key': week_key,
                'season': season,
                'week': week,
                'rank_eliminated': rank_eliminated,
                'percent_eliminated': percent_eliminated,
                'contestants': contestants
            })
    
    # 汇总Spearman相关系数
    spearman_summary = {}
    for key, values in all_rho.items():
        if values:
            spearman_summary[key] = {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'n': len(values)
            }
    
    print(f"\nSpearman秩相关系数汇总:")
    print(f"  ρ1 (Rank vs Fan):     {spearman_summary.get('rho1', {}).get('mean', 'N/A'):.4f}")
    print(f"  ρ2 (Percent vs Fan):  {spearman_summary.get('rho2', {}).get('mean', 'N/A'):.4f}")
    print(f"  ρ3 (Rank vs Judge):   {spearman_summary.get('rho3', {}).get('mean', 'N/A'):.4f}")
    print(f"  ρ4 (Percent vs Judge):{spearman_summary.get('rho4', {}).get('mean', 'N/A'):.4f}")
    print(f"\n淘汰结果反转周数: {len(reversal_weeks)}")
    
    # 计算SPI和FII
    spi_fii = compute_spi_fii(fan_votes_results, total_scores, weekly_contestants)
    
    return {
        'spearman_correlations': spearman_summary,
        'reversal_weeks': reversal_weeks,
        'n_reversals': len(reversal_weeks),
        'spi_fii': spi_fii,
        'rank_results': rank_results,
        'percent_results': percent_results
    }


def compute_spi_fii(fan_votes_results, total_scores, weekly_contestants):
    """
    计算技能保护指数 (SPI) 和粉丝影响指数 (FII)
    
    SPI = 淘汰者中"评委分最低"的比例 → 机制保护技能的程度
    FII = 淘汰者中"粉丝票最低"的比例 → 机制受粉丝影响的程度
    
    对三种机制分别计算:
      - 排名制 (Rank)
      - 百分比制 (Percentage)
      - 排名制 + 评委救人 (Rank + Save)
    
    返回:
        dict: 三种机制的SPI和FII
    """
    # 累计计数
    mechanisms = {
        'rank': {'n_total': 0, 'n_skill': 0, 'n_fan': 0},
        'percentage': {'n_total': 0, 'n_skill': 0, 'n_fan': 0},
        'rank_with_save': {'n_total': 0, 'n_skill': 0, 'n_fan': 0},
    }
    
    for week_key, week_data in fan_votes_results['weekly_estimates'].items():
        season = week_data['season']
        week = week_data['week']
        contestants = week_data['contestants']
        fan_votes = week_data['estimated_fan_votes']
        
        if len(contestants) <= 2:
            continue
        
        # 获取评委分
        judge_scores = {}
        for name in contestants:
            score = total_scores.get((season, name, week), 0)
            judge_scores[name] = score
        
        if all(v == 0 for v in judge_scores.values()):
            continue
        
        # 找出评委分最低和粉丝票最低的选手
        min_judge_name = min(judge_scores, key=judge_scores.get)
        min_fan_name = min(fan_votes, key=fan_votes.get)
        
        # 排名制淘汰者
        rank_score = compute_rank_method_result(judge_scores, fan_votes)
        rank_eliminated = max(rank_score, key=rank_score.get)
        
        mechanisms['rank']['n_total'] += 1
        if rank_eliminated == min_judge_name:
            mechanisms['rank']['n_skill'] += 1
        if rank_eliminated == min_fan_name:
            mechanisms['rank']['n_fan'] += 1
        
        # 百分比制淘汰者
        percent_score = compute_percentage_method_result(judge_scores, fan_votes)
        percent_eliminated = min(percent_score, key=percent_score.get)
        
        mechanisms['percentage']['n_total'] += 1
        if percent_eliminated == min_judge_name:
            mechanisms['percentage']['n_skill'] += 1
        if percent_eliminated == min_fan_name:
            mechanisms['percentage']['n_fan'] += 1
        
        # 排名制 + 评委救人
        rank_ranking = get_ranking_from_scores(rank_score, ascending=True)
        # 找倒数两名
        sorted_names = sorted(rank_ranking, key=rank_ranking.get, reverse=True)
        bottom2 = sorted_names[:2]
        # 评委分低的那个被淘汰
        bottom2_scores = {n: judge_scores[n] for n in bottom2}
        save_eliminated = min(bottom2_scores, key=bottom2_scores.get)
        
        mechanisms['rank_with_save']['n_total'] += 1
        if save_eliminated == min_judge_name:
            mechanisms['rank_with_save']['n_skill'] += 1
        if save_eliminated == min_fan_name:
            mechanisms['rank_with_save']['n_fan'] += 1
    
    # 计算百分比
    results = {}
    for mech_name, counts in mechanisms.items():
        n = counts['n_total']
        if n > 0:
            results[mech_name] = {
                'SPI': round(counts['n_skill'] / n * 100, 1),
                'FII': round(counts['n_fan'] / n * 100, 1),
                'n_weeks': n
            }
    
    print(f"\nSPI / FII 指标:")
    for mech_name, data in results.items():
        print(f"  {mech_name:20s}: SPI={data['SPI']:5.1f}%, FII={data['FII']:5.1f}%")
    
    return results


def run_counterfactual_simulation(fan_votes_results):
    """
    争议案例反事实模拟
    
    对论文中提到的争议选手:
      - Jerry Rice (Season 2)
      - Billy Ray Cyrus (Season 4)
      - Bristol Palin (Season 11)
      - Bobby Bones (Season 27)
    
    模拟在不同机制下的最终排名变化
    """
    print("\n" + "=" * 60)
    print("争议案例反事实模拟")
    print("=" * 60)
    
    df = load_raw_data()
    total_scores, avg_scores = extract_weekly_judge_scores(df)
    weekly_contestants = get_weekly_contestants(df, total_scores)
    eliminations, placements = get_elimination_info(df)
    
    case_results = {}
    
    for case in CONTROVERSIAL_CASES:
        name = case['name']
        season = case['season']
        
        print(f"\n分析: {name} (Season {season})")
        
        # 找到该选手实际参赛的周
        contestant_weeks = []
        for (s, w), contestants in weekly_contestants.items():
            if s == season and name in contestants:
                contestant_weeks.append(w)
        
        if not contestant_weeks:
            print(f"  未找到选手数据")
            continue
        
        # 获取实际最终排名
        actual_placement = placements.get(season, {}).get(name, 'Unknown')
        print(f"  实际最终排名: {actual_placement}")
        
        # 逐周模拟
        weekly_tracking = []
        active_contestants = set(weekly_contestants.get((season, 1), []))
        
        for week in sorted(contestant_weeks):
            current_contestants = [c for c in active_contestants 
                                   if c in weekly_contestants.get((season, week), [])]
            
            if len(current_contestants) <= 1 or name not in current_contestants:
                break
            
            # 获取评委分和粉丝票
            judge_scores = {}
            fan_votes_dict = {}
            
            for c in current_contestants:
                judge_scores[c] = total_scores.get((season, c, week), 0)
            
            # 从Task1结果中获取粉丝票
            week_key = f"S{season}_W{week}"
            if week_key in fan_votes_results['weekly_estimates']:
                fan_votes_dict = fan_votes_results['weekly_estimates'][week_key]['estimated_fan_votes']
            
            # 排名制排名
            rank_score = compute_rank_method_result(judge_scores, fan_votes_dict)
            rank_ranking = get_ranking_from_scores(rank_score, ascending=True)
            rank_position = rank_ranking.get(name, len(current_contestants))
            
            # 百分比制排名
            percent_score = compute_percentage_method_result(judge_scores, fan_votes_dict)
            percent_ranking = get_ranking_from_scores(percent_score, ascending=False)
            percent_position = percent_ranking.get(name, len(current_contestants))
            
            # 排名制 + 评委救人
            rank_sorted = sorted(rank_ranking, key=rank_ranking.get, reverse=True)
            bottom2 = rank_sorted[:2]
            if name in bottom2:
                bottom2_scores = {n: judge_scores[n] for n in bottom2}
                save_eliminated = min(bottom2_scores, key=bottom2_scores.get)
                if save_eliminated == name:
                    active_contestants.discard(name)
                    save_position = rank_position
                else:
                    save_position = rank_position
            else:
                save_position = rank_position
            
            # 排名制淘汰
            rank_eliminated = max(rank_score, key=rank_score.get)
            
            # 百分比制淘汰
            percent_eliminated = min(percent_score, key=percent_score.get)
            
            weekly_tracking.append({
                'week': week,
                'n_contestants': len(current_contestants),
                'judge_score': judge_scores.get(name, 0),
                'fan_vote_share': fan_votes_dict.get(name, 0),
                'rank_position': rank_position,
                'percent_position': percent_position,
                'rank_eliminated': rank_eliminated,
                'percent_eliminated': percent_eliminated,
            })
            
            # 更新活跃选手（排名制淘汰）
            if rank_eliminated in active_contestants:
                active_contestants.discard(rank_eliminated)
        
        # 计算反事实最终排名
        # 在排名制下，该选手能走多远
        rank_final = len(weekly_tracking)
        for i, w in enumerate(weekly_tracking):
            if w['rank_eliminated'] == name:
                rank_final = i + 1
                break
        
        case_results[name] = {
            'season': season,
            'actual_placement': actual_placement,
            'weeks_competed': len(contestant_weeks),
            'weekly_tracking': weekly_tracking,
            'rank_final_week': rank_final
        }
        
        print(f"  参赛周数: {len(contestant_weeks)}")
        if weekly_tracking:
            print(f"  排名制下最高排名: {min(w['rank_position'] for w in weekly_tracking)}")
            print(f"  百分比制下最高排名: {min(w['percent_position'] for w in weekly_tracking)}")
    
    return case_results


def run_task3(fan_votes_results):
    """运行Task 3完整分析"""
    # 全赛季比较
    comparison = run_full_comparison(fan_votes_results)
    
    # 反事实模拟
    counterfactual = run_counterfactual_simulation(fan_votes_results)
    
    results = {
        'spearman_correlations': comparison['spearman_correlations'],
        'reversal_count': comparison['n_reversals'],
        'reversal_weeks': comparison['reversal_weeks'],
        'spi_fii': comparison['spi_fii'],
        'counterfactual_cases': counterfactual
    }
    
    print(f"\n{'=' * 60}")
    print("Task 3 完成!")
    print(f"{'=' * 60}")
    
    return results


if __name__ == '__main__':
    import json
    with open(f'{RESULT_DIR}/task1_fan_vote_reconstruction.json', 'r', encoding='utf-8') as f:
        task1_results = json.load(f)
    results = run_task3(task1_results)
    save_json(results, "task3_mechanism_comparison.json")