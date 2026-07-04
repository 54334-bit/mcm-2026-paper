# -*- coding: utf-8 -*-
"""
完整赛季反事实传播模拟
------------------------
核心思想：
  对每种计分机制（Rank/Percentage/DASS）进行完整赛季模拟，
  即每周根据机制淘汰一人，然后将该选手移除，继续下周模拟。
  这样可以捕捉淘汰结果的传播效应，比单周SPI/FII更准确。

输出：
  - 每种机制下每季的最终排名
  - 争议案例的完整路径对比
  - 技能保护指数 SPI 和 粉丝影响指数 FII
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from config import RESULT_DIR, CONTROVERSIAL_CASES, RANDOM_SEED
from data_loader import (
    load_raw_data, extract_weekly_judge_scores, get_weekly_contestants,
    get_elimination_info, get_week_numbers_for_season, save_json
)
from task3_mechanism_comparison import (
    compute_rank_method_result, compute_percentage_method_result,
    get_ranking_from_scores
)
from task4_dass_system import topsis_scoring, judges_save_mechanism

np.random.seed(RANDOM_SEED)


def simulate_full_season(season, mechanism, fan_votes_results, total_scores, 
                          weekly_contestants, placements, dass_params=None):
    """
    对单个赛季进行完整模拟
    
    参数:
        season: 赛季编号
        mechanism: 机制名称 'rank' | 'percentage' | 'dass'
        fan_votes_results: Task1 粉丝投票结果
        total_scores: 评委分字典
        weekly_contestants: 每周活跃选手
        placements: 实际最终排名
        dass_params: DASS 参数（仅用于dass机制）
    
    返回:
        dict: 模拟结果
    """
    max_week = max([w for (s, w) in weekly_contestants.keys() if s == season], default=0)
    
    # 获取初始活跃选手
    active = set(weekly_contestants.get((season, 1), []))
    
    # 记录每周淘汰
    eliminations_sim = []
    weekly_rankings = []
    
    for week in range(1, max_week + 1):
        # 获取本周仍活跃的选手
        current = [c for c in active if c in weekly_contestants.get((season, week), [])]
        
        if len(current) <= 1:
            break
        
        # 获取评委分和粉丝票
        judge_scores = {name: total_scores.get((season, name, week), 0) for name in current}
        fan_votes = {}
        week_key = f"S{season}_W{week}"
        if week_key in fan_votes_results['weekly_estimates']:
            fan_votes = fan_votes_results['weekly_estimates'][week_key]['estimated_fan_votes']
        
        # 如果缺少某些选手的粉丝票，用均匀分布补全
        for name in current:
            if name not in fan_votes:
                fan_votes[name] = 1.0 / len(current)
        
        # 根据机制计算淘汰者
        if mechanism == 'rank':
            rank_score = compute_rank_method_result(judge_scores, fan_votes)
            eliminated = max(rank_score, key=rank_score.get)
        elif mechanism == 'percentage':
            percent_score = compute_percentage_method_result(judge_scores, fan_votes)
            eliminated = min(percent_score, key=percent_score.get)
        elif mechanism == 'dass':
            closeness = topsis_scoring(judge_scores, fan_votes, dass_params)
            eliminated, _, _ = judges_save_mechanism(judge_scores, fan_votes, closeness)
        else:
            eliminated = current[0]
        
        if eliminated in active:
            active.discard(eliminated)
        
        eliminations_sim.append({
            'week': week,
            'eliminated': eliminated,
            'n_contestants': len(current)
        })
        
        # 记录排名
        if mechanism == 'rank':
            rank_ranking = get_ranking_from_scores(rank_score, ascending=True)
            weekly_rankings.append({'week': week, 'ranking': rank_ranking})
        elif mechanism == 'percentage':
            percent_ranking = get_ranking_from_scores(percent_score, ascending=False)
            weekly_rankings.append({'week': week, 'ranking': percent_ranking})
        elif mechanism == 'dass':
            dass_ranking = get_ranking_from_scores(closeness, ascending=False)
            weekly_rankings.append({'week': week, 'ranking': dass_ranking})
    
    # 根据淘汰顺序确定最终排名
    # 淘汰越早排名越差，未被淘汰的排名基于最后一周
    final_ranking = {}
    eliminated_names = [e['eliminated'] for e in eliminations_sim]
    n_eliminated = len(eliminated_names)
    
    for i, name in enumerate(eliminated_names):
        # 第一个淘汰的 = 最差排名
        final_ranking[name] = n_eliminated - i + 1
    
    # 未被淘汰的选手（实际夺冠者）排名1
    for name in active:
        final_ranking[name] = 1
    
    return {
        'season': season,
        'mechanism': mechanism,
        'eliminations': eliminations_sim,
        'weekly_rankings': weekly_rankings,
        'final_ranking': final_ranking,
        'winner': list(active)[0] if len(active) == 1 else list(active)
    }


def run_full_season_counterfactual(fan_votes_results, mechanisms=None):
    """
    对所有赛季进行完整反事实模拟
    
    参数:
        fan_votes_results: Task1 结果
        mechanisms: 要模拟的机制列表
    
    返回:
        dict: 所有赛季所有机制的模拟结果
    """
    if mechanisms is None:
        mechanisms = ['rank', 'percentage', 'dass']
    
    print("=" * 60)
    print("完整赛季反事实传播模拟")
    print("=" * 60)
    
    df = load_raw_data()
    total_scores, avg_scores = extract_weekly_judge_scores(df)
    weekly_contestants = get_weekly_contestants(df, total_scores)
    eliminations, placements = get_elimination_info(df)
    
    seasons = sorted(df['season'].unique())
    
    all_results = {mech: {} for mech in mechanisms}
    
    for season in seasons:
        season = int(season)
        for mech in mechanisms:
            result = simulate_full_season(
                season, mech, fan_votes_results, total_scores,
                weekly_contestants, placements
            )
            all_results[mech][f"S{season}"] = result
    
    return all_results


def compute_spi_fii_full_season(all_results, placements, total_scores, weekly_contestants, fan_votes_results):
    """
    基于完整赛季模拟计算 SPI 和 FII
    
    定义：
      SPI = 模拟淘汰者中，该周实际评委分最低者的比例
      FII = 模拟淘汰者中，该周实际粉丝票最低者的比例
    
    注意：这里使用完整赛季模拟产生的淘汰序列
    """
    print("\n基于完整赛季模拟计算 SPI/FII...")
    
    metrics = {}
    
    for mech, season_results in all_results.items():
        n_total = 0
        n_skill = 0
        n_fan = 0
        
        for season_key, result in season_results.items():
            season = result['season']
            
            for elim in result['eliminations']:
                week = elim['week']
                eliminated = elim['eliminated']
                
                # 获取该周所有实际活跃选手
                current = weekly_contestants.get((season, week), [])
                if not current:
                    continue
                
                # 获取评委分
                judge_scores = {name: total_scores.get((season, name, week), 0) for name in current}
                if not judge_scores or all(v == 0 for v in judge_scores.values()):
                    continue
                
                # 获取粉丝票（从Task1结果）
                week_key = f"S{season}_W{week}"
                fan_votes = {}
                if week_key in fan_votes_results.get('weekly_estimates', {}):
                    fan_votes = fan_votes_results['weekly_estimates'][week_key]['estimated_fan_votes']
                # 补齐缺失选手
                for name in current:
                    if name not in fan_votes:
                        fan_votes[name] = 1.0 / len(current)
                
                min_judge = min(judge_scores, key=judge_scores.get)
                min_fan = min(fan_votes, key=fan_votes.get)
                
                n_total += 1
                if eliminated == min_judge:
                    n_skill += 1
                if eliminated == min_fan:
                    n_fan += 1
        
        spi = n_skill / n_total * 100 if n_total > 0 else 0
        fii = n_fan / n_total * 100 if n_total > 0 else 0
        
        metrics[mech] = {
            'SPI': round(spi, 1),
            'FII': round(fii, 1),
            'n_total': n_total,
            'n_skill': n_skill,
            'n_fan': n_fan
        }
        
        print(f"  {mech:12s}: SPI={spi:5.1f}%, FII={fii:5.1f}%  (n={n_total})")
    
    return metrics


def analyze_controversial_cases_full_season(all_results, placements):
    """
    分析争议案例在完整赛季模拟中的表现
    """
    print("\n争议案例完整赛季路径分析...")
    
    case_results = {}
    
    for case in CONTROVERSIAL_CASES:
        name = case['name']
        season = case['season']
        season_key = f"S{season}"
        actual_place = placements.get(season, {}).get(name, 'Unknown')
        
        case_results[name] = {
            'season': season,
            'actual_placement': actual_place,
            'simulated_placements': {}
        }
        
        for mech, season_results in all_results.items():
            if season_key in season_results:
                final_rank = season_results[season_key]['final_ranking'].get(name, 'Unknown')
                case_results[name]['simulated_placements'][mech] = final_rank
        
        print(f"  {name}: 实际排名={actual_place}, "
              f"Rank={case_results[name]['simulated_placements'].get('rank', 'N/A')}, "
              f"Percent={case_results[name]['simulated_placements'].get('percentage', 'N/A')}, "
              f"DASS={case_results[name]['simulated_placements'].get('dass', 'N/A')}")
    
    return case_results


def run_counterfactual_full_season(fan_votes_results):
    """运行完整赛季反事实模拟主函数"""
    all_results = run_full_season_counterfactual(fan_votes_results)
    
    df = load_raw_data()
    total_scores, avg_scores = extract_weekly_judge_scores(df)
    weekly_contestants = get_weekly_contestants(df, total_scores)
    eliminations, placements = get_elimination_info(df)
    
    spi_fii = compute_spi_fii_full_season(all_results, placements, total_scores, weekly_contestants, fan_votes_results)
    cases = analyze_controversial_cases_full_season(all_results, placements)
    
    results = {
        'season_simulations': all_results,
        'spi_fii': spi_fii,
        'controversial_cases': cases
    }
    
    print(f"\n{'=' * 60}")
    print("完整赛季反事实模拟完成!")
    print(f"{'=' * 60}")
    
    return results


if __name__ == '__main__':
    import json
    with open(f'{RESULT_DIR}/task1_fan_vote_reconstruction.json', 'r', encoding='utf-8') as f:
        task1_results = json.load(f)
    results = run_counterfactual_full_season(task1_results)
    save_json(results, "counterfactual_full_season.json")