# -*- coding: utf-8 -*-
"""
优化后完整流程运行脚本
----------------------
根据参数搜索结果，自动更新配置并运行完整建模流程，
生成最终结果与对比报告。

运行步骤:
  1. 读取 Task1 多目标参数搜索结果
  2. 更新 config.py 中的 Dirichlet 参数
  3. 用高模拟次数重建粉丝投票 (Task1)
  4. 运行 LMM 分析 (Task2)
  5. 运行机制对比 (Task3)
  6. 运行 DASS 系统 (Task4，使用固定权重最佳配置)
  7. 运行敏感性分析
  8. 运行完整赛季反事实模拟
  9. 生成 summary_report.json
"""

import os
import sys
import json
import time

# 确保在项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from config import RESULT_DIR, RANDOM_SEED, N_SIMULATIONS
from data_loader import save_json
import numpy as np

np.random.seed(RANDOM_SEED)


def update_config_py(new_params):
    """更新 config.py 中的 K1/K2/OMEGA 参数"""
    config_path = os.path.join(BASE_DIR, 'config.py')
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = _replace_config_value(content, 'OMEGA_1', new_params['omega1'])
    content = _replace_config_value(content, 'OMEGA_2', new_params['omega2'])
    content = _replace_config_value(content, 'OMEGA_3', new_params['omega3'])
    content = _replace_config_value(content, 'K1', new_params['K1'])
    content = _replace_config_value(content, 'K2', new_params['K2'])
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[配置更新] config.py 已更新:")
    print(f"  K1={new_params['K1']:.4f}, K2={new_params['K2']:.4f}")
    print(f"  ω=({new_params['omega1']:.4f}, {new_params['omega2']:.4f}, {new_params['omega3']:.4f})")


def _replace_config_value(content, key, value):
    """替换 config.py 中指定 key 的值"""
    import re
    pattern = rf"({key}\s*=\s*)[\d.]+"
    replacement = rf"\g<1>{value:.10g}"
    return re.sub(pattern, replacement, content)


def load_best_task1_params():
    """加载 Task1 多目标搜索的最佳参数"""
    result_file = os.path.join(RESULT_DIR, 'task1_multi_objective_exploration_v2.json')
    if not os.path.exists(result_file):
        # 回退到 v1
        result_file = os.path.join(RESULT_DIR, 'task1_multi_objective_exploration.json')
    
    if not os.path.exists(result_file):
        print("[警告] 未找到 Task1 参数搜索结果，使用当前 config.py 参数")
        from config import K1, K2, OMEGA_1, OMEGA_2, OMEGA_3
        return {
            'K1': K1, 'K2': K2,
            'omega1': OMEGA_1, 'omega2': OMEGA_2, 'omega3': OMEGA_3,
            'source': 'config_fallback'
        }
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 优先选择 good_candidates 中 overlap 最高的
    good = data.get('good_candidates', [])
    if good:
        best = good[0]
        source = 'good_candidate'
    else:
        best = data.get('best_overall', data.get('best_cv'))
        source = 'best_overall'
    
    best['source'] = source
    return best


def run_task1_optimized(params, n_sims=None):
    """运行优化版 Task1"""
    from task1_optimized_v2 import run_task1_single_config, get_cached_data
    
    if n_sims is None:
        n_sims = N_SIMULATIONS
    
    print("\n" + "=" * 70)
    print(f"Task1: 使用优化参数重建粉丝投票 (n_sims={n_sims})")
    print("=" * 70)
    
    # 预热缓存
    get_cached_data()
    
    start = time.time()
    result = run_task1_single_config(
        params['K1'], params['K2'],
        params['omega1'], params['omega2'], params['omega3'],
        n_sims=n_sims
    )
    elapsed = time.time() - start
    
    print(f"Task1 完成: CV={result['median_cv']:.4f}, "
          f"match={result['match_rate']:.4f}, 耗时={elapsed:.1f}s")
    
    return result, elapsed


def run_full_pipeline():
    """运行完整优化流程"""
    overall_start = time.time()
    
    # 1. 加载最佳 Task1 参数
    best_params = load_best_task1_params()
    print(f"[参数来源] {best_params['source']}")
    
    # 2. 更新 config.py
    update_config_py(best_params)
    
    # 3. 重新加载 config 模块（因为已修改）
    import importlib
    import config
    importlib.reload(config)
    
    # 4. 运行 Task1（生成完整结果文件）
    # 注意：task1_optimized_v2 的 run_task1_single_config 只返回汇总，不保存 weekly_estimates
    # 因此需要调用 task1_fan_vote_reconstruction.reconstruct_fan_votes() 来生成完整 JSON
    print("\n" + "=" * 70)
    print(f"Task1: 完整粉丝投票重建 (n_sims={config.N_SIMULATIONS})")
    print("=" * 70)
    
    from task1_fan_vote_reconstruction import reconstruct_fan_votes
    t1_start = time.time()
    task1_results = reconstruct_fan_votes()
    save_json(task1_results, "task1_fan_vote_reconstruction.json")
    t1_time = time.time() - t1_start
    
    # 计算 overlap
    from data_loader import load_raw_data, extract_weekly_judge_scores
    df = load_raw_data()
    total_scores, _ = extract_weekly_judge_scores(df)
    overlap_count = 0
    valid_weeks = 0
    for week_key, week_data in task1_results['weekly_estimates'].items():
        contestants = week_data['contestants']
        fan_votes = week_data['estimated_fan_votes']
        season = week_data['season']
        week = week_data['week']
        judge_scores = {name: total_scores.get((season, name, week), 0) for name in contestants}
        if all(v == 0 for v in judge_scores.values()):
            continue
        min_judge = min(judge_scores, key=judge_scores.get)
        min_fan = min(fan_votes, key=fan_votes.get)
        if min_judge == min_fan:
            overlap_count += 1
        valid_weeks += 1
    overlap_rate = overlap_count / valid_weeks if valid_weeks > 0 else 0
    
    print(f"\nTask1 overlap: {overlap_count}/{valid_weeks} = {overlap_rate:.3f}")
    
    # 5. Task2 LMM
    print("\n" + "=" * 70)
    print("Task2: LMM 分析")
    print("=" * 70)
    from task2_lmm_analysis import run_task2
    t2_start = time.time()
    task2_results = run_task2(task1_results)
    save_json(task2_results, "task2_lmm_analysis.json")
    t2_time = time.time() - t2_start
    
    # 6. Task3 机制对比
    print("\n" + "=" * 70)
    print("Task3: 机制对比")
    print("=" * 70)
    from task3_mechanism_comparison import run_full_comparison
    t3_start = time.time()
    task3_results = run_full_comparison(task1_results)
    save_json(task3_results, "task3_mechanism_comparison.json")
    t3_time = time.time() - t3_start
    
    # 7. Task4 DASS（使用固定权重最佳配置）
    print("\n" + "=" * 70)
    print("Task4: DASS 系统")
    print("=" * 70)
    import task4_dass_system as t4
    from task4_dass_system import run_task4
    
    # 设置最佳DASS参数
    best_dass = {
        'use_entropy': False,
        'fixed_judge_weight': 0.85,
        'min_judge_weight': 0.7,
        'max_judge_weight': 0.95,
        'log_smooth_factor': 0.5,
        'entropy_floor': 0.05,
        'fan_boost_for_low_scores': 0.15
    }
    old_dass = t4.DASS_PARAMS.copy()
    old_ct = t4.CONTROVERSY_THRESHOLD
    t4.DASS_PARAMS = best_dass
    t4.CONTROVERSY_THRESHOLD = 2
    
    t4_start = time.time()
    task4_results = run_task4(task1_results)
    save_json(task4_results, "task4_dass_system.json")
    t4_time = time.time() - t4_start
    
    t4.DASS_PARAMS = old_dass
    t4.CONTROVERSY_THRESHOLD = old_ct
    
    # 8. 敏感性分析
    print("\n" + "=" * 70)
    print("敏感性分析")
    print("=" * 70)
    from sensitivity_analysis import (
        analyze_alpha_robustness, analyze_noise_robustness, analyze_weight_sensitivity
    )
    sa_start = time.time()
    alpha_results = analyze_alpha_robustness(task1_results, sample_weeks=20)
    noise_results = analyze_noise_robustness(task1_results)
    weight_results = analyze_weight_sensitivity(task1_results)
    sensitivity_results = {
        'alpha_robustness': alpha_results,
        'noise_robustness': noise_results,
        'weight_sensitivity': weight_results
    }
    save_json(sensitivity_results, "sensitivity_analysis.json")
    sa_time = time.time() - sa_start
    
    # 9. 完整赛季反事实模拟
    print("\n" + "=" * 70)
    print("完整赛季反事实模拟")
    print("=" * 70)
    from counterfactual_season_sim import run_counterfactual_full_season
    cf_start = time.time()
    cf_results = run_counterfactual_full_season(task1_results)
    save_json(cf_results, "counterfactual_full_season.json")
    cf_time = time.time() - cf_start
    
    # 10. 生成总结报告
    summary = {
        'project': '2026 MCM Problem C: DWTS',
        'team': '2625519 (复现优化版)',
        'execution_time_seconds': round(time.time() - overall_start, 1),
        'include_optimization': True,
        'task1_params': {
            'K1': best_params['K1'],
            'K2': best_params['K2'],
            'omega1': best_params['omega1'],
            'omega2': best_params['omega2'],
            'omega3': best_params['omega3'],
            'source': best_params['source']
        },
        'task_results': {
            'task1_fan_vote_reconstruction': {
                'global_median_cv': task1_results['global_metrics']['global_median_cv'],
                'global_mean_cv': task1_results['global_metrics']['global_mean_cv'],
                'total_weeks_simulated': task1_results['global_metrics']['total_weeks_simulated'],
                'overlap_rate': round(overlap_rate, 4),
                'time_seconds': round(t1_time, 1)
            },
            'task2_lmm_analysis': {
                'n_observations': task2_results['data_summary']['n_observations'],
                'judge_model_type': task2_results['judge_model']['model_type'],
                'fan_model_type': task2_results['fan_model']['model_type'],
                'time_seconds': round(t2_time, 1)
            },
            'task3_mechanism_comparison': {
                'rho1_rank_vs_fan': task3_results['spearman_correlations']['rho1']['mean'],
                'rho2_percent_vs_fan': task3_results['spearman_correlations']['rho2']['mean'],
                'n_reversals': task3_results['n_reversals'],
                'time_seconds': round(t3_time, 1)
            },
            'task4_dass_system': {
                'SPI': task4_results['spi'],
                'FII': task4_results['fii'],
                'save_trigger_rate': task4_results['save_trigger_rate'],
                'in_symbiotic': task4_results['summary']['In_Symbiotic_Interval'],
                'time_seconds': round(t4_time, 1)
            },
            'sensitivity_analysis': {
                'alpha_delta_cv': sensitivity_results['alpha_robustness']['delta_cv'],
                'time_seconds': round(sa_time, 1)
            },
            'counterfactual_full_season': {
                'spi_rank': cf_results['spi_fii']['rank']['SPI'],
                'spi_percentage': cf_results['spi_fii']['percentage']['SPI'],
                'spi_dass': cf_results['spi_fii']['dass']['SPI'],
                'fii_rank': cf_results['spi_fii']['rank']['FII'],
                'fii_percentage': cf_results['spi_fii']['percentage']['FII'],
                'fii_dass': cf_results['spi_fii']['dass']['FII'],
                'time_seconds': round(cf_time, 1)
            }
        }
    }
    
    save_json(summary, "summary_report.json")
    
    print("\n" + "=" * 70)
    print("优化流程全部完成!")
    print("=" * 70)
    print(f"总耗时: {summary['execution_time_seconds']:.1f}s")
    print(f"Task1 CV: {summary['task_results']['task1_fan_vote_reconstruction']['global_median_cv']:.4f}")
    print(f"Task1 overlap: {summary['task_results']['task1_fan_vote_reconstruction']['overlap_rate']:.3f}")
    print(f"Task4 DASS: SPI={task4_results['spi']}%, FII={task4_results['fii']}%, "
          f"共生区间={task4_results['summary']['In_Symbiotic_Interval']}")
    
    return summary


if __name__ == '__main__':
    summary = run_full_pipeline()
