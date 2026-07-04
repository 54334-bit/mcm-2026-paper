# -*- coding: utf-8 -*-
"""
2026 MCM Problem C: DWTS 建模项目 - 主入口（优化版）
=======================================================
完整建模流程:
  Task 1: 粉丝投票逆向估算 (蒙特卡洛 + 贝叶斯 + Dirichlet)
  Task 2: 双分支LMM影响因素分析（真实MixedLM）
  Task 3: 计分机制比较与反事实模拟
  Task 4: DASS动态自适应评分系统设计（参数可扫）
  Task 5: 敏感性分析（真实蒙特卡洛重跑）
  Task 6: 完整赛季反事实传播模拟
  Task 7: α参数迭代优化 + DASS参数扫描

使用方法:
  python main.py              # 运行所有任务（除优化扫描）
  python main.py --optimize   # 包含参数优化扫描（耗时较长）
  python main.py --task 1     # 仅运行Task 1
"""

import sys
import json
import os
import time
import argparse

from config import RESULT_DIR
from data_loader import save_json

# 确保结果目录存在
os.makedirs(RESULT_DIR, exist_ok=True)


def load_task1_results():
    """加载Task 1结果"""
    filepath = os.path.join(RESULT_DIR, 'task1_fan_vote_reconstruction.json')
    if not os.path.exists(filepath):
        print(f"[错误] Task 1结果文件不存在: {filepath}")
        print("   请先运行 Task 1: python main.py --task 1")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_optimized_params():
    """加载优化后的参数"""
    alpha_file = os.path.join(RESULT_DIR, 'alpha_optimization.json')
    if os.path.exists(alpha_file):
        with open(alpha_file, 'r', encoding='utf-8') as f:
            alpha_data = json.load(f)
        return alpha_data
    return None


def run_all(include_optimization=False):
    """运行完整建模流程"""
    total_start = time.time()
    
    print("=" * 70)
    print("  2026 MCM Problem C: DWTS 建模项目")
    print("  Team #2625519 复现 - 优化版")
    print("=" * 70)
    
    # ==================== Task 1 ====================
    print("\n" + "▶" * 35)
    print("  阶段 1/6: 粉丝投票逆向估算")
    print("▶" * 35)
    
    from task1_fan_vote_reconstruction import run_task1
    t1_start = time.time()
    task1_results = run_task1()
    t1_time = time.time() - t1_start
    print(f"  Task 1 耗时: {t1_time:.1f}秒")
    
    # ==================== Task 2 ====================
    print("\n" + "▶" * 35)
    print("  阶段 2/6: 双分支LMM影响因素分析")
    print("▶" * 35)
    
    from task2_lmm_analysis import run_task2
    t2_start = time.time()
    task2_results = run_task2(task1_results)
    save_json(task2_results, "task2_lmm_analysis.json")
    t2_time = time.time() - t2_start
    print(f"  Task 2 耗时: {t2_time:.1f}秒")
    
    # ==================== Task 3 ====================
    print("\n" + "▶" * 35)
    print("  阶段 3/6: 计分机制比较与反事实模拟")
    print("▶" * 35)
    
    from task3_mechanism_comparison import run_task3
    t3_start = time.time()
    task3_results = run_task3(task1_results)
    save_json(task3_results, "task3_mechanism_comparison.json")
    t3_time = time.time() - t3_start
    print(f"  Task 3 耗时: {t3_time:.1f}秒")
    
    # ==================== Task 4 ====================
    print("\n" + "▶" * 35)
    print("  阶段 4/6: DASS动态自适应评分系统")
    print("▶" * 35)
    
    from task4_dass_system import run_task4
    t4_start = time.time()
    task4_results = run_task4(task1_results)
    save_json(task4_results, "task4_dass_system.json")
    t4_time = time.time() - t4_start
    print(f"  Task 4 耗时: {t4_time:.1f}秒")
    
    # ==================== Task 5: 敏感性分析 ====================
    print("\n" + "▶" * 35)
    print("  阶段 5/6: 敏感性分析")
    print("▶" * 35)
    
    from sensitivity_analysis import run_sensitivity_analysis
    t5_start = time.time()
    sensitivity_results = run_sensitivity_analysis(task1_results)
    save_json(sensitivity_results, "sensitivity_analysis.json")
    t5_time = time.time() - t5_start
    print(f"  敏感性分析 耗时: {t5_time:.1f}秒")
    
    # ==================== Task 6: 完整赛季反事实模拟 ====================
    print("\n" + "▶" * 35)
    print("  阶段 6/6: 完整赛季反事实传播模拟")
    print("▶" * 35)
    
    from counterfactual_season_sim import run_counterfactual_full_season
    t6_start = time.time()
    counterfactual_results = run_counterfactual_full_season(task1_results)
    save_json(counterfactual_results, "counterfactual_full_season.json")
    t6_time = time.time() - t6_start
    print(f"  完整赛季反事实模拟 耗时: {t6_time:.1f}秒")
    
    # ==================== 可选：参数优化扫描 ====================
    if include_optimization:
        print("\n" + "▶" * 35)
        print("  附加阶段: α参数优化 + DASS参数扫描")
        print("▶" * 35)
        
        from alpha_optimizer import run_alpha_optimization
        topt_start = time.time()
        alpha_opt_results = run_alpha_optimization()
        save_json(alpha_opt_results, "alpha_optimization.json")
        topt_time = time.time() - topt_start
        print(f"  α参数优化 耗时: {topt_time:.1f}秒")
        
        from parameter_sweep import run_dass_parameter_sweep
        sweep_start = time.time()
        sweep_results = run_dass_parameter_sweep(task1_results)
        save_json(sweep_results, "dass_parameter_sweep.json")
        sweep_time = time.time() - sweep_start
        print(f"  DASS参数扫描 耗时: {sweep_time:.1f}秒")
    else:
        alpha_opt_results = load_optimized_params()
        sweep_results = None
    
    # ==================== 汇总报告 ====================
    total_time = time.time() - total_start
    
    summary = {
        'project': '2026 MCM Problem C: DWTS',
        'team': '2625519 (复现优化版)',
        'execution_time_seconds': round(total_time, 1),
        'include_optimization': include_optimization,
        'task_results': {
            'task1_fan_vote_reconstruction': {
                'global_median_cv': task1_results['global_metrics']['global_median_cv'],
                'total_weeks_simulated': task1_results['global_metrics']['total_weeks_simulated'],
                'time_seconds': round(t1_time, 1)
            },
            'task2_lmm_analysis': {
                'n_observations': task2_results['data_summary']['n_observations'],
                'judge_model_type': task2_results['judge_model'].get('model_type', 'N/A'),
                'fan_model_type': task2_results['fan_model'].get('model_type', 'N/A'),
                'time_seconds': round(t2_time, 1)
            },
            'task3_mechanism_comparison': {
                'rho1_rank_vs_fan': task3_results['spearman_correlations'].get('rho1', {}).get('mean', 'N/A'),
                'rho2_percent_vs_fan': task3_results['spearman_correlations'].get('rho2', {}).get('mean', 'N/A'),
                'n_reversals': task3_results['reversal_count'],
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
                'time_seconds': round(t5_time, 1)
            },
            'counterfactual_full_season': {
                'spi_rank': counterfactual_results['spi_fii'].get('rank', {}).get('SPI', 'N/A'),
                'spi_percentage': counterfactual_results['spi_fii'].get('percentage', {}).get('SPI', 'N/A'),
                'spi_dass': counterfactual_results['spi_fii'].get('dass', {}).get('SPI', 'N/A'),
                'time_seconds': round(t6_time, 1)
            }
        }
    }
    
    if alpha_opt_results:
        summary['optimized_params'] = {
            'k1': alpha_opt_results['recommended_k1'],
            'k2': alpha_opt_results['recommended_k2'],
            'match_rate': alpha_opt_results['recommended_mean_match_rate']
        }
    
    if sweep_results:
        summary['dass_sweep'] = {
            'best_spi': sweep_results['best_params']['spi'],
            'best_fii': sweep_results['best_params']['fii'],
            'best_save_rate': sweep_results['best_params']['save_trigger_rate'],
            'best_params': sweep_results['best_params']['params'],
            'n_combinations': sweep_results['n_combinations']
        }
    
    save_json(summary, "summary_report.json")
    
    print("\n" + "=" * 70)
    print("  全部任务完成!")
    print(f"  总耗时: {total_time:.1f}秒")
    print(f"  结果文件保存在: {RESULT_DIR}")
    print("=" * 70)
    
    # 打印关键结果对比
    print("\n" + "=" * 70)
    print("  关键结果与论文对比")
    print("=" * 70)
    print(f"  {'指标':<35s} {'复现值':<20s} {'论文值':<15s}")
    print(f"  {'-'*70}")
    print(f"  {'全局中位CV':<35s} {task1_results['global_metrics']['global_median_cv']:<20.4f} {'0.293':<15s}")
    print(f"  {'ρ1 (Rank vs Fan)':<35s} {str(task3_results['spearman_correlations'].get('rho1', {}).get('mean', 'N/A')):<20s} {'0.8137':<15s}")
    print(f"  {'ρ2 (Percent vs Fan)':<35s} {str(task3_results['spearman_correlations'].get('rho2', {}).get('mean', 'N/A')):<20s} {'0.8891':<15s}")
    print(f"  {'SPI (DASS, 单周)':<35s} {str(task4_results['spi'])+'%':<20s} {'≥92%':<15s}")
    print(f"  {'FII (DASS, 单周)':<35s} {str(task4_results['fii'])+'%':<20s} {'48%-51%':<15s}")
    print(f"  {'评委救人触发率':<35s} {str(task4_results['save_trigger_rate'])+'%':<20s} {'15%':<15s}")
    print(f"  {'SPI (DASS, 完整赛季)':<35s} {str(counterfactual_results['spi_fii'].get('dass', {}).get('SPI', 'N/A'))+'%':<20s} {'≥92%':<15s}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='2026 MCM Problem C: DWTS 建模复现（优化版）')
    parser.add_argument('--task', type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                        help='指定运行的任务编号 (1-6, 7=参数优化)')
    parser.add_argument('--optimize', action='store_true',
                        help='包含参数优化扫描（耗时较长）')
    args = parser.parse_args()
    
    if args.task == 1:
        from task1_fan_vote_reconstruction import run_task1
        run_task1()
    elif args.task == 2:
        task1_results = load_task1_results()
        if task1_results:
            from task2_lmm_analysis import run_task2
            results = run_task2(task1_results)
            save_json(results, "task2_lmm_analysis.json")
    elif args.task == 3:
        task1_results = load_task1_results()
        if task1_results:
            from task3_mechanism_comparison import run_task3
            results = run_task3(task1_results)
            save_json(results, "task3_mechanism_comparison.json")
    elif args.task == 4:
        task1_results = load_task1_results()
        if task1_results:
            from task4_dass_system import run_task4
            results = run_task4(task1_results)
            save_json(results, "task4_dass_system.json")
    elif args.task == 5:
        task1_results = load_task1_results()
        if task1_results:
            from sensitivity_analysis import run_sensitivity_analysis
            results = run_sensitivity_analysis(task1_results)
            save_json(results, "sensitivity_analysis.json")
    elif args.task == 6:
        task1_results = load_task1_results()
        if task1_results:
            from counterfactual_season_sim import run_counterfactual_full_season
            results = run_counterfactual_full_season(task1_results)
            save_json(results, "counterfactual_full_season.json")
    elif args.task == 7:
        task1_results = load_task1_results()
        if task1_results:
            from alpha_optimizer import run_alpha_optimization
            alpha_results = run_alpha_optimization()
            save_json(alpha_results, "alpha_optimization.json")
            
            from parameter_sweep import run_dass_parameter_sweep
            sweep_results = run_dass_parameter_sweep(task1_results)
            save_json(sweep_results, "dass_parameter_sweep.json")
    else:
        run_all(include_optimization=args.optimize)


if __name__ == '__main__':
    main()