# -*- coding: utf-8 -*-
"""
参数扫描与结果对比框架
----------------------
自动运行多组参数配置，对比DASS的SPI/FII表现，寻找最优参数组合。

可调参数:
  - log_smooth_factor: 对数平滑因子
  - entropy_floor: 熵下限
  - min_judge_weight, max_judge_weight: 权重范围
  - fan_boost_for_low_scores: 低评委分粉丝票补偿
  - controversy_threshold: 评委救人触发阈值

输出:
  - 各参数组合下的 SPI, FII, Save触发率
  - 最优参数组合
"""

import numpy as np
from itertools import product
import time
import json
import os

from config import RESULT_DIR, RANDOM_SEED
from data_loader import save_json
from task4_dass_system import run_dass_simulation, DASS_PARAMS
from task1_fan_vote_reconstruction import reconstruct_fan_votes

np.random.seed(RANDOM_SEED)


def evaluate_dass_params(fan_votes_results, dass_params):
    """
    评估一组DASS参数
    
    返回:
        dict: {spi, fii, save_rate, in_symbiotic}
    """
    result = run_dass_simulation(fan_votes_results)
    
    return {
        'spi': result['spi'],
        'fii': result['fii'],
        'save_trigger_rate': result['save_trigger_rate'],
        'in_symbiotic': result['summary']['In_Symbiotic_Interval'] == 'Yes',
        'params': dass_params.copy()
    }


def grid_search_dass_params(fan_votes_results, param_grid):
    """
    网格扫描DASS参数
    
    参数:
        fan_votes_results: Task1结果
        param_grid: 参数字典 {param_name: [values]}
    
    返回:
        list: 所有组合的评估结果
        dict: 最优组合
    """
    print("=" * 60)
    print("DASS 参数网格扫描")
    print("=" * 60)
    
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    total = np.prod([len(v) for v in param_values])
    
    results = []
    best_spi = -1
    best_params = None
    
    start = time.time()
    
    for idx, values in enumerate(product(*param_values)):
        params = DASS_PARAMS.copy()
        for name, val in zip(param_names, values):
            params[name] = val
        
        print(f"\n[{idx+1}/{total}] 评估参数: {params}")
        
        # 临时修改DASS_PARAMS（通过导入引用修改全局变量）
        import task4_dass_system as t4
        old_params = t4.DASS_PARAMS.copy()
        t4.DASS_PARAMS = params
        
        result = evaluate_dass_params(fan_votes_results, params)
        
        # 恢复
        t4.DASS_PARAMS = old_params
        
        results.append(result)
        
        print(f"  SPI={result['spi']:.1f}%, FII={result['fii']:.1f}%, "
              f"SaveRate={result['save_trigger_rate']:.1f}%, "
              f"Symbiotic={result['in_symbiotic']}")
        
        # 优先考虑共生区间，其次SPI
        if result['in_symbiotic']:
            if result['spi'] > best_spi:
                best_spi = result['spi']
                best_params = result
        elif best_params is None and result['spi'] > best_spi:
            best_spi = result['spi']
            best_params = result
    
    elapsed = time.time() - start
    print(f"\n参数扫描完成，耗时 {elapsed:.1f}秒")
    print(f"最优参数: {best_params['params']}")
    print(f"  SPI={best_params['spi']:.1f}%, FII={best_params['fii']:.1f}%, "
          f"SaveRate={best_params['save_trigger_rate']:.1f}%")
    
    return results, best_params


def run_dass_parameter_sweep(fan_votes_results):
    """
    运行DASS参数扫描
    
    使用较小的参数网格，避免计算量过大
    """
    param_grid = {
        'log_smooth_factor': [0.5, 1.0, 2.0],
        'entropy_floor': [0.0, 0.05, 0.1],
        'min_judge_weight': [0.1, 0.2],
        'max_judge_weight': [0.8, 0.9],
        'fan_boost_for_low_scores': [0.0, 0.05, 0.1]
    }
    
    # 总组合数 = 3*3*2*2*3 = 108，可能较多，先缩小范围
    param_grid = {
        'log_smooth_factor': [0.5, 1.0, 2.0],
        'entropy_floor': [0.0, 0.05],
        'min_judge_weight': [0.15],
        'max_judge_weight': [0.85],
        'fan_boost_for_low_scores': [0.0, 0.05]
    }
    
    results, best = grid_search_dass_params(fan_votes_results, param_grid)
    
    return {
        'all_results': results,
        'best_params': best,
        'n_combinations': len(results)
    }


if __name__ == '__main__':
    import json
    
    # 加载Task1结果
    with open(f'{RESULT_DIR}/task1_fan_vote_reconstruction.json', 'r', encoding='utf-8') as f:
        task1_results = json.load(f)
    
    results = run_dass_parameter_sweep(task1_results)
    save_json(results, "dass_parameter_sweep.json")