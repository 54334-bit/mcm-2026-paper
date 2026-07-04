# -*- coding: utf-8 -*-
"""
数据加载与预处理模块
负责读取CSV数据、清洗、特征工程，为后续所有任务提供统一的数据接口
"""

import pandas as pd
import numpy as np
import json
import os
from config import DATA_FILE, RESULT_DIR, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def load_raw_data():
    """
    读取原始CSV数据
    
    返回:
        DataFrame: 原始数据，421行 × 53列
    """
    df = pd.read_csv(DATA_FILE)
    print(f"[数据加载] 读取到 {df.shape[0]} 条记录, {df.shape[1]} 列")
    return df


def extract_weekly_judge_scores(df):
    """
    从原始数据中提取每周每位选手的评委总分
    
    数据说明:
        - 评委分范围为1-10
        - 部分周有多位评委（通常3-4位），取平均分后乘以评委数得到总分
        - N/A表示无该评委
        - 0分表示已淘汰选手
    
    返回:
        dict: {(season, celebrity_name, week): total_judge_score}
        dict: {(season, celebrity_name, week): avg_judge_score}
    """
    total_scores = {}
    avg_scores = {}
    
    for _, row in df.iterrows():
        season = int(row['season'])
        name = row['celebrity_name']
        
        for week in range(1, 12):  # 最多11周
            scores = []
            for judge in range(1, 5):  # 最多4位评委
                col = f'week{week}_judge{judge}_score'
                val = row.get(col, np.nan)
                if pd.notna(val) and str(val).upper() != 'N/A':
                    try:
                        s = float(val)
                        if s > 0:  # 0分表示淘汰，不计入
                            scores.append(s)
                    except (ValueError, TypeError):
                        pass
            
            if scores:
                avg = np.mean(scores)
                total = sum(scores)
                total_scores[(season, name, week)] = total
                avg_scores[(season, name, week)] = avg
            else:
                total_scores[(season, name, week)] = 0.0
                avg_scores[(season, name, week)] = 0.0
    
    return total_scores, avg_scores


def get_weekly_contestants(df, total_scores):
    """
    获取每季每周的活跃选手列表（评委分 > 0）
    
    返回:
        dict: {(season, week): [contestant_names]}
    """
    weekly_contestants = {}
    
    for season in sorted(df['season'].unique()):
        season_df = df[df['season'] == season]
        season = int(season)
        
        for week in range(1, 12):
            active = []
            for _, row in season_df.iterrows():
                name = row['celebrity_name']
                score = total_scores.get((season, name, week), 0.0)
                if score > 0.0:
                    active.append(name)
            
            if active:
                weekly_contestants[(season, week)] = active
    
    return weekly_contestants


def get_elimination_info(df):
    """
    获取每季每周的淘汰信息
    
    返回:
        dict: {(season, week): eliminated_contestant_name}
        dict: {season: {name: final_placement}}
    """
    eliminations = {}
    placements = {}
    
    for _, row in df.iterrows():
        season = int(row['season'])
        name = row['celebrity_name']
        result = row['results']
        placement = row['placement']
        
        if season not in placements:
            placements[season] = {}
        placements[season][name] = placement
        
        # 解析淘汰周
        if 'Eliminated Week' in str(result):
            week = int(str(result).replace('Eliminated Week', '').strip())
            eliminations[(season, week)] = name
    
    return eliminations, placements


def get_week_numbers_for_season(df, season):
    """
    获取某个赛季实际进行的周数（最后一组非零评委分所在的周）
    
    返回:
        int: 该赛季的周数
    """
    season_df = df[df['season'] == season]
    max_week = 0
    
    for _, row in season_df.iterrows():
        for week in range(1, 12):
            for judge in range(1, 5):
                col = f'week{week}_judge{judge}_score'
                val = row.get(col, np.nan)
                if pd.notna(val) and str(val).upper() != 'N/A':
                    try:
                        if float(val) > 0:
                            max_week = max(max_week, week)
                    except (ValueError, TypeError):
                        pass
    
    return max_week


def build_celebrity_features(df):
    """
    构建选手特征矩阵
    
    特征包括:
        - age: 年龄（标准化后）
        - age_raw: 原始年龄
        - industry: 职业分类（one-hot编码）
        - homestate: 家乡州
        - homecountry: 家乡国家
        - partner: 舞伴名称
        - partner_experience: 舞伴历史出场次数
    
    返回:
        DataFrame: 特征矩阵
        dict: 职业分类映射
        dict: 舞伴经验映射
    """
    features = []
    
    # 计算舞伴经验（历史出场次数）
    partner_count = {}
    for _, row in df.iterrows():
        partner = row['ballroom_partner']
        season = int(row['season'])
        if partner not in partner_count:
            partner_count[partner] = []
        partner_count[partner].append(season)
    
    partner_experience = {}
    for partner, seasons in partner_count.items():
        partner_experience[partner] = len(seasons)
    
    # 构建特征
    for _, row in df.iterrows():
        feat = {
            'celebrity_name': row['celebrity_name'],
            'ballroom_partner': row['ballroom_partner'],
            'celebrity_industry': row['celebrity_industry'],
            'celebrity_homestate': row['celebrity_homestate'] if pd.notna(row['celebrity_homestate']) else 'Unknown',
            'celebrity_homecountry': row['celebrity_homecountry/region'] if pd.notna(row['celebrity_homecountry/region']) else 'Unknown',
            'celebrity_age': row['celebrity_age_during_season'],
            'season': int(row['season']),
            'placement': row['placement'],
            'results': row['results'],
            'partner_experience': partner_experience.get(row['ballroom_partner'], 0),
        }
        features.append(feat)
    
    features_df = pd.DataFrame(features)
    
    # 职业分类映射
    industry_map = {ind: i for i, ind in enumerate(features_df['celebrity_industry'].unique())}
    
    return features_df, industry_map, partner_experience


def compute_industry_popularity(df, features_df):
    """
    计算行业人气分数 S_ind
    基于各行业选手在历史中的平均排名
    
    返回:
        dict: {industry: popularity_score}
    """
    industry_avg_placement = {}
    
    for industry in features_df['celebrity_industry'].unique():
        ind_df = features_df[features_df['celebrity_industry'] == industry]
        avg_place = ind_df['placement'].mean()
        # 排名越小越好，转换为正向分数（最大排名 + 1 - 平均排名）
        max_place = features_df['placement'].max()
        industry_avg_placement[industry] = max_place + 1 - avg_place
    
    # 归一化到 [0, 1]
    max_val = max(industry_avg_placement.values())
    if max_val > 0:
        for k in industry_avg_placement:
            industry_avg_placement[k] /= max_val
    
    return industry_avg_placement


def compute_region_popularity(features_df):
    """
    计算地区人气分数 S_loc
    基于各州/国家选手的出现频率
    
    返回:
        dict: {region: frequency_based_score}
    """
    region_count = {}
    
    for _, row in features_df.iterrows():
        state = row['celebrity_homestate']
        if state and state != 'Unknown':
            region_count[state] = region_count.get(state, 0) + 1
    
    max_freq = max(region_count.values()) if region_count else 1
    
    return {r: c / max_freq for r, c in region_count.items()}


def compute_age_advantage(features_df):
    """
    计算年龄优势分数 S_age
    假设年轻选手更有活力，更受观众欢迎
    S_age = 1 - (age - min_age) / (max_age - min_age)
    
    返回:
        dict: {celebrity_name: age_score} (同一人跨季情况取平均)
    """
    ages = features_df[['celebrity_name', 'celebrity_age']].drop_duplicates()
    min_age = ages['celebrity_age'].min()
    max_age = ages['celebrity_age'].max()
    
    age_range = max_age - min_age
    if age_range == 0:
        age_range = 1
    
    age_scores = {}
    for _, row in ages.iterrows():
        age_scores[row['celebrity_name']] = 1 - (row['celebrity_age'] - min_age) / age_range
    
    return age_scores


def save_json(data, filename):
    """保存数据为JSON文件"""
    filepath = os.path.join(RESULT_DIR, filename)
    
    # 处理numpy类型
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (set,)):
            return list(obj)
        if isinstance(obj, tuple):
            return str(obj)
        return obj
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=convert)
    
    print(f"[保存] {filepath}")


if __name__ == '__main__':
    # 测试数据加载
    df = load_raw_data()
    total_scores, avg_scores = extract_weekly_judge_scores(df)
    weekly_contestants = get_weekly_contestants(df, total_scores)
    eliminations, placements = get_elimination_info(df)
    
    print(f"\n[数据概览]")
    print(f"  总赛季数: {df['season'].nunique()}")
    print(f"  总选手数: {df['celebrity_name'].nunique()}")
    print(f"  总舞伴数: {df['ballroom_partner'].nunique()}")
    
    # 测试数据保存
    test_data = {
        "total_seasons": int(df['season'].nunique()),
        "total_contestants": int(df['celebrity_name'].nunique()),
        "total_partners": int(df['ballroom_partner'].nunique()),
    }
    save_json(test_data, "data_summary.json")
    print("\n数据加载测试完成！")