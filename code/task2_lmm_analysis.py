# -*- coding: utf-8 -*-
"""
Task 2: 双分支线性混合效应模型 (LMM) - 影响因素分析
------------------------------------------------------
优化版:
  - 使用 statsmodels MixedLM 直接接口，避免 formula 解析错误
  - 对地区和行业做适当聚合，减少分类变量维度
  - 正确提取 BLUP 随机效应
  - 含 Q() 公式接口作为备选
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from config import RESULT_DIR
from data_loader import (
    load_raw_data, extract_weekly_judge_scores, get_weekly_contestants,
    build_celebrity_features, save_json
)

import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.linalg


def prepare_lmm_data(fan_votes_results):
    """
    准备LMM所需的输入数据
    
    将每周的数据展开为 (选手, 周) 级别的观测记录
    """
    df = load_raw_data()
    total_scores, avg_scores = extract_weekly_judge_scores(df)
    weekly_contestants = get_weekly_contestants(df, total_scores)
    features_df, industry_map, partner_experience = build_celebrity_features(df)
    
    records = []
    
    for week_key, week_data in fan_votes_results['weekly_estimates'].items():
        season = week_data['season']
        week = week_data['week']
        contestants = week_data['contestants']
        fan_votes = week_data['estimated_fan_votes']
        
        for name in contestants:
            if name not in fan_votes:
                continue
            
            judge_score = avg_scores.get((season, name, week), 0)
            
            feat_row = features_df[
                (features_df['celebrity_name'] == name) & 
                (features_df['season'] == season)
            ]
            
            if len(feat_row) == 0:
                continue
            
            row = feat_row.iloc[0]
            
            # 简化地区分类
            homestate = str(row['celebrity_homestate']).strip()
            homecountry = str(row['celebrity_homecountry']).strip()
            
            if homecountry == 'United States' or homecountry == '':
                if homestate in ['California', 'New York', 'Texas', 'Florida']:
                    region = homestate
                else:
                    region = 'Other_US'
            else:
                region = 'International'
            
            # 简化行业分类
            industry = str(row['celebrity_industry']).strip()
            if industry in ['Social media personality', 'Social Media Personality']:
                industry = 'SocialMedia'
            elif industry in ['Sports Broadcaster', 'News Anchor', 'Radio Personality', 'Journalist']:
                industry = 'Media'
            elif industry in ['Fashion Designer', 'Producer', 'Musician', 'Motivational Speaker']:
                industry = 'OtherCreative'
            elif industry == 'Actor/Actress':
                industry = 'Actor'
            elif industry == 'Singer/Rapper':
                industry = 'Singer'
            elif industry == 'TV Personality':
                industry = 'TV'
            elif industry == 'Beauty Pagent':
                industry = 'Model'
            
            industry = industry.replace(' ', '_').replace('/', '_').replace('-', '_')
            
            record = {
                'season': season,
                'week': week,
                'celebrity_name': name,
                'ballroom_partner': row['ballroom_partner'],
                'celebrity_industry': industry,
                'celebrity_region': region,
                'celebrity_age': row['celebrity_age'],
                'partner_experience': row['partner_experience'],
                'judge_score': judge_score,
                'fan_vote': fan_votes.get(name, 0),
                'placement': row['placement']
            }
            records.append(record)
    
    lmm_df = pd.DataFrame(records)
    
    # 标准化连续变量
    for col in ['judge_score', 'fan_vote', 'celebrity_age', 'partner_experience']:
        if col in lmm_df.columns:
            mean_val = lmm_df[col].mean()
            std_val = lmm_df[col].std()
            if std_val > 0:
                lmm_df[f'{col}_std'] = (lmm_df[col] - mean_val) / std_val
            else:
                lmm_df[f'{col}_std'] = 0
    
    print(f"[LMM数据准备] 共 {len(lmm_df)} 条观测记录")
    print(f"  职业类型数: {lmm_df['celebrity_industry'].nunique()}")
    print(f"  地区类型数: {lmm_df['celebrity_region'].nunique()}")
    print(f"  舞伴数: {lmm_df['ballroom_partner'].nunique()}")
    
    return lmm_df


def create_design_matrix(lmm_df):
    """
    构建LMM的设计矩阵
    
    使用下划线命名法的哑变量，避免 formula 解析问题
    """
    # 选取主要行业和地区
    top_industries = lmm_df['celebrity_industry'].value_counts().head(6).index.tolist()
    top_regions = lmm_df['celebrity_region'].value_counts().head(5).index.tolist()
    
    df_model = lmm_df.copy().reset_index(drop=True)
    
    fixed_cols = []
    
    # 连续变量
    for col in ['celebrity_age_std', 'partner_experience_std']:
        if col in df_model.columns:
            fixed_cols.append(col)
    
    # 行业哑变量
    for ind in top_industries:
        safe_name = f'ind_{ind}'
        df_model[safe_name] = (df_model['celebrity_industry'] == ind).astype(int)
        fixed_cols.append(safe_name)
    
    # 地区哑变量
    for reg in top_regions:
        safe_name = f'reg_{reg}'
        df_model[safe_name] = (df_model['celebrity_region'] == reg).astype(int)
        fixed_cols.append(safe_name)
    
    # 添加截距
    df_model['intercept'] = 1.0
    fixed_cols.insert(0, 'intercept')
    
    return df_model, fixed_cols


def _extract_scalar(value, default=0.0):
    """从可能的标量/数组/DataFrame中提取标量"""
    if value is None:
        return default
    if hasattr(value, 'iloc'):
        return float(value.iloc[0])
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)


def _check_rank_deficiency(X, fixed_cols, tol=1e-10):
    """检查设计矩阵的秩，返回去除共线性列后的列名"""
    X = np.asarray(X, dtype=float)
    u, s, vh = np.linalg.svd(X)
    rank = np.sum(s > tol * s[0])
    if rank == len(fixed_cols):
        return fixed_cols
    
    # 使用QR分解找最大线性无关列集合
    q, r, p = scipy.linalg.qr(X, pivoting=True, mode='economic')
    independent_cols = [fixed_cols[i] for i in p[:rank]]
    return independent_cols


def extract_model_results(result, fixed_cols, target_col):
    """从 statsmodels MixedLM 结果中提取标准化结果（更健壮）"""
    # 固定效应
    fixed_effects = {}
    for i, col in enumerate(fixed_cols):
        fixed_effects[col] = float(result.params[i])
    
    # 随机效应 (BLUP)
    random_effects = {}
    re = result.random_effects
    for partner, effect in re.items():
        random_effects[partner] = _extract_scalar(effect)
    
    # AIC/BIC 处理
    n_obs = result.nobs
    n_params = len(result.params)
    llf = float(result.llf)
    aic = result.aic
    bic = result.bic
    if np.isnan(aic):
        aic = -2 * llf + 2 * n_params
    if np.isnan(bic):
        bic = -2 * llf + n_params * np.log(n_obs)
    
    # 组间方差 (处理标量/矩阵/数组)
    group_var = _extract_scalar(result.cov_re, default=0.0)
    residual_var = _extract_scalar(result.scale, default=1.0)
    
    return {
        'model_type': f'MixedLM direct ({target_col})',
        'fixed_effects': fixed_effects,
        'random_effects': random_effects,
        'log_likelihood': llf,
        'aic': float(aic),
        'bic': float(bic),
        'converged': bool(result.converged),
        'group_var': group_var,
        'residual_var': residual_var,
        'icc': group_var / (group_var + residual_var) if (group_var + residual_var) > 0 else 0,
        'optimization_message': str(result.mle_retvals.get('message', '')) if hasattr(result, 'mle_retvals') else ''
    }


def fit_lmm_direct(df_model, target_col, fixed_cols, group_col):
    """直接使用 MixedLM 接口拟合，含多优化器回退"""
    X = df_model[fixed_cols].values
    y = df_model[target_col].values
    groups = df_model[group_col].values
    
    # 去除共线性列
    independent_cols = _check_rank_deficiency(X, fixed_cols)
    if len(independent_cols) < len(fixed_cols):
        print(f"    去除 {len(fixed_cols) - len(independent_cols)} 个共线性列")
        X = df_model[independent_cols].values
    else:
        independent_cols = fixed_cols
    
    # 尝试多种优化策略
    strategies = [
        {'reml': True},
        {'reml': False},
        {'reml': True, 'method': ['Powell']},
        {'reml': False, 'method': ['Powell']},
    ]
    
    last_error = None
    for strategy in strategies:
        try:
            model = sm.MixedLM(y, X, groups=groups)
            result = model.fit(**strategy)
            if result.converged:
                print(f"    直接MixedLM收敛 (strategy={strategy})")
                return extract_model_results(result, independent_cols, target_col)
            else:
                last_error = f"未收敛: {strategy}"
        except Exception as e:
            last_error = str(e)
            continue
    
    print(f"  [直接MixedLM] 拟合失败: {last_error}")
    return None


def fit_lmm_formula(df_model, target_col, fixed_cols, group_col):
    """使用 formula 接口拟合（备选）"""
    try:
        safe_cols = []
        for col in fixed_cols:
            if col == 'intercept':
                continue
            safe_cols.append(f"Q('{col}')")
        
        formula = f"{target_col} ~ {' + '.join(safe_cols)}"
        model = smf.mixedlm(formula, df_model, groups=df_model[group_col])
        result = model.fit(reml=True)
        
        fixed_effects = {}
        for param, value in result.params.items():
            if param == 'Group Var':
                continue
            clean_param = param
            if param.startswith("Q('") and param.endswith("')"):
                clean_param = param[3:-2]
            elif param == 'Intercept':
                clean_param = 'intercept'
            fixed_effects[clean_param] = float(value)
        
        random_effects = {}
        re = result.random_effects
        for partner, effect in re.items():
            random_effects[partner] = float(effect.iloc[0])
        
        n_obs = result.nobs
        n_params = len(result.params)
        llf = float(result.llf)
        aic = result.aic if not np.isnan(result.aic) else -2 * llf + 2 * n_params
        bic = result.bic if not np.isnan(result.bic) else -2 * llf + n_params * np.log(n_obs)
        
        group_var = _extract_scalar(result.cov_re, default=0.0)
        residual_var = _extract_scalar(result.scale, default=1.0)
        
        return {
            'model_type': f'MixedLM formula ({target_col})',
            'fixed_effects': fixed_effects,
            'random_effects': random_effects,
            'log_likelihood': llf,
            'aic': float(aic),
            'bic': float(bic),
            'converged': bool(result.converged),
            'group_var': group_var,
            'residual_var': residual_var,
            'icc': group_var / (group_var + residual_var) if (group_var + residual_var) > 0 else 0
        }
    except Exception as e:
        print(f"  [Formula MixedLM] 拟合失败: {e}")
        return None


def fit_lmm_judge(lmm_df):
    """拟合评委分LMM"""
    print("  [Judge LMM] 构建设计矩阵...")
    df_model, fixed_cols = create_design_matrix(lmm_df)
    
    result = fit_lmm_direct(df_model, 'judge_score_std', fixed_cols, 'ballroom_partner')
    if result is None:
        print("  回退到 formula 接口...")
        result = fit_lmm_formula(df_model, 'judge_score_std', fixed_cols, 'ballroom_partner')
    if result is None:
        print("  回退到 OLS 近似")
        result = simplified_lmm(lmm_df, 'judge_score_std')
    
    return result


def fit_lmm_fan(lmm_df):
    """拟合粉丝投票LMM"""
    print("  [Fan LMM] 构建设计矩阵...")
    df_model, fixed_cols = create_design_matrix(lmm_df)
    
    result = fit_lmm_direct(df_model, 'fan_vote_std', fixed_cols, 'ballroom_partner')
    if result is None:
        print("  回退到 formula 接口...")
        result = fit_lmm_formula(df_model, 'fan_vote_std', fixed_cols, 'ballroom_partner')
    if result is None:
        print("  回退到 OLS 近似")
        result = simplified_lmm(lmm_df, 'fan_vote_std')
    
    return result


def simplified_lmm(lmm_df, target_col):
    """OLS简化版"""
    print(f"  [简化LMM] 使用OLS近似 {target_col}")
    
    df_model, fixed_cols = create_design_matrix(lmm_df)
    X = df_model[fixed_cols].values
    y = df_model[target_col].values
    
    model = sm.OLS(y, X)
    result = model.fit()
    
    fixed_effects = {}
    for i, col in enumerate(fixed_cols):
        fixed_effects[col] = float(result.params[i])
    
    random_effects = {}
    for partner in df_model['ballroom_partner'].unique():
        partner_data = df_model[df_model['ballroom_partner'] == partner]
        if len(partner_data) > 0:
            random_effects[partner] = float(partner_data[target_col].mean() - df_model[target_col].mean())
    
    return {
        'model_type': f'Simplified LMM ({target_col})',
        'fixed_effects': fixed_effects,
        'random_effects': random_effects,
        'log_likelihood': float(result.llf),
        'aic': float(result.aic),
        'bic': float(result.bic),
        'converged': True,
        'r_squared': float(result.rsquared)
    }


def analyze_coefficient_divergence(judge_result, fan_result):
    """分析评委和粉丝模型之间的系数差异"""
    judge_fe = judge_result.get('fixed_effects', {})
    fan_fe = fan_result.get('fixed_effects', {})
    
    divergence = {}
    common_params = set(judge_fe.keys()) & set(fan_fe.keys())
    for param in sorted(common_params):
        beta_judge = judge_fe[param]
        beta_fan = fan_fe[param]
        diff = beta_fan - beta_judge
        divergence[param] = {
            'beta_judge': round(beta_judge, 6),
            'beta_fan': round(beta_fan, 6),
            'diff': round(diff, 6),
            'favored_by': 'Fans' if diff > 0 else ('Judges' if diff < 0 else 'Neutral'),
            'abs_diff': round(abs(diff), 6)
        }
    
    return divergence


def analyze_partner_impact(judge_result, fan_result):
    """分析舞伴影响力"""
    judge_re = judge_result.get('random_effects', {})
    fan_re = fan_result.get('random_effects', {})
    
    partner_impact = {}
    all_partners = set(list(judge_re.keys()) + list(fan_re.keys()))
    
    for partner in all_partners:
        j_effect = judge_re.get(partner, 0)
        f_effect = fan_re.get(partner, 0)
        partner_impact[partner] = {
            'judge_effect': round(j_effect, 6),
            'fan_effect': round(f_effect, 6),
            'combined_effect': round(j_effect + f_effect, 6),
            'abs_combined': round(abs(j_effect + f_effect), 6)
        }
    
    sorted_partners = sorted(
        partner_impact.items(),
        key=lambda x: abs(x[1]['combined_effect']),
        reverse=True
    )
    
    return dict(sorted_partners[:20])


def run_task2(fan_votes_results):
    """运行Task 2完整分析"""
    print("=" * 60)
    print("Task 2: 双分支LMM影响因素分析")
    print("=" * 60)
    
    lmm_df = prepare_lmm_data(fan_votes_results)
    
    print("\n[1/2] 拟合评委分LMM...")
    judge_result = fit_lmm_judge(lmm_df)
    
    print("\n[2/2] 拟合粉丝投票LMM...")
    fan_result = fit_lmm_fan(lmm_df)
    
    print("\n分析偏好分歧...")
    divergence = analyze_coefficient_divergence(judge_result, fan_result)
    
    print("分析舞伴影响力...")
    partner_impact = analyze_partner_impact(judge_result, fan_result)
    
    results = {
        'judge_model': judge_result,
        'fan_model': fan_result,
        'coefficient_divergence': divergence,
        'partner_impact_top20': partner_impact,
        'data_summary': {
            'n_observations': len(lmm_df),
            'n_seasons': int(lmm_df['season'].nunique()),
            'n_contestants': int(lmm_df['celebrity_name'].nunique()),
            'n_partners': int(lmm_df['ballroom_partner'].nunique()),
            'n_industries': int(lmm_df['celebrity_industry'].nunique()),
        }
    }
    
    print(f"\n{'=' * 60}")
    print("关键发现 (偏好分歧分析):")
    print(f"{'=' * 60}")
    for param, data in sorted(divergence.items(), key=lambda x: abs(x[1]['diff']), reverse=True):
        if abs(data['diff']) > 0.01:
            print(f"  {param:35s}: β_judge={data['beta_judge']:+.4f}, "
                  f"β_fan={data['beta_fan']:+.4f}, "
                  f"diff={data['diff']:+.4f} → 偏向{data['favored_by']}")
    
    print(f"\n评委分模型 AIC: {judge_result.get('aic', 'N/A'):.2f}, ICC: {judge_result.get('icc', 'N/A'):.4f}")
    print(f"粉丝票模型 AIC: {fan_result.get('aic', 'N/A'):.2f}, ICC: {fan_result.get('icc', 'N/A'):.4f}")
    print(f"{'=' * 60}")
    
    return results


if __name__ == '__main__':
    import json
    with open(f'{RESULT_DIR}/task1_fan_vote_reconstruction.json', 'r', encoding='utf-8') as f:
        task1_results = json.load(f)
    results = run_task2(task1_results)
    save_json(results, "task2_lmm_analysis.json")