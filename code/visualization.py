# -*- coding: utf-8 -*-
"""
visualization.py
----------------
Reproducible visualization utilities for the 2026 MCM Problem C paper figures.

Most figures in the original paper were produced with MATLAB. Since the original
MATLAB source is unavailable, this module re-implements the data-driven figures
using the Python model code and matplotlib / seaborn. Diagrammatic figures
(e.g., the overall workflow in Figure 1 and the LMM architecture in Figure 8)
are intentionally omitted because they are best maintained manually in a
drawing tool or LaTeX/TikZ.

Run this module directly to generate all supported figures in ../figures/:

    python visualization.py

Each public function below corresponds (loosely) to one figure or one group of
related figures in the final paper.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from config import RESULT_DIR, RANDOM_SEED, CONTROVERSIAL_CASES
from data_loader import (
    load_raw_data,
    extract_weekly_judge_scores,
    get_weekly_contestants,
    get_elimination_info,
    build_celebrity_features,
)

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['figure.dpi'] = 150
np.random.seed(RANDOM_SEED)

# Output directory for figures (project_root/figures)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def _savefig(name):
    """Save the current figure to the figures directory."""
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, bbox_inches='tight')
    print(f"[Figure saved] {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Comparison of eliminated players' average judge scores
# ---------------------------------------------------------------------------
def fig2_eliminated_vs_survivors():
    """
    Boxplot comparing weekly average judge scores of eliminated contestants
    against the season-wide average judge score.
    """
    df = load_raw_data()
    total_scores, avg_scores = extract_weekly_judge_scores(df)
    eliminations, _ = get_elimination_info(df)
    weekly_contestants = get_weekly_contestants(df, total_scores)

    eliminated_scores = []
    survivor_scores = []

    for (season, week), contestants in weekly_contestants.items():
        elim = eliminations.get((season, week))
        if elim is None:
            continue
        for name in contestants:
            score = avg_scores.get((season, name, week), 0)
            if score <= 0:
                continue
            if name == elim:
                eliminated_scores.append(score)
            else:
                survivor_scores.append(score)

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([eliminated_scores, survivor_scores],
                      labels=['Eliminated', 'Survivors'],
                      patch_artist=True,
                      showmeans=True,
                      meanline=True)
    for patch, color in zip(bp['boxes'], ['#e74c3c', '#2ecc71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Average Judge Score')
    ax.set_title('Figure 2: Judge Scores of Eliminated vs. Survivor Contestants')
    _savefig('fig2_eliminated_vs_survivors.png')


# ---------------------------------------------------------------------------
# Figures 3 & 4: Celebrity home-state and industry distributions
# ---------------------------------------------------------------------------
def fig3_fig4_demographics():
    """Bar charts for celebrity home states and industries."""
    df = load_raw_data()

    # Home states
    states = df['celebrity_homestate'].fillna('Unknown').replace('', 'Unknown')
    state_counts = Counter(states)
    # Keep top 15, group others
    top_states = state_counts.most_common(15)
    other_count = sum(c for _, c in state_counts.items() if c not in dict(top_states).values())
    labels, values = zip(*top_states)
    if other_count > 0:
        labels = list(labels) + ['Others']
        values = list(values) + [other_count]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels[::-1], values[::-1], color='steelblue')
    ax.set_xlabel('Number of Contestants')
    ax.set_title('Figure 3: Distribution of Celebrity Home States')
    for bar in bars:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(int(bar.get_width())), va='center', fontsize=8)
    _savefig('fig3_celebrity_home_states.png')

    # Industries
    industries = df['celebrity_industry'].fillna('Unknown')
    industry_counts = Counter(industries)
    labels2, values2 = zip(*industry_counts.most_common())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels2[::-1], values2[::-1], color='coral')
    ax.set_xlabel('Number of Contestants')
    ax.set_title('Figure 4: Distribution of Celebrity Industries')
    for bar in bars:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(int(bar.get_width())), va='center', fontsize=8)
    _savefig('fig4_celebrity_industries.png')


# ---------------------------------------------------------------------------
# Figure 5: Distribution of professional dancers by appearance count
# ---------------------------------------------------------------------------
def fig5_dancer_distribution():
    """Histogram of professional dancers' appearance frequencies."""
    df = load_raw_data()
    partner_counts = df['ballroom_partner'].value_counts()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(partner_counts.values, bins=range(1, partner_counts.max()+2),
            color='teal', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Number of Seasons Appeared')
    ax.set_ylabel('Number of Dancers')
    ax.set_title('Figure 5: Distribution of Professional Dancers')
    ax.set_xticks(range(1, partner_counts.max()+1))
    _savefig('fig5_dancer_distribution.png')


# ---------------------------------------------------------------------------
# Figure 7: Temporal Evolution of Model Uncertainty (CV over weeks)
# ---------------------------------------------------------------------------
def fig7_temporal_uncertainty():
    """
    Plot median coefficient of variation across weeks/seasons.
    Requires Task 1 results to be present.
    """
    result_path = os.path.join(RESULT_DIR, 'task1_fan_vote_reconstruction.json')
    if not os.path.exists(result_path):
        print('[Figure 7 skipped] Task 1 results not found. Run: python main.py --task 1')
        return

    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    records = []
    for week_key, wd in results.get('weekly_estimates', {}).items():
        cv_values = list(wd.get('cv_per_contestant', {}).values())
        if cv_values:
            records.append({
                'season': wd['season'],
                'week': wd['week'],
                'median_cv': np.median(cv_values),
                'mean_cv': np.mean(cv_values)
            })

    df_cv = pd.DataFrame(records).sort_values(['season', 'week'])
    df_cv['global_week'] = range(1, len(df_cv) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_cv['global_week'], df_cv['median_cv'], label='Median CV', color='#3498db')
    ax.plot(df_cv['global_week'], df_cv['mean_cv'], label='Mean CV', color='#e67e22', alpha=0.7)
    ax.axhline(0.293, color='red', linestyle='--', label='Paper reference CV = 0.293')
    ax.set_xlabel('Global Week Index')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Figure 7: Temporal Evolution of Model Uncertainty')
    ax.legend()
    _savefig('fig7_temporal_evolution_cv.png')


# ---------------------------------------------------------------------------
# Figure 9: Comparative fixed effects (Judge vs Fan LMM)
# ---------------------------------------------------------------------------
def fig9_fixed_effects_comparison():
    """Grouped bar chart of fixed-effect coefficients from Task 2."""
    result_path = os.path.join(RESULT_DIR, 'task2_lmm_analysis.json')
    if not os.path.exists(result_path):
        print('[Figure 9 skipped] Task 2 results not found. Run: python main.py --task 2')
        return

    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    judge_fe = results.get('judge_model', {}).get('fixed_effects', {})
    fan_fe = results.get('fan_model', {}).get('fixed_effects', {})

    common = sorted(set(judge_fe.keys()) & set(fan_fe.keys()))
    if not common:
        print('[Figure 9 skipped] No common fixed effects found.')
        return

    x = np.arange(len(common))
    width = 0.35
    judge_vals = [judge_fe[k] for k in common]
    fan_vals = [fan_fe[k] for k in common]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, judge_vals, width, label='Judge Model', color='#3498db')
    ax.bar(x + width/2, fan_vals, width, label='Fan Model', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=45, ha='right')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Standardized Coefficient')
    ax.set_title('Figure 9: Fixed Effects Comparison (Judge vs. Fan Models)')
    ax.legend()
    _savefig('fig9_fixed_effects_comparison.png')


# ---------------------------------------------------------------------------
# Figure 10: Professional dancer impact matrix (top partners)
# ---------------------------------------------------------------------------
def fig10_partner_impact_matrix():
    """Heatmap of top professional partners' random effects."""
    result_path = os.path.join(RESULT_DIR, 'task2_lmm_analysis.json')
    if not os.path.exists(result_path):
        print('[Figure 10 skipped] Task 2 results not found. Run: python main.py --task 2')
        return

    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    judge_re = results.get('judge_model', {}).get('random_effects', {})
    fan_re = results.get('fan_model', {}).get('random_effects', {})

    partners = sorted(set(judge_re.keys()) & set(fan_re.keys()))
    if len(partners) < 5:
        print('[Figure 10 skipped] Too few partners with both effects.')
        return

    # Select top 20 by combined absolute effect
    combined = {p: abs(judge_re.get(p, 0)) + abs(fan_re.get(p, 0)) for p in partners}
    top_partners = sorted(combined, key=combined.get, reverse=True)[:20]

    matrix = np.array([[judge_re.get(p, 0), fan_re.get(p, 0)] for p in top_partners])

    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdBu_r',
                xticklabels=['Judge Effect', 'Fan Effect'],
                yticklabels=top_partners, center=0, ax=ax)
    ax.set_title('Figure 10: Top Professional Dancer Impact Matrix')
    _savefig('fig10_partner_impact_matrix.png')


# ---------------------------------------------------------------------------
# Figure 11: Boxplot of Spearman correlations by mechanism
# ---------------------------------------------------------------------------
def fig11_spearman_boxplot():
    """Boxplot of weekly Spearman correlations (Task 3)."""
    result_path = os.path.join(RESULT_DIR, 'task3_mechanism_comparison.json')
    if not os.path.exists(result_path):
        print('[Figure 11 skipped] Task 3 results not found. Run: python main.py --task 3')
        return

    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # Detailed weekly correlations are not stored by default; use summary instead
    summary = results.get('spearman_correlations', {})
    labels = []
    values = []
    for key in ['rho1', 'rho2', 'rho3', 'rho4']:
        if key in summary:
            labels.append(key)
            values.append(summary[key].get('mean', 0))

    if not values:
        print('[Figure 11 skipped] No Spearman summary available.')
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(labels, values, color=colors[:len(labels)])
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Figure 11: Mean Spearman Correlations by Mechanism')
    ax.set_ylim([0, 1])
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.4f}',
                ha='center', va='bottom')
    _savefig('fig11_spearman_correlations.png')


# ---------------------------------------------------------------------------
# Figure 13: Radar chart of SPI/FII across mechanisms
# ---------------------------------------------------------------------------
def fig13_spi_fii_radar():
    """Radar chart comparing SPI and FII across Rank, Percentage, and DASS."""
    result_path = os.path.join(RESULT_DIR, 'task3_mechanism_comparison.json')
    if not os.path.exists(result_path):
        print('[Figure 13 skipped] Task 3 results not found. Run: python main.py --task 3')
        return

    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    spi_fii = results.get('spi_fii', {})
    if not spi_fii:
        print('[Figure 13 skipped] SPI/FII data not available.')
        return

    mechanisms = ['rank', 'percentage', 'rank_with_save']
    labels = ['Rank', 'Percentage', 'Rank+Save']
    spi = [spi_fii.get(m, {}).get('SPI', 0) for m in mechanisms]
    fii = [spi_fii.get(m, {}).get('FII', 0) for m in mechanisms]

    categories = ['SPI', 'FII']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    for i, (label, s, fii_val) in enumerate(zip(labels, spi, fii)):
        values = [s, fii_val]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim([0, 100])
    ax.set_title('Figure 13: SPI vs. FII Across Mechanisms', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    _savefig('fig13_spi_fii_radar.png')


# ---------------------------------------------------------------------------
# Figure 15: Bobby Bones popularity inversion under DASS
# ---------------------------------------------------------------------------
def fig15_bobby_bones_case():
    """
    Plot Bobby Bones' simulated ranking trajectory under Percentage vs. DASS.
    Requires Task 1 and Task 4 results.
    """
    t1_path = os.path.join(RESULT_DIR, 'task1_fan_vote_reconstruction.json')
    t4_path = os.path.join(RESULT_DIR, 'task4_dass_system.json')
    if not (os.path.exists(t1_path) and os.path.exists(t4_path)):
        print('[Figure 15 skipped] Task 1 and/or Task 4 results not found.')
        return

    with open(t1_path, 'r', encoding='utf-8') as f:
        t1_results = json.load(f)
    with open(t4_path, 'r', encoding='utf-8') as f:
        t4_results = json.load(f)

    # Extract Bobby Bones' weekly ranks under percentage and DASS
    season = 27
    name = 'Bobby Bones'
    weeks = []
    percent_ranks = []
    dass_ranks = []

    df = load_raw_data()
    total_scores, _ = extract_weekly_judge_scores(df)

    from task3_mechanism_comparison import (
        compute_percentage_method_result, get_ranking_from_scores
    )
    from task4_dass_system import topsis_scoring

    for w in range(1, 12):
        week_key = f'S{season}_W{w}'
        if week_key not in t1_results.get('weekly_estimates', {}):
            continue
        wd = t1_results['weekly_estimates'][week_key]
        if name not in wd['contestants']:
            continue

        contestants = wd['contestants']
        fan_votes = wd['estimated_fan_votes']
        judge_scores = {n: total_scores.get((season, n, w), 0) for n in contestants}

        if all(v == 0 for v in judge_scores.values()):
            continue

        percent_score = compute_percentage_method_result(judge_scores, fan_votes)
        percent_rank = get_ranking_from_scores(percent_score, ascending=False)

        dass_score = topsis_scoring(judge_scores, fan_votes)
        dass_rank = get_ranking_from_scores(dass_score, ascending=False)

        weeks.append(w)
        percent_ranks.append(percent_rank.get(name, len(contestants)))
        dass_ranks.append(dass_rank.get(name, len(contestants)))

    if not weeks:
        print('[Figure 15 skipped] No weekly data found for Bobby Bones.')
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(weeks, percent_ranks, 'o-', label='Percentage System', color='#3498db')
    ax.plot(weeks, dass_ranks, 's-', label='DASS', color='#e74c3c')
    ax.invert_yaxis()
    ax.set_xlabel('Week')
    ax.set_ylabel('Ranking (lower = better)')
    ax.set_title('Figure 15: Bobby Bones — Popularity Inversion Correction by DASS')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig('fig15_bobby_bones_case.png')


# ---------------------------------------------------------------------------
# Figure 17: Ranking stability under Gaussian noise
# ---------------------------------------------------------------------------
def fig17_noise_robustness():
    """Line plot of ranking stability (Spearman rho) vs. noise level."""
    result_path = os.path.join(RESULT_DIR, 'sensitivity_analysis.json')
    if not os.path.exists(result_path):
        print('[Figure 17 skipped] Sensitivity results not found. Run: python main.py --task 5')
        return

    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    noise = results.get('noise_robustness', {})
    levels = noise.get('noise_levels', [])
    if not levels:
        print('[Figure 17 skipped] Noise robustness data missing.')
        return

    mechanisms = ['rank', 'percentage', 'dass']
    labels = ['Rank System', 'Percentage System', 'DASS']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    fig, ax = plt.subplots(figsize=(8, 5))
    for mech, label, color in zip(mechanisms, labels, colors):
        key = f'{mech}_stability'
        if key not in noise:
            continue
        rhos = [item['mean_rho'] for item in noise[key]]
        ax.plot(levels, rhos, 'o-', label=label, color=color)

    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Figure 17: Ranking Stability under Stochastic Noise')
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig('fig17_noise_robustness.png')


# ---------------------------------------------------------------------------
# Figure 18: OAT weight sensitivity
# ---------------------------------------------------------------------------
def fig18_weight_sensitivity():
    """Line plot showing ranking invariance across fan-weight values."""
    result_path = os.path.join(RESULT_DIR, 'sensitivity_analysis.json')
    if not os.path.exists(result_path):
        print('[Figure 18 skipped] Sensitivity results not found. Run: python main.py --task 5')
        return

    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    ws = results.get('weight_sensitivity', {})
    platform = ws.get('stable_platform', {})
    start = platform.get('start')
    end = platform.get('end')

    fig, ax = plt.subplots(figsize=(8, 5))
    # Draw the stable plateau region if available
    if start is not None and end is not None and start < end:
        ax.axvspan(start, end, alpha=0.2, color='green', label=f'Stable Plateau [{start:.2f}, {end:.2f}]')

    # Since detailed per-weight rankings are not serialized, we annotate the conclusion
    ax.text(0.5, 0.5, ws.get('conclusion', 'Stable plateau analysis'), ha='center', va='center',
            transform=ax.transAxes, fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Fan Vote Weight')
    ax.set_ylabel('Ranking Invariance Region')
    ax.set_title('Figure 18: OAT Weight Sensitivity — Stability Plateau')
    ax.legend()
    _savefig('fig18_weight_sensitivity.png')


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_all_figures():
    """Generate all supported figures."""
    print('=' * 60)
    print('Generating reproducible paper figures')
    print('=' * 60)

    fig2_eliminated_vs_survivors()
    fig3_fig4_demographics()
    fig5_dancer_distribution()
    fig7_temporal_uncertainty()
    fig9_fixed_effects_comparison()
    fig10_partner_impact_matrix()
    fig11_spearman_boxplot()
    fig13_spi_fii_radar()
    fig15_bobby_bones_case()
    fig17_noise_robustness()
    fig18_weight_sensitivity()

    print('\nAll supported figures have been saved to:', FIGURES_DIR)


if __name__ == '__main__':
    generate_all_figures()
