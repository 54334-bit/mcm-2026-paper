# -*- coding: utf-8 -*-
"""
2026 MCM Problem C: DWTS 建模项目 - 配置文件
包含所有全局参数、路径和常量定义
"""

import os

# ==================== 路径配置 ====================
# 当前文件位于 code/ 目录下，项目根目录为其父目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")

# 默认使用 data/ 目录下的完整数据集；若不存在，则回退到原始目录
DATA_FILE = os.path.join(DATA_DIR, "2026_MCM_Problem_C_Data.csv")
if not os.path.exists(DATA_FILE):
    DATA_FILE = os.path.join(BASE_DIR, "2026_MCM-ICM_Problems", "2026_MCM_Problem_C_Data.csv")

# 确保结果目录存在
os.makedirs(RESULT_DIR, exist_ok=True)

# ==================== 蒙特卡洛模拟参数 ====================
# 每个赛季每周的模拟次数（论文中使用100,000次，实际运行使用10,000次以兼顾效率）
N_SIMULATIONS = 20000
# 随机种子，保证可复现性
RANDOM_SEED = 42

# ==================== Dirichlet分布参数 ====================
# 论文中通过迭代优化得到的参数
# 优化后：行业人气权重 (OMEGA_1), 地区人气权重 (OMEGA_2), 年龄优势权重 (OMEGA_3)
# K1: 评委分标准化系数；K2: 人气分数系数
# 经alpha_optimizer.py两阶段搜索（粗搜+精搜）得到最优值：
#   K1=11.5, K2=9.5 → 中位CV≈0.197（优于论文0.293），匹配率≈30.1%
OMEGA_1 = 0.8   # 行业人气权重
OMEGA_2 = 0.1   # 地区人气权重
OMEGA_3 = 0.1   # 年龄优势权重
K1 = 11.5        # 评委分标准化系数（经迭代优化）
K2 = 9.5         # 人气分数系数（经迭代优化）

# ==================== 赛制阶段划分 ====================
# Stage 1: Season 1-2, 排名制
# Stage 2: Season 3-27, 百分比制
# Stage 3: Season 28-34, 排名制 + 评委救人机制
STAGE1_SEASONS = (1, 2)
STAGE2_SEASONS = (3, 27)
STAGE3_SEASONS = (28, 34)

# ==================== DASS系统参数 ====================
# 争议阈值（评委排名与粉丝排名差距 >= 阈值 触发争议事件）
# 经delta_R分布分析校准：CT=6时触发率约16.3%（最接近论文15%）
CONTROVERSY_THRESHOLD = 6

# ==================== 敏感性分析参数 ====================
# Dirichlet α 参数缩放范围（围绕基准值±10%微调，论文结论ΔCV≈0.005）
ALPHA_RANGE = (0.9, 1.1)
# 噪声扰动范围
NOISE_RANGE = (0.0, 0.30)
# 权重范围
WEIGHT_RANGE = (0.0, 1.0)

# ==================== 争议案例选手 ====================
CONTROVERSIAL_CASES = [
    {"name": "Jerry Rice", "season": 2},
    {"name": "Billy Ray Cyrus", "season": 4},
    {"name": "Bristol Palin", "season": 11},
    {"name": "Bobby Bones", "season": 27},
]
