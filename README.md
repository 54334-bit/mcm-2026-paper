# 2026 MCM Problem C: DWTS Scoring System Optimization

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX)

This project provides a complete solution for the 2026 Mathematical Contest in Modeling (MCM) Problem C, focusing on **Dancing with the Stars (DWTS)** scoring system optimization. Our approach combines Monte Carlo simulation, Bayesian inference, Linear Mixed Models (LMM), Entropy Weight Method, and TOPSIS to design a fair and robust scoring mechanism.

## 🏆 Project Highlights

- **Fan Vote Reconstruction**: Reverse-engineer unobservable fan votes using Monte Carlo simulation with Bayesian filtering
- **Dual-branch LMM Analysis**: Compare judge and fan scoring preferences
- **Mechanism Comparison**: Evaluate Rank vs. Percentage methods with SPI/FII metrics
- **DASS System**: Design a Dynamic Adaptive Scoring System with entropy-based weighting
- **Counterfactual Simulation**: Full season simulation capturing elimination propagation effects
- **Sensitivity Analysis**: Validate model robustness across parameter variations

## 📁 Project Structure

```
2026-MCM-Problem-C/
├── main.py                 # Main entry point (task orchestration)
├── config.py               # Global configuration and parameters
├── data_loader.py          # Data loading and preprocessing
├── task1_fan_vote_reconstruction.py  # Fan vote reverse estimation
├── task2_lmm_analysis.py   # Dual-branch LMM factor analysis
├── task3_mechanism_comparison.py     # Scoring mechanism comparison
├── task4_dass_system.py    # Dynamic Adaptive Scoring System
├── sensitivity_analysis.py # Sensitivity and robustness analysis
├── counterfactual_season_sim.py      # Full season counterfactual simulation
├── alpha_optimizer.py      # Dirichlet parameter optimization
├── parameter_sweep.py      # DASS parameter grid search
├── run_optimized_pipeline.py         # Optimized full pipeline
├── data/
│   └── 2026_MCM_Problem_C_Data.csv   # Competition dataset
├── results/                # Output directory (auto-generated)
└── README.md
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Dependencies

Install required packages:

```bash
pip install numpy pandas scipy statsmodels
```

For visualization (optional):

```bash
pip install matplotlib seaborn
```

## 🚀 Quick Start

### Run All Tasks

```bash
python main.py
```

### Run Specific Task

```bash
python main.py --task 1    # Task 1: Fan Vote Reconstruction
python main.py --task 2    # Task 2: LMM Analysis
python main.py --task 3    # Task 3: Mechanism Comparison
python main.py --task 4    # Task 4: DASS System
python main.py --task 5    # Task 5: Sensitivity Analysis
python main.py --task 6    # Task 6: Counterfactual Simulation
python main.py --task 7    # Task 7: Parameter Optimization
```

### Include Parameter Optimization

```bash
python main.py --optimize
```

## 📊 Core Algorithms

### 1. Fan Vote Reconstruction (Task 1)

**Three-stage reverse inference model:**
- **Stage 1**: Build Dirichlet prior distribution using judge scores and popularity features
- **Stage 2**: Monte Carlo simulation (20,000 iterations) to generate candidate fan vote vectors
- **Stage 3**: Bayesian posterior filtering - retain only simulations matching actual eliminations

**Key Formula:**
```
α_i = 1 + K₁ × J_norm_i + K₂ × S_pop_i
```
where `S_pop = ω₁ × S_ind + ω₂ × S_loc + ω₃ × S_age`

### 2. Dual-branch LMM Analysis (Task 2)

Compare two Mixed Linear Models:
- **Judge Model**: Predict judge scores using contestant features
- **Fan Model**: Predict estimated fan votes using contestant features

**Key Outputs:**
- Fixed effects (age, industry, region, partner experience)
- Random effects (partner-specific bias)
- Coefficient divergence between judges and fans

### 3. Mechanism Comparison (Task 3)

**Rank Method:** `S = R_Judge + R_Fan` (lower = better)
**Percentage Method:** `S = J/ΣJ + V/ΣV` (higher = better)

**Evaluation Metrics:**
- **SPI (Skill Protection Index)**: % of eliminations where the lowest-judged contestant is eliminated
- **FII (Fan Influence Index)**: % of eliminations where the lowest-voted contestant is eliminated
- **Spearman ρ**: Measure of correlation between mechanism output and judge/fan rankings

### 4. DASS System (Task 4)

**Dynamic Adaptive Scoring System** combining:

1. **Entropy Weight Method**: Adaptively adjust weights based on score diversity
   - High diversity (low entropy) → higher judge weight
   - Low diversity (high entropy) → higher fan weight

2. **TOPSIS**: Multi-criteria decision making with vector normalization

3. **Controversy-triggered Save Mechanism**: When ΔR ≥ threshold, judges can save high-skill contestants

**Symbiotic Interval**: SPI ≥ 92% AND FII ∈ [48%, 51%]

### 5. Sensitivity Analysis (Task 5)

- **α Parameter Robustness**: CV changes within ±10% parameter variation
- **Noise Robustness**: Ranking stability under 0%-30% Gaussian noise
- **Weight Sensitivity**: OAT analysis for DASS weight ranges

### 6. Counterfactual Season Simulation (Task 6)

Full season simulation capturing elimination propagation effects:
- Each week's elimination affects subsequent weeks' competition dynamics
- Compare final rankings across Rank, Percentage, and DASS mechanisms
- Analyze controversial cases (Jerry Rice, Bobby Bones, etc.)

## 📈 Expected Results

| Metric | Paper Value | Our Reproduction |
|--------|-------------|------------------|
| Global Median CV | 0.293 | ~0.197 |
| ρ₁ (Rank vs Fan) | 0.8137 | ~0.81 |
| ρ₂ (Percent vs Fan) | 0.8891 | ~0.89 |
| SPI (DASS) | ≥92% | ≥92% |
| FII (DASS) | 48%-51% | 48%-51% |
| Judge Save Trigger Rate | 15% | ~16% |

## 🎯 Project Objectives

1. **Reverse Engineer Fan Votes**: Estimate unobservable fan voting patterns from elimination data
2. **Analyze Scoring Bias**: Quantify divergence between judge and fan preferences
3. **Evaluate Historical Mechanisms**: Compare Rank and Percentage scoring methods
4. **Design Fair Scoring System**: Develop DASS - a dynamically adaptive mechanism balancing skill and popularity
5. **Validate Robustness**: Ensure the solution is resilient to parameter changes and noise

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📧 Contact

For questions or collaboration, please contact the team at [your-email@example.com].

---

*This project was developed for the 2026 Mathematical Contest in Modeling (MCM) Problem C.*
