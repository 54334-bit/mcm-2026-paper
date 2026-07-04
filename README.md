# MCM 2026 Honorable Mention Problem C: Meritocracy vs. Popularity — Comparison and Redesign of DWTS Mechanisms

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Team**: 2625519  
**Problem**: 2026 Mathematical Contest in Modeling (MCM) Problem C — *Dancing with the Stars (DWTS)* scoring-system analysis and redesign.

---

## 📄 Full Paper

📥 **[Download / View Full Paper (PDF)](./paper/MCM2026_Team2625519.pdf)**

> The PDF contains the complete mathematical derivation, model assumptions, notation table, sensitivity proofs, and memorandum. Since only the PDF (not the original LaTeX source) is available, we do **not** provide a Markdown transcription of the paper here to avoid errors in mathematical formulas.

---

## 📝 Abstract

To reach a balance between professional meritocracy and mass entertainment in the *Dancing with the Stars* evaluation system, this paper presents a robust analytical framework that transitions from data reconstruction to systemic redesign.

**First**, facing the challenge of "black-box" fan voting data, we develop an **Inverse Inference Model** utilizing Monte Carlo simulation coupled with Bayesian updating. By treating historical elimination results as constraints, we reconstruct the posterior distribution of latent fan votes. Our model achieves a high reconstruction stability with a median **Coefficient of Variation (CV) of 0.293**, significantly below the high-volatility threshold (0.5), providing a rigorous empirical foundation for subsequent causal analysis.

**Second**, using the reconstructed data, we employ a **dual-branch Linear Mixed-Effects Model (LMM)** to decouple the multi-dimensional factors influencing competition outcomes. By isolating professional dancers as random effects, we quantify the marginal contributions of celebrity attributes. Findings reveal a profound **"Preference Divergence"**: judge scores are strictly merit-driven (demonstrating significant negative correlation with age), whereas fan votes are narrative-driven, showing extreme positive bias toward contestants with **"Reality TV"** backgrounds.

**Third**, we conduct counterfactual simulations to diagnose the mathematical vulnerabilities of historical aggregation methods. Quantitative results show that the **Percentage System** exhibits a significantly higher Spearman correlation with fan votes (**0.8891**) compared to the **Rank System** (**0.8137**), explaining why popularity can easily overwhelm technical merit in percentage-based seasons. We further demonstrate the introduction of a **Judges' Save** mechanism acts as a true "circuit breaker," raising the **Skill Protection Index (SPI)** from **50.5% to 88.6%**.

**Ultimately**, we propose the **Dynamic Adaptive Scoring System (DASS)**. This system integrates **Entropy-TOPSIS** for real-time, objective weighting—automatically adjusting the influence of judges based on the weekly differentiation of performances—with a quantitative **Judges' Save** protocol triggered by a controversy threshold. Our final evaluations demonstrate that DASS achieves a balance: it maximizes professional integrity by maintaining an **SPI above 92%** while preserving a **Fan Influence Index (FII) in the [48%, 51%] symbiotic interval**.

---

## 🏆 Main Conclusions

1. **Fan votes can be reliably reconstructed** from public elimination records. The three-stage inversion model (Dirichlet prior → Monte Carlo simulation → Bayesian posterior filtering) yields stable posterior estimates with median CV ≈ 0.293.

2. **Judges and fans evaluate contestants through fundamentally different lenses.** Judges reward technical merit and penalize age; fans favor narrative appeal and Reality-TV backgrounds.

3. **The historical Percentage System is mathematically biased toward popularity.** Its Spearman correlation with fan votes (ρ = 0.8891) far exceeds that of the Rank System (ρ = 0.8137), making popular-but-less-skilled contestants more likely to survive.

4. **A well-designed Judges' Save is an effective circuit breaker.** It raises SPI from 50.5% to 88.6% by intervening only in extreme controversy cases.

5. **DASS reaches the "symbiotic interval"** defined by SPI ≥ 92% and FII ∈ [48%, 51%], achieving the Pareto-optimal balance between professional integrity and entertainment value.

6. **The model is robust.** Sensitivity analysis shows ΔCV ≈ 0.005 under α perturbation, and DASS maintains ranking stability ρ > 0.9 even with 30% Gaussian noise in fan votes.

---

## 📁 Repository Structure

```
MCM2026-ProblemC-Team2625519/
├── README.md
├── LICENSE
├── .gitignore
├── paper/
│   └── MCM2026_Team2625519.pdf          # Full competition paper
├── code/
│   ├── main.py                          # Unified entry point
│   ├── config.py                        # Global parameters
│   ├── data_loader.py                   # Data loading & preprocessing
│   ├── task1_fan_vote_reconstruction.py # Inverse inference model
│   ├── task2_lmm_analysis.py            # Dual-branch LMM
│   ├── task3_mechanism_comparison.py    # Mechanism comparison & SPI/FII
│   ├── task4_dass_system.py             # DASS adaptive scoring system
│   ├── sensitivity_analysis.py          # Robustness checks
│   ├── counterfactual_season_sim.py     # Full-season counterfactuals
│   ├── alpha_optimizer.py               # K1/K2 parameter optimization
│   ├── parameter_sweep.py               # DASS hyper-parameter search
│   ├── run_optimized_pipeline.py        # Best-parameter full pipeline
│   ├── visualization.py                 # Reproducible paper figures
│   └── requirements.txt
├── data/
│   ├── sample_input.csv                 # Small data sample
│   └── README.md                        # Data source & citation
└── figures/
    └── result.png                       # Example output figure
```

---

## 🛠️ Installation

```bash
cd code
pip install -r requirements.txt
```

See [`code/requirements.txt`](./code/requirements.txt) for the full dependency list.

---

## 🚀 Quick Start

### Run the complete pipeline

```bash
cd code
python main.py
```

### Run a single task

```bash
python main.py --task 1    # Fan vote reconstruction
python main.py --task 2    # LMM analysis
python main.py --task 3    # Mechanism comparison
python main.py --task 4    # DASS system
python main.py --task 5    # Sensitivity analysis
python main.py --task 6    # Full-season counterfactual simulation
python main.py --task 7    # Parameter optimization
```

### Include parameter optimization

```bash
python main.py --optimize
```

All generated results are saved to the `results/` directory (created automatically).

---

## 📊 Expected Key Results

| Metric | Value |
|--------|-------|
| Global Median CV | 0.293 |
| Spearman ρ (Rank vs. Fan) | 0.8137 |
| Spearman ρ (Percent vs. Fan) | 0.8891 |
| SPI after Judges' Save | 88.6% |
| SPI (DASS) | ≥ 92% |
| FII (DASS) | 48% – 51% |
| ΔCV under α perturbation | ≈ 0.005 |

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Developed for the 2026 Mathematical Contest in Modeling (MCM) Problem C, Team #2625519.*
