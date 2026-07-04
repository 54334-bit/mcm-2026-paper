# Data Source and Description

## Full Dataset

The complete dataset used in this project is the official data released for the **2026 Mathematical Contest in Modeling (MCM) Problem C**.

- **File name**: `2026_MCM_Problem_C_Data.csv`
- **Source**: COMAP 2026 MCM/ICM Competition, Problem C
- **Official download**: https://www.comap.com/undergraduate/contests/mcm/contests/2026/problems/
  (or the official MCM problem archive provided to registered teams)

## Data Description

The dataset contains weekly judge scores and contestant information for **34 seasons** of *Dancing with the Stars (DWTS)*, spanning from Season 1 to Season 34.

### Columns

| Column Group | Description |
|--------------|-------------|
| `celebrity_name` | Name of the celebrity contestant |
| `ballroom_partner` | Name of the professional dance partner |
| `celebrity_industry` | Industry / profession of the celebrity (e.g., Actor/Actress, Singer/Rapper, Athlete) |
| `celebrity_homestate` | Home state (for U.S. contestants) |
| `celebrity_homecountry/region` | Home country or region |
| `celebrity_age_during_season` | Age of the celebrity during the season |
| `season` | Season number (1–34) |
| `results` | Final result (e.g., "1st Place", "Eliminated Week X") |
| `placement` | Final placement number |
| `week{k}_judge{n}_score` | Judge score awarded in week `k` by judge `n` (1–4). `N/A` or `0` indicates no score / already eliminated |

### Key Notes

- Judge scores are on a scale of **1–10**.
- A value of **0** indicates the contestant had already been eliminated in that week.
- A value of **N/A** indicates that the particular judge did not score in that week.
- Some contestants withdrew from the competition; these cases are handled in the preprocessing step (see `code/data_loader.py`).

## Sample File

The file `sample_input.csv` contains only the first 8 rows of the full dataset (covering Season 1 and part of Season 2). It is provided for quick inspection and testing only. To reproduce all results, download the complete official dataset and place it at:

```
code/../2026_MCM-ICM_Problems/2026_MCM_Problem_C_Data.csv
```

or update `DATA_FILE` in `code/config.py` to point to the full dataset location.

## Citation

If you use this data in your own work, please cite the original MCM 2026 Problem C materials:

> COMAP. (2026). *2026 Mathematical Contest in Modeling: Problem C — Dancing with the Stars*. Consortium for Mathematics and Its Applications.
