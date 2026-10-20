# 🌾 AgriMarket Intelligence
### Crop Yield Analysis & Prediction — Data Analysis & Visualization (DAV) Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![Tableau](https://img.shields.io/badge/Dashboard-Tableau-blue)](https://public.tableau.com)

---

## 📌 Problem Statement

India's agricultural productivity is determined by a complex interplay of soil chemistry, climate conditions, and farming practices. Yet most farmers and planners lack access to data-driven insights to guide optimal crop selection and resource allocation.

**Research Question:** *What combination of soil, climate, and farming inputs most significantly determines crop yield across Indian regions and seasons?*

---

## 👥 Key Stakeholders

| Stakeholder | Key Question |
|---|---|
| **Farmers / Cooperatives** | Which crop + season maximizes yield given my soil conditions? |
| **State Agricultural Departments** | Which regions underperform and need interventions? |
| **Agri-Input Suppliers** | What is the optimal fertilizer/pesticide dose? |

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `data/raw_agmarknet.csv` |
| Records | 10,000 |
| Features | 20 (soil, climate, farm inputs) |
| Target | `Crop_Yield_ton_per_hectare` |

**Feature categories:** Soil nutrients (N, P, K), Soil properties (pH, moisture, type), Climate (temperature, humidity, rainfall, sunlight, wind), Farm context (region, altitude, season, crop type, irrigation), Inputs (fertilizer, pesticide)

---

## 🗂️ Project Structure

```
AgriMarket-Intelligence/
├── data/
│   ├── raw_agmarknet.csv                  ← Original dataset
│   ├── cleaned_agriculture_data.csv       ← Cleaned + feature-engineered
│   └── cleaned_agriculture_data.xlsx      ← Tableau-ready export
├── notebooks/
│   └── 01_dav_analysis.ipynb              ← Main analysis notebook ⭐
├── models/
│   └── crop_yield_random_forest.joblib    ← Saved RF model
├── reports/
│   ├── Project_Report.md                  ← Full project report ⭐
│   ├── Medium_Article_Draft.md            ← Medium article draft ⭐
│   ├── correlation_matrix.csv
│   ├── feature_importance.csv
│   ├── model_metrics.json
│   └── viz_*.png                          ← All visualizations
├── data_pipeline.py                       ← Full Python pipeline
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python data_pipeline.py
```

This will:
- Load and clean the raw dataset
- Engineer composite features
- Run Pearson correlation analysis
- Run Chi-square test (Season vs Crop_Type)
- Train Random Forest Regressor
- Save cleaned CSV + Excel (for Tableau)
- Save all reports and metrics

### 3. Open the Jupyter Notebook
```bash
jupyter notebook notebooks/01_dav_analysis.ipynb
```

Run all cells for the complete analysis with visualizations.

### 4. Open Tableau Dashboard
1. Open Tableau Desktop or Tableau Public
2. Connect to: `data/cleaned_agriculture_data.xlsx`
3. Build views: Yield by Region, Yield by Season, Climate vs Yield, Nutrient Analysis

---

## 📈 Results Summary

### Machine Learning Model (Random Forest Regressor)
| Metric | Value |
|---|---|
| R² Score | **0.9978** |
| MAE | **0.9022 ton/ha** |
| RMSE | **1.1312 ton/ha** |
| Model | Random Forest (300 trees) |
| Split | 80% train / 20% test |

### Statistical Tests
| Test | Variables | Result |
|---|---|---|
| Pearson Correlation | All numeric vs Yield | K, Rainfall strongest drivers |
| Chi-Square | Season vs Crop_Type | p = 0.41 (independent) |
| One-Way ANOVA | Yield across Regions | Significant regional differences (p < 0.05) |

---

## 💡 Five Key Insights

1. **Irrigation type** is the most controllable yield driver — drip irrigation outperforms rainfed
2. **Potassium (K)** is the most yield-relevant macronutrient among N, P, K
3. **Rabi season** yields are more stable/predictable than Kharif
4. **Central region** achieves high yields with below-average input intensity
5. **Soil pH** outside 6.5–7.5 silently suppresses yield regardless of other inputs

### 🔍 Unexpected Finding
Higher fertilizer use does NOT always mean higher yield. An inverted-U relationship shows optimal yield at 160–220 kg/ha, with declining returns beyond 280 kg/ha.

---

## 📋 Three Recommendations

1. **Expand drip irrigation** in rainfed belts (15–25% yield uplift expected)
2. **Require soil pH testing** as prerequisite for fertilizer subsidies (10–20% efficiency gain)
3. **Deploy RF model** as farmer-facing mobile advisory tool

---

## 📁 Deliverables (Faculty Submission)

| Deliverable | File | Status |
|---|---|---|
| Dataset (Excel) | `data/cleaned_agriculture_data.xlsx` | ✅ |
| Python Notebook | `notebooks/01_dav_analysis.ipynb` | ✅ |
| Tableau Dashboard | Built from `data/*.xlsx` | ✅ |
| Project Report | `reports/Project_Report.md` | ✅ |
| Medium Article | `reports/Medium_Article_Draft.md` | ✅ |

---

## 📚 References

1. Agmarknet — Ministry of Agriculture, Government of India
2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
3. scikit-learn: Machine Learning in Python, JMLR 12, 2825–2830, 2011.
4. Indian Council of Agricultural Research (ICAR) — Fertilizer Use Efficiency Reports
