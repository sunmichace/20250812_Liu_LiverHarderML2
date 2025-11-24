# Predicting Next Liver Stiffness from Longitudinal Records

## Project Goal
- Predict each patient's next liver stiffness measurement using historical follow-up data.
- Dataset: 11,947 raw records from 1,102 patients; 1,045 patients with ≥2 visits kept for modeling, producing 2,984 supervised samples (80/20 train/test, 69 time-series features per sample).
- Stack: Python 3.8+, pandas, scikit-learn, matplotlib, seaborn.

## Repository Layout
```
project/
├── data/
│   ├── hepatitis_b_longitudinal_data.xlsx   # Raw data
│   └── processed/                           # Intermediate & model artifacts
│       ├── *.csv                            # Per-phase outputs
│       ├── *.png                            # Figures
│       └── models/                          # Trained models
├── phase1_longitudinal_preprocessing.py     # Data filtering, target/feature framing
├── phase2_longitudinal_eda.py               # Exploratory time-series analysis
├── phase3_feature_engineering_new.py        # Feature construction & scaling
├── phase4_simple_modeling.py                # Tabular baselines
├── phase4_time_series_modeling.py           # Time-series models
├── phase5_model_evaluation.py               # Hold-out evaluation & reporting
├── result/                                  # Evaluation summaries
└── README_en.md / README.md                 # Docs (EN / CN)
```

## Prerequisites
- Python 3.8+; ≥4 GB RAM; ~500 MB free disk.
- Install Python dependencies as defined in the scripts (no external services required).

## How to Run (end-to-end)
1) Preprocess longitudinal data  
   `python phase1_longitudinal_preprocessing.py`  
   Outputs cleaned sequences, target framing, and QC plots to `data/processed/`.

2) Explore time-series patterns  
   `python phase2_longitudinal_eda.py`  
   Generates trajectory visuals, predictive feature correlations, clustering, and time-span analyses.

3) Build and clean features  
   `python phase3_feature_engineering_new.py`  
   Creates comprehensive feature sets (`comprehensive_features_*` CSVs) and scalers.

4) Train models  
   - Baseline/tabular models: `python phase4_simple_modeling.py`  
   - Temporal models: `python phase4_time_series_modeling.py`  
   Artifacts saved under `data/processed/models/` and plots under `data/processed/`.

5) Evaluate on hold-out data  
   `python phase5_model_evaluation.py`  
   Writes metrics, comparison tables, and charts to `result/` and `data/processed/`.

## Key Outputs
- `data/processed/phase1_*`: cleaned longitudinal data, prediction datasets, QC reports.
- `data/processed/phase2_*`: EDA figures (trajectories, feature importance, time-dependency) and CSVs.
- `data/processed/comprehensive_features_*`: engineered feature matrices and scalers.
- `data/processed/models/`: trained model files.
- `result/`: evaluation summaries for the final models.

## Notes
- Scripts are independent; you can re-run specific phases after modifying data or parameters.
- Plots referenced in the Chinese README are produced automatically into `data/processed/`.
