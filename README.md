# Intelligent Repair Cost Estimator

A tabular + image-feature regression baseline for estimating repair costs from
damage class, component, severity, labour/parts indicators and optional image
features produced by Project 1.

## Run
```bash
pip install -r requirements.txt
python scripts/generate_demo_data.py
python scripts/train.py --data data/sample/repair_records.csv
python scripts/evaluate.py --data data/sample/repair_records.csv --model artifacts/cost_model.joblib
python scripts/predict.py --model artifacts/cost_model.joblib
streamlit run app/demo.py
pytest
```

The generated dataset is synthetic and only tests the pipeline. Replace it with
approved, de-identified repair records. Define currency, VAT basis, price date,
regional labour rate and repair operation mapping. Do not present demo estimates
as ABL prices or customer quotes.

## Schema
`damage_type, component, severity, panels_affected, labour_hours, parts_required,
paint_required, vehicle_age_years, image_damage_area_ratio,
image_model_confidence, target_cost_gbp`

## Methodology
One-hot categorical variables, impute and scale numeric features, train a
RandomForestRegressor baseline, and report MAE, RMSE and R² on held-out records.
For robust research, split by vehicle/job and time; compare CatBoost/XGBoost,
quantile regression and calibrated prediction intervals. Analyse error by class,
component, site and price band. SHAP can be added for assessor-facing explanations.
