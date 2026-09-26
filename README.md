# House Rent Prediction — ML Subject Project

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Project files

- `REPORT.md` — complete project write-up aligned with the college guideline
- `presentation.md` — 12-slide presentation content
- `viva.md` — viva questions and concise answers
- `train_model.py` — preprocessing, training, evaluation, and model saving
- `app.py` — Streamlit demo
- `model_results.csv` — actual test-set results
- `dataset_audit.csv` — dataset audit
- `figures/` — required EDA and feature-importance graphs
- `models/` — saved Random Forest model
- `data/House_Rent_Dataset.csv` — supplied dataset

## Algorithms

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

## Target

`Rent` — a regression target.
