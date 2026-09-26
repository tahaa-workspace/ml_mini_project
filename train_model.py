import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "House_Rent_Dataset.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

def floor_parts(s):
    s = str(s).strip()
    if " out of " not in s:
        return np.nan, np.nan
    first, total = s.split(" out of ", 1)
    mapping = {"ground": 0, "lower basement": -1, "upper basement": -2}
    try:
        floor = mapping[first.lower()] if first.lower() in mapping else int(first)
    except:
        floor = np.nan
    try:
        total_floors = int(total)
    except:
        total_floors = np.nan
    return floor, total_floors

def prepare_dataframe(df):
    x = df.copy()
    parsed = pd.DataFrame(x["Floor"].map(floor_parts).tolist(),
                          columns=["Floor_Number", "Total_Floors"], index=x.index)
    invalid = parsed["Floor_Number"] > parsed["Total_Floors"]
    parsed.loc[invalid, ["Floor_Number", "Total_Floors"]] = np.nan
    x = pd.concat([x, parsed], axis=1)
    dt = pd.to_datetime(x["Posted On"], errors="coerce")
    x["Posted_Month"] = dt.dt.month
    x["Posted_DayOfWeek"] = dt.dt.dayofweek
    return x

FEATURES = [
    "BHK", "Size", "Floor_Number", "Total_Floors", "Area Type", "City",
    "Furnishing Status", "Tenant Preferred", "Bathroom",
    "Posted_Month", "Posted_DayOfWeek"
]
CATEGORICAL = ["Area Type", "City", "Furnishing Status", "Tenant Preferred"]
NUMERICAL = [c for c in FEATURES if c not in CATEGORICAL]

def make_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), NUMERICAL),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    ])

df = prepare_dataframe(pd.read_csv(DATA))
X, y = df[FEATURES], df["Rent"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=12, min_samples_leaf=3),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1, min_samples_leaf=2),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42, n_estimators=200, max_depth=3, learning_rate=0.05, loss="huber")
}

rows = []
for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({
        "Algorithm": name,
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred)
    })
    if name == "Random Forest Regressor":
        joblib.dump(pipe, MODEL_DIR / "random_forest_model.joblib")

pd.DataFrame(rows).to_csv(BASE / "model_results.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print("\nSaved:", MODEL_DIR / "random_forest_model.joblib")
