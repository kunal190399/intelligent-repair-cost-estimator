from pathlib import Path
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

CAT=["damage_type","component","severity"]
NUM=["panels_affected","labour_hours","parts_required","paint_required",
     "vehicle_age_years","image_damage_area_ratio","image_model_confidence"]
FEATURES=CAT+NUM
TARGET="target_cost_gbp"

def build_model():
    cat=Pipeline([("impute",SimpleImputer(strategy="most_frequent")),
                  ("encode",OneHotEncoder(handle_unknown="ignore"))])
    num=Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    prep=ColumnTransformer([("cat",cat,CAT),("num",num,NUM)])
    return Pipeline([("prep",prep),("regressor",RandomForestRegressor(
        n_estimators=300,min_samples_leaf=2,random_state=42,n_jobs=-1))])

def fit(frame):
    missing=set(FEATURES+[TARGET])-set(frame.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    model=build_model()
    model.fit(frame[FEATURES],frame[TARGET])
    return model

def save(model,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True); joblib.dump(model,path)

def predict(model,record):
    row=pd.DataFrame([record])
    for col in FEATURES:
        if col not in row: row[col]=None
    return float(model.predict(row[FEATURES])[0])
