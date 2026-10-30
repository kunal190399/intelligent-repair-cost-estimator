import argparse,json,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from pathlib import Path
from cost_ai.model import FEATURES,TARGET
import joblib
p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--model",required=True)
p.add_argument("--output",default="outputs/metrics.json");a=p.parse_args()
df=pd.read_csv(a.data);_,te=train_test_split(df,test_size=.2,random_state=42)
pred=joblib.load(a.model).predict(te[FEATURES])
m={"mae_gbp":mean_absolute_error(te[TARGET],pred),"rmse_gbp":mean_squared_error(te[TARGET],pred)**.5,"r2":r2_score(te[TARGET],pred)}
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(m,indent=2));print(m)
