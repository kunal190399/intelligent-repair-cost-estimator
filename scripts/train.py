import argparse,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from cost_ai.model import fit,save,FEATURES,TARGET
p=argparse.ArgumentParser();p.add_argument("--data",required=True)
p.add_argument("--output",default="artifacts/cost_model.joblib");a=p.parse_args()
df=pd.read_csv(a.data); tr,te=train_test_split(df,test_size=.2,random_state=42)
m=fit(tr); pred=m.predict(te[FEATURES])
print({"MAE":mean_absolute_error(te[TARGET],pred),
       "RMSE":mean_squared_error(te[TARGET],pred)**.5,"R2":r2_score(te[TARGET],pred)})
save(m,a.output); print("Saved",a.output)
