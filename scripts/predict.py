import argparse
from cost_ai.model import predict
import joblib
p=argparse.ArgumentParser();p.add_argument("--model",required=True)
p.add_argument("--damage_type",default="dent");p.add_argument("--component",default="door")
p.add_argument("--severity",default="moderate");p.add_argument("--labour_hours",type=float,default=3.5)
a=p.parse_args()
record={"damage_type":a.damage_type,"component":a.component,"severity":a.severity,
"panels_affected":1,"labour_hours":a.labour_hours,"parts_required":0,"paint_required":1,
"vehicle_age_years":5,"image_damage_area_ratio":None,"image_model_confidence":None}
print(f"Illustrative estimate: £{predict(joblib.load(a.model),record):,.2f} — not a customer quote.")
