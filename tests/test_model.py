import pandas as pd
from cost_ai.model import fit,predict,FEATURES
def test_fit_and_predict():
    rows=[]
    for i in range(30):
        rows.append(dict(damage_type="dent",component="door",severity="minor" if i%2 else "moderate",
        panels_affected=1,labour_hours=2+i*.1,parts_required=0,paint_required=1,
        vehicle_age_years=i%8,image_damage_area_ratio=.1,image_model_confidence=.8,target_cost_gbp=200+i*8))
    m=fit(pd.DataFrame(rows)); record={k:rows[0][k] for k in FEATURES}
    assert predict(m,record)>0
