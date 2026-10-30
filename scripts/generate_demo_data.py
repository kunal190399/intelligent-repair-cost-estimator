from pathlib import Path
import numpy as np,pandas as pd
r=np.random.default_rng(12); n=900
damage=r.choice(["scratch","dent","crack","broken_glass","missing_part"],n)
component=r.choice(["bumper","door","bonnet","wing","windscreen","light"],n)
severity=r.choice(["minor","moderate","severe"],n,p=[.45,.4,.15])
panels=r.integers(1,4,n); labour=np.maximum(.5,r.normal(1+panels*1.5+(severity=="severe")*2,1,n))
parts=((damage=="broken_glass")|(damage=="missing_part")).astype(int)
paint=((damage=="scratch")|(damage=="dent")|(damage=="crack")).astype(int)
age=r.integers(0,18,n); area=np.clip(r.beta(2,8,n)+(severity=="severe")*.15,0,1)
conf=r.uniform(.55,.99,n)
cost=70+labour*62+parts*180+paint*125+panels*45+area*300+(severity=="severe")*150+r.normal(0,40,n)
df=pd.DataFrame(dict(damage_type=damage,component=component,severity=severity,
 panels_affected=panels,labour_hours=labour,parts_required=parts,paint_required=paint,
 vehicle_age_years=age,image_damage_area_ratio=area,image_model_confidence=conf,
 target_cost_gbp=cost.clip(50)))
Path("data/sample").mkdir(parents=True,exist_ok=True)
df.to_csv("data/sample/repair_records.csv",index=False)
print("Wrote synthetic demonstration data; not actual repair pricing.")
