from pathlib import Path
import sys,streamlit as st
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cost_ai.model import predict
import joblib
st.title("Intelligent Repair Cost Estimator")
st.warning("Prototype estimate only. Not a customer quotation.")
path=st.sidebar.text_input("Model","artifacts/cost_model.joblib")
damage=st.selectbox("Damage",["scratch","dent","crack","broken_glass","missing_part"])
component=st.selectbox("Component",["bumper","door","bonnet","wing","windscreen","light"])
severity=st.selectbox("Severity",["minor","moderate","severe"])
labour=st.number_input("Labour hours",0.5,100.0,3.5)
if st.button("Estimate"):
    if not Path(path).exists(): st.error("Train model first; see README.")
    else:
        row=dict(damage_type=damage,component=component,severity=severity,panels_affected=1,
        labour_hours=labour,parts_required=int(damage in ["broken_glass","missing_part"]),
        paint_required=int(damage in ["scratch","dent","crack"]),vehicle_age_years=5,
        image_damage_area_ratio=None,image_model_confidence=None)
        st.metric("Illustrative estimate",f"£{predict(joblib.load(path),row):,.2f}")
