# app.py
import pandas as pd, joblib, streamlit as st
from pathlib import Path

st.set_page_config(page_title="Riesgo Crediticio", page_icon="📊", layout="wide")
st.title("📊 Riesgo Crediticio – Demo UCI")

bundle = joblib.load("models/uci_lgbm.joblib")
clf, cols = bundle["model"], bundle["columns"]

up = st.file_uploader("Sube un CSV con mismas columnas que el train", type=["csv"])
if up:
    df = pd.read_csv(up)
    df.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in df.columns]
    # si tu CSV incluye la columna target, la quitamos
    tgt = [c for c in df.columns if "default" in c and "next" in c]
    if tgt: df = df.drop(columns=tgt)
    X = df.reindex(columns=cols, fill_value=0)
    proba = clf.predict_proba(X)[:,1]
    out = df.copy()
    out["risk_probability"] = proba.round(4)
    out["risk_bucket"] = pd.cut(out["risk_probability"], [-.01,.33,.66,1], labels=["Bajo","Medio","Alto"])
    st.dataframe(out.head(200))
    st.download_button("⬇️ Descargar resultados", out.to_csv(index=False).encode("utf-8"),
                       file_name="predicciones.csv", mime="text/csv")
else:
    st.info("Sube un CSV para ver probabilidades de riesgo por cliente.")
