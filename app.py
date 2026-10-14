"""Streamlit app: run with  streamlit run app.py"""
import json
import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from traffic.data import CLASS_NAMES, NUMERIC, WEATHER, build
from traffic.evaluate import load_dl, predict_dl
from traffic.graph import AREA_COORDS
from traffic.plots import FIG
from traffic.train import ART

COLORS = {"Low": "#2e9e5b", "Medium": "#e0a526", "High": "#d1432f"}

st.set_page_config(page_title="Bangalore Traffic Surrogate", layout="wide")


@st.cache_resource
def load():
    ds = build()
    models = {k: load_dl(k, ds).cpu() for k in ("gnn", "mlp") if (ART / f"{k}.pth").exists()}
    return ds, models


def road_coords(ds):
    jitter = np.linspace(-0.006, 0.006, len(ds.roads))
    return [(AREA_COORDS[a][0] + j, AREA_COORDS[a][1] - j) for a, j in zip(ds.areas, jitter)]


def run_model(model, x, mask):
    xt, mt = torch.as_tensor(x, dtype=torch.float32)[None], torch.as_tensor(mask)[None]
    t = time.perf_counter()
    with torch.no_grad():
        logits, sp = model(xt, mt)
    ms = (time.perf_counter() - t) * 1000
    return torch.softmax(logits[0], -1).numpy(), sp[0].numpy(), ms


def draw_map(ds, df):
    lat, lon = zip(*road_coords(ds))
    df = df.assign(lat=[lat[ds.roads.index(r)] for r in df["Road"]], lon=[lon[ds.roads.index(r)] for r in df["Road"]])
    fig = px.scatter_map(df, lat="lat", lon="lon", color="Predicted", color_discrete_map=COLORS,
                         category_orders={"Predicted": CLASS_NAMES}, hover_name="Road",
                         hover_data={"lat": False, "lon": False, "Pred. speed (km/h)": ":.1f", "P(High)": ":.2f"},
                         zoom=10.2, height=520)
    fig.update_traces(marker=dict(size=18))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Predicted congestion")
    st.plotly_chart(fig, width="stretch")


ds, models = load()
if not models:
    st.error("No trained models found. Run `python -m traffic.train --model gnn` and `--model mlp` first.")
    st.stop()

st.title("Bangalore road-network traffic surrogate")
st.caption("A deep-learning surrogate that predicts congestion class and speed for all 16 monitored roads in one forward "
           "pass, in place of running a slow traffic simulation for every scenario.")
with st.sidebar:
    kind = st.selectbox("Model", list(models), format_func=lambda k: {"gnn": "Graph network (GNN)", "mlp": "Per-road MLP"}[k])
    st.markdown("Congestion classes are cut at Congestion Level 60 and 90 (Low < 60 <= Medium < 90 <= High).")
model = models[kind]

tab1, tab2, tab3 = st.tabs(["Network forecast (test period)", "What-if scenario", "Model comparison"])

with tab1:
    s = ds.test
    day = st.select_slider("Test date", options=list(range(len(s.dates))),
                           format_func=lambda i: str(s.dates[i])[:10], value=0)
    proba, sp, ms = run_model(model, s.x[day], s.mask[day])
    m = s.mask[day]
    df = pd.DataFrame({
        "Road": np.array(ds.roads)[m], "Area": np.array(ds.areas)[m],
        "Actual": np.array(CLASS_NAMES)[s.cls[day][m]], "Predicted": np.array(CLASS_NAMES)[proba[m].argmax(-1)],
        "P(High)": proba[m][:, 2], "Pred. speed (km/h)": sp[m] * ds.speed_std + ds.speed_mean,
        "Actual speed (km/h)": s.speed[day][m] * ds.speed_std + ds.speed_mean})
    c1, c2, c3 = st.columns(3)
    c1.metric("Roads observed this day", int(m.sum()))
    c2.metric("Class accuracy this day", f"{(df['Actual'] == df['Predicted']).mean():.0%}")
    c3.metric("Forward pass (whole network)", f"{ms:.2f} ms")
    draw_map(ds, df)
    st.dataframe(df.round(2), hide_index=True, width="stretch")

with tab2:
    st.write("Start from each road's typical conditions, change the scenario, and re-predict the whole network instantly.")
    train = ds.frame[ds.frame["split"] == "train"]
    base = train.groupby("Road")[NUMERIC].mean().reindex(ds.roads)
    c1, c2, c3, c4 = st.columns(4)
    vol = c1.slider("Traffic volume vs typical", 0.4, 1.8, 1.0, 0.05)
    inc = c2.slider("Extra incidents per road", 0, 6, 0)
    weather = c3.selectbox("Weather", WEATHER)
    rw = c4.checkbox("Roadwork on all roads")
    sel = st.multiselect("Apply volume change to (default: all roads)", ds.roads, default=[])
    raw = base.copy()
    hit = raw.index.isin(sel) if sel else np.ones(len(raw), bool)
    raw.loc[hit, "Traffic Volume"] *= vol
    raw["Incident Reports"] += inc
    feats = np.concatenate([raw.values, np.full((len(raw), 1), float(rw)),
                            np.tile([float(w == weather) for w in WEATHER], (len(raw), 1))], 1)
    x = ((feats - ds.feat_mean) / ds.feat_std).astype(np.float32)
    proba, sp, ms = run_model(model, x, np.ones(len(ds.roads), bool))
    out = pd.DataFrame({"Road": ds.roads, "Area": ds.areas, "Traffic volume": raw["Traffic Volume"].round(0).values,
                        "Predicted": np.array(CLASS_NAMES)[proba.argmax(-1)], "P(High)": proba[:, 2],
                        "Pred. speed (km/h)": sp * ds.speed_std + ds.speed_mean})
    c1, c2, c3 = st.columns(3)
    c1.metric("Roads predicted High", int((out["Predicted"] == "High").sum()), help="of 16")
    c2.metric("Mean predicted speed", f"{out['Pred. speed (km/h)'].mean():.1f} km/h")
    c3.metric("Whole-network prediction time", f"{ms:.2f} ms")
    draw_map(ds, out)
    st.dataframe(out.round(2), hide_index=True, width="stretch")
    st.caption("Speed is only weakly predictable from these inputs (R2 about 0.15 on held-out data), so treat predicted speed as indicative.")

with tab3:
    mp = ART / "metrics.json"
    if not mp.exists():
        st.warning("Run `python -m traffic.evaluate` to generate metrics.")
    else:
        met = json.loads(mp.read_text())
        rows = [dict(Model=k, Accuracy=v["accuracy"], **{"Macro precision": v["macro_precision"], "Macro recall": v["macro_recall"],
                 "Macro F1": v["macro_f1"], "Speed MAE": v["speed_mae"], "Speed RMSE": v["speed_rmse"], "Speed R2": v["speed_r2"]})
                for k, v in met["results"].items()]
        st.subheader(f"Held-out test set ({met['n_test_rows']} rows, last 10% of dates)")
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        lat = pd.DataFrame({k: {d: f"{ms:.2f}" for d, ms in v.items()} for k, v in met["latency_ms"].items()}).T
        st.subheader("Inference latency (ms, one 16-road snapshot)")
        st.dataframe(lat.rename_axis("Model"), width="stretch")
        for f, cap in (("model_comparison.png", "Model comparison"), ("confusion_matrices.png", "Confusion matrices (test)"),
                       ("loss_curves.png", "Training vs validation curves")):
            if (FIG / f).exists():
                st.image(str(FIG / f), caption=cap)
        if (ART / "tuning.csv").exists():
            st.subheader("Hyperparameter search (validation)")
            st.dataframe(pd.read_csv(ART / "tuning.csv").round(4), hide_index=True, width="stretch")
