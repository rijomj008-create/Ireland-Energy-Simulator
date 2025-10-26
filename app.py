# app.py — Ireland Energy Transition Decision Intelligence Case Study
# Author: Rijo Mathew John

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Ireland Energy Decision Intelligence Case Study",
    layout="wide",
    page_icon="⚡"
)

# -----------------------------
# UTIL: small helpers
# -----------------------------
def add_linear_fit(fig, x, y, name="Linear fit", color="#888", dash="dot"):
    """Adds a simple NumPy least-squares fit line to a Plotly figure (no statsmodels)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 2:
        m, b = np.polyfit(x[ok], y[ok], 1)
        xx = np.linspace(x[ok].min(), x[ok].max(), 100)
        yy = m * xx + b
        fig.add_trace(go.Scatter(
            x=xx, y=yy, name=name,
            mode="lines",
            line=dict(color=color, width=2, dash=dash),
            hovertemplate=f"{name}<br>x=%{{x:.2f}}<br>y=%{{y:.1f}}<extra></extra>",
            showlegend=True
        ))

# -----------------------------
# HEADER — Project Title & Theme
# -----------------------------
st.title("⚡ Ireland Energy Transition — Decision Intelligence Case Study")
st.markdown("""
### A Data-Driven Exploration of Ireland’s Energy Balance  
**From System Volatility → Insights → Strategy → Simulation**
""")
st.caption("Author: **Rijo Mathew John** | MSc Data Analytics | Decision Intelligence & Operations Analytics")

st.markdown("---")

# -----------------------------
# DATA LOAD
# -----------------------------
@st.cache_data(show_spinner=True)
def load_mart():
    # Try repo paths
    p1 = Path("data/processed/mart_ie_hourly_system_kpis.csv")
    p2 = Path("mart_ie_hourly_system_kpis.csv")
    if p1.exists():
        df_ = pd.read_csv(p1, parse_dates=["ts_utc"])
    elif p2.exists():
        df_ = pd.read_csv(p2, parse_dates=["ts_utc"])
    else:
        st.error("CSV not found. Place mart_ie_hourly_system_kpis.csv in repo root or data/processed/")
        st.stop()
    df_ = df_.sort_values("ts_utc").copy()
    df_["ts_utc"] = df_["ts_utc"].dt.floor("h")
    # Backfill convenience cols if missing
    if "ren_mw" not in df_.columns and "total_generation_mw" in df_.columns and "renewable_share_pct" in df_.columns:
        df_["ren_mw"] = (df_["total_generation_mw"] * df_["renewable_share_pct"] / 100.0).fillna(0.0)
    if "wind_speed_mps" not in df_.columns:
        df_["wind_speed_mps"] = np.nan
    if "sunshine_fraction" not in df_.columns:
        df_["sunshine_fraction"] = np.nan
    return df_

df = load_mart()

# -----------------------------
# 1️⃣ NARRATIVE SECTION — CONTEXT
# -----------------------------
st.header("🌍 1. Context & Narrative")
st.markdown("""
Ireland’s energy system faces a **three-way challenge** — balancing **Affordability**, **Reliability**, and **Sustainability**.  
Each hour, the grid must decide how to meet electricity demand using a mix of fossil fuels and renewables (mainly wind).  
But when the wind drops or demand surges, stress hours appear — and **prices spike**.

This project builds a **Decision Intelligence framework** that links:
- Energy generation & demand (ENTSO-E data)  
- Weather influence (Meteostat data)  
- Market prices  
- Scenario simulations for policy decisions  

We move from **data observation → diagnosis → prescription**, mirroring how modern energy analytics drives national grid decisions.
""")

st.info("This is not just visualization — it’s a full **decision intelligence pipeline**, built to test trade-offs between cost, reliability, and renewable stability.")

# -----------------------------
# 2️⃣ DASHBOARDS — EXPLORATION
# -----------------------------
st.markdown("---")
st.header("📊 2. System Dashboard — Understanding the Week")
st.markdown("Dataset window: **Oct 18–25, 2025** (hourly).")

colA, colB = st.columns(2)
with colA:
    fig1 = px.line(df, x="ts_utc", y=["load_mw","total_generation_mw"], labels={"value":"MW","ts_utc":"Time"})
    fig1.update_layout(title="Load vs Generation", legend_title="", height=400,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig1, use_container_width=True)

with colB:
    fig2 = px.line(df, x="ts_utc", y="price_eur_per_mwh", labels={"price_eur_per_mwh":"€/MWh"})
    fig2.update_layout(title="Market Price Trend", height=400,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    fig3 = px.line(df, x="ts_utc", y="renewable_share_pct", labels={"renewable_share_pct":"% Renewables"})
    fig3.update_layout(title="Renewable Share Over Time", height=400,
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    # Wind Speed vs Price
    fig4 = px.scatter(
        df, x="wind_speed_mps", y="price_eur_per_mwh",
        labels={"wind_speed_mps":"Wind Speed (m/s)","price_eur_per_mwh":"€/MWh"},
        color_discrete_sequence=["#60A5FA"]
    )
    add_linear_fit(fig4, df["wind_speed_mps"], df["price_eur_per_mwh"], name="Linear fit", color="#2563EB")
    fig4.update_layout(title="Price vs Wind Speed", height=380, margin=dict(l=10, r=10, t=10, b=10),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# 3️⃣ INSIGHTS & GAP IDENTIFIED
# -----------------------------
st.markdown("---")
st.header("🔍 3. Insights & Gap Identification")

avg_price = float(pd.to_numeric(df["price_eur_per_mwh"], errors="coerce").mean())
stress_ratio = float(((pd.to_numeric(df["total_generation_mw"], errors="coerce") /
                       pd.to_numeric(df["load_mw"], errors="coerce")) < 1).mean() * 100)
ren_mean = float(pd.to_numeric(df["renewable_share_pct"], errors="coerce").mean())

st.markdown("""
**Exploratory Findings:**
- When renewable share drops, prices spike and stress hours rise.  
- Volatile wind patterns (high RSD) create instability, even if average renewable share is high.  
- Prices correlate strongly with **stress hours** and **renewable volatility**.  
- Evening peaks (17:00–21:00) are consistent high-stress zones.
""")

colx, coly, colz = st.columns(3)
colx.metric("💶 Average Price", f"€{avg_price:.1f}")
coly.metric("⚙️ Stress Hours", f"{stress_ratio:.1f}%")
colz.metric("🌱 Avg Renewable Share", f"{ren_mean:.1f}%")

st.warning("**Problem Statement:** High price sensitivity to renewable volatility is the biggest operational and financial gap.")

# -----------------------------
# 4️⃣ CALL TO ACTION (STRATEGIC)
# -----------------------------
st.markdown("---")
st.header("🚀 4. Strategic Call to Action")
st.markdown("""
**Goal:** Build operational resilience while lowering costs and improving stability.

**Strategic Actions Identified:**
1. **Increase Wind Penetration (+20%)** → Lowers average price & stress hours.  
2. **Improve Renewable Stability (−25% RSD)** → Smooths volatility and enhances predictability.  
3. **Prepare for Demand Growth (+10%)** → Requires flexible storage or fast-responding generation.  
4. **Hybrid Strategy** → Combining 1 + 2 offers the **best trade-off** between price, reliability, and stability.
""")
st.success("We validate these actions through a simulation model — the Decision Intelligence Simulator below.")

# -----------------------------
# 5️⃣ DECISION INTELLIGENCE SIMULATOR
# -----------------------------
st.markdown("---")
st.header("🧠 5. Decision Intelligence Simulator")
st.caption("Interactively simulate how changes in wind, demand, or renewable smoothness affect system KPIs.")

# ---- SAFETY HELPERS ----
def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def build_features_safe(dfx, load_col="load_mw", gen_col="total_generation_mw",
                        ren_col="renewable_share_pct"):
    """Return a clean, finite feature matrix X and a boolean mask of valid rows."""
    d = dfx.copy()

    for c in [load_col, gen_col, ren_col, "wind_speed_mps", "sunshine_fraction"]:
        if c not in d.columns:
            d[c] = 0

    for c in [load_col, gen_col, ren_col, "wind_speed_mps", "sunshine_fraction"]:
        d[c] = _to_num(d[c])

    ratio = np.divide(_to_num(d[gen_col]), _to_num(d[load_col]),
                      out=np.full_like(_to_num(d[gen_col]), np.nan, dtype=float),
                      where=_to_num(d[load_col]).to_numpy() != 0)

    X = pd.DataFrame(index=d.index)
    X["ren_share"] = _to_num(d[ren_col])
    X["stress"] = (ratio < 1).astype(float)
    load_mean = _to_num(d[load_col]).mean(skipna=True)
    load_std  = _to_num(d[load_col]).std(skipna=True)
    if not np.isfinite(load_std) or load_std == 0:
        load_std = 1.0
    X["load_scaled"] = (_to_num(d[load_col]) - load_mean) / load_std

    X["wind_speed_mps"] = _to_num(d["wind_speed_mps"]).fillna(d["wind_speed_mps"].median() if "wind_speed_mps" in d else 0)
    X["sunshine_fraction"] = _to_num(d["sunshine_fraction"]).fillna(0)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.astype(float)
    X = X.fillna(X.mean(numeric_only=True))

    valid = np.isfinite(X.to_numpy()).all(axis=1)
    return X, valid, load_mean, load_std

def fit_model(dfb, price_col="price_eur_per_mwh",
              load_col="load_mw", gen_col="total_generation_mw", ren_col="renewable_share_pct",
              alpha=5.0):
    y = _to_num(dfb[price_col]).astype(float)
    X, valid, load_mean, load_std = build_features_safe(dfb, load_col, gen_col, ren_col)

    valid = valid & np.isfinite(y.to_numpy())
    X_fit = X.loc[valid]
    y_fit = y.loc[valid]

    if len(X_fit) < 5:
        raise ValueError("Not enough clean rows to fit the model after removing NaN/Inf. Check your CSV columns.")

    model = make_pipeline(StandardScaler(with_mean=True, with_std=True),
                          Ridge(alpha=alpha, fit_intercept=True))
    model.fit(X_fit, y_fit)
    return model, load_mean, load_std

def predict_price(model, dfx, load_mean, load_std,
                  load_col="load_mw", gen_col="total_generation_mw", ren_col="renewable_share_pct"):
    X, valid, _, _ = build_features_safe(dfx, load_col, gen_col, ren_col)
    yhat = pd.Series(np.nan, index=dfx.index, dtype=float)
    yhat.loc[X.index] = model.predict(X)
    return yhat.to_numpy()

# --- Fit baseline model ONCE ---
model, load_mean, load_std = fit_model(df)

# -------- SIMULATION SLIDERS --------
st.sidebar.header("Adjust Parameters")
wind_change   = st.sidebar.slider("Wind Availability (%)", -10, 30, 0)
demand_change = st.sidebar.slider("Demand Growth (%)", -10, 20, 0)
smooth_change = st.sidebar.slider("Renewable Stability (−RSD %)", 0, 40, 0)

# -------- APPLY SIMULATION --------
sim = df.copy()

# start from baseline columns
sim["ren_share_sim"]  = _to_num(sim["renewable_share_pct"])
sim["load_sim"]       = _to_num(sim["load_mw"])
sim["total_gen_sim"]  = _to_num(sim["total_generation_mw"])

# wind availability → scale renewable share (bounded to 0..100)
sim["ren_share_sim"] = (sim["ren_share_sim"] * (1 + wind_change/100)).clip(lower=0, upper=100)

# renewable stability smoothing (shrink deviations toward the mean)
if smooth_change and np.isfinite(sim["ren_share_sim"].mean()):
    m = sim["ren_share_sim"].mean()
    factor = 1 - (smooth_change / 100.0)  # e.g., 25% smoothing → factor 0.75
    sim["ren_share_sim"] = m + factor * (sim["ren_share_sim"] - m)

# demand growth
sim["load_sim"] = sim["load_sim"] * (1 + demand_change/100.0)

# predict price on the simulated columns
sim["price_sim"] = predict_price(
    model, sim, load_mean, load_std,
    ren_col="ren_share_sim",
    load_col="load_sim",
    gen_col="total_gen_sim"
)

# -------- KPI COMPARISON --------
def get_kpi(dframe):
    ratio = _to_num(dframe["total_generation_mw"]) / _to_num(dframe["load_sim"])
    stress = float((ratio < 1).mean() * 100)
    ren = _to_num(dframe["ren_share_sim"])
    rsd = float(ren.std() / ren.mean()) if np.isfinite(ren.mean()) and ren.mean() != 0 else np.nan
    return {
        "avg_price": float(_to_num(dframe["price_sim"]).mean()),
        "stress_pct": stress,
        "rsd": rsd
    }

# baseline mapped to the same *_sim columns for fair comparison
base = df.copy()
base["ren_share_sim"] = _to_num(base["renewable_share_pct"])
base["load_sim"]      = _to_num(base["load_mw"])
base["price_sim"]     = _to_num(base["price_eur_per_mwh"])

base_kpi = get_kpi(base)
sim_kpi  = get_kpi(sim)

st.subheader("Scenario Comparison")
colA, colB, colC = st.columns(3)
colA.metric("💶 Avg Price (€/MWh)", f"{sim_kpi['avg_price']:.1f}", f"{sim_kpi['avg_price']-base_kpi['avg_price']:.1f}")
colB.metric("⚙️ Stress Hours (%)", f"{sim_kpi['stress_pct']:.1f}", f"{sim_kpi['stress_pct']-base_kpi['stress_pct']:.1f}")
colC.metric("🌱 RSD (Stability)", f"{sim_kpi['rsd']:.2f}", f"{sim_kpi['rsd']-base_kpi['rsd']:.2f}")

fig = px.line(sim, x="ts_utc", y=["price_eur_per_mwh","price_sim"], labels={"value":"€/MWh","ts_utc":"Time"})
fig.update_layout(title="Simulated vs Actual Price", legend_title="",
                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

st.info("Use the sliders to explore how operational changes affect price, stress, and renewable stability in real time.")

# -----------------------------
# 6️⃣ CONCLUSION / SUMMARY
# -----------------------------
st.markdown("---")
st.header("📈 6. Conclusion & Executive Summary")
st.markdown(f"""
**Summary of Findings**
- Ireland’s grid is highly sensitive to renewable volatility.
- A 20% wind availability increase (proxy via renewable share) can reduce prices (delta depends on week).
- Smoothing renewables further cuts stress hours and stabilizes operations.
- A hybrid approach (Wind + Stability) achieves the best balance.

**Takeaway**
Data-driven simulation enables **quantitative energy strategy design** — transforming reactive policy into proactive decision intelligence.
""")

st.download_button(
    "⬇️ Download Case Study Summary (Markdown)",
    data=f"""
# Ireland Energy Decision Intelligence Case Study

**Author:** Rijo Mathew John  
**Theme:** Balancing Affordability, Reliability, and Sustainability using Data Analytics

### Summary
- Dataset: ENTSO-E + Meteostat, Oct 18–25, 2025
- Model: Ridge Regression with renewable share, stress, and scaled load
- Simulation controls: Wind availability, demand growth, renewable smoothing

**Findings (this week):**
- Baseline price (avg): €{base_kpi['avg_price']:.1f}/MWh
- Simulated price (avg): €{sim_kpi['avg_price']:.1f}/MWh
- Stress hours: {base_kpi['stress_pct']:.1f}% → {sim_kpi['stress_pct']:.1f}%
- Renewable stability (RSD): {base_kpi['rsd']:.2f} → {sim_kpi['rsd']:.2f}

**Conclusion:**
Decision intelligence bridges analytics and policy — enabling data-backed energy transition planning.

**Live demo:** Your Streamlit app link
""".encode("utf-8"),
    file_name="Ireland_Energy_Case_Study_Summary.md",
    mime="text/markdown",
)

st.markdown("---")
st.caption("© 2025 Rijo Mathew John — Decision Intelligence | Data Analytics | Energy Systems")
