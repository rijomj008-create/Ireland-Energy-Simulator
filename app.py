%%writefile app.py
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# -----------------------------
# Setup & data load
# -----------------------------
st.set_page_config(page_title="Ireland Energy Decision Simulator", layout="wide")
st.title("⚡ Ireland Energy Decision Intelligence Simulator")
st.caption("Data: ENTSO-E (load, gen, price) + Meteostat (weather) → week of Oct 18–25, 2025")

# Use your exact path from the screenshot
DATA_PATH = Path("data/processed/mart_ie_hourly_system_kpis.csv")
if not DATA_PATH.exists():
    # fallback if you run from a different cwd
    DATA_PATH = Path("/content/data/processed/mart_ie_hourly_system_kpis.csv")

df = pd.read_csv(DATA_PATH, parse_dates=["ts_utc"]).sort_values("ts_utc")
df["ts_utc"] = df["ts_utc"].dt.floor("h")

df = pd.read_csv(DATA_PATH, parse_dates=["ts_utc"]).sort_values("ts_utc")
df["ts_utc"] = df["ts_utc"].dt.floor("h")
df = df.loc[:, ~df.columns.duplicated()].copy()   # <— add this line

# -----------------------------
# Helper functions (same logic you used)
# -----------------------------
def kpis(df, price_col, ren_col, load_col="load_mw", gen_col="total_generation_mw"):
    ratio = df[gen_col] / df[load_col]
    stress = (ratio < 1).astype(int)
    rsd = df[ren_col].std() / (df[ren_col].mean() if df[ren_col].mean()!=0 else np.nan)
    return {
        "avg_price": df[price_col].mean(),
        "p95_price": df[price_col].quantile(0.95),
        "stress_pct": 100*stress.mean(),
        "rsd": rsd,
        "ren_mean": df[ren_col].mean()
    }

def build_features(dfx, load_mean, load_std, ren_col, load_col, gen_col):
    X = pd.DataFrame(index=dfx.index)
    X["ren_share"] = dfx[ren_col]
    X["stress"] = ((dfx[gen_col] / dfx[load_col]) < 1).astype(int)
    X["load_scaled"] = (dfx[load_col] - load_mean) / load_std
    if "wind_speed_mps" in dfx.columns:
        X["wind_speed_mps"] = dfx["wind_speed_mps"].fillna(dfx["wind_speed_mps"].mean())
    if "sunshine_fraction" in dfx.columns:
        X["sunshine_fraction"] = dfx["sunshine_fraction"].fillna(0)
    for c in X.columns:
        X[c] = X[c].fillna(X[c].mean())
    return X

def fit_ridge(dfb, ren_col="renewable_share_pct", load_col="load_mw", gen_col="total_generation_mw", alpha=5.0):
    load_mean, load_std = dfb[load_col].mean(), dfb[load_col].std()
    X = build_features(dfb, load_mean, load_std, ren_col, load_col, gen_col)
    y = dfb["price_eur_per_mwh"].values
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True))
    model.fit(X, y)
    return model, load_mean, load_std, X.columns.tolist()

def predict_price(model, dfx, load_mean, load_std, ren_col="renewable_share_pct",
                  load_col="load_mw", gen_col="total_generation_mw"):
    X = build_features(dfx, load_mean, load_std, ren_col, load_col, gen_col)
    return model.predict(X)

# Split renewable into wind/not (assumption ~80% wind)
df["ren_mw"]  = (df["total_generation_mw"] * df["renewable_share_pct"] / 100.0).fillna(0.0)
df["nren_mw"] = (df["total_generation_mw"] - df["ren_mw"]).clip(lower=0)
WIND_FRACTION_IN_REN = 0.80

# Fit baseline model
model, load_mean, load_std, feat_names = fit_ridge(df)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Custom Scenario Controls")
wind_pct   = st.sidebar.slider("Wind availability change (%)", -10, 30, 0)
demand_pct = st.sidebar.slider("Demand change (%)", -10, 20, 0)
smooth_pct = st.sidebar.slider("Storage smoothing (reduce RSD by %) ", 0, 40, 0)

# -----------------------------
# Preset scenarios (same 4)
# -----------------------------
def scenario_baseline(dfx):
    out = dfx.copy()
    out["ren_share_sim"]  = out["renewable_share_pct"]
    out["load_sim"]       = out["load_mw"]
    out["total_gen_sim"]  = out["total_generation_mw"]
    out["price_sim"] = predict_price(
        model, out, load_mean, load_std,
        ren_col="ren_share_sim", load_col="load_sim", gen_col="total_gen_sim"
    )
    return out


def scenario_wind_up(dfx, pct=20):
    out = dfx.copy()
    wind_mw   = out["ren_mw"] * WIND_FRACTION_IN_REN
    other_ren = out["ren_mw"] * (1 - WIND_FRACTION_IN_REN)
    wind_mw_new = wind_mw * (1 + pct/100)
    ren_mw_new  = wind_mw_new + other_ren
    gen_delta   = ren_mw_new - out["ren_mw"]

    out["total_gen_sim"] = out["total_generation_mw"] + gen_delta
    out["ren_share_sim"] = 100 * ren_mw_new / out["total_gen_sim"].replace(0, np.nan)
    out["load_sim"]      = out["load_mw"]

    out["price_sim"] = predict_price(
        model, out, load_mean, load_std,
        ren_col="ren_share_sim", load_col="load_sim", gen_col="total_gen_sim"
    )
    return out


def scenario_smoothing(dfx, reduce_pct=25):
    out = dfx.copy()
    m = out["renewable_share_pct"].mean()
    factor = 1 - max(0, min(100, reduce_pct))/100
    out["ren_share_sim"] = m + factor*(out["renewable_share_pct"] - m)
    out["total_gen_sim"] = out["total_generation_mw"]
    out["load_sim"]      = out["load_mw"]

    out["price_sim"] = predict_price(
        model, out, load_mean, load_std,
        ren_col="ren_share_sim", load_col="load_sim", gen_col="total_gen_sim"
    )
    return out


def scenario_demand_up(dfx, pct=10):
    out = dfx.copy()
    out["load_sim"]      = out["load_mw"]*(1 + pct/100)
    out["total_gen_sim"] = out["total_generation_mw"]
    out["ren_share_sim"] = 100 * out["ren_mw"] / out["total_gen_sim"].replace(0, np.nan)

    out["price_sim"] = predict_price(
        model, out, load_mean, load_std,
        ren_col="ren_share_sim", load_col="load_sim", gen_col="total_gen_sim"
    )
    return out


def scenario_hybrid(dfx, wind_pct=20, smooth_pct=25):
    s1 = scenario_wind_up(dfx, pct=wind_pct)
    m = s1["ren_share_sim"].mean()
    factor = 1 - max(0, min(100, smooth_pct))/100
    s1["ren_share_sim"] = m + factor*(s1["ren_share_sim"] - m)

    s1["price_sim"] = predict_price(
        model, s1, load_mean, load_std,
        ren_col="ren_share_sim", load_col="load_sim", gen_col="total_gen_sim"
    )
    return s1


# Build all scenarios
scen = {
    "Baseline": scenario_baseline(df),
    "Wind +20%": scenario_wind_up(df, 20),
    "Storage smoothing (−25% RSD)": scenario_smoothing(df, 25),
    "Demand +10%": scenario_demand_up(df, 10),
    "Hybrid: Wind +20% & smoothing": scenario_hybrid(df, 20, 25),
    "Your custom": None  # filled next
}

# Custom scenario from sliders
custom = scenario_hybrid(
    scenario_demand_up(df, demand_pct),
    wind_pct, smooth_pct
)
scen["Your custom"] = custom

# -----------------------------
# KPI table
# -----------------------------
rows = []
for name, d in scen.items():
    m = kpis(d, price_col="price_sim", ren_col="ren_share_sim",
             load_col=("load_sim" if "load_sim" in d else "load_mw"),
             gen_col=("total_gen_sim" if "total_gen_sim" in d else "total_generation_mw"))
    m["scenario"] = name
    rows.append(m)
kpi = pd.DataFrame(rows).set_index("scenario").loc[
    ["Baseline","Wind +20%","Storage smoothing (−25% RSD)","Demand +10%","Hybrid: Wind +20% & smoothing","Your custom"]
]
st.subheader("Scenario KPIs")
st.dataframe(kpi.style.format({
    "avg_price":"€{:.1f}", "p95_price":"€{:.1f}",
    "stress_pct":"{:.1f}%", "rsd":"{:.2f}", "ren_mean":"{:.1f}%"
}), use_container_width=True)

# -----------------------------
# Price trajectory — polished (Plotly)
# -----------------------------
st.subheader("Price trajectory (Baseline vs Your custom)")

# Optional smoothing toggle
smooth_on = st.checkbox("Show 3-hour smoothing", value=True)

base = scen["Baseline"][["ts_utc","price_sim"]].rename(columns={"price_sim":"Baseline"})
cust = scen["Your custom"][["ts_utc","price_sim"]].rename(columns={"price_sim":"Your custom"})

line_wide = base.merge(cust, on="ts_utc", how="inner").sort_values("ts_utc")
line_long = line_wide.melt(id_vars="ts_utc", var_name="scenario", value_name="price")

# 3h centered smoothing (only for display)
if smooth_on:
    for col in ["Baseline","Your custom"]:
        line_wide[f"{col} (3h MA)"] = line_wide[col].rolling(3, center=True, min_periods=1).mean()

# Build figure
fig = go.Figure()

# main lines
color_map = {"Baseline":"#3B82F6", "Your custom":"#F59E0B"}  # blue / amber
for col in ["Baseline", "Your custom"]:
    fig.add_trace(go.Scatter(
        x=line_wide["ts_utc"], y=line_wide[col],
        mode="lines",
        name=col,
        line=dict(width=3, color=color_map[col]),
        hovertemplate=f"{col}<br>%{{x|%Y-%m-%d %H:%M}}<br>€%{{y:.1f}}/MWh<extra></extra>",
    ))

# smoothing overlay (thinner, dashed)
if smooth_on:
    for col in ["Baseline", "Your custom"]:
        fig.add_trace(go.Scatter(
            x=line_wide["ts_utc"], y=line_wide[f"{col} (3h MA)"],
            mode="lines",
            name=f"{col} (3h MA)",
            line=dict(width=2, dash="dot", color=color_map[col]),
            hovertemplate=f"{col} (3h MA)<br>%{{x|%Y-%m-%d %H:%M}}<br>€%{{y:.1f}}/MWh<extra></extra>",
            showlegend=False
        ))

fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    xaxis=dict(
        title=None,
        showgrid=False,
        rangeselector=dict(buttons=[
            dict(count=2, label="2d", step="day", stepmode="backward"),
            dict(step="all", label="All")
        ]),
        rangeslider=dict(visible=True, thickness=0.08)
    ),
    yaxis=dict(
        title="€/MWh",
        gridcolor="rgba(180,180,180,0.25)",
        tickprefix="€"
    ),
)

st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Bar comparison (cleaned x-axis)
# -----------------------------
st.subheader("Bar comparison")

# Consistent order
SCEN_ORDER = [
    "Baseline",
    "Wind +20%",
    "Storage smoothing (−25% RSD)",
    "Demand +10%",
    "Hybrid: Wind +20% & smoothing",
    "Your custom"
]
kpi_plot = kpi.reindex(SCEN_ORDER)

def bar_metric(df, col, title, yfmt="raw", decimals=1):
    fig = px.bar(
        df.reset_index(),
        x="scenario", y=col,
        height=360,
        text=df[col].round(decimals),
        color_discrete_sequence=["#4F46E5"],
        hover_data={"scenario": True, col: True},
    )
    # Highlight your scenario
    colors = ["#22C55E" if s == "Your custom" else "#4F46E5" for s in df.index]
    fig.update_traces(marker_color=colors, textposition="outside")

    # Simplify layout
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=({"euro": "€ / MWh", "pct": "%"}).get(yfmt, col),
        bargap=0.35,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(size=13),
        showlegend=False,
    )

    # Hide x-axis labels but keep hover
    fig.update_xaxes(
        categoryorder="array", categoryarray=SCEN_ORDER,
        tickvals=[], showticklabels=False,
        showgrid=False, zeroline=False
    )

    # Format y-axis and add units
    if yfmt == "euro":
        fig.update_yaxes(tickprefix="€", showgrid=True, gridcolor="rgba(180,180,180,0.25)")
    elif yfmt == "pct":
        fig.update_yaxes(ticksuffix="%", showgrid=True, gridcolor="rgba(180,180,180,0.25)")
        fig.update_traces(texttemplate="%{text:.1f}%")
    else:
        fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.25)")
        fig.update_traces(texttemplate=f"%{{text:.{decimals}f}}")

    return fig

colA, colB, colC = st.columns(3)
with colA:
    st.plotly_chart(bar_metric(kpi_plot, "avg_price", "Average Price", yfmt="euro", decimals=1), use_container_width=True)
with colB:
    st.plotly_chart(bar_metric(kpi_plot, "stress_pct", "Stress Hours", yfmt="pct", decimals=1), use_container_width=True)
with colC:
    st.plotly_chart(bar_metric(kpi_plot, "rsd", "Renewable Stability (RSD)", yfmt="raw", decimals=2), use_container_width=True)

st.caption("Tip: Green bar = your custom scenario. Hover to see scenario names.")
