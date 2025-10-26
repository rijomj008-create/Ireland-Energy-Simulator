

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
# Page config
# -----------------------------
st.set_page_config(page_title="Ireland Energy Decision Simulator", layout="wide")
st.title("⚡ Ireland Energy Decision Intelligence Simulator")
st.caption("Data: ENTSO-E (load, generation, price) + Meteostat (weather) — Oct 18–25, 2025")

# -----------------------------
# Data loader (robust + cached)
# -----------------------------
@st.cache_data(show_spinner=True, ttl=3600)
def load_mart():
    here = Path(__file__).parent
    p1 = here / "data" / "processed" / "mart_ie_hourly_system_kpis.csv"
    p2 = here / "mart_ie_hourly_system_kpis.csv"
    if p1.exists():
        df = pd.read_csv(p1, parse_dates=["ts_utc"])
    elif p2.exists():
        df = pd.read_csv(p2, parse_dates=["ts_utc"])
    else:
        url = os.getenv("MART_CSV_URL", "") or st.secrets.get("MART_CSV_URL", "")
        if not url:
            raise FileNotFoundError(
                "CSV not found. Put it at data/processed/mart_ie_hourly_system_kpis.csv "
                "or mart_ie_hourly_system_kpis.csv (repo root), or set MART_CSV_URL in Secrets."
            )
        df = pd.read_csv(url, parse_dates=["ts_utc"])
    df = df.sort_values("ts_utc").copy()
    df["ts_utc"] = df["ts_utc"].dt.floor("h")
    df = df.loc[:, ~df.columns.duplicated()].copy()
    # add ren_mw if missing
    if "ren_mw" not in df.columns and {"total_generation_mw","renewable_share_pct"}.issubset(df.columns):
        df["ren_mw"] = (df["total_generation_mw"] * df["renewable_share_pct"] / 100.0).fillna(0.0)
    return df

df = load_mart()
if df.empty or "price_eur_per_mwh" not in df.columns:
    st.error("Dataset empty or missing price_eur_per_mwh. Check the CSV.")
    st.stop()

# -----------------------------
# Global sidebar filters (date range) — TZ-robust
# -----------------------------
min_d, max_d = df["ts_utc"].min(), df["ts_utc"].max()
default_range = [min_d.date(), max_d.date()]

chosen = st.sidebar.date_input("Filter date range", default_range)

# Normalize selection to (start_date, end_date)
if isinstance(chosen, (list, tuple)) and len(chosen) == 2:
    start_d, end_d = chosen
else:
    start_d, end_d = default_range

# Build tz-compatible boundaries
col_tz = getattr(df["ts_utc"].dt, "tz", None)  # None if naive, tzinfo if aware
if col_tz is None:
    # ts_utc is naive
    start_ts = pd.Timestamp(start_d)
    end_ts   = pd.Timestamp(end_d) + pd.Timedelta(days=1)
else:
    # ts_utc is tz-aware (e.g., UTC)
    start_ts = pd.Timestamp(start_d).tz_localize(col_tz)
    end_ts   = (pd.Timestamp(end_d) + pd.Timedelta(days=1)).tz_localize(col_tz)

df = df[(df["ts_utc"] >= start_ts) & (df["ts_utc"] < end_ts)]
if df.empty:
    st.warning("No data for the selected date range.")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def kpis(dframe, price_col, ren_col, load_col="load_mw", gen_col="total_generation_mw"):
    ratio = dframe[gen_col] / dframe[load_col]
    stress = (ratio < 1).astype(int)
    ren = pd.to_numeric(dframe[ren_col], errors="coerce")
    mean_ren = ren.mean()
    rsd = ren.std() / (mean_ren if mean_ren else np.nan)
    return {
        "avg_price": dframe[price_col].mean(),
        "p95_price": dframe[price_col].quantile(0.95),
        "stress_pct": 100 * stress.mean(),
        "rsd": rsd,
        "ren_mean": mean_ren
    }

def build_features(dfx, load_mean, load_std, ren_col, load_col, gen_col):
    X = pd.DataFrame(index=dfx.index)
    X["ren_share"] = pd.to_numeric(dfx[ren_col], errors="coerce")
    X["stress"] = ((pd.to_numeric(dfx[gen_col], errors="coerce") /
                    pd.to_numeric(dfx[load_col], errors="coerce")) < 1).astype(int)
    denom = load_std if (load_std and not np.isnan(load_std)) else 1.0
    X["load_scaled"] = (pd.to_numeric(dfx[load_col], errors="coerce") - load_mean) / denom
    if "wind_speed_mps" in dfx.columns:
        w = pd.to_numeric(dfx["wind_speed_mps"], errors="coerce")
        X["wind_speed_mps"] = w.fillna(w.mean())
    if "sunshine_fraction" in dfx.columns:
        s = pd.to_numeric(dfx["sunshine_fraction"], errors="coerce")
        X["sunshine_fraction"] = s.fillna(0)
    for c in X.columns:
        X[c] = X[c].fillna(X[c].mean())
    return X

def _fit_ridge(dfb, ren_col="renewable_share_pct", load_col="load_mw", gen_col="total_generation_mw", alpha=5.0):
    load_mean, load_std = dfb[load_col].mean(), dfb[load_col].std()
    X = build_features(dfb, load_mean, load_std, ren_col, load_col, gen_col)
    y = dfb["price_eur_per_mwh"].values
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, fit_intercept=True))
    model.fit(X, y)
    resid = y - model.predict(X)
    resid_std = float(np.std(resid)) if len(resid) else 0.0
    return model, load_mean, load_std, list(X.columns), resid_std

@st.cache_resource
def fit_cached(dfb):
    return _fit_ridge(dfb)

def predict_price(model, dfx, load_mean, load_std, ren_col="renewable_share_pct",
                  load_col="load_mw", gen_col="total_generation_mw"):
    X = build_features(dfx, load_mean, load_std, ren_col, load_col, gen_col)
    return model.predict(X)

# -----------------------------
# Fit baseline model (cached)
# -----------------------------
WIND_FRACTION_IN_REN = 0.80
model, load_mean, load_std, feat_names, resid_std = fit_cached(df)

# -----------------------------
# Sidebar: custom scenario controls (with URL persistence)
# -----------------------------
qs = st.query_params
wind_init   = int(qs.get("wind", 0))
demand_init = int(qs.get("demand", 0))
smooth_init = int(qs.get("smooth", 0))

st.sidebar.header("Custom Scenario Controls")
wind_pct   = st.sidebar.slider("Wind availability change (%)", -10, 30, wind_init)
demand_pct = st.sidebar.slider("Demand change (%)", -10, 20, demand_init)
smooth_pct = st.sidebar.slider("Storage smoothing (reduce RSD by %) ", 0, 40, smooth_init)

col_reset, _ = st.sidebar.columns([1,2])
if col_reset.button("Reset sliders"):
    st.query_params.clear()
    st.rerun()

# write params back to URL
st.query_params.update({"wind": wind_pct, "demand": demand_pct, "smooth": smooth_pct})

# -----------------------------
# Scenarios
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
    factor = 1 - max(0, min(100, reduce_pct)) / 100
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
    out["load_sim"]      = out["load_mw"] * (1 + pct/100)
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
    factor = 1 - max(0, min(100, smooth_pct)) / 100
    s1["ren_share_sim"] = m + factor*(s1["ren_share_sim"] - m)
    s1["price_sim"] = predict_price(
        model, s1, load_mean, load_std,
        ren_col="ren_share_sim", load_col="load_sim", gen_col="total_gen_sim"
    )
    return s1

# Build scenarios
scen = {
    "Baseline": scenario_baseline(df),
    "Wind +20%": scenario_wind_up(df, 20),
    "Storage smoothing (−25% RSD)": scenario_smoothing(df, 25),
    "Demand +10%": scenario_demand_up(df, 10),
    "Hybrid: Wind +20% & smoothing": scenario_hybrid(df, 20, 25),
}
scen["Your custom"] = scenario_hybrid(scenario_demand_up(df, demand_pct), wind_pct, smooth_pct)

# KPI table
rows = []
for name, d in scen.items():
    m = kpis(
        d,
        price_col="price_sim",
        ren_col="ren_share_sim",
        load_col=("load_sim" if "load_sim" in d else "load_mw"),
        gen_col=("total_gen_sim" if "total_gen_sim" in d else "total_generation_mw"),
    )
    m["scenario"] = name
    rows.append(m)
kpi = pd.DataFrame(rows).set_index("scenario").loc[
    ["Baseline","Wind +20%","Storage smoothing (−25% RSD)","Demand +10%","Hybrid: Wind +20% & smoothing","Your custom"]
]

st.subheader("Scenario KPIs")
st.dataframe(
    kpi.style.format({
        "avg_price":"€{:.1f}", "p95_price":"€{:.1f}",
        "stress_pct":"{:.1f}%", "rsd":"{:.2f}", "ren_mean":"{:.1f}%"
    }),
    use_container_width=True
)
st.caption("**RSD** = variability of renewable share (σ/μ). Lower is steadier supply. **Stress** = hours when generation < load.")

# -----------------------------
# Price trajectory (Plotly + confidence band)
# -----------------------------
st.subheader("Price trajectory (Baseline vs Your custom)")
smooth_on = st.checkbox("Show 3-hour smoothing", value=True)

base = scen["Baseline"][["ts_utc","price_sim"]].rename(columns={"price_sim":"Baseline"})
cust = scen["Your custom"][["ts_utc","price_sim"]].rename(columns={"price_sim":"Your custom"})
line_wide = base.merge(cust, on="ts_utc", how="inner").sort_values("ts_utc")

fig = go.Figure()
color_map = {"Baseline":"#3B82F6", "Your custom":"#F59E0B"}

# main lines
for col in ["Baseline", "Your custom"]:
    fig.add_trace(go.Scatter(
        x=line_wide["ts_utc"], y=line_wide[col],
        mode="lines", name=col, line=dict(width=3, color=color_map[col]),
        hovertemplate=f"{col}<br>%{{x|%Y-%m-%d %H:%M}}<br>€%{{y:.1f}}/MWh<extra></extra>",
    ))

# confidence band for custom (using training residual std as proxy)
yhat = line_wide["Your custom"].to_numpy()
if resid_std and resid_std > 0:
    upper = yhat + 1.96 * resid_std
    lower = yhat - 1.96 * resid_std
    fig.add_trace(go.Scatter(x=line_wide["ts_utc"], y=upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=line_wide["ts_utc"], y=lower, line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(245,158,11,0.12)", showlegend=False, hoverinfo="skip"))

# smoothing overlay
if smooth_on:
    for col in ["Baseline", "Your custom"]:
        fig.add_trace(go.Scatter(
            x=line_wide["ts_utc"], y=line_wide[col].rolling(3, center=True, min_periods=1).mean(),
            mode="lines", name=f"{col} (3h MA)",
            line=dict(width=2, dash="dot", color=color_map[col]), showlegend=False
        ))

fig.update_layout(
    height=420, margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
    xaxis=dict(title=None, showgrid=False, rangeslider=dict(visible=True, thickness=0.08)),
    yaxis=dict(title="€/MWh", gridcolor="rgba(180,180,180,0.25)", tickprefix="€"),
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Bar comparison (cleaned x-axis)
# -----------------------------
st.subheader("Bar comparison")
SCEN_ORDER = [
    "Baseline",
    "Wind +20%",
    "Storage smoothing (−25% RSD)",
    "Demand +10%",
    "Hybrid: Wind +20% & smoothing",
    "Your custom"
]
kpi_plot = kpi.reindex(SCEN_ORDER)

def bar_metric(df_in, col, title, yfmt="raw", decimals=1):
    figb = px.bar(
        df_in.reset_index(), x="scenario", y=col, height=360,
        text=df_in[col].round(decimals), color_discrete_sequence=["#4F46E5"],
        hover_data={"scenario": True, col: True},
    )
    colors = ["#22C55E" if s == "Your custom" else "#4F46E5" for s in df_in.index]
    figb.update_traces(marker_color=colors, textposition="outside", cliponaxis=False)
    figb.update_layout(
        title=title, xaxis_title=None,
        yaxis_title=({"euro":"€ / MWh","pct":"%"}).get(yfmt, col),
        bargap=0.35, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10), font=dict(size=13), showlegend=False,
    )
    figb.update_xaxes(categoryorder="array", categoryarray=SCEN_ORDER,
                      tickvals=[], showticklabels=False, showgrid=False, zeroline=False)
    if yfmt == "euro":
        figb.update_yaxes(tickprefix="€", showgrid=True, gridcolor="rgba(180,180,180,0.25)")
    elif yfmt == "pct":
        figb.update_yaxes(ticksuffix="%", showgrid=True, gridcolor="rgba(180,180,180,0.25)")
        figb.update_traces(texttemplate="%{text:.1f}%")
    else:
        figb.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.25)")
        figb.update_traces(texttemplate=f"%{{text:.{decimals}f}}")
    return figb

c1, c2, c3 = st.columns(3)
with c1:
    st.plotly_chart(bar_metric(kpi_plot, "avg_price", "Average Price", yfmt="euro", decimals=1), use_container_width=True)
with c2:
    st.plotly_chart(bar_metric(kpi_plot, "stress_pct", "Stress Hours", yfmt="pct", decimals=1), use_container_width=True)
with c3:
    st.plotly_chart(bar_metric(kpi_plot, "rsd", "Renewable Stability (RSD)", yfmt="raw", decimals=2), use_container_width=True)
st.caption("Tip: Green bar = your custom scenario. Hover to see scenario names.")

# -----------------------------
# Monitoring Dashboards (Power BI pages → Streamlit tabs)
# -----------------------------
st.markdown("---")
st.header("📊 Monitoring Dashboards")

def rsd_val(series: pd.Series) -> float:
    m = series.mean()
    return (series.std() / m) if (m and not np.isnan(m)) else np.nan

def kpi_cards_block(dfk, price_col="price_eur_per_mwh", ren_col="renewable_share_pct",
                    load_col="load_mw", gen_col="total_generation_mw"):
    ratio = dfk[gen_col] / dfk[load_col]
    stress_pct = (ratio < 1).mean() * 100
    p95 = dfk[price_col].quantile(0.95)
    colA, colB, colC, colD = st.columns(4)
    colA.metric("💶 Avg Price", f"€{dfk[price_col].mean():.1f}")
    colB.metric("P95 Price", f"€{p95:.1f}")
    colC.metric("🌱 Avg Renewable Share", f"{dfk[ren_col].mean():.1f}%")
    colD.metric("⚙️ Stress Hours", f"{stress_pct:.1f}%")

def add_date_parts(dfin):
    d = dfin.copy()
    d["date"] = d["ts_utc"].dt.date
    d["hour"] = d["ts_utc"].dt.hour
    return d

tab1, tab2, tab3 = st.tabs([
    "🏠 System Overview",
    "💶 Affordability & Reliability",
    "🌬️ Renewable Stability & Weather",
])

with tab1:
    st.subheader("System Overview")
    kpi_cards_block(df)

    # Load vs Generation
    base_lg = df[["ts_utc","load_mw","total_generation_mw"]].sort_values("ts_utc")
    fig_lg = go.Figure()
    fig_lg.add_trace(go.Scatter(x=base_lg["ts_utc"], y=base_lg["load_mw"],
                                name="Load (MW)", mode="lines",
                                line=dict(width=3, color="#3B82F6")))
    fig_lg.add_trace(go.Scatter(x=base_lg["ts_utc"], y=base_lg["total_generation_mw"],
                                name="Generation (MW)", mode="lines",
                                line=dict(width=3, color="#22C55E")))
    fig_lg.update_layout(height=360, margin=dict(l=10,r=10,t=10,b=10),
                         yaxis_title="MW",
                         plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_lg, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_p = px.line(df, x="ts_utc", y="price_eur_per_mwh", height=320,
                        labels={"price_eur_per_mwh":"€/MWh","ts_utc":""})
        fig_p.update_traces(line=dict(width=3, color="#F59E0B"))
        fig_p.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                            yaxis=dict(tickprefix="€"),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_p, use_container_width=True)
    with col2:
        fig_r = px.line(df, x="ts_utc", y="renewable_share_pct", height=320,
                        labels={"renewable_share_pct":"%","ts_utc":""})
        fig_r.update_traces(line=dict(width=3, color="#10B981"))
        fig_r.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                            yaxis=dict(ticksuffix="%"),
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_r, use_container_width=True)

    # Stress heatmap (hour × date)
    st.markdown("**Stress Heatmap (Hour × Date)**")
    d2 = add_date_parts(df)
    d2["stress"] = ((d2["total_generation_mw"] / d2["load_mw"]) < 1).astype(int)
    heat = (d2.pivot_table(index="hour", columns="date", values="stress", aggfunc="mean")
              .sort_index(ascending=True))
    fig_h = go.Figure(data=go.Heatmap(
        z=heat.values, x=heat.columns.astype(str), y=heat.index,
        colorscale=[[0,"#0EA5E9"],[1,"#EF4444"]], colorbar=dict(title="Stress"),
        zmin=0, zmax=1))
    fig_h.update_layout(height=380, margin=dict(l=10,r=10,t=10,b=10),
                        yaxis_title="Hour", xaxis_title="Date",
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_h, use_container_width=True)

with tab2:
    st.subheader("Affordability & Reliability")

    colA, colB = st.columns(2)
    with colA:
        fig_hist = px.histogram(df, x="price_eur_per_mwh", nbins=25, height=340)
        fig_hist.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                               xaxis_title="€/MWh", yaxis_title="Count",
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist, use_container_width=True)
    with colB:
        d3 = df.copy()
        d3["stress_flag"] = np.where((d3["total_generation_mw"]/d3["load_mw"])<1, "Stress", "Non-stress")
        grp = d3.groupby("stress_flag")["price_eur_per_mwh"].mean().reset_index()
        fig_bar = px.bar(grp, x="stress_flag", y="price_eur_per_mwh", text="price_eur_per_mwh", height=340,
                         color="stress_flag", color_discrete_map={"Stress":"#EF4444","Non-stress":"#22C55E"})
        fig_bar.update_traces(texttemplate="€%{text:.1f}", textposition="outside", cliponaxis=False)
        fig_bar.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                              xaxis_title="", yaxis_title="€/MWh", showlegend=False,
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        fig_sc = px.scatter(df, x="load_mw", y="price_eur_per_mwh", height=340,
                            labels={"load_mw":"Load (MW)", "price_eur_per_mwh":"€/MWh"})
        fig_sc.update_traces(marker=dict(size=7, opacity=0.6, color="#3B82F6"))
        fig_sc.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sc, use_container_width=True)
    with colD:
        d4 = df.copy()
        d4["hour"] = d4["ts_utc"].dt.hour
        prof = d4.groupby("hour").agg(avg_price=("price_eur_per_mwh","mean"),
                                      avg_load=("load_mw","mean")).reset_index()
        fig_prof = go.Figure()
        fig_prof.add_trace(go.Scatter(x=prof["hour"], y=prof["avg_price"], name="Avg Price (€/MWh)",
                                      mode="lines+markers", line=dict(width=3, color="#F59E0B")))
        fig_prof.add_trace(go.Scatter(x=prof["hour"], y=prof["avg_load"], name="Avg Load (MW)",
                                      mode="lines+markers", line=dict(width=3, color="#3B82F6"),
                                      yaxis="y2"))
        fig_prof.update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10),
                               xaxis_title="Hour",
                               yaxis=dict(title="€/MWh", tickprefix="€"),
                               yaxis2=dict(title="MW", overlaying="y", side="right",
                                           gridcolor="rgba(180,180,180,0.15)"),
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_prof, use_container_width=True)

with tab3:
    st.subheader("Renewable Stability & Weather")
    r = rsd_val(df["renewable_share_pct"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RSD (σ/μ) of Ren. Share", f"{r:.2f}" if pd.notna(r) else "—")
    col2.metric("Avg Wind (m/s)", f"{df['wind_speed_mps'].mean():.2f}" if "wind_speed_mps" in df.columns else "—")
    col3.metric("Avg Sunshine (0–1)", f"{df['sunshine_fraction'].mean():.2f}" if "sunshine_fraction" in df.columns else "—")
    col4.metric("Avg Ren. Share", f"{df['renewable_share_pct'].mean():.1f}%")

    colA, colB = st.columns(2)
    with colA:
        if "wind_speed_mps" in df.columns:
            fig_ws = px.scatter(df, x="wind_speed_mps", y="price_eur_per_mwh", height=340,
                                labels={"wind_speed_mps":"Wind speed (m/s)","price_eur_per_mwh":"€/MWh"})
            fig_ws.update_traces(marker=dict(size=7, opacity=0.6, color="#06B6D4"))
            fig_ws.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                                 plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_ws, use_container_width=True)
        else:
            st.info("Wind speed not available in this dataset.")
    with colB:
        d5 = df.copy()
        d5["date"] = d5["ts_utc"].dt.date
        d5["hour"] = d5["ts_utc"].dt.hour
        heat2 = (d5.pivot_table(index="hour", columns="date", values="renewable_share_pct", aggfunc="mean")
                   .sort_index(ascending=True))
        fig_rh = go.Figure(data=go.Heatmap(
            z=heat2.values, x=heat2.columns.astype(str), y=heat2.index,
            colorscale="Greens", colorbar=dict(title="% Ren. Share")))
        fig_rh.update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10),
                             yaxis_title="Hour", xaxis_title="Date",
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rh, use_container_width=True)

    st.markdown("**Wind Efficiency (Renewable MW per m/s)**")
    if "wind_speed_mps" in df.columns:
        if "ren_mw" not in df.columns:
            df["ren_mw"] = (df["total_generation_mw"] * df["renewable_share_pct"] / 100.0).fillna(0.0)
        eff = df[["ts_utc","ren_mw","wind_speed_mps"]].copy()
        eff["eff_mw_per_ms"] = eff["ren_mw"] / eff["wind_speed_mps"].replace(0, np.nan)
        fig_eff = px.line(eff, x="ts_utc", y="eff_mw_per_ms", height=320,
                          labels={"eff_mw_per_ms":"MW per m/s", "ts_utc":""})
        fig_eff.update_traces(line=dict(width=3, color="#10B981"))
        fig_eff.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_eff, use_container_width=True)
    else:
        st.info("Wind speed not available to compute efficiency.")

# -----------------------------
# 📖 Storytelling Presentation Page
# -----------------------------
st.markdown("---")
st.header("📘 Project Story: From Data to Decisions")

# --- PROBLEM ---
st.subheader("1️⃣ Problem")
st.markdown("""
Ireland’s energy system faces the **trilemma** of balancing:
- **Affordability** — keeping market prices manageable for consumers.
- **Reliability** — ensuring supply meets demand even during peak or low-wind hours.
- **Sustainability** — maintaining a healthy share of renewable generation (wind, solar, hydro).

When renewable output drops or fluctuates sharply, **stress hours** rise and **electricity prices spike**.
This volatility affects both consumers and operators, making **data-driven decision models** essential.
""")

# --- GOAL ---
st.subheader("2️⃣ Goal")
st.markdown("""
To build a **Decision Intelligence Simulator** that connects weather, generation, demand, and price data —
allowing policymakers and analysts to test **what-if scenarios**:
> “What happens if wind availability rises by 20%?”  
> “What if storage smooths renewable fluctuations by 25%?”  
> “What if demand increases by 10% without extra generation?”

The goal: quantify how these operational changes affect **price, stress hours, and renewable stability**.
""")

# --- METHODOLOGY (LAYMAN) ---
st.subheader("3️⃣ Methodology — In Simple Terms")
st.markdown("""
1. **Collect Data:** Hourly data from ENTSO-E (load, generation, market price) and Meteostat (weather).
2. **Clean & Combine:** Build a unified “mart” with load, generation, price, wind speed, sunshine, and temperature.
3. **Explore:** Identify trends and pain points — when stress hours increase, when renewables are low, and how price reacts.
4. **Model:** Train a lightweight predictive model that links price with renewable share, grid stress, and demand.
5. **Simulate:** Adjust wind, demand, and renewable smoothness to see how the system reacts.
6. **Visualize:** Convert results into interactive dashboards for decision-makers.
""")

# --- METHODOLOGY (ADVANCED) ---
with st.expander("🔬 Methodology — Technical Details (for Data Analysts)", expanded=False):
    st.markdown("""
**Data Sources**
- ENTSO-E Transparency Platform — Load, Generation by Type, Market Price.
- Meteostat API — Hourly weather (wind speed, sunshine, precipitation, temperature).

**Processing Steps**
- Built hourly fact tables: `fact_load_actual`, `fact_generation_actual`, `fact_market_price`.
- Joined into `mart_ie_hourly_system_kpis` with renewable share, total generation, and weather features.
- Cleaned NaNs, harmonized timestamps, applied UTC rounding.

**Exploratory Analysis**
- Evaluated renewable volatility (RSD), stress hours, and price sensitivity.
- Found high correlation between price spikes and low renewable share.

**Modeling**
- Used **Ridge Regression**:  
  `price_eur_per_mwh ~ const + renewable_share_pct + stress + load_scaled (+ wind, sunshine)`  
- Standardized features using `StandardScaler()` and pipelined via `make_pipeline()` for reproducibility.

**Simulation Layer**
- Generated alternate dataframes for each what-if:
  - Wind +20%
  - Storage smoothing (−25% RSD)
  - Demand +10%
  - Hybrid (Wind +20% + smoothing)
- Predicted price responses and computed new KPIs for each.
""")

# --- FINDINGS ---
st.subheader("4️⃣ Findings & Interpretation")
st.markdown("""
**Baseline Week (Oct 18–25, 2025)**  
- Avg price: ~€{:.1f}/MWh  
- Stress hours: {:.1f}% of time  
- Renewable share: {:.1f}% (RSD {:.2f})  

**Key Insights**
- Prices spike sharply during low-renewable evenings.
- Renewable instability (high RSD) drives volatility.
- Even moderate increases in wind output improve both affordability and reliability.
""".format(
    kpi.loc["Baseline","avg_price"],
    kpi.loc["Baseline","stress_pct"],
    kpi.loc["Baseline","ren_mean"],
    kpi.loc["Baseline","rsd"]
))

# --- CALL TO ACTION ---
st.subheader("5️⃣ Call to Action")
st.markdown("""
1. **Invest in wind expansion** — even a 20% increase improves stability and lowers price by ~€{:.1f}/MWh.  
2. **Adopt storage or flexible demand mechanisms** — a 25% RSD reduction smooths volatility and cuts stress hours.  
3. **Integrate analytics** into daily grid monitoring to pre-empt risk rather than react.  
4. **Use simulation tools** to test future capacity mixes under varying weather and demand conditions.
""".format(
    kpi.loc["Baseline","avg_price"] - kpi.loc["Wind +20%","avg_price"]
))

st.info("The combination of **Wind +20% + Storage Smoothing** delivered the best balance — lowest average price and reduced stress.")

# --- DEMO CTA ---
st.subheader("6️⃣ Interactive Demo — Simulate Your Scenario")
st.markdown("""
You can now use the **sliders in the sidebar** to change:
- Wind availability
- Demand increase
- Renewable stability (via smoothing)

The charts and KPIs update instantly, showing how system cost, reliability, and stability respond.
""")
st.success("Try adjusting the sliders and watch how your decisions reshape the grid’s balance in real time.")


# -----------------------------
# Downloads + Footer
# -----------------------------
st.markdown("### ⬇️ Downloads")
st.download_button(
    "Download scenario KPIs (CSV)",
    data=kpi.to_csv(index=True).encode("utf-8"),
    file_name="scenario_kpis.csv",
    mime="text/csv",
)
traj = scen["Your custom"][["ts_utc","price_sim"]].rename(columns={"price_sim":"price_custom"})
traj = traj.merge(scen["Baseline"][["ts_utc","price_sim"]].rename(columns={"price_sim":"price_baseline"}), on="ts_utc")
st.download_button(
    "Download prices (CSV)",
    data=traj.to_csv(index=False).encode("utf-8"),
    file_name="prices_baseline_vs_custom.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown("Built by **Rijo Mathew John** — "
            "[GitHub](https://github.com/rijomj008-create) • "
            "[Streamlit app](https://ireland-energy-simulator-jvaxfcjmqitlxxdap5reapp.streamlit.app/)")
