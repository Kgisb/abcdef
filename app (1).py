# app.py — Predictability only (M0 + Carryover with partial-month extrapolation)
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from datetime import date, timedelta

st.set_page_config(page_title="JetLearn Predictability", layout="wide", page_icon="🔮")
st.markdown("""
<style>
:root{ --text:#0f172a; --muted:#64748b; --border: rgba(15,23,42,.10); --card:#fff; }
html, body, [class*="css"] { font-family: ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial; }
.block-container { padding-top:.6rem; padding-bottom:.8rem; }
.head{ position:sticky; top:0; z-index:50; display:flex; gap:10px; align-items:center; padding:10px 12px; background:#0b1220; color:#fff; border-radius:12px; margin-bottom:10px;}
.head .title{ font-weight:800; font-size:1.02rem; margin-right:auto; }
.kpi{ padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--card); }
.kpi .label{ color:var(--muted); font-size:.78rem; margin-bottom:4px; }
.kpi .value{ font-size:1.45rem; font-weight:800; line-height:1.05; color:var(--text); }
hr.soft{ border:0; height:1px; background:var(--border); margin:.6rem 0 1rem; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2563eb","#06b6d4","#10b981","#f59e0b","#ef4444","#8b5cf6","#0ea5e9"]

# ---------- Helpers ----------
def robust_read_csv(file_or_path):
    for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
        try:
            if hasattr(file_or_path,"read"):
                file_or_path.seek(0)
                return pd.read_csv(file_or_path, encoding=enc)
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError("Could not read the CSV with tried encodings.")

def detect_payment_col(cols):
    for c in cols:
        cl=c.lower()
        if "payment" in cl and "received" in cl and "date" in cl: return c
    for c in cols:
        cl=c.lower()
        if "payment" in cl and "date" in cl: return c
    return None

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def alt_line(df,x,y,color=None,tooltip=None,height=260):
    enc=dict(x=alt.X(x,title=None), y=alt.Y(y,title=None), tooltip=tooltip or [])
    if color: enc["color"]=alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=height)

# ---------- Header ----------
st.markdown('<div class="head"><div class="title">🔮 JetLearn — Predictability (Monthly)</div></div>', unsafe_allow_html=True)

# ---------- DATA: master (preloaded) + current-month partial (separate) ----------
HIST_DEFAULT_PATH = "Master_sheet_DB_10percent.csv"  # <-- your preloaded master file

c1,c2 = st.columns([3,3])
with c1:
    st.caption("Master modeling data (preloaded)")
    hist_path = st.text_input("Master CSV path (preloaded, used for modeling only)", value=HIST_DEFAULT_PATH)
with c2:
    st.caption("Running month partial (upload separately)")
    uploaded_curr = st.file_uploader("Upload current/running month PARTIAL CSV (optional)", type=["csv"], key="CURR_UP")

exclude_invalid = st.checkbox("Exclude '1.2 Invalid Deal' from both files", value=True)

# Load master
try:
    df_master = robust_read_csv(hist_path)
    st.success(f"Master loaded ✅ — rows: {len(df_master):,}")
except Exception as e:
    st.error(f"Failed to load master: {e}")
    st.stop()

df_master.columns = [c.strip() for c in df_master.columns]
required = ["Pipeline","JetLearn Deal Source","Country","Student/Academic Counsellor","Deal Stage","Create Date"]
miss = [c for c in required if c not in df_master.columns]
if miss:
    st.error(f"Missing required columns in master: {miss}\nAvailable: {list(df_master.columns)}")
    st.stop()

if exclude_invalid and "Deal Stage" in df_master.columns:
    df_master = df_master[~df_master["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()

# Coerce dates
df_master["Create Date"] = pd.to_datetime(df_master["Create Date"], errors="coerce", dayfirst=True)
df_master["Create_Month"] = df_master["Create Date"].dt.to_period("M").dt.to_timestamp()

# Load current/running month partial (separate)
df_curr = None
if uploaded_curr is not None:
    try:
        df_curr = robust_read_csv(BytesIO(uploaded_curr.getvalue()))
        df_curr.columns = [c.strip() for c in df_curr.columns]
        if exclude_invalid and "Deal Stage" in df_curr.columns:
            df_curr = df_curr[~df_curr["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()
        if "Create Date" in df_curr.columns:
            df_curr["Create Date"] = pd.to_datetime(df_curr["Create Date"], errors="coerce", dayfirst=True)
            df_curr["Create_Month"] = df_curr["Create Date"].dt.to_period("M").dt.to_timestamp()
        st.success(f"Running month partial loaded ✅ — rows: {len(df_curr):,}")
    except Exception as e:
        st.warning(f"Could not load current partial: {e}. Proceeding without it.")

# ---------- Predictability controls ----------
PAY_COL = detect_payment_col(df_master.columns)
if PAY_COL is None:
    st.error("Couldn't find a payment date column in master. Add 'Payment Received Date'.")
    st.stop()

base = df_master.copy()
base["Payment Received Date"] = pd.to_datetime(base[PAY_COL], errors="coerce", dayfirst=True)
base["Payment_Month"] = base["Payment Received Date"].dt.to_period("M").dt.to_timestamp()

today = pd.Timestamp.today().normalize()
run_month = pd.Timestamp(year=today.year, month=today.month, day=1)  # running month

c3,c4,c5 = st.columns(3)
with c3:
    # allow a window from history into a few months ahead
    mmin = min(base["Create_Month"].dropna().min(), base["Payment_Month"].dropna().min())
    mmax = max(base["Create_Month"].dropna().max(), base["Payment_Month"].dropna().max())
    options = list(pd.date_range(start=mmin, end=(mmax + pd.offsets.MonthBegin(6)), freq="MS"))
    default_idx = options.index(run_month) if run_month in options else len(options)-1
    target_month = st.selectbox("Target month", options=options, index=default_idx,
                                format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"))
    target_month = pd.Timestamp(target_month)
with c4:
    split_by = st.selectbox("Split forecast by", ["None","JetLearn Deal Source","Student/Academic Counsellor","Country","Pipeline"])
with c5:
    lookback_split = st.slider("Split lookback (months)", 3, 12, 6, 1)

st.caption("Forecast = **M0 (same-month from creates in target month)** + **Carryover (lagged from prior months)**.")

# ---------- IMPORTANT: build the model on master EXCLUDING the running month ----------
hist = base[base["Create_Month"] < run_month].copy()

# Creates-by-CM and cohort payments
creates_by_CM = hist.groupby("Create_Month")["Create Date"].count().rename("Creates_CM")
cohort = (
    hist.dropna(subset=["Payment Received Date"])
        .groupby(["Create_Month","Payment_Month"])["Payment Received Date"]
        .count().rename("Paid").reset_index()
)
cohort["lag_k"] = ((cohort["Payment_Month"].dt.year - cohort["Create_Month"].dt.year)*12
                   + (cohort["Payment_Month"].dt.month - cohort["Create_Month"].dt.month)).astype(int)
cohort = cohort[cohort["lag_k"] >= 0]

paid_by_k = cohort.groupby("lag_k")["Paid"].sum()
total_creates = creates_by_CM.sum()
if total_creates <= 0:
    st.error("No creates in master (before running month).")
    st.stop()

lag_prob = (paid_by_k / max(total_creates,1.0)).reindex(range(0,18), fill_value=0.0).clip(lower=0.0)
M0_rate = float(lag_prob.get(0, 0.0))

# ---------- Estimate creates in target month ----------
def month_days(ts: pd.Timestamp) -> int:
    m0 = ts.to_period("M").to_timestamp()
    m1 = (m0 + pd.offsets.MonthBegin(1))
    return int((m1 - m0).days)

def estimate_full_month_creates(df_current, target_month_ts: pd.Timestamp) -> float:
    """Use running-month partial only if it matches the target month; else use seasonal average from master history (excluding running month)."""
    if (df_current is not None) and ("Create Date" in df_current.columns):
        cur = df_current.copy()
        cur["CM"] = cur["Create Date"].dt.to_period("M").dt.to_timestamp()
        cur_tm = cur[cur["CM"] == target_month_ts]
        if not cur_tm.empty and (target_month_ts == run_month):
            # day-weighted extrapolation
            last_day = int(cur_tm["Create Date"].dt.day.max())
            days_in_month = month_days(target_month_ts)
            observed = int(len(cur_tm))
            frac = max(min(last_day/days_in_month, 0.999), 1.0/days_in_month)
            return observed/frac

    # seasonal fallback on historical master (already excludes running month)
    hist_cm = hist[hist["Create_Month"].notna()].copy()
    if hist_cm["Create_Month"].nunique() == 0:
        return 0.0
    hist_cm["moy"] = hist_cm["Create_Month"].dt.month
    moy_avg = hist_cm.groupby("moy")["Create Date"].count() / hist_cm["Create_Month"].nunique()
    return float(moy_avg.get(int(target_month_ts.month), moy_avg.mean() if len(moy_avg) else 0.0))

est_creates_TM = estimate_full_month_creates(df_curr, target_month)
M0_expected = float(est_creates_TM * M0_rate)

# ---------- Carryover into target month from prior months (model uses hist only) ----------
carry = 0.0
for j in creates_by_CM.index[creates_by_CM.index < target_month]:
    k = (target_month.year - j.year)*12 + (target_month.month - j.month)
    if k <= 0: continue
    p = float(lag_prob.get(k, 0.0))
    carry += float(creates_by_CM.get(j, 0.0)) * p
Carry_expected = float(carry)
total_forecast = float(M0_expected + Carry_expected)

# ---------- KPIs ----------
k1,k2,k3 = st.columns(3)
with k1: st.markdown(f"<div class='kpi'><div class='label'>M0 (same-month) — {target_month:%b %Y}</div><div class='value'>{int(round(M0_expected)):,}</div></div>", unsafe_allow_html=True)
with k2: st.markdown(f"<div class='kpi'><div class='label'>Carryover (lagged) — {target_month:%b %Y}</div><div class='value'>{int(round(Carry_expected)):,}</div></div>", unsafe_allow_html=True)
with k3: st.markdown(f"<div class='kpi'><div class='label'>Total Forecast — {target_month:%b %Y}</div><div class='value'>{int(round(total_forecast)):,}</div></div>", unsafe_allow_html=True)

# ---------- History + point forecast chart ----------
paid_hist = hist.dropna(subset=["Payment Received Date"]).copy()
monthly_paid = paid_hist["Payment_Month"].value_counts().rename_axis("Month").sort_index().rename("y").reset_index()
if not monthly_paid.empty:
    monthly_paid["MonthStr"] = pd.to_datetime(monthly_paid["Month"]).dt.strftime("%Y-%m")
fut_df = pd.DataFrame({"Month":[target_month], "yhat":[total_forecast]})
fut_df["MonthStr"] = fut_df["Month"].dt.strftime("%Y-%m")

if not monthly_paid.empty:
    hist_line = alt_line(monthly_paid, "MonthStr:O", "y:Q", tooltip=["MonthStr","y"]).encode(color=alt.value("#0ea5e9"))
    fut_line  = alt_line(fut_df,       "MonthStr:O", "yhat:Q", tooltip=["MonthStr","yhat"]).encode(color=alt.value("#ef4444"))
    st.altair_chart(alt.layer(hist_line, fut_line).resolve_scale(y='shared'), use_container_width=True)
else:
    st.info("No historical payments to plot (before running month).")

# ---------- Optional split of forecast (monthly) ----------
st.markdown("### Split of forecast (optional)")
if split_by == "None":
    st.info("No split selected.")
else:
    def eb_smooth_props(counts_by_cat: pd.Series, prior_props: pd.Series, prior_strength: float = 5.0):
        counts = counts_by_cat.astype(float); total = counts.sum()
        if total <= 0:
            pp = prior_props.fillna(0).clip(0,1)
            return (pp/pp.sum()) if pp.sum()>0 else pp
        cats = counts.index; prior = prior_props.reindex(cats).fillna(0.0)
        smoothed = (counts + prior_strength * prior) / (total + prior_strength)
        s = smoothed.sum(); return smoothed / s if s>0 else smoothed

    def historical_split_props(df_paid: pd.DataFrame, split_col: str, lookback_months: int = 6):
        if split_col not in df_paid.columns: return pd.Series(dtype=float)
        dfp = df_paid.copy()
        dfp["PaymentMonth"] = dfp["Payment Received Date"].dt.to_period("M").dt.to_timestamp()
        global_counts = dfp.groupby(split_col)["Payment Received Date"].count()
        gp_total = global_counts.sum()
        prior_props = (global_counts / gp_total) if gp_total>0 else global_counts
        months = sorted(dfp["PaymentMonth"].dropna().unique())
        take = months[-lookback_months:] if len(months)>=lookback_months else months
        recent = dfp[dfp["PaymentMonth"].isin(take)]
        recent_counts = recent.groupby(split_col)["Payment Received Date"].count().sort_values(ascending=False)
        return eb_smooth_props(recent_counts, prior_props, prior_strength=5.0)

    props = historical_split_props(paid_hist, split_by, lookback_months=lookback_split)
    if props.empty or props.sum() == 0:
        st.warning("Not enough data to compute split proportions; using uniform split across observed categories.")
        cats = paid_hist[split_by].dropna().astype(str).value_counts().index
        if len(cats) > 0:
            props = pd.Series(1/len(cats), index=cats)
    if len(props) > 0:
        split_table = props.rename("Prop").reset_index().rename(columns={"index":split_by})
        split_table["Forecast"] = (total_forecast * split_table["Prop"]).round(0)
        st.dataframe(split_table.sort_values("Forecast", ascending=False), use_container_width=True)
        st.download_button("Download split CSV", to_csv_bytes(split_table), file_name="forecast_split.csv", mime="text/csv")
