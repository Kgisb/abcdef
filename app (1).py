# app.py — JetLearn Predictability (M0 + Carryover) only
# - Preloads Master_sheet_DB.csv as modeling data
# - Running month data uploaded separately (partial current month file)
# - No hourly/daily features, no today/tomorrow blocks

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from datetime import date, timedelta

# --------------------- Page & Style ---------------------
st.set_page_config(page_title="JetLearn — Predictability", layout="wide", page_icon="🔮")
st.markdown("""
<style>
:root{ --text:#0f172a; --muted:#64748b; --border: rgba(15,23,42,.10); --card:#fff; }
html, body, [class*="css"] { font-family: ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial; }
.block-container { padding-top:.6rem; padding-bottom:.75rem; }
.head{ position:sticky; top:0; z-index:50; display:flex; gap:10px; align-items:center;
       padding:10px 12px; background:#0b1220; color:#fff; border-radius:12px; margin-bottom:10px; }
.head .title{ font-weight:800; font-size:1.02rem; margin-right:auto; }
.section-title{ font-weight:800; margin:.25rem 0 .6rem; color:var(--text); }
.kpi{ padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--card); }
.kpi .label{ color:var(--muted); font-size:.78rem; margin-bottom:4px; }
.kpi .value{ font-size:1.45rem; font-weight:800; line-height:1.05; color:var(--text); }
hr.soft{ border:0; height:1px; background:var(--border); margin:.6rem 0 1rem; }
.warn{ padding:8px 10px; border-left:4px solid #ef4444; background:#fff5f5; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2563eb","#06b6d4","#10b981","#f59e0b","#ef4444","#8b5cf6","#0ea5e9"]
REQUIRED_COLS = ["Pipeline","JetLearn Deal Source","Country","Student/Academic Counsellor","Deal Stage","Create Date"]

# --------------------- Utilities ---------------------
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

def alt_line(df,x,y,color=None,tooltip=None,height=260):
    enc=dict(x=alt.X(x,title=None), y=alt.Y(y,title=None), tooltip=tooltip or [])
    if color: enc["color"]=alt.Color(color, scale=alt.Scale(range=PALETTE))
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(height=height)

def month_start(ts): 
    return pd.Timestamp(ts).to_period("M").to_timestamp()

def month_days(ts: pd.Timestamp) -> int:
    m0 = month_start(ts)
    m1 = (m0 + pd.offsets.MonthBegin(1))
    return int((m1 - m0).days)

# --------------------- Header ---------------------
st.markdown('<div class="head"><div class="title">🔮 JetLearn — Predictability (Monthly)</div></div>', unsafe_allow_html=True)
st.markdown("<div class='warn'><b>Modeling data is preloaded from <code>Master_sheet_DB.csv</code>.</b> Upload the running month partial CSV separately below.</div>", unsafe_allow_html=True)

# --------------------- Global switches ---------------------
c0a, c0b = st.columns([2,2])
with c0a:
    exclude_invalid = st.checkbox("Exclude '1.2 Invalid Deal' from master", value=True)
with c0b:
    st.caption("No hourly/daily breakdowns. Forecast = M0 (same-month) + Carryover (lagged).")

# --------------------- Load Master (preloaded) ---------------------
MASTER_PATH = "Master_sheet_DB.csv"   # fixed path
try:
    df = robust_read_csv(MASTER_PATH)
    st.success(f"Master loaded ✅ ({MASTER_PATH}) — rows: {len(df):,}")
except Exception as e:
    st.error(f"Failed to load {MASTER_PATH}: {e}")
    st.stop()

df.columns = [c.strip() for c in df.columns]
missing=[c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns in master: {missing}\nAvailable: {list(df.columns)}")
    st.stop()

if exclude_invalid and "Deal Stage" in df.columns:
    df = df[~df["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()

df["Create Date"]  = pd.to_datetime(df["Create Date"], errors="coerce", dayfirst=True)
df["Create_Month"] = df["Create Date"].dt.to_period("M").dt.to_timestamp()

# --------------------- Current month partial uploader ---------------------
st.subheader("Current month partial file (upload here)")
uploaded_curr = st.file_uploader("Upload running month partial CSV (optional)", type=["csv"], key="curr")
df_curr = None
if uploaded_curr is not None:
    try:
        df_curr = robust_read_csv(BytesIO(uploaded_curr.getvalue()))
        df_curr.columns = [c.strip() for c in df_curr.columns]
        if exclude_invalid and "Deal Stage" in df_curr.columns:
            df_curr = df_curr[~df_curr["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()
        if "Create Date" in df_curr.columns:
            df_curr["Create Date"]  = pd.to_datetime(df_curr["Create Date"], errors="coerce", dayfirst=True)
            df_curr["Create_Month"] = df_curr["Create Date"].dt.to_period("M").dt.to_timestamp()
        st.success(f"Current month partial loaded ✅ — rows: {len(df_curr):,}")
    except Exception as e:
        st.warning(f"Could not read current-month partial: {e}")

st.markdown("<hr class='soft'/>", unsafe_allow_html=True)

# --------------------- Predictability controls ---------------------
st.subheader("Forecast controls")
c1,c2,c3 = st.columns(3)
with c1:
    today = pd.Timestamp.today().normalize()
    running_month = pd.Timestamp(year=today.year, month=today.month, day=1)
    # target month defaults to running month
    # allow a modest forward horizon
    mmin = df["Create_Month"].dropna().min()
    mmax = df["Create_Month"].dropna().max()
    if pd.isna(mmin) or pd.isna(mmax):
        st.error("No valid Create Date values in master."); st.stop()
    options = list(pd.date_range(start=mmin, end=(mmax + pd.offsets.MonthBegin(6)), freq="MS"))
    idx = options.index(running_month) if running_month in options else len(options)-1
    target_month = st.selectbox("Target month", options=options, index=idx, format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"))
    target_month = pd.Timestamp(target_month)
with c2:
    split_by = st.selectbox("Split forecast by (optional)", ["None","JetLearn Deal Source","Student/Academic Counsellor","Country","Pipeline"])
with c3:
    lookback_split = st.slider("Split lookback (months)", 3, 12, 6, 1)

st.caption("Master is used for modeling **excluding the running month**; the running month data should be uploaded above (partial).")

# --------------------- Build modeling frame (exclude running month from master) ---------------------
# Running month is the *calendar* current month (not target). Per requirement, do not use master rows of running month.
hist = df.copy()
hist = hist[hist["Create_Month"] < running_month]  # exclude running month rows from master modeling

# Detect payment column
def detect_payment_col(cols):
    for c in cols:
        cl=c.lower()
        if "payment" in cl and "received" in cl and "date" in cl: return c
    for c in cols:
        cl=c.lower()
        if "payment" in cl and "date" in cl: return c
    return None

PAY_COL = detect_payment_col(hist.columns)
if PAY_COL is None:
    st.error("Couldn't find a payment date column in master. Expected something like 'Payment Received Date'.")
    st.stop()

hist["Payment Received Date"] = pd.to_datetime(hist[PAY_COL], errors="coerce", dayfirst=True)
hist["Payment_Month"]        = hist["Payment Received Date"].dt.to_period("M").dt.to_timestamp()

# ---------- Cohort mechanics from master (historical only) ----------
hist["CM"] = hist["Create_Month"]
hist["PM"] = hist["Payment_Month"]
creates_by_CM = hist.groupby("CM")["Create Date"].count().rename("Creates_CM")

cohort = (
    hist.dropna(subset=["PM"])
        .groupby(["CM","PM"])["Payment Received Date"]
        .count().rename("Paid").reset_index()
)
cohort["lag_k"] = ((cohort["PM"].dt.year - cohort["CM"].dt.year)*12 + (cohort["PM"].dt.month - cohort["CM"].dt.month)).astype(int)
cohort = cohort[cohort["lag_k"] >= 0]

paid_by_k = cohort.groupby("lag_k")["Paid"].sum()
total_creates = float(creates_by_CM.sum())
if total_creates <= 0:
    st.error("No creates in master (before running month)."); st.stop()

lag_prob = (paid_by_k / max(total_creates,1.0)).reindex(range(0,18), fill_value=0.0).clip(lower=0.0)
M0_rate = float(lag_prob.get(0, 0.0))

# ---------- Estimate full-month creates for target_month ----------
def estimate_full_month_creates(df_current, target_month_ts: pd.Timestamp) -> float:
    """If target == running month and partial is uploaded, extrapolate from partial.
       Else use month-of-year average from master (history window)."""
    tm_moy = int(target_month_ts.month)

    # If we have partial current month and we're forecasting the running month, use it.
    if (df_current is not None) and ("Create Date" in df_current.columns) and (target_month_ts == running_month):
        cur = df_current.copy()
        cur["CM"] = cur["Create Date"].dt.to_period("M").dt.to_timestamp()
        cur_tm = cur[cur["CM"]==target_month_ts]
        if not cur_tm.empty:
            last_day = int(cur_tm["Create Date"].dt.day.max())
            days_in_month = month_days(target_month_ts)
            observed = int(len(cur_tm))
            # simple linear extrapolation by days elapsed
            frac = max(min(last_day/days_in_month, 0.999), 1.0/days_in_month)
            return float(observed/frac)

    # Fallback: month-of-year average creates from master (history-only)
    hist_cm = hist[hist["Create_Month"].notna()].copy()
    if hist_cm["Create_Month"].nunique() > 0:
        hist_cm["moy"] = hist_cm["Create_Month"].dt.month
        moy_avg = hist_cm.groupby("moy")["Create Date"].count() / hist_cm["Create_Month"].nunique()
        return float(moy_avg.get(tm_moy, moy_avg.mean() if len(moy_avg) else 0.0))
    return 0.0

est_creates_TM = estimate_full_month_creates(df_curr, target_month)
M0_expected    = float(est_creates_TM * M0_rate)

# ---------- Carryover into target_month from prior months (using master history only) ----------
carry = 0.0
for j in creates_by_CM.index[creates_by_CM.index < target_month]:
    k = (target_month.year - j.year)*12 + (target_month.month - j.month)
    if k <= 0: 
        continue
    p = float(lag_prob.get(k, 0.0))
    carry += float(creates_by_CM.get(j, 0.0)) * p
Carry_expected = float(carry)

total_forecast = float(M0_expected + Carry_expected)

# --------------------- KPIs ---------------------
k1,k2,k3 = st.columns(3)
with k1:
    st.markdown(f"<div class='kpi'><div class='label'>M0 (same-month) — {target_month:%b %Y}</div><div class='value'>{int(round(M0_expected)):,}</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><div class='label'>Carryover (lagged) — {target_month:%b %Y}</div><div class='value'>{int(round(Carry_expected)):,}</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><div class='label'>Total Forecast — {target_month:%b %Y}</div><div class='value'>{int(round(total_forecast)):,}</div></div>", unsafe_allow_html=True)

# --------------------- History + point forecast (lightweight chart) ---------------------
paid_hist = hist.dropna(subset=["Payment Received Date"]).copy()
monthly_paid = paid_hist["Payment_Month"].value_counts().rename_axis("Month").sort_index().rename("y").reset_index()
fut_df = pd.DataFrame({"Month":[target_month], "yhat":[total_forecast]})
if not monthly_paid.empty:
    monthly_paid["MonthStr"] = pd.to_datetime(monthly_paid["Month"]).dt.strftime("%Y-%m")
    fut_df["MonthStr"] = fut_df["Month"].dt.strftime("%Y-%m")

    hist_line = alt_line(monthly_paid, "MonthStr:O", "y:Q", tooltip=["MonthStr","y"]).encode(color=alt.value("#0ea5e9"))
    fut_line  = alt_line(fut_df,       "MonthStr:O", "yhat:Q", tooltip=["MonthStr","yhat"]).encode(color=alt.value("#ef4444"))
    st.altair_chart(alt.layer(hist_line, fut_line).resolve_scale(y='shared'), use_container_width=True)
else:
    st.info("No historical payments to plot (before running month).")

# --------------------- Optional split (monthly only) ---------------------
st.subheader("Split of forecast (optional)")
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
        st.warning("Not enough data to compute split proportions; using uniform split.")
        cats = paid_hist[split_by].dropna().astype(str).value_counts().index
        if len(cats) > 0:
            props = pd.Series(1/len(cats), index=cats)
    if len(props) > 0:
        split_table = props.rename("Prop").reset_index().rename(columns={"index":split_by})
        split_table["Forecast"] = (total_forecast * split_table["Prop"]).round(0)
        st.dataframe(split_table.sort_values("Forecast", ascending=False), use_container_width=True)
        st.download_button("Download split CSV", split_table.to_csv(index=False).encode("utf-8"),
                           file_name="forecast_split.csv", mime="text/csv")
