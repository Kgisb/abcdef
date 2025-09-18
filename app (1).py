# app.py — JetLearn Insights (MTD/Cohort) + Predictability (M0 + Carryover)
# - Preloads Master_sheet_DB.csv as master dataset
# - Current month partial uploaded separately
# - Defensive datetime conversion to avoid .dt errors

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO
from datetime import date, timedelta

# --------------------- Page & Style ---------------------
st.set_page_config(page_title="JetLearn Insights + Predictability", layout="wide", page_icon="📊")
st.markdown("""
<style>
:root{ --text:#0f172a; --muted:#64748b; --blue:#2563eb; --border: rgba(15,23,42,.10);
       --card:#fff; --bg:#f8fafc; }
html, body, [class*="css"] { font-family: ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial; }
.block-container { padding-top:.6rem; padding-bottom:.75rem; }
.head { position:sticky; top:0; z-index:50; display:flex; gap:10px; align-items:center;
        padding:10px 12px; background:#0b1220; color:#fff; border-radius:12px; margin-bottom:10px; }
.head .title { font-weight:800; font-size:1.02rem; margin-right:auto; }
.section-title { font-weight:800; margin:.25rem 0 .6rem; color:var(--text); }
.kpi { padding:10px 12px; border:1px solid var(--border); border-radius:12px; background:var(--card); }
.kpi .label { color:var(--muted); font-size:.78rem; margin-bottom:4px; }
.kpi .value { font-size:1.45rem; font-weight:800; line-height:1.05; color:var(--text); }
hr.soft { border:0; height:1px; background:var(--border); margin:.6rem 0 1rem; }
.warn { padding:8px 10px; border-left:4px solid #ef4444; background:#fff5f5; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

PALETTE = ["#2563eb","#06b6d4","#10b981","#f59e0b","#ef4444","#8b5cf6","#0ea5e9"]
REQUIRED_COLS = ["Pipeline","JetLearn Deal Source","Country","Student/Academic Counsellor","Deal Stage","Create Date"]

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

def detect_measure_date_columns(df: pd.DataFrame):
    """Find all date-like columns except Create Date; coerce to datetime."""
    date_like=[]
    for col in df.columns:
        if col == "Create Date": continue
        cl = col.lower()
        if any(k in cl for k in ["date","time","timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum()>0:
                df[col] = parsed
                date_like.append(col)
    if "Payment Received Date" in date_like:
        date_like = ["Payment Received Date"] + [c for c in date_like if c!="Payment Received Date"]
    return date_like

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
st.markdown('<div class="head"><div class="title">📊 JetLearn — Insights & Predictability</div></div>', unsafe_allow_html=True)

# --------------------- Master Data (preloaded) ---------------------
MASTER_PATH = "Master_sheet_DB.csv"   # fixed path
exclude_invalid = st.checkbox("Exclude '1.2 Invalid Deal'", value=True)

try:
    df_master = robust_read_csv(MASTER_PATH)
    st.success(f"Master loaded ✅ ({MASTER_PATH}) — rows: {len(df_master):,}")
except Exception as e:
    st.error(f"Failed to load {MASTER_PATH}: {e}")
    st.stop()

df_master.columns = [c.strip() for c in df_master.columns]
missing=[c for c in REQUIRED_COLS if c not in df_master.columns]
if missing:
    st.error(f"Missing required columns in master: {missing}\nAvailable: {list(df_master.columns)}")
    st.stop()

if exclude_invalid and "Deal Stage" in df_master.columns:
    df_master = df_master[~df_master["Deal Stage"].astype(str).str.strip().eq("1.2 Invalid Deal")].copy()

df_master["Create Date"]  = pd.to_datetime(df_master["Create Date"], errors="coerce", dayfirst=True)
df_master["Create_Month"] = df_master["Create Date"].dt.to_period("M").dt.to_timestamp()

# Coerce date-like cols in-place
date_like_cols = detect_measure_date_columns(df_master)

# --------------------- Current month partial uploader ---------------------
st.subheader("Current month partial file (upload here)")
uploaded_curr = st.file_uploader("Upload running month partial CSV", type=["csv"], key="curr")
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

# --------------------- Tabs ---------------------
tab_insights, tab_predict = st.tabs(["📋 Insights (MTD/Cohort)", "🔮 Predictability (M0 + Carryover)"])

# =========================================================
# ===============  INSIGHTS (MTD / Cohort)  ===============
# =========================================================
with tab_insights:
    st.subheader("Insights (MTD & Cohort)")

    if not date_like_cols:
        st.error("No usable date-like columns found in master (other than Create Date).")
        st.stop()

    base_ins = df_master.copy()

    # Defensive coercion of measures
    measures = ["Payment Received Date"] if "Payment Received Date" in date_like_cols else [date_like_cols[0]]
    for m in measures:
        if m in base_ins.columns and not pd.api.types.is_datetime64_any_dtype(base_ins[m]):
            base_ins[m] = pd.to_datetime(base_ins[m], errors="coerce", dayfirst=True)
        mn=f"{m}_Month"
        if m in base_ins.columns and mn not in base_ins.columns:
            base_ins[mn] = base_ins[m].dt.to_period("M").dt.to_timestamp()

    st.write("### Simple Counts (MTD)")
    this_month = pd.Timestamp.today().to_period("M").to_timestamp()
    mtd = base_ins[base_ins["Create_Month"]==this_month]
    st.metric("Deals created this month", len(mtd))

    st.write("### Cohort Counts")
    if "Payment Received Date" in base_ins.columns:
        cohort = base_ins.groupby("Payment Received Date").size()
        st.line_chart(cohort)

# =========================================================
# ===========  PREDICTABILITY (M0 + Carryover) ============
# =========================================================
with tab_predict:
    st.subheader("Predictability (M0 + Carryover)")

    today = pd.Timestamp.today().normalize()
    running_month = pd.Timestamp(year=today.year, month=today.month, day=1)

    mmin = df_master["Create_Month"].dropna().min()
    mmax = df_master["Create_Month"].dropna().max()
    options = list(pd.date_range(start=mmin, end=(mmax + pd.offsets.MonthBegin(6)), freq="MS"))
    idx = options.index(running_month) if running_month in options else len(options)-1
    target_month = st.selectbox("Target month", options=options, index=idx, format_func=lambda d: pd.Timestamp(d).strftime("%b %Y"))

    target_month = pd.Timestamp(target_month)

    if target_month == running_month and (df_curr is None or df_curr[df_curr["Create_Month"]==running_month].empty):
        st.error("Please upload the running month partial file to forecast the current month.")
        st.stop()

    # Exclude running month from master modeling
    hist = df_master[df_master["Create_Month"] < running_month].copy()

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
        st.error("Couldn't find a payment date column in master. Expected 'Payment Received Date'.")
        st.stop()

    hist["Payment Received Date"] = pd.to_datetime(hist[PAY_COL], errors="coerce", dayfirst=True)
    hist["Payment_Month"]        = hist["Payment Received Date"].dt.to_period("M").dt.to_timestamp()

    hist["CM"] = hist["Create_Month"]
    hist["PM"] = hist["Payment_Month"]
    creates_by_CM = hist.groupby("CM")["Create Date"].count().rename("Creates_CM")

    cohort = (
        hist.dropna(subset=["PM"])
            .groupby(["CM","PM"])["Payment Received Date"].count().rename("Paid").reset_index()
    )
    cohort["lag_k"] = ((cohort["PM"].dt.year - cohort["CM"].dt.year)*12 + (cohort["PM"].dt.month - cohort["CM"].dt.month)).astype(int)
    cohort = cohort[cohort["lag_k"] >= 0]

    paid_by_k = cohort.groupby("lag_k")["Paid"].sum()
    total_creates = float(creates_by_CM.sum())
    lag_prob = (paid_by_k / max(total_creates,1.0)).reindex(range(0,18), fill_value=0.0).clip(lower=0.0)
    M0_rate = float(lag_prob.get(0, 0.0))

    # Forecast: simple M0 + carryover
    est_creates = creates_by_CM.mean()
    M0_expected = est_creates * M0_rate
    carry = 0.0
    for j in creates_by_CM.index[creates_by_CM.index < target_month]:
        k = (target_month.year - j.year)*12 + (target_month.month - j.month)
        if k<=0: continue
        carry += creates_by_CM.get(j,0) * lag_prob.get(k,0)
    total_forecast = M0_expected + carry

    st.metric("M0 forecast", round(M0_expected))
    st.metric("Carryover forecast", round(carry))
    st.metric("Total forecast", round(total_forecast))
