"""
Smart Data Analyst — v5 + ngrok
Full chat interface with public URL via ngrok tunnel.
Run: streamlit run app_with_ngrok.py
"""

# ─────────────────────────────────────────────────────────────────
import os
import sys
from dotenv import load_dotenv
load_dotenv()

# ── UTF-8 ENCODING FIX ────────────────────────────────────────────────────────
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

# ── NGROK DETECTION ──────────────────────────────────────────────────────────
def get_ngrok_url():
    """Detects active ngrok URL from env or shared file."""
    url = os.environ.get("_NGROK_URL")
    if url: return url
    if os.path.exists("ngrok_url.txt"):
        try:
            with open("ngrok_url.txt", "r") as f:
                return f.read().strip()
        except Exception: pass
    return ""

_NGROK_URL = get_ngrok_url()

# ─────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings, re, time
from dotenv import load_dotenv

from smolagents import tool, ToolCallingAgent, LiteLLMModel
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

load_dotenv()
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────
P    = ["#2D6A4F","#40916C","#52B788","#74C69D","#B7E4C7","#1B4332","#D8F3DC","#95D5B2"]
WARN = "#E76F51"
MID  = "#F4A261"
BG   = "#FAFAF8"
GRID = "#E0E5DD"
TXT  = "#1A2E1A"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: #F2F4F1;
  color: #1A2E1A;
}
.app-header {
  background: linear-gradient(120deg, #1B4332 0%, #2D6A4F 55%, #52B788 100%);
  border-radius: 18px;
  padding: 1.6rem 2rem 1.4rem;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 1.2rem;
  box-shadow: 0 10px 40px rgba(27,67,50,.22);
}
.app-header .icon { font-size: 2.4rem; line-height: 1; }
.app-header h1    { font-family: 'DM Serif Display', serif; color: #D8F3DC; font-size: 1.8rem; margin: 0; letter-spacing: -.02em; }
.app-header p     { color: #B7E4C7; margin: .15rem 0 0; font-size: .9rem; font-weight: 300; }
.ngrok-banner {
  background: #1B4332; color: #D8F3DC;
  border-radius: 10px; padding: .6rem 1.1rem;
  margin-bottom: .8rem; font-size: .88rem;
  display: flex; align-items: center; gap: .6rem;
}
.ngrok-banner a { color: #74C69D; font-weight: 600; word-break: break-all; }
.ngrok-banner.warn { background: #FFF4F0; color: #7A3A2A; border: 1px solid #F4A261; }
.row-user { display: flex; justify-content: flex-end; align-items: flex-end; gap: .5rem; margin: .4rem 0; }
.row-bot  { display: flex; justify-content: flex-start; align-items: flex-start; gap: .5rem; margin: .4rem 0; }
.avatar      { width:32px; height:32px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:1.1rem; flex-shrink:0; }
.avatar-user { background: #2D6A4F; color: #D8F3DC; }
.avatar-bot  { background: #fff; border: 1.5px solid #B7E4C7; color: #2D6A4F; }
.bubble-user {
  background: #2D6A4F; color: #fff;
  border-radius: 18px 18px 4px 18px;
  padding: .7rem 1rem; max-width: 72%;
  font-size: .92rem; line-height: 1.55;
  box-shadow: 0 3px 12px rgba(45,106,79,.28);
}
.bubble-bot {
  background: #fff; color: #1A2E1A;
  border-radius: 18px 18px 18px 4px;
  padding: .75rem 1.1rem; max-width: 80%;
  font-size: .92rem; line-height: 1.7;
  border: 1px solid #D8F3DC;
  box-shadow: 0 2px 10px rgba(0,0,0,.07);
}
.bubble-bot strong { color: #1B4332; }
.bubble-bot code   { background: #EAF4ED; border-radius: 4px; padding: 1px 5px; font-size: .88rem; color: #2D6A4F; }
.ibox      { background:#F0FAF3; border:1px solid #B7E4C7; border-radius:10px; padding:1rem 1.3rem; margin:.4rem 0; font-size:.92rem; line-height:1.6; }
.ibox.warn { background:#FFF4F0; border-color:#F4A261; }
.ibox.info { background:#EEF2FF; border-color:#ADB5BD; }
.stButton>button {
  background: #2D6A4F !important; color: #fff !important;
  border-radius: 50px !important; font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important; border: none !important;
  padding: .42rem 1rem !important; transition: all .15s !important; font-size: .84rem !important;
}
.stButton>button:hover    { background: #1B4332 !important; }
.stButton>button:disabled { background: #95D5B2 !important; color: #1B4332 !important; opacity: .6 !important; }
[data-testid="stSidebar"] { background: #F0FAF3 !important; }
.stTabs [data-baseweb="tab-list"]  { background: #EAF4ED; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"]       { border-radius: 8px; font-family: 'DM Sans', sans-serif; color: #2D6A4F; }
.stTabs [aria-selected="true"]     { background: #2D6A4F !important; color: #fff !important; }
</style>
"""


# ─────────────────────────────────────────────────────────────────
#  EMOJI SAFETY WRAPPER
# ─────────────────────────────────────────────────────────────────
def safe_str(text: str) -> str:
    try:
        enc = sys.stdout.encoding or "utf-8"
        return text.encode(enc, errors="replace").decode(enc, errors="replace")
    except Exception:
        return text.encode("ascii", errors="replace").decode("ascii")


# ─────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def read_csv_safe(file) -> pd.DataFrame:
    for enc in ["utf-8", "latin1", "cp1252", "ISO-8859-1"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    raise ValueError("Cannot read the CSV file. Please check the encoding.")


def detect_time_col(df: pd.DataFrame) -> str | None:
    hints = ["date", "time", "timestamp", "year", "month", "day", "period"]
    for col in df.columns:
        if any(h in col.lower() for h in hints):
            try:
                if pd.to_datetime(df[col], errors="coerce").notnull().mean() > 0.6:
                    return col
            except Exception:
                pass
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            if pd.to_datetime(df[col], errors="coerce").notnull().mean() > 0.7:
                return col
        except Exception:
            pass
    return None


def detect_target_col(df: pd.DataFrame) -> str | None:
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in df.select_dtypes(include=[np.number]).columns:
        if 2 <= df[col].nunique() <= 20:
            cats.append(col)
    cands = [(c, df[c].nunique()) for c in cats
             if 2 <= df[c].nunique() <= 30 and df[c].isnull().mean() < 0.5]
    if not cands:
        return None
    for col, _ in cands:
        if any(h in col.lower() for h in ["label","target","class","category","type","status","group","outcome"]):
            return col
    return min(cands, key=lambda x: x[1])[0]


def fmt(n) -> str:
    if pd.isna(n): return "N/A"
    n = float(n)
    if abs(n) >= 1e6: return f"{n/1e6:.1f}M"
    if abs(n) >= 1e3: return f"{n/1e3:.1f}K"
    return f"{n:,.2f}"


def chart_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis="y", color=GRID, linestyle="--", alpha=.7)
    ax.tick_params(colors=TXT, labelsize=9)
    if title:  ax.set_title(title,  fontsize=12, fontweight="bold", color=TXT, pad=9)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9,  color=TXT)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9,  color=TXT)


def build_data_context(df: pd.DataFrame) -> str:
    rows, ncols  = df.shape
    num_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = df.select_dtypes(include=["object", "category"]).columns.tolist()
    time_col     = detect_time_col(df)
    dups         = int(df.duplicated().sum())
    miss_tot     = df.isnull().sum().sum()
    completeness = 100 - miss_tot / (rows * ncols) * 100 if rows * ncols else 100

    lines = [
        "=== VERIFIED DATASET STATISTICS ===",
        f"Rows: {rows:,}  |  Columns: {ncols}  |  Duplicates: {dups}",
        f"Completeness: {completeness:.1f}%",
        f"Numeric columns  ({len(num_cols)}): {', '.join(num_cols)}",
        f"Categorical cols ({len(cat_cols)}): {', '.join(cat_cols)}",
        f"Date/time column: {time_col or 'None detected'}",
        "", "--- Per-Column Stats ---",
    ]
    for col in num_cols:
        s = df[col].dropna()
        lines.append(
            f"[NUM] {col}: min={fmt(s.min())} max={fmt(s.max())} "
            f"mean={fmt(s.mean())} median={fmt(s.median())} "
            f"std={fmt(s.std())} missing={df[col].isnull().mean()*100:.1f}%"
        )
    for col in cat_cols:
        top5 = ", ".join(f"'{v}'({c})" for v, c in df[col].value_counts().head(5).items())
        lines.append(
            f"[CAT] {col}: {df[col].nunique()} unique | top: {top5} | "
            f"missing={df[col].isnull().mean()*100:.1f}%"
        )
    if time_col:
        try:
            dt = pd.to_datetime(df[time_col], errors="coerce").dropna()
            lines.append(
                f"[TIME] {time_col}: {dt.min().date()} to {dt.max().date()} | "
                f"span={(dt.max()-dt.min()).days} days"
            )
        except Exception:
            pass
    if len(num_cols) >= 2:
        try:
            corr = df[num_cols].corr().abs()
            pairs = (
                corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack().sort_values(ascending=False).head(5)
            )
            lines += ["", "--- Top Correlations ---"]
            for (c1, c2), v in pairs.items():
                lines.append(f"{c1} <-> {c2}: r={v:.3f}")
        except Exception:
            pass
    lines += ["", "--- Sample (first 5 rows) ---", df.head(5).to_string(index=False), "", "=== END ==="]
    return "\n".join(lines)


def clean_output(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(
        r"(?im)^[ \t]*(thoughts?|code|observation|action\s*input|action)\s*:.*?(?=\n\n|\Z)",
        "", text, flags=re.DOTALL
    )
    py_pat = re.compile(
        r"^\s*(import |from |def |class |for |while |if |print\(|plt\.|pd\.|np\.|df\.|"
        r"result\s*=|labels\s*=|model\s*=|fig\s*=|ax\s*=|X\s*=|y\s*=)",
        re.IGNORECASE
    )
    lines = [l for l in text.split("\n") if not py_pat.match(l)]
    text  = "\n".join(lines)
    text  = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text if len(text) > 15 else "Analysis complete. See the chart above."


# ─────────────────────────────────────────────────────────────────
#  TOOL DEFINITIONS
# ─────────────────────────────────────────────────────────────────

@tool
def summarise_dataset() -> str:
    """
    Generates a complete visual and statistical summary of the uploaded dataset.
    Produces distribution charts, missing-value bars, and categorical breakdowns.
    Call this when the user asks for: overview, summary, describe the data,
    what is in the data, show me the data, data profile, explore the data.
    Returns verified key statistics in plain English.
    """
    df = st.session_state.get("data")
    if df is None or df.empty:
        return "No dataset loaded. Please upload a CSV file using the sidebar first."

    rows, ncols  = df.shape
    num_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = df.select_dtypes(include=["object","category"]).columns.tolist()
    missing      = df.isnull().sum()
    missing      = missing[missing > 0].sort_values(ascending=False)
    dups         = int(df.duplicated().sum())
    time_col     = detect_time_col(df)
    completeness = 100 - df.isnull().sum().sum() / (rows * ncols) * 100

    n_num = min(len(num_cols), 4)
    n_cat = min(len(cat_cols), 3)
    n_fig = max(1, 1 + (1 if n_num>0 else 0) + (1 if n_cat>0 or len(missing)>0 else 0))

    fig = plt.figure(figsize=(16, 4*n_fig), facecolor="#F2F4F1")
    fig.suptitle(f"Dataset Summary  |  {rows:,} rows  |  {ncols} columns",
                 fontsize=15, fontweight="bold", color=TXT, y=0.99)
    rc = 0

    if n_num > 0:
        ax0  = fig.add_subplot(n_fig, 1, rc+1); rc += 1
        cs   = num_cols[:6]
        avgs = [df[c].mean() for c in cs]
        bars = ax0.barh(cs, avgs, color=P[:len(cs)], edgecolor="white", lw=.8, height=.5)
        for bar, col in zip(bars, cs):
            ax0.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                     f"avg {fmt(df[col].mean())}  |  min {fmt(df[col].min())}  |  max {fmt(df[col].max())}",
                     va="center", fontsize=8.5, color=TXT)
        chart_style(ax0, title="Numeric Columns - Average Values")
        ax0.grid(axis="x", color=GRID, linestyle="--", alpha=.7)
        ax0.grid(axis="y", visible=False)
        sm = max(avgs) if max(avgs) > 0 else 1
        ax0.set_xlim(0, sm*1.55)

    if n_num > 0:
        for i, col in enumerate(num_cols[:n_num]):
            ax = fig.add_subplot(n_fig, n_num, rc*n_num+i+1)
            d  = df[col].dropna()
            ax.hist(d, bins=25, color=P[i%len(P)], edgecolor="white", lw=.5, alpha=.9)
            ax.axvline(d.mean(), color=WARN, linestyle="--", lw=1.5, label=f"avg {fmt(d.mean())}")
            chart_style(ax, title=col, xlabel="Value", ylabel="Count")
            ax.legend(fontsize=7, framealpha=.6)
        rc += 1

    if n_cat > 0 or len(missing) > 0:
        tc2 = n_cat + (1 if len(missing)>0 else 0)
        for i, col in enumerate(cat_cols[:n_cat]):
            ax = fig.add_subplot(n_fig, tc2, rc*tc2+i+1)
            vc = df[col].value_counts().head(6)
            ax.bar(vc.index.astype(str), vc.values,
                   color=[P[j%len(P)] for j in range(len(vc))], edgecolor="white", lw=.6)
            chart_style(ax, title=f"{col} (top values)", ylabel="Count")
            ax.tick_params(axis="x", rotation=30, labelsize=8)
        if len(missing) > 0:
            ax_m = fig.add_subplot(n_fig, tc2, rc*tc2+n_cat+1)
            pcts = (missing/rows*100).head(8)
            bc   = [WARN if v>40 else MID if v>15 else "#74C69D" for v in pcts]
            ax_m.barh(pcts.index.astype(str), pcts.values, color=bc, edgecolor="white", lw=.6)
            ax_m.axvline(15, color=MID,  linestyle="--", lw=1, alpha=.8, label="15%")
            ax_m.axvline(40, color=WARN, linestyle="--", lw=1, alpha=.8, label="40%")
            chart_style(ax_m, title="Missing Values (%)", xlabel="% Missing")
            ax_m.grid(axis="x", color=GRID, linestyle="--", alpha=.7)
            ax_m.grid(axis="y", visible=False)
            ax_m.legend(fontsize=7, framealpha=.6)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    st.session_state["pending_plot"] = fig
    plt.close(fig)

    out = [
        f"[OK] {rows:,} rows | {ncols} columns",
        f"- Completeness: {completeness:.1f}%{' -- no missing values!' if missing.empty else ''}",
        f"- Duplicates: {'None found.' if dups==0 else f'{dups:,} duplicate rows detected.'}",
    ]
    if num_cols:
        items = [f"`{c}`: avg {fmt(df[c].mean())}, range {fmt(df[c].min())}--{fmt(df[c].max())}"
                 for c in num_cols[:6]]
        out.append("- Numeric columns: " + " | ".join(items))
    if cat_cols:
        items = [f"`{c}`: {df[c].nunique()} values (top: '{df[c].value_counts().idxmax()}')"
                 for c in cat_cols[:5]]
        out.append("- Categorical columns: " + " | ".join(items))
    if not missing.empty:
        ms = [f"`{c}` ({v/rows*100:.0f}%)" for c, v in missing.head(5).items()]
        out.append(f"- WARNING: Missing values in: {', '.join(ms)}")
    if time_col:
        try:
            dt = pd.to_datetime(df[time_col], errors="coerce").dropna()
            out.append(f"- Date column `{time_col}`: {dt.min().date()} to {dt.max().date()} "
                       f"({(dt.max()-dt.min()).days} days)")
        except Exception:
            pass
    tips = []
    if time_col: tips.append("ask me to **analyse trends over time**")
    tgt = detect_target_col(df)
    if tgt: tips.append(f"ask me to **classify {tgt}**")
    if tips:
        out.append(f"\nTip: You can also: {' or '.join(tips)}.")
    return "\n".join(out)


@tool
def analyse_time_series(
    time_column: str,
    target_column: str,
    frequency: str,
    aggregation: str,
) -> str:
    """
    Analyses how a numeric value changes over time and produces a two-panel trend chart.
    Call this when the user asks about: trends, time series, over time, changes over time,
    monthly trends, weekly patterns, how X changed, time analysis, seasonality.

    Args:
        time_column: The date/timestamp column name. Use 'auto' to auto-detect it.
        target_column: The numeric column to track. Use 'auto' to auto-select the best one.
        frequency: Resampling period -- 'D' daily, 'W' weekly, 'M' monthly, 'Q' quarterly, 'Y' yearly.
        aggregation: How to group values -- 'mean', 'sum', 'median', 'max', 'min'.
    """
    df = st.session_state.get("data")
    if df is None or df.empty:
        return "No dataset loaded. Please upload a CSV file using the sidebar first."

    if time_column.strip().lower() == "auto":
        time_column = detect_time_col(df)
    if not time_column or time_column not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return (
            "ERROR: No date column found in this dataset.\n\n"
            "Time series analysis requires a column with dates or timestamps.\n\n"
            f"Available numeric columns: {', '.join(num_cols[:8])}\n\n"
            "Try asking me to summarise the dataset or run classification instead."
        )
    if target_column.strip().lower() == "auto":
        num_opts = [c for c in df.select_dtypes(include=[np.number]).columns if c != time_column]
        if not num_opts:
            return "No numeric column found to track over time."
        target_column = num_opts[0]
    if target_column not in df.columns:
        return f"Column `{target_column}` not found. Available columns: {', '.join(df.columns.tolist())}"
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        return f"`{target_column}` is not numeric. Please specify a numeric column to track."

    valid_freqs = {"D","W","M","Q","Y"}
    valid_aggs  = {"mean","sum","median","max","min"}
    frequency   = frequency.upper()   if frequency.upper()   in valid_freqs else "M"
    aggregation = aggregation.lower() if aggregation.lower() in valid_aggs  else "mean"

    df2 = df.copy()
    df2[time_column] = pd.to_datetime(df2[time_column], errors="coerce")
    df2 = df2.dropna(subset=[time_column]).set_index(time_column).sort_index()

    try:
        series = df2[target_column].resample(frequency).agg(aggregation)
    except Exception as e:
        return f"Resampling failed: {e}. Try frequency='M' for monthly."
    if len(series) < 2:
        return "Not enough time periods to draw a trend. Try frequency='D' (daily) or 'W' (weekly)."

    avg    = series.mean()
    std    = series.std()
    pk_d   = series.idxmax()
    lw_d   = series.idxmin()
    pct    = ((series.iloc[-1]-series.iloc[0])/abs(series.iloc[0])*100 if series.iloc[0] != 0 else 0)
    trend  = "upward" if series.iloc[-1] > series.iloc[0] else "downward"
    roll_w = max(2, len(series)//6)
    roll   = series.rolling(window=roll_w, min_periods=1).mean()

    FL = {"D":"Daily","W":"Weekly","M":"Monthly","Q":"Quarterly","Y":"Yearly"}
    AL = {"mean":"Average","sum":"Total","median":"Median","max":"Max","min":"Min"}

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), facecolor="#F2F4F1",
                             gridspec_kw={"height_ratios": [3, 1]})
    trend_icon = "(up)" if trend == "upward" else "(down)"
    fig.suptitle(
        f"{target_column} Over Time {trend_icon}  |  "
        f"{AL.get(aggregation,aggregation)} by {FL.get(frequency,frequency)}",
        fontsize=14, fontweight="bold", color=TXT
    )
    ax = axes[0]; ax.set_facecolor(BG)
    ax.fill_between(series.index, series.values, alpha=.12, color=P[0])
    ax.plot(series.index, series.values, lw=2, color=P[0], marker="o", markersize=4, label=target_column)
    ax.plot(roll.index,   roll.values,   lw=2.5, color=WARN, linestyle="--", label=f"Trend (window={roll_w})")
    ax.axhline(avg, color=P[3], linestyle=":", lw=1.8, label=f"Avg: {fmt(avg)}")
    ax.annotate(f"Peak\n{fmt(series.max())}", xy=(pk_d, series.max()),
                xytext=(10, 14), textcoords="offset points", fontsize=8, color=P[0],
                arrowprops=dict(arrowstyle="->", color=P[0], lw=1.2))
    chart_style(ax, ylabel=target_column)
    ax.legend(fontsize=9, framealpha=.7)

    ax2 = axes[1]; ax2.set_facecolor(BG)
    bar_c = [P[0] if v >= avg else P[4] for v in series.values]
    try:
        import pandas.tseries.frequencies as tsf
        bw = tsf.to_offset(frequency).nanos / 1e9 / 86400 * 0.7
        ax2.bar(series.index, series.values, width=bw, color=bar_c, alpha=.8, edgecolor="white", lw=.4)
    except Exception:
        ax2.bar(range(len(series)), series.values, color=bar_c, alpha=.8, edgecolor="white", lw=.4)
    ax2.axhline(avg, color=P[3], linestyle=":", lw=1.5)
    chart_style(ax2, xlabel="Date", ylabel="Value")
    plt.tight_layout()
    st.session_state["pending_plot"] = fig
    plt.close(fig)

    vol = "high" if std > avg*.3 else "moderate" if std > avg*.1 else "low"
    return (
        f"[CHART] Time Series: `{target_column}` -- "
        f"{AL.get(aggregation,aggregation)} by {FL.get(frequency,frequency)}\n\n"
        f"- Overall trend: {trend} ({pct:+.1f}% change from start to end)\n"
        f"- Peak: {fmt(series.max())} on {pk_d.strftime('%Y-%m-%d')}\n"
        f"- Lowest: {fmt(series.min())} on {lw_d.strftime('%Y-%m-%d')}\n"
        f"- Average: {fmt(avg)} | Std deviation: {fmt(std)} ({vol} volatility)\n"
        f"- Periods analysed: {len(series)} {FL.get(frequency,'').lower()} periods\n\n"
        f"The chart above shows the full trend line, smoothed average (window={roll_w}), and period bars."
    )


@tool
def run_classification(
    target_column: str,
    feature_columns: str,
    model_name: str,
) -> str:
    """
    Trains a machine learning model to predict a categorical column from numeric features.
    Produces a confusion matrix, per-class F1 chart, and feature importance chart.
    ONLY call this when user says: classify, predict, supervised learning, train model,
    what predicts X, labels, target column.
    NEVER call this for cluster / segment / group / KMeans / DBSCAN requests.

    Args:
        target_column: The column to predict. Use 'auto' to auto-detect the best target.
        feature_columns: Comma-separated numeric feature column names, or 'auto' for all numeric columns.
        model_name: 'RandomForest', 'GradientBoosting', or 'LogisticRegression'.
    """
    df = st.session_state.get("data")
    if df is None or df.empty:
        return "No dataset loaded. Please upload a CSV file using the sidebar first."

    if target_column.strip().lower() == "auto":
        target_column = detect_target_col(df)
    if not target_column or target_column not in df.columns:
        cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
        return (
            f"Target column not found. Categorical columns available: "
            f"{', '.join(cat_cols) if cat_cols else 'none found'}.\n"
            "Specify one, e.g. \"classify species using RandomForest\"."
        )
    if feature_columns.strip().lower() == "auto":
        feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    else:
        feat_cols = [c.strip() for c in feature_columns.split(",") if c.strip() in df.columns]
    if not feat_cols:
        return "No valid numeric feature columns found. Please check the column names."

    valid_models = {"RandomForest", "GradientBoosting", "LogisticRegression"}
    if model_name not in valid_models:
        model_name = "RandomForest"

    le = LabelEncoder()
    try:
        y = le.fit_transform(df[target_column].astype(str).fillna("Unknown"))
    except Exception as e:
        return f"Could not encode target column: {e}"

    X = StandardScaler().fit_transform(
        SimpleImputer(strategy="mean").fit_transform(df[feat_cols])
    )
    if len(np.unique(y)) < 2:
        return "Target column has only one class. At least 2 classes are needed."
    if len(y) < 20:
        return "Not enough rows (minimum 20 needed for classification)."

    MODELS = {
        "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
        "GradientBoosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    }
    clf = MODELS[model_name]
    tsz = min(0.25, max(0.1, 50/len(y)))
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=tsz, random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=tsz, random_state=42)

    clf.fit(Xtr, ytr)
    yp  = clf.predict(Xte)
    acc = accuracy_score(yte, yp)
    cv  = cross_val_score(clf, X, y, cv=min(5, len(np.unique(y))), scoring="accuracy")
    rep = classification_report(yte, yp, target_names=le.classes_, output_dict=True, zero_division=0)

    has_imp = hasattr(clf, "feature_importances_")
    feat_df = None
    if has_imp:
        feat_df = (pd.DataFrame({"Feature": feat_cols, "Importance": clf.feature_importances_})
                   .sort_values("Importance", ascending=False).head(10))

    n_cls = len(le.classes_)
    fig   = plt.figure(figsize=(16, 10 if has_imp else 6), facecolor="#F2F4F1")
    gs    = gridspec.GridSpec(2 if has_imp else 1, 2, figure=fig, hspace=.45, wspace=.4)

    ax_cm = fig.add_subplot(gs[0, 0]); ax_cm.set_facecolor(BG)
    cm    = confusion_matrix(yte, yp)
    cmap  = LinearSegmentedColormap.from_list("g", ["#D8F3DC","#1B4332"])
    im    = ax_cm.imshow(cm, cmap=cmap, aspect="auto")
    ax_cm.set_xticks(range(n_cls)); ax_cm.set_yticks(range(n_cls))
    lbs = [str(c)[:12] for c in le.classes_]
    ax_cm.set_xticklabels(lbs, rotation=35, ha="right", fontsize=8)
    ax_cm.set_yticklabels(lbs, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=9,
                       color="white" if cm[i,j] > cm.max()*.5 else TXT)
    ax_cm.set_title("Confusion Matrix\n(rows=actual, cols=predicted)", fontsize=11, fontweight="bold", color=TXT)
    ax_cm.set_xlabel("Predicted", fontsize=9, color=TXT)
    ax_cm.set_ylabel("Actual",    fontsize=9, color=TXT)
    plt.colorbar(im, ax=ax_cm, fraction=.046, pad=.04)

    ax_f1 = fig.add_subplot(gs[0, 1]); ax_f1.set_facecolor(BG)
    cls_n = list(le.classes_)
    f1s   = [rep.get(str(c), {}).get("f1-score", 0) for c in cls_n]
    bfs   = ax_f1.bar([str(c)[:12] for c in cls_n], f1s,
                       color=[P[i%len(P)] for i in range(len(cls_n))], edgecolor="white", lw=.6)
    for bar, val in zip(bfs, f1s):
        ax_f1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.01,
                   f"{val:.2f}", ha="center", fontsize=9, color=TXT)
    ax_f1.set_ylim(0, 1.15)
    ax_f1.axhline(acc, color=WARN, linestyle="--", lw=1.5, label=f"Accuracy: {acc:.2%}")
    chart_style(ax_f1, title="F1-Score per Class", xlabel="Class", ylabel="F1")
    ax_f1.tick_params(axis="x", rotation=30, labelsize=8)
    ax_f1.legend(fontsize=8, framealpha=.7)

    if has_imp and feat_df is not None:
        ax_i = fig.add_subplot(gs[1, :]); ax_i.set_facecolor(BG)
        ax_i.barh(feat_df["Feature"][::-1], feat_df["Importance"][::-1],
                  color=[P[i%len(P)] for i in range(len(feat_df))][::-1],
                  edgecolor="white", lw=.6)
        chart_style(ax_i, title="Feature Importance -- What Drives Predictions",
                    xlabel="Importance Score", ylabel="Feature")
        ax_i.grid(axis="x", color=GRID, linestyle="--", alpha=.7)
        ax_i.grid(axis="y", visible=False)

    fig.suptitle(f"Classification  |  {model_name}  |  Predicting: '{target_column}'",
                 fontsize=14, fontweight="bold", color=TXT, y=0.99)
    plt.tight_layout(rect=[0,0,1,0.97])
    st.session_state["pending_plot"] = fig
    plt.close(fig)

    acc_lbl = "excellent" if acc>=.9 else "good" if acc>=.75 else "moderate" if acc>=.6 else "weak"
    top3    = feat_df["Feature"].head(3).tolist() if feat_df is not None else feat_cols[:3]
    weak    = [c for c in cls_n if rep.get(str(c),{}).get("f1-score",1) < .5]

    return (
        f"[RESULT] Classification -- Predicting `{target_column}`\n\n"
        f"- Accuracy: {acc:.2%} ({acc_lbl} performance)\n"
        f"- Cross-validation: {cv.mean():.2%} +/- {cv.std():.2%} "
        f"({'stable generalisation' if cv.std()<.05 else 'some variance between folds'})\n"
        f"- Train / Test split: {len(ytr)} train | {len(yte)} test rows\n"
        f"- Classes: {n_cls}  |  Features used: {len(feat_cols)}\n"
        f"- Most predictive features: {', '.join(f'`{f}`' for f in top3)}\n"
        + (f"- WARNING: Weak classes: {', '.join(str(c) for c in weak)} (F1<0.5)\n" if weak
           else "- All classes scored F1 >= 0.5 -- model is well balanced\n")
        + f"\nModel used: {model_name}. Charts above show confusion matrix, F1 per class, and feature importance."
    )


# ─────────────────────────────────────────────────────────────────
#  CLUSTERING TOOL 
# ─────────────────────────────────────────────────────────────────

@tool
def run_clustering(
    feature_columns: str,
    n_clusters: str,
    algorithm: str,
) -> str:
    """
    UNSUPERVISED clustering — groups rows WITHOUT a target/label column.
    Produces a PCA scatter plot, cluster size chart, and feature profile heatmap.
    ALWAYS call this (NEVER run_classification) when user mentions:
    KMeans, DBSCAN, cluster, clustering, segment, group, unsupervised,
    find groups, similar records, natural groups, customer segments.
    MANDATORY: You must call this tool to answer "Can data be grouped?" or "Are there segments?". 
    Do not guess if segments exist without running this.

    Args:
        feature_columns: Comma-separated numeric column names, or 'auto' for all numeric columns.
        n_clusters: Number of clusters (2-10). Use 0 to auto-select via silhouette method.
        algorithm: 'KMeans' or 'DBSCAN'.
    """
    df = st.session_state.get("data")
    if df is None or df.empty:
        return "No dataset loaded. Please upload a CSV file using the sidebar first."

    # Convert n_clusters to int if string/any (prevents Groq schema errors)
    try:
        n_clusters = int(n_clusters)
    except (ValueError, TypeError):
        n_clusters = 0

    # ── Feature selection  ──────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if feature_columns.strip().lower() == "auto":
        feat_cols = [c for c in num_cols if not any(
            x in c.lower() for x in ["id", "index", "code", "number", "no"]
        )]
        if not feat_cols:
            feat_cols = num_cols
    else:
        feat_cols = [c.strip() for c in feature_columns.split(",") if c.strip() in df.columns]

    if not feat_cols:
        return f"No valid numeric feature columns found. Available: {', '.join(num_cols)}."
    if len(feat_cols) < 2:
        return "Clustering needs at least 2 numeric feature columns."

    # ── Preprocess ────────────────────────────────────────────────
    X_imp = SimpleImputer(strategy="mean").fit_transform(df[feat_cols].copy())
    X_sc  = StandardScaler().fit_transform(X_imp)

    valid_algos = {"KMeans", "DBSCAN"}
    algorithm   = algorithm if algorithm in valid_algos else "KMeans"

    # ── Auto-select k (KMeans) ────────────────────────────────────
    if algorithm == "KMeans" and n_clusters == 0:
        sil_scores, k_range = [], range(2, min(9, len(df)//5 + 2))
        for k in k_range:
            lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_sc)
            sil_scores.append(silhouette_score(X_sc, lbl))
        n_clusters = list(k_range)[int(np.argmax(sil_scores))]
    elif n_clusters < 2:
        n_clusters = 3

    # ── Fit model ─────────────────────────────────────────────────
    if algorithm == "KMeans":
        model  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_sc)

    else:
        # ── DBSCA auto-eps ────────────────────────────────
        min_samp = max(3, len(df) // 50)
        nn       = NearestNeighbors(n_neighbors=min_samp).fit(X_sc)
        dists, _ = nn.kneighbors(X_sc)

        # percentiles eps
        for pct in [90, 95, 97, 99]:
            eps_try = float(np.percentile(dists[:, -1], pct))
            lbl_try = DBSCAN(eps=eps_try, min_samples=min_samp).fit_predict(X_sc)
            n_try   = len(set(lbl_try)) - (1 if -1 in lbl_try else 0)
            noise_r = (lbl_try == -1).sum() / len(lbl_try)
            if n_try >= 2 and noise_r <= 0.4:
                labels     = lbl_try
                n_clusters = n_try
                model      = DBSCAN(eps=eps_try, min_samples=min_samp)
                break
        else:
            return (
                "DBSCAN could not find clear clusters in this data even after auto-tuning eps.\n"
                "Recommendation: Use KMeans instead -- it works better for this dataset.\n"
                "Try: 'cluster using KMeans with 3 clusters'"
            )

    # ── Silhouette ────────────────────────────────────────────────
    clean_mask = labels >= 0
    sil = (silhouette_score(X_sc[clean_mask], labels[clean_mask])
           if clean_mask.sum() > 1 and len(set(labels[clean_mask])) > 1 else None)

    # ── PCA ───────────────────────────────────────────────────────
    pca    = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_sc)
    var_ex = pca.explained_variance_ratio_ * 100

    # ── Cluster profiles ──────────────────────────────────────────
    df_prof             = df[feat_cols].copy()
    df_prof["_cluster"] = labels
    profile             = df_prof.groupby("_cluster")[feat_cols].mean()
    profile_norm        = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    unique_labels = sorted([l for l in set(labels) if l >= 0])
    cluster_names = [f"Cluster {l}" for l in unique_labels]
    cluster_sizes = [int((labels == l).sum()) for l in unique_labels]
    noise_count   = int((labels == -1).sum())

    # ── Figure: 3 panels ─────────────────────────────────────────
    fig = plt.figure(figsize=(17, 11), facecolor="#F2F4F1")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    # Panel 1: PCA scatter
    ax_sc = fig.add_subplot(gs[0, 0]); ax_sc.set_facecolor(BG)
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax_sc.scatter(coords[mask, 0], coords[mask, 1],
                      c=P[idx % len(P)], label=f"Cluster {lbl} (n={mask.sum()})",
                      s=28, alpha=0.72, edgecolors="white", linewidths=0.4)
    if noise_count > 0:
        mask = labels == -1
        ax_sc.scatter(coords[mask, 0], coords[mask, 1],
                      c="#BBBBBB", label=f"Noise (n={noise_count})", s=18, alpha=0.5, marker="x")
    if algorithm == "KMeans":
        c2d = pca.transform(model.cluster_centers_)
        ax_sc.scatter(c2d[:, 0], c2d[:, 1], c="white", s=160, marker="*",
                      edgecolors=TXT, linewidths=1.2, zorder=5, label="Centroids")
    chart_style(ax_sc, title=f"Cluster Map (PCA)  |  {algorithm}",
                xlabel=f"PC1 ({var_ex[0]:.1f}% var)", ylabel=f"PC2 ({var_ex[1]:.1f}% var)")
    ax_sc.legend(fontsize=8, framealpha=0.75)
    ax_sc.grid(visible=True, color=GRID, linestyle="--", alpha=0.5)

    # Panel 2: Cluster sizes
    ax_sz = fig.add_subplot(gs[0, 1]); ax_sz.set_facecolor(BG)
    bars  = ax_sz.bar(cluster_names, cluster_sizes,
                      color=[P[i % len(P)] for i in range(len(cluster_names))],
                      edgecolor="white", lw=0.7, width=0.55)
    for bar, val in zip(bars, cluster_sizes):
        ax_sz.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=9, color=TXT)
    if noise_count > 0:
        ax_sz.bar(["Noise"], [noise_count], color="#CCCCCC", edgecolor="white", width=0.55)
        ax_sz.text(len(cluster_names), noise_count + 0.5,
                   f"{noise_count}\n({noise_count/len(df)*100:.1f}%)",
                   ha="center", fontsize=9, color="#888")
    chart_style(ax_sz, title="Cluster Sizes", xlabel="Cluster", ylabel="Records")
    ax_sz.set_ylim(0, max(cluster_sizes + [noise_count if noise_count > 0 else 0]) * 1.22)

    # Panel 3: Heatmap
    ax_hm   = fig.add_subplot(gs[1, :]); ax_hm.set_facecolor(BG)
    hm_data = profile_norm.loc[unique_labels]
    cmap_hm = LinearSegmentedColormap.from_list("gn", ["#D8F3DC", "#1B4332"])
    im2     = ax_hm.imshow(hm_data.values, cmap=cmap_hm, aspect="auto", vmin=0, vmax=1)
    ax_hm.set_xticks(range(len(feat_cols))); ax_hm.set_xticklabels(feat_cols, rotation=35, ha="right", fontsize=9)
    ax_hm.set_yticks(range(len(unique_labels))); ax_hm.set_yticklabels([f"Cluster {l}" for l in unique_labels], fontsize=9)
    for i in range(len(unique_labels)):
        for j in range(len(feat_cols)):
            v = hm_data.values[i, j]
            ax_hm.text(j, i, f"{profile.values[i, j]:.2g}", ha="center", va="center",
                       fontsize=8, color="white" if v > 0.6 else TXT)
    plt.colorbar(im2, ax=ax_hm, fraction=0.02, pad=0.02, label="Normalised mean (0-1)")
    chart_style(ax_hm, title="Feature Profile per Cluster", xlabel="Feature", ylabel="Cluster")

    fig.suptitle(f"Clustering  |  {algorithm}  |  {n_clusters} clusters  |  "
                 f"{len(feat_cols)} features  |  {len(df):,} records",
                 fontsize=14, fontweight="bold", color=TXT, y=1.01)
    plt.tight_layout()
    st.session_state["pending_plot"] = fig
    plt.close(fig)

    # Save labels to data so other tools can query by cluster
    df["_cluster_tag"] = labels
    st.session_state["data"] = df

    # ── Text summary ──────────────────────────────────────────────
    sil_str = f"{sil:.3f}" if sil else "N/A"
    sil_lbl = ("excellent" if sil and sil>.7 else "good" if sil and sil>.5
               else "moderate" if sil and sil>.3 else "weak") if sil else "N/A"

    lines = [
        f"[RESULT] Clustering -- {algorithm} | {n_clusters} clusters\n",
        f"- Records: {len(df):,} | Features: {', '.join(f'`{c}`' for c in feat_cols)}",
        f"- Silhouette score: **{sil_str}** ({sil_lbl} separation)",
        f"- PCA captures **{var_ex[0]+var_ex[1]:.1f}%** of variance in the 2-D plot",
    ]
    if noise_count > 0:
        lines.append(f"- Noise points: {noise_count} ({noise_count/len(df)*100:.1f}%)")

    lines.append("\n**Average features per cluster (Direct Answer):**")
    for lbl in unique_labels:
        means = profile.loc[lbl]
        m_str = " | ".join([f"{c}: **{fmt(means[c])}**" for c in feat_cols[:6]])
        lines.append(f"- **Cluster {lbl}** ({int((labels==lbl).sum()):,} rows): {m_str}")

    lines.append("\nCharts: PCA scatter, size bars, and feature profile heatmap.")
    return "\n".join(lines)


@tool
def answer_data_question(question: str) -> str:
    """
    Answers any factual or exploratory question about the dataset using verified statistics.
    Call this for questions like: how many rows, what columns are there, average of X,
    which column has most nulls, are columns correlated, max/min value of X, data quality.
    Also use this to check if clustering has already been performed (existence of _cluster_tag).

    Args:
        question: The user's plain-English question about the dataset.
    """
    df = st.session_state.get("data")
    if df is None or df.empty:
        return "No dataset loaded. Please upload a CSV file using the sidebar first."

    ctx   = st.session_state.get("data_context", build_data_context(df))
    q     = question.lower()
    rows, ncols = df.shape

    # ── ANTI-HALLUCINATION GUARDRAIL ──────────────────────────────
    cluster_keywords = ["cluster", "clustering", "segment", "segmentation", "kmeans", "dbscan", "silhouette", "centroid", "group"]
    if any(kw in q for kw in cluster_keywords) and "_cluster_tag" not in df.columns:
        return (
            "Clustering has not been performed yet (no `_cluster_tag` column found). "
            "I cannot provide cluster metrics or averages until the data is segmented.\n\n"
            "**Please run clustering first.** For example, type:\n"
            "- *'Cluster the data into 3 groups using KMeans'* \n"
            "- *'Run DBSCAN clustering feature1, feature2'*"
        )

    # ── Handle cluster-specific questions if labels exist ──────────
    if "_cluster_tag" in df.columns and any(kw in q for kw in ["cluster", "segment", "group"]):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        tag_col  = "_cluster_tag"
        means    = df.groupby(tag_col)[num_cols].mean()
        counts   = df[tag_col].value_counts()
        
        lines = ["Average Values per Cluster (Verified Statistics)"]
        for lbl in sorted(df[tag_col].unique()):
            if lbl == -1: continue # skip noise
            row = means.loc[lbl]
            # filter for columns mentioned in question or just top ones
            target_cols = [c for c in num_cols if c.lower() in q] or num_cols[:5]
            avg_str = " | ".join([f"{c}: **{fmt(row[c])}**" for c in target_cols])
            lines.append(f"- **Cluster {lbl}** ({counts.get(lbl, 0):,} rows): {avg_str}")
        return "\n".join(lines)

    if any(w in q for w in ["how many rows","row count","number of rows","size"]):
        return f"The dataset has **{rows:,} rows**."
    if any(w in q for w in ["how many columns","column count","number of columns"]):
        return f"The dataset has **{ncols} columns**: {', '.join(df.columns.tolist())}."
    if "column" in q and any(w in q for w in ["name","list","what are","show"]):
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        cat = df.select_dtypes(include=["object","category"]).columns.tolist()
        return (f"All columns ({ncols}):\n"
                f"- Numeric ({len(num)}): {', '.join(num) or 'none'}\n"
                f"- Categorical ({len(cat)}): {', '.join(cat) or 'none'}")
    if any(w in q for w in ["missing","null","nan","empty","incomplete","blank"]):
        miss = df.isnull().sum().sort_values(ascending=False)
        miss = miss[miss > 0]
        if miss.empty:
            return "No missing values -- the dataset is 100% complete."
        total_miss = miss.sum()
        items = [f"- `{c}`: {v:,} missing ({v/rows*100:.1f}%)" for c, v in miss.head(10).items()]
        return (f"Missing values found in {len(miss)} out of {ncols} columns "
                f"(total {total_miss:,} missing cells):\n" + "\n".join(items))
    if "duplicate" in q:
        dups = int(df.duplicated().sum())
        return ("No duplicate rows found." if dups == 0
                else f"WARNING: {dups:,} duplicate rows detected ({dups/rows*100:.1f}% of data).")
    if any(w in q for w in ["correlat","relationship","related to","association"]):
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num) < 2:
            return "Need at least 2 numeric columns to compute correlations."
        corr = df[num].corr().abs()
        pairs = (corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
                 .stack().sort_values(ascending=False).head(6))
        lines = [f"- `{c1}` <-> `{c2}`: r = {v:.3f} "
                 f"({'strong' if v>.7 else 'moderate' if v>.4 else 'weak'})"
                 for (c1,c2), v in pairs.items()]
        return "Top column correlations:\n" + "\n".join(lines)
    if any(w in q for w in ["average","mean","avg"]):
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num: return "No numeric columns found."
        for col in num:
            if col.lower() in q:
                return (f"`{col}` -- mean: {fmt(df[col].mean())}, "
                        f"median: {fmt(df[col].median())}, std: {fmt(df[col].std())}")
        lines = [f"- `{c}`: {fmt(df[c].mean())}" for c in num[:10]]
        return "Column averages:\n" + "\n".join(lines)
    if any(w in q for w in ["max","maximum","highest","largest"]):
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num:
            if col.lower() in q:
                return f"Maximum of `{col}`: {fmt(df[col].max())}"
        lines = [f"- `{c}`: {fmt(df[c].max())}" for c in num[:10]]
        return "Column maximums:\n" + "\n".join(lines)
    if any(w in q for w in ["min","minimum","lowest","smallest"]):
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num:
            if col.lower() in q:
                return f"Minimum of `{col}`: {fmt(df[col].min())}"
        lines = [f"- `{c}`: {fmt(df[c].min())}" for c in num[:10]]
        return "Column minimums:\n" + "\n".join(lines)
    if any(w in q for w in ["unique","distinct","categories","how many values"]):
        for col in df.columns:
            if col.lower() in q:
                vc = df[col].value_counts().head(10)
                lines = [f"- '{v}': {c:,}" for v, c in vc.items()]
                return (f"`{col}` has {df[col].nunique()} unique values. Top values:\n"
                        + "\n".join(lines))
        cat = df.select_dtypes(include=["object","category"]).columns.tolist()
        lines = [f"- `{c}`: {df[c].nunique()} unique" for c in cat[:8]]
        return "Unique value counts per categorical column:\n" + "\n".join(lines)

    return (f"Here are the verified statistics for this dataset:\n\n"
            f"{ctx}\n\nQuestion asked: {question}")


# ─────────────────────────────────────────────────────────────────
#  AGENT
# ─────────────────────────────────────────────────────────────────

CUSTOM_INSTRUCTIONS = """
You are a friendly, expert Data Analyst assistant.
YOU HAVE NO MEMORY OF PREVIOUS CLUSTERING ACTIONS UNLESS YOU VERIFY THEM NOW.

TOOL SELECTION RULES:
- summarise_dataset()        -> overview, summary, describe, explore, profile data
- analyse_time_series(...)   -> trends, time patterns, how X changed over time
- run_classification(...)    -> classify, predict, supervised, train model, labels, target column
- run_clustering(...)        -> cluster, KMeans, DBSCAN, segment, group, unsupervised
- answer_data_question(...)  -> factual stats: rows, columns, missing, averages, correlations
- final_answer(answer: str)  -> ONLY call this once you have the results to provide the terminal response.

STRICT CLUSTERING & ANTI-HALLUCINATION RULES:
1. FORBIDDEN: Never mention "segments", "groups", "cluster counts", or "silhouette scores" in final_answer UNLESS you have just called a tool in this specific turn.
2. VERIFY FIRST: If a user asks "Can the data be grouped?", you MUST call `run_clustering` or `answer_data_question` to actually check. Do not answer "Yes/No" based on intuition.
3. NO GUESSING: If you have not called a tool, you do not know if segments exist. Never invent statistics or silhouette scores.
4. If a tool returns "Clustering has not been performed", your response must be to guide the user to run it.

BREVITY & STABILITY RULES (CRITICAL):
1. Groq models fail on very long tool inputs. KEEP YOUR FINAL ANSWER CONCISE.
2. Do not repeat every single number from the tool output in the `final_answer`. 
3. Summarize the key takeaway and refer the user to the "Direct Answer" or "Chart" above for details.
4. If a tool call succeeded (like `run_clustering`), your `final_answer` should just be a 2-3 sentence summary of the main finding.
5. Use maximum 3-4 bullet points in `final_answer`.

GENERAL RULES:
1. ALWAYS fetch data using a tool first. Never answer without data.
2. NEVER use the clustering tool if a target column or prediction is requested (use classification).
3. NEVER use the classification tool for general grouping/segments (use clustering).
4. Use ONLY numbers from tool results. Do not invent statistics.
5. Write your final response in clear plain English with bullet points.
6. Never output Python code or internal thinking sections.
7. If no dataset is loaded, tell the user to upload a CSV from the sidebar.
"""


def make_agent(model_id: str, api_key: str) -> ToolCallingAgent:
    llm = LiteLLMModel(model_id=model_id, api_key=api_key, temperature=0.1)
    return ToolCallingAgent(
        tools=[summarise_dataset, analyse_time_series, run_classification,
               run_clustering, answer_data_question],
        model=llm,
        instructions=CUSTOM_INSTRUCTIONS,
        max_steps=4,
    )


def run_agent(prompt: str, model_id: str, api_key: str) -> str:
    import litellm
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            agent = make_agent(model_id, api_key)
            st.session_state["pending_plot"] = None
            raw = agent.run(prompt)
            return clean_output(str(raw))
        except litellm.RateLimitError as e:
            wm = re.search(r"try again in (\d+(?:\.\d+)?)s", str(e))
            ws = int(float(wm.group(1))) + 3 if wm else 35
            if attempt < max_retries:
                time.sleep(ws)
            else:
                return (f"WARNING: Rate limit reached. Please wait ~{ws}s and try again, "
                        "or switch to a different model in the sidebar.")
        except Exception as e:
            err = str(e)
            if "charmap" in err.lower() or "codec" in err.lower():
                return "WARNING: A text encoding error occurred. Please try again."
            return f"WARNING: Error: {err[:500]}"


# ─────────────────────────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Smart Data Analyst",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

for k, v in {
    "data": None, "last_file": None, "data_context": None,
    "pending_plot": None, "messages": [],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Upload your CSV")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded and st.session_state["last_file"] != uploaded.name:
        try:
            df_new = read_csv_safe(uploaded)
            st.session_state.update({
                "data":         df_new,
                "last_file":    uploaded.name,
                "data_context": build_data_context(df_new),
                "pending_plot": None,
                "messages": [{
                    "role":    "assistant",
                    "content": (
                        f"[OK] **{uploaded.name}** loaded -- "
                        f"**{df_new.shape[0]:,} rows x {df_new.shape[1]} columns**.\n\n"
                        "Ask me anything! For example:\n"
                        "- \"Summarise the dataset\"\n"
                        "- \"Analyse trends over time\"\n"
                        "- \"Run a classification analysis (supervised learning, predict labels)\"*\n"
                        "- \"Cluster the data (KMeans, DBSCAN, segment, unsupervised)\"*\n"
                        "- \"How many missing values are there?\""
                    ),
                    "plot": None,
                }],
            })
            st.success("File loaded!")
        except Exception as e:
            st.error(str(e))

    df = st.session_state["data"]

    if df is not None:
        st.info(f"**{df.shape[0]:,} rows x {df.shape[1]} cols**")
        with st.expander("Preview (5 rows)"):
            st.dataframe(df.head(), use_container_width=True)
        st.divider()

        st.markdown("### Time Series Settings")
        ts_c_opts = ["Auto-detect"] + df.columns.tolist()
        ts_col    = st.selectbox("Date column", ts_c_opts, key="ts_col")
        num_all   = df.select_dtypes(include=[np.number]).columns.tolist()
        ts_val    = st.selectbox("Value column", ["Auto-detect"] + num_all, key="ts_val")
        ts_freq   = st.selectbox("Frequency", ["M -- Monthly","D -- Daily","W -- Weekly","Q -- Quarterly","Y -- Yearly"], key="ts_freq")
        ts_agg    = st.selectbox("Aggregation", ["mean -- Average","sum -- Total","median -- Median","max -- Max","min -- Min"], key="ts_agg")
        ts_col_v  = ts_col  if ts_col  != "Auto-detect" else "auto"
        ts_val_v  = ts_val  if ts_val  != "Auto-detect" else "auto"
        ts_freq_v = ts_freq.split(" -- ")[0]
        ts_agg_v  = ts_agg.split(" -- ")[0]
        st.divider()

        st.markdown("### Classification Settings")
        tgt_auto  = detect_target_col(df)
        all_c     = df.columns.tolist()
        clf_tgt   = st.selectbox("Target column", all_c,
                                  index=all_c.index(tgt_auto) if tgt_auto in all_c else 0,
                                  key="clf_tgt")
        avail_f   = [c for c in df.select_dtypes(include=[np.number]).columns if c != clf_tgt]
        sel_feats = st.multiselect("Feature columns", avail_f,
                                   default=avail_f[:min(8, len(avail_f))], key="clf_feats")
        clf_model = st.selectbox("Model", ["RandomForest","GradientBoosting","LogisticRegression"], key="clf_model")
        st.divider()

        st.markdown("### Clustering Settings")
        clust_feats   = st.multiselect("Cluster features", num_all,
                                        default=num_all[:min(5, len(num_all))], key="clust_feats")
        clust_k       = st.slider("Number of clusters (0 = auto)", 0, 10, 3, key="clust_k")
        clust_algo    = st.selectbox("Algorithm", ["KMeans", "DBSCAN"], key="clust_algo")
        clust_feats_v = ", ".join(clust_feats) if clust_feats else "auto"
        st.divider()

        st.markdown("### AI Model")
        model_opts = {
            "Llama 3.3 70B -- Smartest": "groq/llama-3.3-70b-versatile",
            "Llama 3.1 8B -- Fastest":  "groq/llama-3.1-8b-instant",
        }
        sel_model    = st.selectbox("", list(model_opts.keys()), index=0,
                                     label_visibility="collapsed", key="ai_model")
        sel_model_id = model_opts[sel_model]
        st.divider()

        st.markdown("### Export")
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "data.csv", "text/csv")
        st.divider()

        st.markdown("### ngrok Tunnel")
        ngrok_url = get_ngrok_url()
        if ngrok_url:
            st.success("Tunnel active!")
            st.code(ngrok_url, language=None)
            st.caption("Share this URL with anyone to access the app remotely.")
        else:
            st.warning("Tunnel not active. Start the Ngrok process locally.")
        st.divider()

        if st.button("Clear chat history"):
            st.session_state["messages"] = []
            st.rerun()
    else:
        sel_model_id  = "groq/llama-3.1-8b-instant"
        ts_col_v = ts_val_v = ts_freq_v = ts_agg_v = "auto"
        clf_tgt = "auto"; sel_feats = []; clf_model = "RandomForest"
        clust_feats_v = "auto"; clust_k = 3; clust_algo = "KMeans"


# ── Main area ─────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="icon">🌿</div>
  <div>
    <h1>Smart Data Analyst</h1>
    <p>Upload a CSV &rarr; chat to summarise, find trends, classify, cluster, or ask any question about your data.</p>
  </div>
</div>""", unsafe_allow_html=True)

ngrok_url = get_ngrok_url()
if ngrok_url:
    st.markdown(
        f'<div class="ngrok-banner">&#127760; Public URL: '
        f'<a href="{ngrok_url}" target="_blank">{ngrok_url}</a>'
        f'&nbsp;&mdash;&nbsp;share this link to access the app from anywhere.</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="ngrok-banner warn">&#9888; ngrok tunnel not active. '
        'Run <code>python start_ngrok.py</code> to enable public access.</div>',
        unsafe_allow_html=True
    )

groq_key = os.environ.get("GROQ_API_KEY", "")
if not groq_key:
    try:
        from google.colab import userdata
        groq_key = userdata.get("GROQ_API_KEY", "")
    except Exception:
        pass

if not groq_key:
    st.markdown("""<div class="ibox warn">
    <strong>GROQ_API_KEY is not set.</strong><br>
    Create a <code>.env</code> file in the same folder as this script and add:<br>
    <code>GROQ_API_KEY=your_key_here</code><br>
    Get a free key at <a href="https://console.groq.com" target="_blank">console.groq.com</a>
    </div>""", unsafe_allow_html=True)
    st.stop()

if st.session_state["data"] is None:
    st.markdown("""<div class="ibox info" style="text-align:center;padding:2rem;">
    <h3 style="color:#1B4332;margin-bottom:.5rem;">Welcome!</h3>
    <p>Upload a CSV file using the <strong>sidebar on the left</strong> to get started.</p>
    <p style="color:#52B788;margin-top:.5rem;">
    Then type anything in the chat &mdash; like:<br>
    <em>"summarise the data"</em> &middot; <em>"show trends"</em> &middot;
    <em>"classify the target"</em> &middot; <em>"cluster into 3 groups"</em> &middot;
    <em>"how many missing values?"</em>
    </p>
    </div>""", unsafe_allow_html=True)
    st.stop()

df       = st.session_state["data"]
has_time = detect_time_col(df) is not None

# ── Quick-action chips ────────────────────────────────────────────
chip_defs = {
    "Summarise":      "Give me a complete overview and summary of the dataset.",
    "Trends":         f"Analyse the time series. Time column: '{ts_col_v}', value column: '{ts_val_v}', frequency: '{ts_freq_v}', aggregation: '{ts_agg_v}'.",
    "Classify":       f"Run classification (supervised learning) to predict '{clf_tgt}' using features: {', '.join(sel_feats) if sel_feats else 'auto'}, model: {clf_model}.",
    "Cluster":        f"Run clustering (unsupervised KMeans/DBSCAN) using features: {clust_feats_v}, n_clusters: {clust_k}, algorithm: {clust_algo}.",
    "Missing values": "How many missing values are there in each column?",
    "Correlations":   "What are the strongest correlations between columns?",
}

st.markdown("**Quick actions — click any button to run instantly:**")
chip_cols = st.columns(len(chip_defs))
triggered_prompt = None
for idx, (label, prompt) in enumerate(chip_defs.items()):
    disabled = (label == "Trends" and not has_time)
    if chip_cols[idx].button(label, key=f"chip_{idx}", use_container_width=True, disabled=disabled):
        triggered_prompt = prompt

if not has_time:
    st.caption("'Trends' is disabled -- no date column detected in this dataset.")
st.divider()

# ── Chat history ──────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="row-user"><div class="bubble-user">{msg["content"]}</div>'
            f'<div class="avatar avatar-user">U</div></div>',
            unsafe_allow_html=True
        )
    else:
        content = msg["content"]
        content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'`(.*?)`',        r'<code>\1</code>',    content)
        lines_out = []
        for line in content.split("\n"):
            stripped = line.strip()
            if stripped.startswith("- "):
                lines_out.append(f"&bull; {stripped[2:]}<br>")
            else:
                lines_out.append(stripped + ("<br>" if stripped else "<br>"))
        content_html = "\n".join(lines_out)
        st.markdown(
            f'<div class="row-bot"><div class="avatar avatar-bot">A</div>'
            f'<div class="bubble-bot">{content_html}</div></div>',
            unsafe_allow_html=True
        )
        if msg.get("plot") is not None:
            st.pyplot(msg["plot"])
            plt.close("all")
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    col_in, col_btn = st.columns([6, 1])
    user_text   = col_in.text_input(
        "msg",
        placeholder="e.g. 'cluster into 3 groups', 'classify order_status', 'show revenue trends'",
        label_visibility="collapsed",
    )
    send_clicked = col_btn.form_submit_button("Send")

final_prompt = None
if send_clicked and user_text.strip():
    final_prompt = user_text.strip()
elif triggered_prompt:
    final_prompt = triggered_prompt

if final_prompt:
    st.session_state["messages"].append({"role": "user", "content": final_prompt, "plot": None})
    with st.spinner("Thinking..."):
        reply = run_agent(final_prompt, sel_model_id, groq_key)
    produced_plot = st.session_state.get("pending_plot")
    st.session_state["pending_plot"] = None
    st.session_state["messages"].append({"role": "assistant", "content": reply, "plot": produced_plot})
    st.rerun()
