"""
Smart Data Analyst 
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
                      color=[P[i % len(P)] for i in range(len(cl
