# app.py - AI Data Platform V7

import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
from langchain_openai import ChatOpenAI
import json, re, os, html
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide", page_title="AI Data Platform", page_icon="🚀")

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container{padding-top:1.1rem;padding-bottom:.8rem;}

/* header */
.platform-header{
    background:linear-gradient(100deg,#0f2027 0%,#203a43 55%,#2c5364 100%);
    border-radius:12px;padding:16px 26px;margin-bottom:16px;
}
.platform-header h1{margin:0;font-size:22px;font-weight:700;color:#fff;}
.platform-header p{margin:2px 0 0;font-size:11px;color:#90caf9;}

/* stat cards */
.stat-row{display:flex;gap:10px;margin-bottom:14px;}
.stat-card{
    flex:1;background:#0d1117;border:1px solid #21262d;
    border-radius:10px;padding:11px 14px;text-align:center;
}
.stat-card .sv{font-size:20px;font-weight:700;color:#4fc3f7;line-height:1.2;}
.stat-card .sl{font-size:10px;color:#8b949e;margin-top:2px;text-transform:uppercase;letter-spacing:.5px;}

/* sql intent strip */
.sql-strip{
    background:#0d1117;border:1px solid #21262d;border-left:3px solid #3b82f6;
    border-radius:0 8px 8px 0;padding:8px 14px;
    margin-bottom:10px;display:flex;align-items:center;gap:10px;
}
.sql-strip .badge{
    background:#1d4ed8;color:#bfdbfe;border-radius:4px;
    padding:2px 8px;font-size:10px;font-weight:700;letter-spacing:.6px;white-space:nowrap;
}
.sql-strip .sql-text{
    font-family:'SF Mono','Fira Code',monospace;font-size:11px;color:#8b949e;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;
}

/* exec summary */
.exec-box{
    background:#f0f9ff;border-left:3px solid #0369a1;
    border-radius:0 8px 8px 0;padding:14px 18px;
    color:#0c2340;font-size:13.5px;line-height:1.8;margin-top:8px;
}

/* join score pill */
.score-high{color:#22c55e;font-weight:700;}
.score-med {color:#f59e0b;font-weight:700;}
.score-low {color:#ef4444;font-weight:700;}

/* suggestion chip buttons — tighten default padding */
div[data-testid="stButton"]>button{border-radius:6px !important;}
div[data-testid="stMetric"]{background:transparent !important;border:none !important;padding:0 !important;}
div[data-testid="stTextInput"] input{border-radius:8px !important;font-size:14px !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LLM  (unchanged)
# ─────────────────────────────────────────────────────────────────
# llm = ChatOpenAI(
#     base_url=os.getenv("LLM_BASE_URL"),
#     api_key=os.getenv("LLM_API_KEY"),
#     default_headers={"x-api-key": os.getenv("LLM_HEADER_KEY")},
#     model="openai.gpt-4o",
#     temperature=0,
#     max_completion_tokens=600,
# )


llm = ChatOpenAI(
    base_url=st.secrets["LLM_BASE_URL"],
    api_key=st.secrets["LLM_API_KEY"],
    default_headers={
        "x-api-key": st.secrets["LLM_HEADER_KEY"]
    },
    model="openai.gpt-4o",
    temperature=0,
    max_tokens=600,
)


# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
DEFAULTS = {
    "dfs": {}, "join_mode": "auto",
    "manual_joins": {}, "sql_join_text": "",
    "memory": {}, "query_history": [],
    "last_query": "", "last_plan": None,
    "last_result": None, "last_exec_summary": None,
    "llm_calls": 0, "total_tokens": 0,
    "max_llm_calls": 60, "max_tokens": 30000,
    "query_input": "",
    # suggestion routing fix: store pending suggestion separately
    "pending_suggestion": None,
    "last_insights": None,      # ← ADD THIS LINE
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────
def clean_name(n): return n.replace(".csv","").lower().strip()
def norm(t):       return re.sub(r'[^a-z0-9]','',str(t).lower())

def call_llm(prompt: str) -> str | None:
    if st.session_state.llm_calls >= st.session_state.max_llm_calls:
        st.error("🚫 LLM call limit reached.")
        return None
    resp = llm.invoke(prompt)
    text = getattr(resp, "content", str(resp))
    st.session_state.llm_calls   += 1
    st.session_state.total_tokens += int((len(prompt)+len(text))/4)
    return text

def load_files(files):
    for f in files:
        df = pd.read_csv(f)
        for col in df.columns:
            if "date" in col.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
        st.session_state.dfs[clean_name(f.name)] = df

def update_history(q: str, plan: dict):
    st.session_state.last_query = q
    st.session_state.last_plan  = plan
    if q not in st.session_state.query_history:
        st.session_state.query_history.insert(0, q)
    st.session_state.query_history = st.session_state.query_history[:8]

# ─────────────────────────────────────────────────────────────────
# SQL GUARDRAILS  — block any mutating / DDL statements
# ─────────────────────────────────────────────────────────────────
_BLOCKED = re.compile(
    r'^\s*(drop|delete|truncate|update|insert|alter|create|replace|merge|call|exec)\b',
    re.IGNORECASE | re.MULTILINE,
)

def sql_is_safe(sql: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Blocks any destructive SQL."""
    if _BLOCKED.search(sql):
        keyword = _BLOCKED.search(sql).group(1).upper()
        return False, f"Statement contains blocked keyword: **{keyword}**. Only SELECT queries are allowed."
    if not re.search(r'\bSELECT\b', sql, re.IGNORECASE):
        return False, "Only SELECT queries are permitted."
    return True, ""

# ─────────────────────────────────────────────────────────────────
# SCHEMA BUILDER  (unchanged — column-level + value samples)
# ─────────────────────────────────────────────────────────────────
def build_rich_schema(df: pd.DataFrame) -> str:
    lines = []
    for col in df.columns:
        s = df[col]; nn = s.notna().sum(); uniq = s.nunique()
        if pd.api.types.is_numeric_dtype(s):
            mn = round(float(s.min()),2) if nn else "N/A"
            mx = round(float(s.max()),2) if nn else "N/A"
            lines.append(f"  {col} ({s.dtype}): range=[{mn},{mx}]")
        elif pd.api.types.is_datetime64_any_dtype(s):
            mn = str(s.min())[:10] if nn else "N/A"
            mx = str(s.max())[:10] if nn else "N/A"
            lines.append(f"  {col} (date): range=[{mn},{mx}]")
        else:
            top = s.dropna().value_counts().head(5).index.tolist()
            lines.append(f"  {col} (text,{uniq} unique): top_values={top}")
    return "COLUMNS:\n" + "\n".join(lines)

# ─────────────────────────────────────────────────────────────────
# CORE NLQ ENGINE  (unchanged logic, same prompt)
# ─────────────────────────────────────────────────────────────────
def nlq_to_sql(question: str, df: pd.DataFrame) -> str | None:
    schema     = build_rich_schema(df)
    name_cols  = [c for c in df.columns if any(x in c.lower() for x in
                  ["first","last","fname","lname","name","full"])]
    prompt = f"""You are an expert DuckDB SQL generator. Given a dataset schema and a natural language question, generate the best DuckDB SQL query.

TABLE NAME: df
{schema}

RULES:
1. Always SELECT meaningful labels. If there are separate first_name and last_name columns, concatenate: first_name || ' ' || last_name AS salesperson_name
2. For "best/top/worst" queries: always ORDER BY metric DESC/ASC with LIMIT (default 10 if not specified)
3. For trend queries: use DATE_TRUNC or strftime to group by month/year
4. For "by X and Y" queries: GROUP BY both X and Y columns
5. For count queries: use COUNT(*) or COUNT(DISTINCT col)
6. For comparison queries (vs/compare): use CASE or multiple aggregations
7. Always use meaningful column aliases
8. If question involves a specific value (ford, red, SUV), use WHERE col ILIKE '%value%'
9. Never return more than 500 rows unless explicitly asked
10. For salesperson/person queries: combine first+last name if both exist, never show just first_name if last_name also exists
11. For date columns, handle NULL safely with IS NOT NULL where needed
12. Multi-column group: if user says "by brand and type", GROUP BY make, car_type
13. Return ONLY the SQL string, no explanation, no markdown fences.

NAME COLUMNS DETECTED: {name_cols}

QUESTION: {question}

SQL:"""
    return call_llm(prompt)


def run_sql(sql: str, df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """Execute SQL against df via DuckDB. Guardrail-checked before execution."""
    safe, reason = sql_is_safe(sql)
    if not safe:
        return None, f"🔒 Blocked: {reason}"
    try:
        con = duckdb.connect()
        con.register("df", df)
        result = con.execute(sql.strip()).df()
        con.close()
        return result, None
    except Exception as e:
        return None, str(e)


def enrich_query(q: str) -> str:
    if not st.session_state.get("last_plan"):
        return q
    triggers = ["top","lowest","highest","now","only","for","in","show","filter","same","also"]
    if any(w in q.lower() for w in triggers) and len(q.split()) <= 7:
        prev = st.session_state.get("last_query","")
        if prev:
            return prev + " " + q
    return q


def run_query(working_df: pd.DataFrame, question: str):
    """NLQ → SQL → DuckDB → DataFrame. Returns (result_df, sql_str, error_str)."""
    if working_df is None or working_df.empty:
        return None, "", "No data loaded."

    q = enrich_query(question)
    cache_key = f"nlq_{q}"
    if cache_key in st.session_state.memory:
        cached = st.session_state.memory[cache_key]
        update_history(question, {"sql": cached[1]})
        return cached

    sql = nlq_to_sql(q, working_df)
    if not sql:
        return None, "", "LLM did not return SQL."

    sql = sql.strip().strip("`").strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()

    result, err = run_sql(sql, working_df)

    if err and not err.startswith("🔒"):
        retry_prompt = f"""The following DuckDB SQL failed with error: {err}
SQL: {sql}
Schema columns: {list(working_df.columns)}
Fix and return ONLY corrected SQL:"""
        sql2 = call_llm(retry_prompt)
        if sql2:
            sql2 = sql2.strip().strip("`").strip()
            result, err = run_sql(sql2, working_df)
            sql = sql2

    if result is not None:
        st.session_state.memory[cache_key] = (result, sql, None)
        update_history(question, {"sql": sql})

    return result, sql, err

# ─────────────────────────────────────────────────────────────────
# IMPROVED AUTO-JOIN
# Improvements over V6:
#   - normalizes column names before matching (order_id == OrderID)
#   - scores candidate join keys by: data-type match, value overlap %, cardinality
#   - prefers integer/id-like columns over free-text columns
#   - shows match quality per pair so user can see what happened
# ─────────────────────────────────────────────────────────────────
def _col_norm_map(df: pd.DataFrame) -> dict:
    """Returns {normalized_name: actual_column_name}"""
    return {norm(c): c for c in df.columns}

def _join_score(left_series: pd.Series, right_series: pd.Series) -> float:
    """
    Score a candidate join pair 0–100.
    Combines: value overlap, cardinality match, dtype match.
    """
    try:
        l_vals = set(left_series.dropna().astype(str).unique())
        r_vals = set(right_series.dropna().astype(str).unique())
        if not l_vals or not r_vals:
            return 0.0
        overlap = len(l_vals & r_vals) / min(len(l_vals), len(r_vals))
        # cardinality ratio — prefer columns where both sides have similar unique count
        card_ratio = min(len(l_vals), len(r_vals)) / max(len(l_vals), len(r_vals))
        # dtype bonus
        dtype_match = 1.0 if left_series.dtype == right_series.dtype else 0.7
        # id-like name bonus
        name_bonus = 1.1 if any(x in norm(left_series.name) for x in ["id","key","code","num"]) else 1.0
        score = overlap * 0.6 + card_ratio * 0.3 + (dtype_match - 1) * 0.1
        return round(min(score * dtype_match * name_bonus * 100, 100), 1)
    except Exception:
        return 0.0

def auto_join(dfs: dict) -> tuple[pd.DataFrame, list[dict]]:
    """
    Returns (joined_df, join_log).
    join_log = [{"left_col":..,"right_col":..,"score":..,"left_table":..,"right_table":..}]
    """
    tables = list(dfs.items())
    if len(tables) == 1:
        return tables[0][1], []

    base_name, base = tables[0][0], tables[0][1].copy()
    join_log = []

    for r_name, right in tables[1:]:
        l_map = _col_norm_map(base)
        r_map = _col_norm_map(right)
        common_norms = set(l_map.keys()) & set(r_map.keys())

        if not common_norms:
            # fallback: try fuzzy — any norm key of left that is a substring of a norm key of right
            for lk in l_map:
                for rk in r_map:
                    if lk in rk or rk in lk:
                        common_norms.add(lk)
                        r_map[lk] = r_map.pop(rk, r_map.get(lk))
                        break

        if not common_norms:
            join_log.append({"left_table": base_name, "right_table": r_name,
                              "left_col": "—", "right_col": "—", "score": 0,
                              "note": "No matching columns found"})
            continue

        # score each candidate, pick the best
        best_score, best_lc, best_rc = -1, None, None
        for n_key in common_norms:
            lc = l_map.get(n_key)
            rc = r_map.get(n_key)
            if lc and rc and lc in base.columns and rc in right.columns:
                s = _join_score(base[lc], right[rc])
                if s > best_score:
                    best_score, best_lc, best_rc = s, lc, rc

        if best_lc is None or best_score < 5:
            join_log.append({"left_table": base_name, "right_table": r_name,
                              "left_col": "—", "right_col": "—", "score": best_score,
                              "note": "Score too low — skipped"})
            continue

        try:
            merged = pd.merge(base, right, left_on=best_lc, right_on=best_rc,
                              how="left", suffixes=("", f"_{r_name}"))
            # drop duplicated suffix columns
            merged = merged[[c for c in merged.columns
                              if not (c.endswith(f"_{r_name}") and c[:-len(f"_{r_name}")] in merged.columns)]]
            base = merged
            join_log.append({"left_table": base_name, "right_table": r_name,
                              "left_col": best_lc, "right_col": best_rc,
                              "score": best_score, "note": "OK"})
        except Exception as e:
            join_log.append({"left_table": base_name, "right_table": r_name,
                              "left_col": best_lc, "right_col": best_rc,
                              "score": best_score, "note": f"Merge error: {e}"})

    return base, join_log


def manual_join(dfs: dict, joins: dict) -> pd.DataFrame:
    if not joins:
        return list(dfs.values())[0]
    first = list(joins.values())[0]
    if first["left"] not in dfs:
        return list(dfs.values())[0]
    base = dfs[first["left"]].copy()
    for j in joins.values():
        if not j.get("left_on") or not j.get("right_on"): continue
        if j["right"] not in dfs: continue
        try:
            base = pd.merge(base, dfs[j["right"]],
                            left_on=j["left_on"], right_on=j["right_on"],
                            how=j.get("type","inner"), suffixes=("","_r"))
            base = base[[c for c in base.columns if not c.endswith("_r")]]
        except Exception as e:
            st.warning(f"Join error: {e}")
    return base


def sql_join(dfs: dict, sql: str) -> pd.DataFrame | None:
    safe, reason = sql_is_safe(sql)
    if not safe:
        st.error(f"🔒 Blocked: {reason}")
        return None
    try:
        con = duckdb.connect()
        for name, df in dfs.items():
            con.register(name, df)
        result = con.execute(sql).df()
        con.close()
        return result
    except Exception as e:
        st.error(f"SQL join error: {e}")
        return None


def get_working_df() -> pd.DataFrame | None:
    dfs = st.session_state.dfs
    if not dfs: return None
    if len(dfs) == 1: return list(dfs.values())[0]
    mode = st.session_state.join_mode
    if mode == "auto":
        df, _ = auto_join(dfs)
        return df
    elif mode == "manual":
        return manual_join(dfs, st.session_state.manual_joins)
    elif mode == "sql":
        sql = st.session_state.sql_join_text
        return sql_join(dfs, sql) if sql.strip() else list(dfs.values())[0]
    return list(dfs.values())[0]

# ─────────────────────────────────────────────────────────────────
# CHART BUILDER  (unchanged)
# ─────────────────────────────────────────────────────────────────
def auto_chart_type(result: pd.DataFrame, question: str) -> str:
    q = question.lower()
    n = len(result)
    num_cols = result.select_dtypes(include="number").columns.tolist()
    str_cols = result.select_dtypes(exclude="number").columns.tolist()
    if any(w in q for w in ["trend","monthly","yearly","over time","growth"]): return "Line"
    if any(w in q for w in ["share","proportion","percent","breakdown","distribution"]) and n<=10: return "Pie"
    if any(w in q for w in ["compare","vs","versus"]): return "Bar"
    if len(num_cols)>=2 and len(str_cols)==0: return "Scatter"
    if n>30: return "Line"
    return "Bar"

def build_chart(result: pd.DataFrame, chart_type: str, x_col: str, y_col: str):
    try:
        df_plot = result.copy()
        df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors="coerce")
        colors = px.colors.qualitative.Plotly
        layout = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10,r=10,t=30,b=10), font=dict(size=12))
        if chart_type=="Bar":
            fig = px.bar(df_plot,x=x_col,y=y_col,text_auto=True,
                         color=x_col if df_plot[x_col].nunique()<=20 else None,
                         color_discrete_sequence=colors)
        elif chart_type=="Line":
            fig = px.line(df_plot,x=x_col,y=y_col,markers=True)
        elif chart_type=="Pie":
            fig = px.pie(df_plot,names=x_col,values=y_col,hole=0.35,
                         color_discrete_sequence=colors)
        elif chart_type=="Scatter":
            fig = px.scatter(df_plot,x=x_col,y=y_col)
        elif chart_type=="Area":
            fig = px.area(df_plot,x=x_col,y=y_col)
        else:
            fig = px.bar(df_plot,x=x_col,y=y_col)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")

# ─────────────────────────────────────────────────────────────────
# EXECUTIVE SUMMARY  (unchanged — LLM only on demand)
# ─────────────────────────────────────────────────────────────────
def generate_executive_summary(result: pd.DataFrame, question: str) -> str:
    rows   = min(15, len(result))
    sample = result.head(rows).to_dict(orient="records")
    num_cols = result.select_dtypes(include="number").columns.tolist()
    stats = {}
    for c in num_cols[:4]:
        col_s = result[c].dropna()
        if len(col_s):
            stats[c] = {"total":round(float(col_s.sum()),2),"avg":round(float(col_s.mean()),2),
                        "max":round(float(col_s.max()),2),"min":round(float(col_s.min()),2)}
    prompt = f"""You are a senior business analyst writing an executive summary.
Question asked: "{question}"
Dataset ({len(result)} rows, showing top {rows}): {json.dumps(sample, default=str)}
Statistics: {json.dumps(stats)}

Write a concise executive summary (4-6 sentences) covering:
- Key finding / direct answer to the question
- Top performer or notable value
- Any significant pattern or trend
- Business recommendation

Be factual. Use only the data above. Executive tone. Plain text, no bullet points."""
    return call_llm(prompt) or "Could not generate summary."

# ─────────────────────────────────────────────────────────────────
# ═══════════════  UI  ═══════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="platform-header">
  <h1>🚀 AI Data Platform</h1>
  <p>Natural Language &rarr; SQL &rarr; Visualization &nbsp;|&nbsp; Powered by GPT-4o &amp; DuckDB</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Data")
    files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
    if files:
        load_files(files)
        st.success(f"✅ {len(st.session_state.dfs)} file(s) loaded")

    if st.session_state.dfs:
        st.markdown("---")
        st.markdown("**Loaded Tables**")
        for tname, tdf in st.session_state.dfs.items():
            st.markdown(f"- `{tname}` — {tdf.shape[0]:,} rows × {tdf.shape[1]} cols")

        st.markdown("---")
        st.markdown("**Usage**")
        c1, c2 = st.columns(2)
        c1.metric("LLM Calls",   st.session_state.llm_calls)
        c2.metric("Tokens Used", f"{st.session_state.total_tokens:,}")
        st.progress(
            min(st.session_state.llm_calls / st.session_state.max_llm_calls, 1.0),
            text=f"{max(st.session_state.max_llm_calls - st.session_state.llm_calls, 0)} calls remaining"
        )
        if st.button("🔄 Reset Usage"):
            st.session_state.llm_calls = 0
            st.session_state.total_tokens = 0
            st.rerun()
        if st.button("🗑️ Clear Cache"):
            st.session_state.memory        = {}
            st.session_state.last_plan     = None
            st.session_state.last_query    = ""
            st.session_state.query_history = []
            st.rerun()

# ─────────────────────────────────────────────────────────────────
# GATE
# ─────────────────────────────────────────────────────────────────
if not st.session_state.dfs:
    st.info("👈 Upload one or more CSV files to get started.")
    st.stop()

tables = list(st.session_state.dfs.keys())
tab_join, tab_preview, tab_query = st.tabs(["🔗 Join / Combine", "📄 Data Preview", "⚡ AI Query"])

# ═══════════════════════════════════════════════════════════
# TAB 1 — JOIN / COMBINE  (upgraded auto-join with scoring)
# ═══════════════════════════════════════════════════════════
with tab_join:
    if len(st.session_state.dfs) == 1:
        st.info("Only one table loaded — no joining needed. Go to AI Query.")
    else:
        st.subheader("🔗 Combine Tables")

        mode_label = st.radio(
            "Join Method",
            ["🤖 Auto-detect (recommended)", "🛠️ Manual UI", "📝 SQL Query"],
            horizontal=True,
        )
        st.session_state.join_mode = (
            "auto"   if "Auto"   in mode_label else
            "manual" if "Manual" in mode_label else "sql"
        )
        st.markdown("---")

        # ── AUTO ──────────────────────────────────────────
        if st.session_state.join_mode == "auto":
            st.markdown(
                "Auto-join detects the best key between each table pair using "
                "**column name similarity + value overlap scoring**. "
                "A score ≥ 60 is a reliable join."
            )
            if st.button("▶️ Preview Auto-Join"):
                with st.spinner("Analysing tables and scoring join keys…"):
                    joined, join_log = auto_join(st.session_state.dfs)

                if joined is not None:
                    st.success(f"✅ Result: {joined.shape[0]:,} rows × {joined.shape[1]} cols")

                    # Show join quality report
                    if join_log:
                        st.markdown("**Join Quality Report**")
                        for entry in join_log:
                            score = entry["score"]
                            cls   = "score-high" if score>=60 else ("score-med" if score>=30 else "score-low")
                            icon  = "✅" if score>=60 else ("⚠️" if score>=30 else "❌")
                            st.markdown(
                                f"{icon} `{entry['left_table']}`.`{entry['left_col']}` ↔ "
                                f"`{entry['right_table']}`.`{entry['right_col']}` — "
                                f"<span class='{cls}'>score {score}</span> &nbsp; _{entry.get('note','')}_",
                                unsafe_allow_html=True,
                            )
                    st.dataframe(joined.head(100), use_container_width=True)

        # ── MANUAL UI ────────────────────────────────────
        elif st.session_state.join_mode == "manual":
            joins = st.session_state.manual_joins
            if not joins:
                joins[0] = {"left":tables[0],"right":tables[min(1,len(tables)-1)],
                            "left_on":"","right_on":"","type":"inner"}
            to_del = []
            for i, j in joins.items():
                c0,c1,c2,c3,c4,c5 = st.columns([2,2,2,2,1,0.5])
                j["left"]  = c0.selectbox("Base",      tables, index=tables.index(j["left"])  if j["left"]  in tables else 0, key=f"l{i}")
                j["right"] = c1.selectbox("Join Table", tables, index=tables.index(j["right"]) if j["right"] in tables else 0, key=f"r{i}")
                lc = list(st.session_state.dfs[j["left"]].columns)
                rc = list(st.session_state.dfs[j["right"]].columns)
                j["left_on"]  = c2.selectbox("Left Key",  lc, key=f"lk{i}")
                j["right_on"] = c3.selectbox("Right Key", rc, key=f"rk{i}")
                j["type"]     = c4.selectbox("Type", ["inner","left","right","outer"], key=f"jt{i}")
                if c5.button("❌", key=f"d{i}"): to_del.append(i)
            for r in to_del:
                del st.session_state.manual_joins[r]
            if to_del: st.rerun()
            ca, cb = st.columns(2)
            if ca.button("➕ Add Join"):
                nk = max(joins.keys(), default=-1)+1
                joins[nk] = {"left":tables[0],"right":tables[0],"left_on":"","right_on":"","type":"inner"}
                st.rerun()
            if cb.button("▶️ Preview"):
                jdf = manual_join(st.session_state.dfs, joins)
                if jdf is not None:
                    st.success(f"✅ {jdf.shape[0]:,} rows × {jdf.shape[1]} cols")
                    st.dataframe(jdf.head(100), use_container_width=True)

        # ── SQL JOIN ─────────────────────────────────────
        elif st.session_state.join_mode == "sql":
            st.markdown("**Available tables:** " + ", ".join([f"`{t}`" for t in tables]))
            sql_text = st.text_area(
                "SQL Join Query",
                value=st.session_state.sql_join_text or
                      f"SELECT *\nFROM {tables[0]}\n" +
                      (f"LEFT JOIN {tables[1]} ON {tables[0]}.id = {tables[1]}.id" if len(tables)>1 else ""),
                height=140,
            )
            st.session_state.sql_join_text = sql_text
            if st.button("▶️ Execute & Preview"):
                jdf = sql_join(st.session_state.dfs, sql_text)
                if jdf is not None:
                    st.success(f"✅ {jdf.shape[0]:,} rows × {jdf.shape[1]} cols")
                    st.dataframe(jdf.head(100), use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2 — DATA PREVIEW  (unchanged)
# ═══════════════════════════════════════════════════════════
with tab_preview:
    st.subheader("📄 Data Preview")
    sel    = st.selectbox("Select Table", tables)
    search = st.text_input("🔍 Search columns", "")
    pdf    = st.session_state.dfs[sel]
    if search:
        matched = [c for c in pdf.columns if search.lower() in c.lower()]
        pdf = pdf[matched] if matched else pdf
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card"><div class="sv">{pdf.shape[0]:,}</div><div class="sl">Rows</div></div>
      <div class="stat-card"><div class="sv">{pdf.shape[1]}</div><div class="sl">Columns</div></div>
      <div class="stat-card"><div class="sv">{pdf.select_dtypes(include='number').shape[1]}</div><div class="sl">Numeric Cols</div></div>
    </div>""", unsafe_allow_html=True)
    st.dataframe(pdf.head(200), use_container_width=True)
    with st.expander("📌 Column Details"):
        info = []
        for col in st.session_state.dfs[sel].columns:
            s = st.session_state.dfs[sel][col]
            info.append({"Column":col,"Type":str(s.dtype),
                         "Non-Null":int(s.notna().sum()),"Null":int(s.isna().sum()),
                         "Unique":int(s.nunique()),
                         "Sample":str(s.dropna().iloc[0]) if s.notna().any() else "N/A"})
        st.dataframe(pd.DataFrame(info), use_container_width=True)
    if len(tables)>1:
        st.markdown("---")
        if st.button("▶️ Show Working Dataset (after join)"):
            wdf = get_working_df()
            if wdf is not None:
                st.info(f"{wdf.shape[0]:,} rows × {wdf.shape[1]} cols")
                st.dataframe(wdf.head(100), use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — AI QUERY
# ═══════════════════════════════════════════════════════════
with tab_query:
    st.subheader("⚡ AI Query")

    # ── working df — computed ONCE, before any widgets ──────
    working_df = get_working_df()
    if working_df is None or working_df.empty:
        st.warning("⚠️ No data available.")
        st.stop()

    # ── QUICK FILTERS ────────────────────────────────────────
    with st.expander("🎛️ Quick Filters", expanded=False):
        qf    = {}
        qcols = st.columns(4)
        idx   = 0
        cat_candidates = [c for c in working_df.columns
                          if working_df[c].dtype == object and working_df[c].nunique() <= 40]
        for col in cat_candidates[:3]:
            vals    = ["All"] + sorted(working_df[col].dropna().unique().tolist())
            sel_val = qcols[idx%4].selectbox(col, vals, key=f"qf_{col}")
            if sel_val != "All": qf[col] = sel_val
            idx += 1
        date_cols = [c for c in working_df.columns
                     if pd.api.types.is_datetime64_any_dtype(working_df[c])]
        if date_cols:
            dc = date_cols[0]
            mn = int(working_df[dc].dt.year.min())
            mx = int(working_df[dc].dt.year.max())
            if mn < mx:
                yr = st.slider("Year Range", mn, mx, (mn, mx), key="qf_yr")
                qf["__year__"] = (dc, yr)
        if qf:
            for col, val in qf.items():
                if col == "__year__":
                    dc2, (y1,y2) = val
                    working_df = working_df[working_df[dc2].dt.year.between(y1,y2)]
                else:
                    working_df = working_df[
                        working_df[col].astype(str).str.strip().str.lower()
                        == str(val).strip().lower()
                    ]
            st.success(f"✅ Filters applied — {working_df.shape[0]:,} rows")

    # ── snapshot AFTER filters, used by ALL query paths ─────
    _wdf = working_df

    st.markdown("---")

    # ── QUERY HISTORY ────────────────────────────────────────
    if st.session_state.query_history:
        st.markdown("**Recent Queries**")
        hcols = st.columns(min(len(st.session_state.query_history), 4))
        for i, hq in enumerate(st.session_state.query_history[:4]):
            if hcols[i].button(f"↩ {hq[:28]}{'…' if len(hq)>28 else ''}", key=f"h{i}"):
                st.session_state.query_input = hq
                st.rerun()

    # ── QUERY INPUT ──────────────────────────────────────────
    qcol, runcol, clrcol = st.columns([8,1,1])
    q = qcol.text_input(
        "Ask anything",
        key="query_input",
        placeholder="e.g. top 10 salespersons by revenue, monthly sales trend for Ford in 2023, sales by brand and car type",
        label_visibility="collapsed",
    )
    run_clicked = runcol.button("▶️ Run",   use_container_width=True)
    clr_clicked = clrcol.button("🗑️ Clear", use_container_width=True)

    if clr_clicked:
        st.session_state.last_result       = None
        st.session_state.last_exec_summary = None
        st.session_state.pending_suggestion = None
        st.rerun()

    # ── RUN from text input ──────────────────────────────────
    if run_clicked and q.strip():
        with st.spinner("Generating SQL & fetching results…"):
            result, sql, err = run_query(_wdf, q.strip())
        st.session_state.last_result        = (result, sql, err, q.strip())
        st.session_state.last_exec_summary  = None
        st.session_state.pending_suggestion = None
        st.session_state.view_toggle        = "📋 Table"   ##rk1 ← ADD THIS LINE 


    # ── SUGGESTED QUERIES ────────────────────────────────────
    # FIX: suggestions set pending_suggestion in session_state and rerun.
    # The actual query runs at the TOP of the next render cycle,
    # before any expander re-evaluation — this eliminates the
    # "df evaluated inside expander" scope bug.
    # with st.expander("💡 Example Queries", expanded=False):
    #     suggestions = [
    #         "top 10 salespersons by total sales",
    #         "sales by brand and car type",
    #         "monthly sales trend",
    #         "compare sales by colour",
    #         "top 5 car models by revenue in 2023",
    #         "average price by car type",
    #         "sales by make and colour",
    #         "total revenue by year",
    #         "which salesperson sold the most SUVs",
    #         "bottom 5 performing models",
    #     ]
    #     sg_cols = st.columns(5)
    #     for i, sg in enumerate(suggestions):
    #         if sg_cols[i%5].button(sg, key=f"sg{i}"):
    #             st.session_state.pending_suggestion = sg
    #             st.session_state.query_input        = sg
    #             st.rerun()   # rerun — pending_suggestion is handled below

    # ── PROCESS PENDING SUGGESTION (top-level, safe scope) ──
    # This block runs after the full widget tree is evaluated,
    # so _wdf is always the correctly filtered working_df.
    if st.session_state.pending_suggestion:
        _sg = st.session_state.pending_suggestion
        st.session_state.pending_suggestion = None   # clear before running
        with st.spinner(f"Running: {_sg}"):
            _r, _s, _e = run_query(_wdf, _sg)
        st.session_state.last_result       = (_r, _s, _e, _sg)
        st.session_state.last_exec_summary = None
        st.session_state.view_toggle       = "📋 Table"   ##rk1 ← ADD THIS LINE
        st.rerun()

    st.markdown("---")

    # ── RESULTS ──────────────────────────────────────────────
    if st.session_state.last_result is not None:
        result, sql, err, asked_q = st.session_state.last_result

        if err and result is None:
            st.error(f"❌ {err}")
            if sql:
                with st.expander("🔍 SQL Attempted"):
                    st.code(sql, language="sql")

        elif result is not None and not result.empty:

            # ── SQL INTENT STRIP ─────────────────────────────
            sql_safe_preview = html.escape((sql or "").strip().replace("\n"," "))
            first80 = sql_safe_preview[:120] + ("…" if len(sql_safe_preview)>120 else "")
            st.markdown(f"""
            <div class="sql-strip">
              <span class="badge">SQL</span>
              <span class="sql-text">{first80}</span>
            </div>""", unsafe_allow_html=True)

            # ── EDITABLE SQL EXPANDER ────────────────────────
            with st.expander("✏️ View / Edit & Re-run SQL", expanded=False):
                edited_sql = st.text_area(
                    "Edit then Re-run — no LLM call, direct DuckDB execution",
                    value=(sql or "").strip(),
                    height=140,
                    key="edited_sql_area",
                )
                rcol, _ = st.columns([2,8])
                if rcol.button("▶️ Re-run SQL", key="rerun_sql_btn"):
                    safe, reason = sql_is_safe(edited_sql.strip())
                    if not safe:
                        st.error(f"🔒 Blocked: {reason}")
                    else:
                        with st.spinner("Running edited SQL…"):
                            new_result, new_err = run_sql(edited_sql.strip(), _wdf)
                        if new_err:
                            st.error(f"SQL error: {new_err}")
                        elif new_result is not None:
                            st.session_state.last_result       = (new_result, edited_sql.strip(), None, asked_q)
                            st.session_state.last_exec_summary = None
                            st.session_state.view_toggle       = "📋 Table"   ##rk1 ← ADD THIS LINE
                            st.rerun()

            # ── STAT CARDS ───────────────────────────────────
            num_c   = result.select_dtypes(include="number").columns.tolist()
            total_v = f"{result[num_c[0]].sum():,.1f}" if num_c else "—"
            total_l = num_c[0] if num_c else "Total"
            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-card"><div class="sv">{result.shape[0]:,}</div><div class="sl">Rows</div></div>
              <div class="stat-card"><div class="sv">{result.shape[1]}</div><div class="sl">Columns</div></div>
              <div class="stat-card"><div class="sv">{total_v}</div><div class="sl">{total_l}</div></div>
            </div>""", unsafe_allow_html=True)

            # ── VIEW TOGGLE ──────────────────────────────────
            view = st.radio("View", ["📊 Chart","📋 Table"], horizontal=True, key="view_toggle")

            if view == "📊 Chart":
                all_cols = list(result.columns)
                num_cols = result.select_dtypes(include="number").columns.tolist()
                str_cols = result.select_dtypes(exclude="number").columns.tolist()
                ctrl_col, chart_col = st.columns([2,8])
                auto_ct    = auto_chart_type(result, asked_q)
                chart_type = ctrl_col.selectbox("Chart Type",["Bar","Line","Pie","Scatter","Area"],
                                                 index=["Bar","Line","Pie","Scatter","Area"].index(auto_ct),
                                                 key="ct_sel")
                default_x  = str_cols[0] if str_cols else all_cols[0]
                default_y  = num_cols[0] if num_cols else all_cols[-1]
                x_axis = ctrl_col.selectbox("X Axis", all_cols,
                                             index=all_cols.index(default_x), key="xa")
                y_axis = ctrl_col.selectbox("Y Axis", all_cols,
                                             index=all_cols.index(default_y) if default_y in all_cols else 0,
                                             key="ya")
                with chart_col:
                    build_chart(result[[x_axis,y_axis]], chart_type, x_axis, y_axis)
            else:
                st.dataframe(result, use_container_width=True)
                st.download_button("⬇️ Download CSV",
                                   data=result.to_csv(index=False).encode(),
                                   file_name="result.csv", mime="text/csv")

            #

            # ── EXECUTIVE SUMMARY + INSIGHTS — on demand ─────────────
            st.markdown("---")
            sum_col, ins_col = st.columns([1, 1])

            # Executive Summary button + display
            with sum_col:
                if st.button("📋 Generate Executive Summary", use_container_width=True):
                    with st.spinner("Generating summary…"):
                        st.session_state.last_exec_summary = generate_executive_summary(result, asked_q)
                        # Clear insights when a fresh summary is generated
                        st.session_state.last_insights = None

                if st.session_state.get("last_exec_summary"):
                    st.markdown(
                        f"<div class='exec-box'>{html.escape(st.session_state.last_exec_summary)}</div>",
                        unsafe_allow_html=True,
                    )

            # Insights button — only active after summary exists
            with ins_col:
                insights_disabled = not bool(st.session_state.get("last_exec_summary"))
                if st.button(
                    "💡 Generate Insights",
                    use_container_width=True,
                    disabled=insights_disabled,
                    help="Generate Executive Summary first to unlock Insights",
                ):
                    with st.spinner("Generating insights…"):
                        rows   = min(15, len(result))
                        sample = result.head(rows).to_dict(orient="records")
                        insight_prompt = f"""You are a senior data analyst.
            Executive Summary already provided:
            \"\"\"{st.session_state.last_exec_summary}\"\"\"

            Original question: "{asked_q}"
            Data sample ({len(result)} rows, showing {rows}): {json.dumps(sample, default=str)}

            Now provide 3-5 concise, actionable business insights NOT already covered in the summary above.
            Format as a numbered list. Be specific, data-driven, and avoid repeating the summary."""
                        st.session_state.last_insights = call_llm(insight_prompt) or "Could not generate insights."

                if st.session_state.get("last_insights"):
                    st.markdown(
                        f"<div class='exec-box' style='border-left-color:#7c3aed;background:#f5f3ff;'>"
                        f"{html.escape(st.session_state.last_insights)}</div>",
                        unsafe_allow_html=True,
                    )
                    
    #         st.markdown("---")
    #         exc, _ = st.columns([3,7])
    #         if exc.button("📋 Generate Executive Summary", use_container_width=True):
    #             with st.spinner("Generating…"):
    #                 st.session_state.last_exec_summary = generate_executive_summary(result, asked_q)

    #         if st.session_state.last_exec_summary:
    #             st.markdown(
    #                 f"<div class='exec-box'>{html.escape(st.session_state.last_exec_summary)}</div>",
    #                 unsafe_allow_html=True,
    #             )

        elif result is not None and result.empty:
            st.warning("⚠️ Query returned no rows. Try broadening filters or rephrasing.")
            with st.expander("🔍 Generated SQL"):
                st.code(sql, language="sql")

    elif not run_clicked:
        st.markdown(
            "<div style='text-align:center;padding:56px;color:#8b949e;font-size:15px;'>"
            "💬 Type a question above and press <b>Run</b>"
            "</div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:11px;'>"
    "🚀Capgemini AI Data Platform &nbsp;|&nbsp; DuckDB · GPT-4o · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
