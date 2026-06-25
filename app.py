# app.py - AI Data Platform V10

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go
import json, re, os, html
from dotenv import load_dotenv

load_dotenv()

try:
    from langchain_openai import ChatOpenAI
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

st.set_page_config(layout="wide", page_title="AI Data Platform", page_icon="🚀")

st.markdown("""
<style>
.block-container{padding-top:1.1rem;padding-bottom:.8rem;}
.platform-header{
    background:linear-gradient(100deg,#0f2027 0%,#203a43 55%,#2c5364 100%);
    border-radius:12px;padding:16px 26px;margin-bottom:16px;
}
.platform-header h1{margin:0;font-size:22px;font-weight:700;color:#fff;}
.platform-header p{margin:2px 0 0;font-size:11px;color:#90caf9;}
.stat-row{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap;}
.stat-card{
    flex:1;min-width:120px;background:#0d1117;border:1px solid #21262d;
    border-radius:10px;padding:11px 14px;text-align:center;
}
.stat-card .sv{font-size:20px;font-weight:700;color:#4fc3f7;line-height:1.2;}
.stat-card .sl{font-size:10px;color:#8b949e;margin-top:2px;text-transform:uppercase;letter-spacing:.5px;}
.kpi-row{display:flex;gap:18px;margin-bottom:22px;flex-wrap:wrap;}
.kpi-card{
    flex:1;min-width:160px;
    background:#0d1117;
    border:1px solid #21262d;
    border-radius:12px;
    padding:20px 20px 16px 20px;
    text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,0.35);
    transition: border-color .2s;
}
.kpi-card:hover{border-color:#4fc3f7;}
.kpi-card .kv{font-size:24px;font-weight:700;color:#4fc3f7;line-height:1.25;margin-bottom:6px;}
.kpi-card .kl{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px;}
.kpi-card .ks{font-size:11px;color:#22c55e;margin-top:4px;}
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
.exec-box{
    background:#f0f9ff;border-left:3px solid #0369a1;
    border-radius:0 8px 8px 0;padding:14px 18px;
    color:#0c2340;font-size:13.5px;line-height:1.8;margin-top:8px;
}
.score-high{color:#22c55e;font-weight:700;}
.score-med {color:#f59e0b;font-weight:700;}
.score-low {color:#ef4444;font-weight:700;}
.dq-badge-green{
    display:inline-block;background:#052e16;color:#22c55e;
    border:1px solid #166534;border-radius:6px;
    padding:3px 10px;font-size:11px;font-weight:700;margin:2px;
}
.dq-badge-amber{
    display:inline-block;background:#1c1204;color:#f59e0b;
    border:1px solid #92400e;border-radius:6px;
    padding:3px 10px;font-size:11px;font-weight:700;margin:2px;
}
.dq-badge-red{
    display:inline-block;background:#1c0505;color:#ef4444;
    border:1px solid #991b1b;border-radius:6px;
    padding:3px 10px;font-size:11px;font-weight:700;margin:2px;
}
.dq-score-ring{
    font-size:48px;font-weight:800;text-align:center;
    line-height:1.1;padding:10px 0 4px 0;
}
.dq-issue-row{
    background:#0d1117;border:1px solid #21262d;border-radius:8px;
    padding:10px 14px;margin-bottom:8px;font-size:13px;
}
div[data-testid="stButton"]>button{border-radius:6px !important;}
div[data-testid="stMetric"]{background:transparent !important;border:none !important;padding:0 !important;}
div[data-testid="stTextInput"] input{border-radius:8px !important;font-size:14px !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────────
if _LLM_AVAILABLE:
    try:
        llm = ChatOpenAI(
            base_url=st.secrets["LLM_BASE_URL"],
            api_key=st.secrets["LLM_API_KEY"],
            default_headers={
                "x-api-key": st.secrets["LLM_HEADER_KEY"]
            },
            model="openai.gpt-5",
            temperature=0,
            max_tokens=600,
        )
    except Exception:
        llm = None
else:
    llm = None

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
DEFAULTS = {
    "dfs": {}, "join_mode": "auto",
    "manual_joins": {}, "sql_join_text": "",
    "memory": {}, "query_history": [],
    "last_query": "", "last_plan": None,
    "last_result": None, 
    # "last_exec_summary": None,
    "llm_calls": 0, "total_tokens": 0,
    "max_llm_calls": 60, "max_tokens": 30000,
    "query_input": "",
    "pending_suggestion": None,
    # "last_insights": None,
    "auto_join_base": None,
    "last_analysis": None,
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
    if llm is None:
        st.error("LLM not configured.")
        return None
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
# FORMAT RESULT DATES → YYYY-MM
# ─────────────────────────────────────────────────────────────────
def format_result_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if str(df[col].dtype).startswith('period'):
            df[col] = df[col].astype(str).str[:7]
            continue
        if df[col].dtype == object:
            sample = df[col].dropna().head(10)
            ts_count = sum(
                1 for v in sample
                if isinstance(v, str) and re.match(r'\d{4}-\d{2}-\d{2}', str(v)) and len(str(v)) > 7
            )
            col_lower = col.lower()
            if ts_count >= max(1, len(sample) // 2) and any(
                x in col_lower for x in ['month','period','date','time','ym','year_month']
            ):
                df[col] = df[col].astype(str).str[:7]
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if any(x in col.lower() for x in ['month','period','ym','year_month']):
                df[col] = df[col].dt.strftime('%Y-%m')
    return df

# ─────────────────────────────────────────────────────────────────
# SQL GUARDRAILS
# ─────────────────────────────────────────────────────────────────
_BLOCKED = re.compile(
    r'^\s*(drop|delete|truncate|update|insert|alter|create|replace|merge|call|exec)\b',
    re.IGNORECASE | re.MULTILINE,
)

def sql_is_safe(sql: str) -> tuple[bool, str]:
    if _BLOCKED.search(sql):
        keyword = _BLOCKED.search(sql).group(1).upper()
        return False, f"Statement contains blocked keyword: **{keyword}**. Only SELECT queries are allowed."
    if not re.search(r'\bSELECT\b', sql, re.IGNORECASE):
        return False, "Only SELECT queries are permitted."
    return True, ""

# ─────────────────────────────────────────────────────────────────
# SCHEMA BUILDER
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
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
# CORE NLQ ENGINE
# ─────────────────────────────────────────────────────────────────
def nlq_to_sql(question: str, df: pd.DataFrame) -> str | None:
    schema    = build_rich_schema(df)
    name_cols = [c for c in df.columns if any(x in c.lower() for x in
                 ["first","last","fname","lname","name","full"])]
    prompt = f"""You are an expert DuckDB SQL generator. Given a dataset schema and a natural language question, generate the best DuckDB SQL query.

TABLE NAME: df
{schema}

RULES:
1. Always SELECT meaningful labels. If there are separate first_name and last_name columns, concatenate: first_name || ' ' || last_name AS salesperson_name
2. For "best/top/worst" queries: always ORDER BY metric DESC/ASC with LIMIT (default 10 if not specified)
3. For trend queries: use strftime('%Y-%m', date_col) AS month to group by month — always alias as 'month'
4. For "by X and Y" queries: GROUP BY both X and Y columns
5. For count queries: use COUNT(*) or COUNT(DISTINCT col)
6. For comparison queries (vs/compare): use CASE or multiple aggregations
7. Always use meaningful column aliases
8. If question involves a specific value (ford, red, SUV), use WHERE col ILIKE '%value%'
9. Never return more than 500 rows unless explicitly asked
10. For salesperson/person queries: combine first+last name if both exist
11. For date columns, handle NULL safely with IS NOT NULL where needed
12. Multi-column group: if user says "by brand and type", GROUP BY make, car_type
13. Return ONLY the SQL string, no explanation, no markdown fences.
14. For month/period grouping ALWAYS use strftime('%Y-%m', date_col) AS month — never DATE_TRUNC which returns timestamps

NAME COLUMNS DETECTED: {name_cols}

QUESTION: {question}

SQL:"""
    return call_llm(prompt)


def run_sql(sql: str, df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    safe, reason = sql_is_safe(sql)
    if not safe:
        return None, f"🔒 Blocked: {reason}"
    try:
        con = duckdb.connect()
        con.register("df", df)
        result = con.execute(sql.strip()).df()
        con.close()
        result = format_result_dates(result)
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
# AUTO-JOIN
# ─────────────────────────────────────────────────────────────────
def _col_norm_map(df: pd.DataFrame) -> dict:
    return {norm(c): c for c in df.columns}

def _join_score(left_series: pd.Series, right_series: pd.Series) -> float:
    try:
        l_vals = set(left_series.dropna().astype(str).unique())
        r_vals = set(right_series.dropna().astype(str).unique())
        if not l_vals or not r_vals:
            return 0.0
        overlap     = len(l_vals & r_vals) / min(len(l_vals), len(r_vals))
        card_ratio  = min(len(l_vals), len(r_vals)) / max(len(l_vals), len(r_vals))
        dtype_match = 1.0 if left_series.dtype == right_series.dtype else 0.7
        name_bonus  = 1.1 if any(x in norm(left_series.name) for x in ["id","key","code","num"]) else 1.0
        score = overlap * 0.6 + card_ratio * 0.3 + (dtype_match - 1) * 0.1
        return round(min(score * dtype_match * name_bonus * 100, 100), 1)
    except Exception:
        return 0.0

def auto_join(dfs: dict, base_name: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    tables = list(dfs.items())
    if len(tables) == 1:
        return tables[0][1], []
    if base_name and base_name in dfs:
        other_tables = [(n, d) for n, d in tables if n != base_name]
        tables = [(base_name, dfs[base_name])] + other_tables
    base_name_actual, base = tables[0][0], tables[0][1].copy()
    join_log = []
    for r_name, right in tables[1:]:
        l_map = _col_norm_map(base)
        r_map = _col_norm_map(right)
        common_norms = set(l_map.keys()) & set(r_map.keys())
        if not common_norms:
            for lk in l_map:
                for rk in r_map:
                    if lk in rk or rk in lk:
                        common_norms.add(lk)
                        r_map.setdefault(lk, r_map.get(rk))
                        break
        if not common_norms:
            join_log.append({"left_table": base_name_actual, "right_table": r_name,
                              "left_col": "—", "right_col": "—", "score": 0,
                              "note": "No matching columns found"})
            continue
        best_score, best_lc, best_rc = -1, None, None
        for n_key in common_norms:
            lc = l_map.get(n_key)
            rc = r_map.get(n_key)
            if lc and rc and lc in base.columns and rc in right.columns:
                s = _join_score(base[lc], right[rc])
                if s > best_score:
                    best_score, best_lc, best_rc = s, lc, rc
        if best_lc is None or best_score < 5:
            join_log.append({"left_table": base_name_actual, "right_table": r_name,
                              "left_col": "—", "right_col": "—", "score": best_score,
                              "note": "Score too low — skipped"})
            continue
        try:
            merged = pd.merge(base, right, left_on=best_lc, right_on=best_rc,
                              how="left", suffixes=("", f"_{r_name}"))
            merged = merged[[c for c in merged.columns
                              if not (c.endswith(f"_{r_name}") and
                                      c[:-len(f"_{r_name}")] in merged.columns)]]
            base = merged
            join_log.append({"left_table": base_name_actual, "right_table": r_name,
                              "left_col": best_lc, "right_col": best_rc,
                              "score": best_score, "note": "OK"})
        except Exception as e:
            join_log.append({"left_table": base_name_actual, "right_table": r_name,
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
        base = st.session_state.get("auto_join_base") or list(dfs.keys())[0]
        df, _ = auto_join(dfs, base_name=base)
        return df
    elif mode == "manual":
        return manual_join(dfs, st.session_state.manual_joins)
    elif mode == "sql":
        sql = st.session_state.sql_join_text
        return sql_join(dfs, sql) if sql.strip() else list(dfs.values())[0]
    return list(dfs.values())[0]

# ─────────────────────────────────────────────────────────────────
# CHART BUILDER
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
        if not isinstance(x_col, str) or not isinstance(y_col, str):
            st.warning("⚠️ Invalid axis selection.")
            return
        all_cols = list(result.columns)
        if x_col not in all_cols:
            st.warning(f"⚠️ X-axis column '{x_col}' not found in result.")
            return
        if y_col not in all_cols:
            st.warning(f"⚠️ Y-axis column '{y_col}' not found in result.")
            return
        if x_col == y_col:
            st.warning("⚠️ X and Y axes must be different columns.")
            return
        df_plot = result[[x_col, y_col]].copy()
        df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors="coerce")
        if df_plot[y_col].isna().all():
            st.warning(f"⚠️ Column '{y_col}' has no numeric values to plot.")
            return
        colors = px.colors.qualitative.Plotly
        layout = dict(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=30,b=10), font=dict(size=12)
        )
        if chart_type == "Bar":
            fig = px.bar(df_plot, x=x_col, y=y_col, text_auto=True,
                         color=x_col if df_plot[x_col].nunique() <= 20 else None,
                         color_discrete_sequence=colors)
        elif chart_type == "Line":
            fig = px.line(df_plot, x=x_col, y=y_col, markers=True)
        elif chart_type == "Pie":
            fig = px.pie(df_plot, names=x_col, values=y_col, hole=0.35,
                         color_discrete_sequence=colors)
        elif chart_type == "Scatter":
            fig = px.scatter(df_plot, x=x_col, y=y_col)
        elif chart_type == "Area":
            fig = px.area(df_plot, x=x_col, y=y_col)
        else:
            fig = px.bar(df_plot, x=x_col, y=y_col)
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Chart could not be rendered: {e}")
        st.info("Try selecting different X / Y columns or switching chart type.")

# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# COMBINED ANALYSIS ENGINE — Summary + Insights + Recommendation
# Single LLM call replaces two separate calls
# ─────────────────────────────────────────────────────────────────
def generate_analysis(result: pd.DataFrame, question: str) -> dict:
    """
    Single LLM call that returns summary, facts and recommendation.
    Returns dict with keys: 'summary', 'facts', 'recommendation'
    Falls back gracefully if parsing fails.
    """
    num_cols = result.select_dtypes(include="number").columns.tolist()
    str_cols = result.select_dtypes(exclude="number").columns.tolist()

    # ── Smart sample — send ALL cols for small results ──────────
    if len(result) <= 15 and len(result.columns) <= 12:
        sample_df = result.head(10)          # full columns, small result
    else:
        keep_cols = str_cols[:1] + num_cols[:3]
        sample_df = result[keep_cols].head(8) if keep_cols else result.head(8)
    sample_str = sample_df.to_csv(index=False)

    # ── Compact stats — top 3 numeric cols only ─────────────────
    stats_lines = []
    for c in num_cols[:3]:
        col_s = pd.to_numeric(result[c], errors="coerce").dropna()
        if len(col_s):
            stats_lines.append(
                f"{c}: total={round(float(col_s.sum()),2)}, "
                f"avg={round(float(col_s.mean()),2)}, "
                f"max={round(float(col_s.max()),2)}, "
                f"min={round(float(col_s.min()),2)}"
            )

    # ── Compact pre-analysis ─────────────────────────────────────
    context_lines = []
    if num_cols and str_cols:
        col = pd.to_numeric(result[num_cols[0]], errors="coerce").fillna(0)
        total = col.sum()
        if total > 0:
            top1_pct = round(col.nlargest(1).sum() / total * 100, 1)
            top3_pct = round(col.nlargest(3).sum() / total * 100, 1)
            context_lines.append(
                f"Concentration: top-1={top1_pct}%, top-3={top3_pct}% of {num_cols[0]}"
            )
    if num_cols:
        col = pd.to_numeric(result[num_cols[0]], errors="coerce").dropna()
        if len(col) >= 4:
            half   = len(col) // 2
            h1_avg = round(float(col.iloc[:half].mean()), 2)
            h2_avg = round(float(col.iloc[half:].mean()), 2)
            chg    = round((h2_avg - h1_avg) / h1_avg * 100, 1) if h1_avg else 0
            direction = "accelerating" if chg > 5 else ("decelerating" if chg < -5 else "stable")
            context_lines.append(f"Trend: {direction} ({chg:+.1f}%), H1={h1_avg}, H2={h2_avg}")

    # ── Single compact prompt ────────────────────────────────────
    prompt = (
        f'QUESTION: "{question}"\n'
        f"TOTAL ROWS: {len(result)} | COLUMNS: {list(result.columns)}\n\n"
        f"DATA (CSV):\n{sample_str}\n"
        f"STATS: {'; '.join(stats_lines) or 'N/A'}\n"
        f"CONTEXT: {' | '.join(context_lines) or 'N/A'}\n\n"
        f"You are a senior business analyst. Analyse the data above and respond in EXACTLY this format:\n\n"
        f"SUMMARY: [2-3 sentences directly answering the question with exact numbers from data]\n\n"
        f"FACTS:\n"
        f"• [Fact 1 with specific number]\n"
        f"• [Fact 2 with specific number]\n"
        f"• [Fact 3 with specific number]\n\n"
        f"RECOMMENDATION: [1-2 sentences of specific actionable advice based on data]\n\n"
        f"RULES:\n"
        f"- Use ONLY numbers from the data provided\n"
        f"- SUMMARY must directly answer the question asked\n"
        f"- Each FACT must cite a real number\n"
        f"- No vague words like 'significant' or 'many'\n"
        f"- No repetition between sections\n"
        f"- Total response must be under 180 words"
    )

    raw = call_llm(prompt)
    if not raw:
        return {
            "summary": "Could not generate summary.",
            "facts": [],
            "recommendation": ""
        }

    # ── Parse structured response ────────────────────────────────
    return _parse_analysis_response(raw)


def _parse_analysis_response(raw: str) -> dict:
    """
    Parses the structured LLM response into summary, facts, recommendation.
    Robust fallback if format is not followed perfectly.
    """
    result = {"summary": "", "facts": [], "recommendation": ""}

    try:
        # Extract SUMMARY
        summary_match = re.search(
            r'SUMMARY:\s*(.+?)(?=FACTS:|RECOMMENDATION:|$)',
            raw, re.DOTALL | re.IGNORECASE
        )
        if summary_match:
            result["summary"] = summary_match.group(1).strip()

        # Extract FACTS — bullet points
        facts_match = re.search(
            r'FACTS:\s*(.+?)(?=RECOMMENDATION:|$)',
            raw, re.DOTALL | re.IGNORECASE
        )
        if facts_match:
            facts_block = facts_match.group(1).strip()
            bullets = re.findall(r'[•\-\*]\s*(.+)', facts_block)
            if bullets:
                result["facts"] = [b.strip() for b in bullets[:3]]
            else:
                # Fallback — split by newline if no bullets found
                lines = [l.strip() for l in facts_block.split('\n') if l.strip()]
                result["facts"] = lines[:3]

        # Extract RECOMMENDATION
        rec_match = re.search(
            r'RECOMMENDATION:\s*(.+?)$',
            raw, re.DOTALL | re.IGNORECASE
        )
        if rec_match:
            result["recommendation"] = rec_match.group(1).strip()

        # Fallback — if parsing totally failed, put everything in summary
        if not result["summary"] and not result["facts"]:
            result["summary"] = raw.strip()

    except Exception:
        result["summary"] = raw.strip()

    return result

# ─────────────────────────────────────────────────────────────────
# KPI ENGINE — column candidates
# ─────────────────────────────────────────────────────────────────
_REV_CANDIDATES    = ["revenue","sales_amount","revenue_value","turnover","sale_price",
                      "price","amount","total","value"]
_VOL_CANDIDATES    = ["units_sold","quantity","vehicle_sales","sales_volume","units","count","qty"]
_DATE_CANDIDATES   = ["date","month","quarter","year","sale_date","order_date","transaction_date"]
_SEG_CANDIDATES    = ["segment","category","type","car_type","vehicle_type","class"]
_MODEL_CANDIDATES  = ["model","vehicle_model","product_name","make","brand","car_model"]
_REGION_CANDIDATES = ["region","territory","state","country","area","zone","city"]
_SALES_CANDIDATES  = ["salesperson","sales_person","sales_rep","agent",
                      "seller","employee","staff","rep"]
_MKTSH_CANDIDATES  = ["market_share","marketshare","share"]
_FIRST_NAME_CANDIDATES = ["first_name","firstname","fname","first"]
_LAST_NAME_CANDIDATES  = ["last_name","lastname","lname","last","surname"]

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    col_norms = {norm(c): c for c in df.columns}
    for cand in candidates:
        if norm(cand) in col_norms:
            return col_norms[norm(cand)]
    for cand in candidates:
        for c in df.columns:
            if cand in norm(c):
                return c
    return None

def _safe_sum(series: pd.Series) -> float:
    try:
        return float(pd.to_numeric(series, errors="coerce").dropna().sum())
    except Exception:
        return 0.0

def _fmt_currency(v: float) -> str:
    if v >= 1_000_000_000: return f"₹{v/1_000_000_000:.2f}B"
    if v >= 1_000_000:     return f"₹{v/1_000_000:.2f}M"
    if v >= 1_000:         return f"₹{v/1_000:.1f}K"
    return f"₹{v:,.0f}"

def _fmt_number(v: float) -> str:
    if v >= 1_000_000: return f"{v/1_000_000:.2f}M"
    if v >= 1_000:     return f"{v/1_000:.1f}K"
    return f"{v:,.0f}"

# ─────────────────────────────────────────────────────────────────
# DATA QUALITY ENGINE  (zero LLM — pure Pandas)
# ─────────────────────────────────────────────────────────────────


def _is_metric_col(col_name: str, series: pd.Series) -> bool:
    """
    Returns True only if column is a genuine business metric.
    Filters out ID, key, index and code columns.
    """
    col_lower = col_name.lower()

    # ── Explicit ID/key patterns to exclude ──────────────────
    exclude_patterns = [
        "id", "_id", "id_", "key", "_key",
        "code", "_code", "code_",
        "index", "idx", "_idx",
        "num", "_num", "number",
        "ref", "_ref", "seq",
        "row", "record", "pk", "fk"
    ]
    for pattern in exclude_patterns:
        if col_lower == pattern:                    return False
        if col_lower.endswith(f"_{pattern}"):       return False
        if col_lower.startswith(f"{pattern}_"):     return False
        if col_lower.endswith(pattern):             return False

    # ── Statistical check — IDs have near-unique values ──────
    if series.nunique() / max(len(series), 1) > 0.95:
        return False   # 95%+ unique → almost certainly an ID

    # ── IDs are usually sequential integers — check range ────
    if pd.api.types.is_integer_dtype(series):
        col_min = series.min()
        col_max = series.max()
        col_mean = series.mean()
        # If min value is 1 or 0 and values are evenly spread → ID
        if col_min >= 0 and col_max == series.nunique():
            return False

    # ── Explicit metric keywords to include ──────────────────
    metric_patterns = [
        "sale", "sales", "revenue", "amount", "total",
        "price", "cost", "value", "profit", "margin",
        "unit", "units", "qty", "quantity", "count",
        "volume", "rate", "score", "age", "salary",
        "income", "expense", "tax", "discount", "share",
        "growth", "change", "pct", "percent", "ratio"
    ]
    for pattern in metric_patterns:
        if pattern in col_lower:
            return True

    # ── Default — if none of the above matched ────────────────
    # Small range of values = likely a metric not an ID
    if series.nunique() < 1000:
        return True

    return False

@st.cache_data(show_spinner=False)
def compute_data_quality(df: pd.DataFrame) -> dict:
    report = {}
    total_rows  = len(df)
    total_cells = total_rows * len(df.columns)

    null_counts = df.isnull().sum()
    null_pct    = (null_counts / total_rows * 100).round(2)
    report["null_summary"] = {
        col: {"count": int(null_counts[col]), "pct": float(null_pct[col])}
        for col in df.columns if null_counts[col] > 0
    }
    report["total_null_cells"] = int(null_counts.sum())
    report["total_null_pct"]   = round(float(null_counts.sum()) / total_cells * 100, 2)

    dup_mask = df.duplicated()
    report["duplicate_count"] = int(dup_mask.sum())
    report["duplicate_pct"]   = round(float(dup_mask.sum()) / total_rows * 100, 2)
    if dup_mask.sum() > 0:
        report["duplicate_sample"] = df[dup_mask].head(5).to_dict(orient="records")

    outlier_report = {}
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        if not _is_metric_col(col, df[col]):
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 10: continue
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0: continue
        lower    = Q1 - 3.0 * IQR
        upper    = Q3 + 3.0 * IQR
        out_mask = (s < lower) | (s > upper)
        out_count = int(out_mask.sum())
        if out_count > 0:
            outlier_report[col] = {
                "count":       out_count,
                "pct":         round(out_count / len(s) * 100, 2),
                "lower_fence": round(float(lower), 2),
                "upper_fence": round(float(upper), 2),
                "min_outlier": round(float(s[out_mask].min()), 2),
                "max_outlier": round(float(s[out_mask].max()), 2),
                "sample":      [round(float(x), 2) for x in s[out_mask].head(5).tolist()],
            }
    report["outliers"] = outlier_report

    date_col     = _find_col(df, _DATE_CANDIDATES)
    spike_report = {}
    if date_col:
        try:
            tmp = df.copy()
            tmp["__dt__"] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=["__dt__"])
            tmp["__period__"] = tmp["__dt__"].dt.to_period("M")
            for col in [c for c in num_cols if _is_metric_col(c, df[c])][:6]:
                try:
                    tmp[col]     = pd.to_numeric(tmp[col], errors="coerce")
                    monthly      = tmp.groupby("__period__")[col].sum().sort_index()
                    if len(monthly) < 4: continue
                    rolling_mean = monthly.rolling(3, min_periods=2).mean().shift(1)
                    rolling_std  = monthly.rolling(3, min_periods=2).std().shift(1)
                    z_scores     = ((monthly - rolling_mean) / rolling_std.replace(0, np.nan)).abs()
                    spikes       = z_scores[z_scores > 2.5].dropna()
                    if not spikes.empty:
                        spike_report[col] = [
                            {
                                "period":    str(p),
                                "value":     round(float(monthly[p]), 2),
                                "z_score":   round(float(z_scores[p]), 2),
                                "direction": "▲ Spike UP" if monthly[p] > rolling_mean[p] else "▼ Spike DOWN",
                            }
                            for p in spikes.index
                        ]
                except Exception:
                    continue
        except Exception:
            pass
    report["spikes"]   = spike_report
    report["date_col"] = date_col

    type_issues = []
    for col in df.select_dtypes(include="object").columns:
        sample_vals = df[col].dropna().head(500)
        if len(sample_vals) == 0: continue
        numeric_convertible = pd.to_numeric(sample_vals, errors="coerce").notna().sum()
        pct_numeric         = numeric_convertible / len(sample_vals)
        if pct_numeric >= 0.85:
            type_issues.append({
                "column": col, "issue": "Stored as text but looks numeric",
                "pct_numeric": round(pct_numeric * 100, 1), "sample": sample_vals.head(3).tolist(),
            })
        date_convertible = pd.to_datetime(sample_vals, errors="coerce").notna().sum()
        pct_date         = date_convertible / len(sample_vals)
        if pct_date >= 0.85 and pct_numeric < 0.5:
            type_issues.append({
                "column": col, "issue": "Stored as text but looks like a date",
                "pct_date": round(pct_date * 100, 1), "sample": sample_vals.head(3).tolist(),
            })
    report["type_issues"] = type_issues

    cardinality_flags = []
    for col in df.select_dtypes(include="object").columns:
        uniq  = df[col].nunique()
        total = df[col].notna().sum()
        if total == 0: continue
        uniq_ratio = uniq / total
        if uniq_ratio > 0.95 and uniq > 100:
            cardinality_flags.append({"column": col, "issue": "Very high cardinality — possible free-text or ID column", "unique": uniq, "ratio": round(uniq_ratio * 100, 1)})
        elif uniq == 1:
            cardinality_flags.append({"column": col, "issue": "Only 1 unique value — constant column, no analytical value", "unique": 1, "ratio": round(uniq_ratio * 100, 1)})
        elif uniq == total and total > 50:
            cardinality_flags.append({"column": col, "issue": "All values unique — likely an ID/key column", "unique": uniq, "ratio": 100.0})
    report["cardinality_flags"] = cardinality_flags

    date_gaps = []
    if date_col:
        try:
            dt_series = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if len(dt_series) >= 3:
                periods   = dt_series.dt.to_period("M").unique()
                all_range = pd.period_range(periods.min(), periods.max(), freq="M")
                missing   = all_range.difference(periods)
                if len(missing) > 0:
                    date_gaps = [str(p) for p in missing[:12]]
        except Exception:
            pass
    report["date_gaps"] = date_gaps

    score = 100.0
    score -= min(report["total_null_pct"] * 1.5, 25)
    score -= min(report["duplicate_pct"] * 2.0, 20)
    score -= min(len(outlier_report) * 3, 15)
    score -= min(len(type_issues) * 4, 16)
    score -= min(len(cardinality_flags) * 2, 10)
    score -= min(len(date_gaps) * 1, 10)
    
    report["health_score"] = max(round(score, 1), 0.0)
    report["total_rows"]   = total_rows
    report["total_cols"]   = len(df.columns)
    return report


def render_data_quality(df: pd.DataFrame, table_name: str):
    with st.spinner("Running data quality checks…"):
        dq = compute_data_quality(df)

    st.markdown("### 🔬 Data Quality Intelligence")
    st.caption("Automated data health checks — zero AI involvement. Pure statistical analysis.")

    score = dq["health_score"]
    if score >= 80:
        score_color = "#22c55e"; score_label = "Excellent";      badge_cls = "dq-badge-green"
    elif score >= 60:
        score_color = "#f59e0b"; score_label = "Needs Attention"; badge_cls = "dq-badge-amber"
    else:
        score_color = "#ef4444"; score_label = "Poor Quality";    badge_cls = "dq-badge-red"

    g_col1, g_col2, g_col3 = st.columns([1, 2, 3])
    with g_col1:
        st.markdown(
            f"<div style='text-align:center;padding:16px 0;'>"
            f"<div style='font-size:56px;font-weight:800;color:{score_color};line-height:1;'>{score}%</div>"
            f"<div style='font-size:12px;color:#8b949e;margin-top:6px;'>Data Health Score</div>"
            f"<div class='{badge_cls}' style='margin-top:8px;display:inline-block;'>{score_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with g_col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            gauge={
                "axis":  {"range": [0, 100], "tickwidth": 1},
                "bar":   {"color": score_color},
                "steps": [{"range": [0,60], "color": "#1c0505"}, {"range": [60,80], "color": "#1c1204"}, {"range": [80,100], "color": "#052e16"}],
                "threshold": {"line": {"color": score_color, "width": 3}, "thickness": 0.8, "value": score},
            },
            number={"suffix": "%", "font": {"size": 28, "color": score_color}},
        ))
        fig_gauge.update_layout(height=180, margin=dict(l=10,r=10,t=20,b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e"))
        st.plotly_chart(fig_gauge, use_container_width=True)
    with g_col3:
        st.markdown(f"""
        <div class="stat-row" style="margin-top:12px;">
          <div class="stat-card"><div class="sv">{dq['total_rows']:,}</div><div class="sl">Total Rows</div></div>
          <div class="stat-card"><div class="sv">{dq['total_cols']}</div><div class="sl">Columns</div></div>
          <div class="stat-card"><div class="sv" style="color:{'#ef4444' if dq['total_null_pct']>10 else '#4fc3f7'};">{dq['total_null_pct']}%</div><div class="sl">Null Rate</div></div>
          <div class="stat-card"><div class="sv" style="color:{'#ef4444' if dq['duplicate_count']>0 else '#22c55e'};">{dq['duplicate_count']:,}</div><div class="sl">Duplicates</div></div>
          <div class="stat-card"><div class="sv" style="color:{'#f59e0b' if len(dq['outliers'])>0 else '#22c55e'};">{len(dq['outliers'])}</div><div class="sl">Outlier Cols</div></div>
        
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if dq["null_summary"]:
        st.markdown("#### 🕳️ Null Value Analysis")
        null_df = pd.DataFrame([
            {"Column": col, "Null Count": v["count"], "Null %": v["pct"],
             "Status": "🔴 Critical" if v["pct"] > 30 else ("🟡 Warning" if v["pct"] > 10 else "🟢 Minor")}
            for col, v in dq["null_summary"].items()
        ]).sort_values("Null %", ascending=False)
        nc1, nc2 = st.columns([3, 2])
        with nc1:
            fig_null = px.bar(null_df, x="Null %", y="Column", orientation="h",
                              color="Null %", color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
                              range_color=[0,100], text="Null %", title="Missing Data % by Column")
            fig_null.update_layout(height=max(250, len(null_df)*32), margin=dict(l=0,r=0,t=30,b=0),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
            fig_null.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            st.plotly_chart(fig_null, use_container_width=True)
        with nc2:
            st.dataframe(null_df, use_container_width=True, hide_index=True)
        st.markdown("---")
    else:
        st.success("✅ No null values detected — data is complete!")
        st.markdown("---")

    st.markdown("#### 👥 Duplicate Row Detection")
    if dq["duplicate_count"] > 0:
        st.warning(f"⚠️ **{dq['duplicate_count']:,} duplicate rows** found ({dq['duplicate_pct']}% of data). These may skew aggregations and KPIs.")
        if "duplicate_sample" in dq:
            with st.expander("👁️ Preview Duplicate Rows"):
                st.dataframe(pd.DataFrame(dq["duplicate_sample"]), use_container_width=True, hide_index=True)
    else:
        st.success("✅ No duplicate rows found — data is unique!")
    st.markdown("---")

    st.markdown("#### 📊 Statistical Outlier Detection")
    st.caption("Using IQR × 3.0 method — extreme values only flagged, not standard variation.")
    if dq["outliers"]:
        out_rows = [{"Column": col, "Outlier Count": info["count"], "Outlier %": info["pct"],
                     "Lower Fence": info["lower_fence"], "Upper Fence": info["upper_fence"],
                     "Min Outlier": info["min_outlier"], "Max Outlier": info["max_outlier"],
                     "Sample Values": str(info["sample"])} for col, info in dq["outliers"].items()]
        out_df = pd.DataFrame(out_rows).sort_values("Outlier %", ascending=False)
        oc1, oc2 = st.columns([2, 3])
        with oc1:
            fig_out = px.bar(out_df, x="Column", y="Outlier %", color="Outlier %",
                             color_continuous_scale=["#f59e0b","#ef4444"], text="Outlier Count", title="Outlier Count by Column")
            fig_out.update_layout(margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
            st.plotly_chart(fig_out, use_container_width=True)
        with oc2:
            st.dataframe(out_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No significant outliers detected across numeric columns!")
    st.markdown("---")

    # st.markdown("#### ⚡ Time-Series Spike Detection")
    # st.caption("Flags months where values deviate > 2.5 standard deviations from rolling average.")
    # if dq["spikes"]:
    #     for col, spike_list in dq["spikes"].items():
    #         with st.expander(f"📌 **{col}** — {len(spike_list)} spike(s) detected", expanded=True):
    #             spike_df = pd.DataFrame(spike_list)
    #             spike_df.columns = ["Period","Value","Z-Score","Direction"]
    #             sc1, sc2 = st.columns([3, 2])
    #             with sc1:
    #                 fig_spike = px.bar(spike_df, x="Period", y="Value", color="Z-Score",
    #                                    color_continuous_scale=["#f59e0b","#ef4444"], text="Direction", title=f"Spikes in {col}")
    #                 fig_spike.update_layout(margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
    #                 st.plotly_chart(fig_spike, use_container_width=True)
    #             with sc2:
    #                 st.dataframe(spike_df, use_container_width=True, hide_index=True)
    # else:
    #     st.success("✅ No unusual spikes detected in time-series columns!")
    # st.markdown("---")

    st.markdown("#### 🔧 Data Type Issue Detection")
    st.caption("Columns stored in wrong format — affects calculations and joins.")
    if dq["type_issues"]:
        for issue in dq["type_issues"]:
            col_name = issue["column"]; issue_txt = issue["issue"]
            if "numeric" in issue_txt:
                pct_info = f"{issue.get('pct_numeric','')}% of values are numeric"; badge = "dq-badge-amber"; icon = "🔢"
            else:
                pct_info = f"{issue.get('pct_date','')}% of values look like dates"; badge = "dq-badge-amber"; icon = "📅"
            st.markdown(f"<div class='dq-issue-row'>{icon} <b>{col_name}</b> — <span class='{badge}'>{issue_txt}</span> &nbsp; <span style='color:#8b949e;font-size:12px;'>{pct_info} &nbsp;|&nbsp; Sample: {issue.get('sample', [])}</span></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:12px;color:#8b949e;margin-top:4px;'>💡 <b>Recommendation:</b> Convert these columns to their correct data types before analysis to ensure accurate aggregations and joins.</div>", unsafe_allow_html=True)
    else:
        st.success("✅ All columns appear to be stored in the correct data type!")
    st.markdown("---")

    st.markdown("#### 🏷️ Column Cardinality Analysis")
    st.caption("Detects columns with suspiciously high or low unique value counts.")
    if dq["cardinality_flags"]:
        card_df = pd.DataFrame([{"Column": f["column"], "Issue": f["issue"], "Unique Values": f["unique"], "Unique %": f["ratio"]} for f in dq["cardinality_flags"]])
        cc1, cc2 = st.columns([2, 3])
        with cc1:
            fig_card = px.bar(card_df, x="Column", y="Unique %", color="Unique %",
                              color_continuous_scale=["#4fc3f7","#7c3aed"], text="Unique Values", title="Unique Value % by Column")
            fig_card.update_layout(margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
            st.plotly_chart(fig_card, use_container_width=True)
        with cc2:
            st.dataframe(card_df, use_container_width=True, hide_index=True)
    else:
        st.success("✅ All categorical columns have healthy cardinality!")
    st.markdown("---")

    st.markdown("#### 📅 Time-Series Continuity Check")
    st.caption("Checks for missing months in your date column — gaps can distort trend analysis.")
    if dq.get("date_col"):
        if dq["date_gaps"]:
            st.warning(f"⚠️ **{len(dq['date_gaps'])} missing month(s)** detected in column `{dq['date_col']}`. This may affect trend analysis and forecasting accuracy.")
            gap_cols = st.columns(min(len(dq["date_gaps"]), 6))
            for i, gap in enumerate(dq["date_gaps"][:6]):
                gap_cols[i].markdown(f"<div class='dq-badge-red' style='text-align:center;'>📭 {gap}</div>", unsafe_allow_html=True)
            if len(dq["date_gaps"]) > 6:
                st.caption(f"… and {len(dq['date_gaps']) - 6} more missing periods.")
            st.markdown("<div style='font-size:12px;color:#8b949e;margin-top:8px;'>💡 <b>Recommendation:</b> Fill missing periods with zero values or interpolated estimates before running time-series analysis.</div>", unsafe_allow_html=True)
        else:
            st.success(f"✅ No date gaps found in `{dq['date_col']}` — time series is continuous!")
    else:
        st.info("ℹ️ No date column detected — skipping time-series continuity check.")
    st.markdown("---")

    st.markdown("#### 📋 Complete Issues Summary")
    all_issues = []
    for col, v in dq["null_summary"].items():
        severity = "🔴 Critical" if v["pct"] > 30 else ("🟡 Warning" if v["pct"] > 10 else "🟢 Minor")
        all_issues.append({"Column": col, "Issue": "Missing / Null Values", "Detail": f"{v['count']:,} nulls ({v['pct']}%)", "Severity": severity})
    if dq["duplicate_count"] > 0:
        all_issues.append({"Column": "— (row level)", "Issue": "Duplicate Rows", "Detail": f"{dq['duplicate_count']:,} rows ({dq['duplicate_pct']}%)", "Severity": "🔴 Critical" if dq["duplicate_pct"] > 10 else "🟡 Warning"})
    for col, info in dq["outliers"].items():
        all_issues.append({"Column": col, "Issue": "Statistical Outliers", "Detail": f"{info['count']} values outside [{info['lower_fence']}, {info['upper_fence']}]", "Severity": "🟡 Warning" if info["pct"] < 5 else "🔴 Critical"})
    # for col, spike_list in dq["spikes"].items():
    #     for sp in spike_list:
    #         all_issues.append({"Column": col, "Issue": "Time-Series Spike", "Detail": f"{sp['period']} — value {sp['value']:,} (Z={sp['z_score']}) {sp['direction']}", "Severity": "🟡 Warning"})
    for issue in dq["type_issues"]:
        all_issues.append({"Column": issue["column"], "Issue": "Data Type Mismatch", "Detail": issue["issue"], "Severity": "🟡 Warning"})
    for flag in dq["cardinality_flags"]:
        all_issues.append({"Column": flag["column"], "Issue": "Cardinality Anomaly", "Detail": flag["issue"], "Severity": "🟢 Minor"})
    for gap in dq["date_gaps"]:
        all_issues.append({"Column": dq.get("date_col","date"), "Issue": "Date Gap", "Detail": f"Missing period: {gap}", "Severity": "🟡 Warning"})

    if all_issues:
        issues_df = pd.DataFrame(all_issues)
        sev_order = {"🔴 Critical": 0, "🟡 Warning": 1, "🟢 Minor": 2}
        issues_df["_sort"] = issues_df["Severity"].map(sev_order)
        issues_df = issues_df.sort_values("_sort").drop(columns=["_sort"])
        st.dataframe(issues_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Quality Report (CSV)", data=issues_df.to_csv(index=False).encode(), file_name=f"data_quality_{table_name}.csv", mime="text/csv")
    else:
        st.success("🎉 No issues found — this dataset is in excellent shape!")

# ─────────────────────────────────────────────────────────────────
# KPI ENGINE
# ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)  
def compute_kpis(df: pd.DataFrame) -> dict:
    kpis = {}
    rev_col    = _find_col(df, _REV_CANDIDATES)
    vol_col    = _find_col(df, _VOL_CANDIDATES)
    date_col   = _find_col(df, _DATE_CANDIDATES)
    seg_col    = _find_col(df, _SEG_CANDIDATES)
    model_col  = _find_col(df, _MODEL_CANDIDATES)
    region_col = _find_col(df, _REGION_CANDIDATES)
    sales_col  = _find_col(df, _SALES_CANDIDATES)
    mktsh_col  = _find_col(df, _MKTSH_CANDIDATES)
    first_name_col = _find_col(df, _FIRST_NAME_CANDIDATES)
    last_name_col  = _find_col(df, _LAST_NAME_CANDIDATES)

    if rev_col:
        kpis["revenue"] = _safe_sum(df[rev_col]); kpis["revenue_col"] = rev_col
    if vol_col:
        kpis["units_sold"] = _safe_sum(df[vol_col]); kpis["vol_col"] = vol_col

    if date_col and (sales_col or first_name_col) and (rev_col or vol_col):
        try:
            metric_col = rev_col or vol_col
            tmp = df.copy()
            tmp["__dt__"] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
            tmp = tmp.dropna(subset=["__dt__", metric_col])
            latest_month = tmp["__dt__"].dt.to_period("M").max()
            month_df = tmp[tmp["__dt__"].dt.to_period("M") == latest_month]
            if not month_df.empty:
                if first_name_col and last_name_col and first_name_col in month_df.columns and last_name_col in month_df.columns:
                    month_df = month_df.copy()
                    month_df["__full_name__"] = (month_df[first_name_col].fillna("").astype(str).str.strip() + " " + month_df[last_name_col].fillna("").astype(str).str.strip()).str.strip()
                    sp_col = "__full_name__"
                elif first_name_col and first_name_col in month_df.columns:
                    sp_col = first_name_col
                else:
                    sp_col = sales_col
                best_sp = month_df.groupby(sp_col)[metric_col].sum().idxmax()
                kpis["best_salesperson"] = str(best_sp)
                kpis["best_salesperson_month"] = str(latest_month)
        except Exception:
            pass

    if date_col and (rev_col or vol_col):
        try:
            tmp = df.copy()
            tmp["__year__"] = pd.to_datetime(tmp[date_col], errors="coerce").dt.year
            metric_col = rev_col or vol_col
            yearly = (tmp.groupby("__year__")[metric_col].apply(lambda s: pd.to_numeric(s, errors="coerce").sum()).sort_index())
            if len(yearly) >= 2:
                cy = float(yearly.iloc[-1]); py = float(yearly.iloc[-2])
                kpis["yoy_growth"]    = ((cy - py) / py * 100) if py else None
                kpis["yoy_curr_year"] = int(yearly.index[-1])
        except Exception:
            pass

    if mktsh_col:
        try:
            kpis["market_share"] = float(pd.to_numeric(df[mktsh_col], errors="coerce").dropna().mean())
        except Exception:
            pass

    if model_col and vol_col:
        try:
            tmp = df.copy(); tmp[vol_col] = pd.to_numeric(tmp[vol_col], errors="coerce")
            kpis["top_model"] = str(tmp.groupby(model_col)[vol_col].sum().idxmax())
        except Exception:
            pass
    elif model_col and rev_col:
        try:
            tmp = df.copy(); tmp[rev_col] = pd.to_numeric(tmp[rev_col], errors="coerce")
            kpis["top_model"] = str(tmp.groupby(model_col)[rev_col].sum().idxmax())
        except Exception:
            pass

    if region_col:
        try: kpis["active_regions"] = int(df[region_col].nunique())
        except Exception: pass

    if seg_col and vol_col:
        try:
            tmp = df.copy(); tmp[vol_col] = pd.to_numeric(tmp[vol_col], errors="coerce")
            seg_totals = tmp.groupby(seg_col)[vol_col].sum().sort_values(ascending=False)
            grand = seg_totals.sum()
            if grand > 0:
                kpis["segment_mix"] = {str(k): {"units": float(v), "pct": round(float(v)/grand*100, 1)} for k, v in seg_totals.items()}
        except Exception:
            pass

    if model_col and rev_col:
        try:
            tmp = df.copy(); tmp[rev_col] = pd.to_numeric(tmp[rev_col], errors="coerce")
            m5 = tmp.groupby(model_col)[rev_col].sum().nlargest(5)
            total_rev = kpis.get("revenue", m5.sum()) or m5.sum()
            kpis["top5_models"] = [{"model": str(k), "revenue": float(v), "pct": round(float(v)/total_rev*100,1) if total_rev else 0} for k, v in m5.items()]
        except Exception:
            pass

    if seg_col and date_col and (vol_col or rev_col):
        try:
            metric_col = vol_col or rev_col
            tmp = df.copy()
            tmp["__year__"] = pd.to_datetime(tmp[date_col], errors="coerce").dt.year
            tmp[metric_col] = pd.to_numeric(tmp[metric_col], errors="coerce")
            years = sorted(tmp["__year__"].dropna().unique())
            if len(years) >= 2:
                cy, py = years[-1], years[-2]
                curr = tmp[tmp["__year__"]==cy].groupby(seg_col)[metric_col].sum()
                prev = tmp[tmp["__year__"]==py].groupby(seg_col)[metric_col].sum()
                growth = ((curr - prev) / prev.replace(0, np.nan) * 100).dropna()
                if not growth.empty:
                    kpis["fastest_seg"] = str(growth.idxmax()); kpis["fastest_seg_pct"] = round(float(growth.max()), 1)
        except Exception:
            pass

    kpis["_date_col"] = date_col; kpis["_rev_col"] = rev_col
    kpis["_vol_col"]  = vol_col;  kpis["_model_col"] = model_col; kpis["_region_col"] = region_col
    return kpis


def render_kpi_tab(df: pd.DataFrame):
    if df is None or df.empty:
        st.warning("No data available for KPI analysis.")
        return

    date_col_detect = _find_col(df, _DATE_CANDIDATES)
    filtered_df     = df.copy()

    if date_col_detect:
        try:
            filtered_df[date_col_detect] = pd.to_datetime(filtered_df[date_col_detect], errors="coerce")
            all_years = sorted(filtered_df[date_col_detect].dt.year.dropna().unique().astype(int).tolist())
            month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
            all_months_in_data = sorted(filtered_df[date_col_detect].dt.month.dropna().unique().astype(int).tolist())
            st.markdown("#### 🗓️ Filter Charts by Period")
            fcol1, fcol2, _ = st.columns([2, 2, 6])
            sel_year  = fcol1.selectbox("Year",  options=["All"] + [str(y) for y in all_years],  key="kpi_year_filter")
            sel_month = fcol2.selectbox("Month", options=["All"] + [month_names[m] for m in all_months_in_data], key="kpi_month_filter")
            if sel_year != "All":
                filtered_df = filtered_df[filtered_df[date_col_detect].dt.year == int(sel_year)]
            if sel_month != "All":
                month_num = {v: k for k, v in month_names.items()}[sel_month]
                filtered_df = filtered_df[filtered_df[date_col_detect].dt.month == month_num]
            if sel_year != "All" or sel_month != "All":
                st.caption(f"📌 Showing: {'Year ' + sel_year if sel_year != 'All' else 'All Years'} | {'Month: ' + sel_month if sel_month != 'All' else 'All Months'} — {filtered_df.shape[0]:,} rows")
            st.markdown("---")
        except Exception:
            filtered_df = df.copy()

    with st.spinner("Computing KPIs…"):
        kpis = compute_kpis(filtered_df)

    if not kpis:
        st.info("Could not detect standard KPI columns.")
        return

    st.subheader("📊 Executive KPI Summary")
    st.caption("All metrics computed directly from your data — no AI involved.")

    cards = []
    if "revenue"         in kpis: cards.append(("💰 Total Revenue",    _fmt_currency(kpis["revenue"]),           kpis.get("revenue_col","")))
    if "units_sold"      in kpis: cards.append(("🚗 Units Sold",        _fmt_number(kpis["units_sold"]),           kpis.get("vol_col","")))
    if "yoy_growth"      in kpis and kpis["yoy_growth"] is not None:
        g = kpis["yoy_growth"]; arrow = "▲" if g >= 0 else "▼"
        cards.append(("📈 YoY Growth", f"{arrow} {abs(g):.1f}%", f"vs {kpis.get('yoy_curr_year','') - 1}"))
    if "market_share"    in kpis: cards.append(("🏷️ Market Share",      f"{kpis['market_share']:.1f}%",           "avg"))
    if "best_salesperson"in kpis: cards.append(("🌟 Best Salesperson",  kpis["best_salesperson"],                 f"Month: {kpis.get('best_salesperson_month','')}"))
    if "top_model"       in kpis: cards.append(("🏆 Top Model",          kpis["top_model"],                        "by sales"))
    if "active_regions"  in kpis: cards.append(("🌍 Active Regions",    str(kpis["active_regions"]),              "unique"))

    for i in range(0, len(cards), 4):
        row  = cards[i:i+4]
        cols = st.columns(len(row))
        for col, (label, value, sub) in zip(cols, row):
            with col:
                st.markdown(f"""<div class="kpi-card"><div class="kv">{value}</div><div class="kl">{label}</div><div class="ks">{sub}</div></div>""", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:10px;'></div>", unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("📋 Show Detailed KPIs ▼", expanded=False):
        if "segment_mix" in kpis:
            st.markdown("#### 🥧 Vehicle Sales Mix by Segment")
            seg_df = pd.DataFrame([{"Segment": k, "Units Sold": v["units"], "Share %": v["pct"]} for k, v in kpis["segment_mix"].items()])
            c1, c2 = st.columns([1,1])
            with c1:
                fig = px.pie(seg_df, names="Segment", values="Units Sold", hole=0.35, color_discrete_sequence=px.colors.qualitative.Plotly, title="Sales Mix")
                fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(seg_df, use_container_width=True, hide_index=True)
            st.markdown("---")

        if "top5_models" in kpis:
            st.markdown("#### 🏎️ Top 5 Models by Revenue")
            m5_df = pd.DataFrame(kpis["top5_models"]); m5_df.columns = ["Model","Revenue","Contribution %"]
            m5_df["Revenue Display"] = m5_df["Revenue"].apply(_fmt_currency)
            c1, c2 = st.columns([1,1])
            with c1:
                fig = px.bar(m5_df, y="Model", x="Revenue", orientation="h", text="Revenue Display", color_discrete_sequence=["#4fc3f7"], title="Revenue by Model")
                fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(m5_df[["Model","Revenue Display","Contribution %"]], use_container_width=True, hide_index=True)
            st.markdown("---")

        if "fastest_seg" in kpis:
            st.markdown("#### 🚀 Fastest Growing Segment")
            st.success(f"**{kpis['fastest_seg']}** grew **{kpis['fastest_seg_pct']}%** YoY")
            st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="platform-header">
  <h1>🚀 AI Data Platform</h1>
  <p>Natural Language &rarr; SQL &rarr; Visualization &nbsp;|&nbsp; Powered by GPT-4o &amp; DuckDB</p>
</div>
""", unsafe_allow_html=True)

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
        st.progress(min(st.session_state.llm_calls / st.session_state.max_llm_calls, 1.0), text=f"{max(st.session_state.max_llm_calls - st.session_state.llm_calls, 0)} calls remaining")
        if st.button("🔄 Reset Usage"):
            st.session_state.llm_calls = 0; st.session_state.total_tokens = 0; st.rerun()
        if st.button("🗑️ Clear Cache"):
            st.session_state.memory = {}; st.session_state.last_plan = None; st.session_state.last_query = ""; st.session_state.query_history = []; st.rerun()

if not st.session_state.dfs:
    st.info("👈 Upload one or more CSV files to get started.")
    st.stop()

tables = list(st.session_state.dfs.keys())
working_df = get_working_df()
tab_join, tab_preview, tab_kpi, tab_query = st.tabs(["🔗 Join / Combine", "📄 Data Preview", "📊 KPI Summary", "⚡ AI Query"])

# ═══════════════════════════════════════════════════════════
# TAB 1 — JOIN / COMBINE
# ═══════════════════════════════════════════════════════════
with tab_join:
    if len(st.session_state.dfs) == 1:
        st.info("Only one table loaded — no joining needed. Go to AI Query.")
    else:
        st.subheader("🔗 Combine Tables")
        mode_label = st.radio("Join Method", ["🤖 Auto-detect (recommended)", "🛠️ Manual UI", "📝 SQL Query"], horizontal=True)
        st.session_state.join_mode = "auto" if "Auto" in mode_label else ("manual" if "Manual" in mode_label else "sql")
        st.markdown("---")

        if st.session_state.join_mode == "auto":
            st.markdown("Auto-join detects the best key between each table pair using **column name similarity + value overlap scoring**. A score ≥ 60 is a reliable join.")
            current_base = st.session_state.get("auto_join_base") or tables[0]
            if current_base not in tables: current_base = tables[0]
            base_choice = st.selectbox("📌 Base Table (left-side anchor for all joins)", options=tables, index=tables.index(current_base), help="All other tables will be joined onto this table.", key="auto_join_base_select")
            st.session_state.auto_join_base = base_choice
            other_tables = [t for t in tables if t != base_choice]
            if other_tables: st.caption(f"Will join **{', '.join(other_tables)}** onto **{base_choice}**")
            if st.button("▶️ Preview Auto-Join"):
                with st.spinner("Analysing tables…"):
                    joined, join_log = auto_join(st.session_state.dfs, base_name=base_choice)
                if joined is not None:
                    st.success(f"✅ Result: {joined.shape[0]:,} rows × {joined.shape[1]} cols")
                    if join_log:
                        st.markdown("**Join Quality Report**")
                        for entry in join_log:
                            score = entry["score"]
                            cls  = "score-high" if score >= 60 else ("score-med" if score >= 30 else "score-low")
                            icon = "✅" if score >= 60 else ("⚠️" if score >= 30 else "❌")
                            st.markdown(f"{icon} `{entry['left_table']}`.`{entry['left_col']}` ↔ `{entry['right_table']}`.`{entry['right_col']}` — <span class='{cls}'>score {score}</span> &nbsp; _{entry.get('note','')}_", unsafe_allow_html=True)
                    st.dataframe(joined.head(100), use_container_width=True)

        elif st.session_state.join_mode == "manual":
            joins = st.session_state.manual_joins
            if not joins:
                joins[0] = {"left": tables[0], "right": tables[min(1, len(tables)-1)], "left_on": "", "right_on": "", "type": "inner"}
            to_del = []
            for i, j in joins.items():
                c0,c1,c2,c3,c4,c5 = st.columns([2,2,2,2,1,0.5])
                j["left"]    = c0.selectbox("Base",       tables, index=tables.index(j["left"])  if j["left"]  in tables else 0, key=f"l{i}")
                j["right"]   = c1.selectbox("Join Table", tables, index=tables.index(j["right"]) if j["right"] in tables else 0, key=f"r{i}")
                lc = list(st.session_state.dfs[j["left"]].columns); rc = list(st.session_state.dfs[j["right"]].columns)
                j["left_on"]  = c2.selectbox("Left Key",  lc, key=f"lk{i}")
                j["right_on"] = c3.selectbox("Right Key", rc, key=f"rk{i}")
                j["type"]     = c4.selectbox("Type", ["inner","left","right","outer"], key=f"jt{i}")
                if c5.button("❌", key=f"d{i}"): to_del.append(i)
            for r in to_del: del st.session_state.manual_joins[r]
            if to_del: st.rerun()
            ca, cb = st.columns(2)
            if ca.button("➕ Add Join"):
                nk = max(joins.keys(), default=-1) + 1
                joins[nk] = {"left": tables[0], "right": tables[0], "left_on": "", "right_on": "", "type": "inner"}; st.rerun()
            if cb.button("▶️ Preview"):
                jdf = manual_join(st.session_state.dfs, joins)
                if jdf is not None: st.success(f"✅ {jdf.shape[0]:,} rows × {jdf.shape[1]} cols"); st.dataframe(jdf.head(100), use_container_width=True)

        elif st.session_state.join_mode == "sql":
            st.markdown("**Available tables:** " + ", ".join([f"`{t}`" for t in tables]))
            sql_text = st.text_area("SQL Join Query", value=st.session_state.sql_join_text or f"SELECT *\nFROM {tables[0]}\n" + (f"LEFT JOIN {tables[1]} ON {tables[0]}.id = {tables[1]}.id" if len(tables) > 1 else ""), height=140)
            st.session_state.sql_join_text = sql_text
            if st.button("▶️ Execute & Preview"):
                jdf = sql_join(st.session_state.dfs, sql_text)
                if jdf is not None: st.success(f"✅ {jdf.shape[0]:,} rows × {jdf.shape[1]} cols"); st.dataframe(jdf.head(100), use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 2 — DATA PREVIEW
# ═══════════════════════════════════════════════════════════
with tab_preview:
    st.subheader("📄 Data Preview")
    # working_preview_df = get_working_df()
    working_preview_df = working_df
    if working_preview_df is not None:
        multi = len(st.session_state.dfs) > 1
        label = "🔗 Joined / Working Dataset" if multi else "📋 Working Dataset"
        st.markdown(f"#### {label}")
        st.markdown(f"""<div class="stat-row">
          <div class="stat-card"><div class="sv">{working_preview_df.shape[0]:,}</div><div class="sl">Rows</div></div>
          <div class="stat-card"><div class="sv">{working_preview_df.shape[1]}</div><div class="sl">Columns</div></div>
          <div class="stat-card"><div class="sv">{working_preview_df.select_dtypes(include='number').shape[1]}</div><div class="sl">Numeric Cols</div></div>
        </div>""", unsafe_allow_html=True)
        st.dataframe(working_preview_df.head(200), use_container_width=True)
        with st.expander("🔬 Data Quality Intelligence ▼", expanded=False):
            render_data_quality(working_preview_df, "working_dataset")

    st.markdown("---")
    st.markdown("#### 🗂️ Individual Table View")
    sel    = st.selectbox("Select Table", tables, key="preview_table_sel")
    search = st.text_input("🔍 Search columns", "")
    pdf    = st.session_state.dfs[sel]
    if search:
        matched = [c for c in pdf.columns if search.lower() in c.lower()]
        pdf = pdf[matched] if matched else pdf
    st.markdown(f"""<div class="stat-row">
      <div class="stat-card"><div class="sv">{pdf.shape[0]:,}</div><div class="sl">Rows</div></div>
      <div class="stat-card"><div class="sv">{pdf.shape[1]}</div><div class="sl">Columns</div></div>
      <div class="stat-card"><div class="sv">{pdf.select_dtypes(include='number').shape[1]}</div><div class="sl">Numeric Cols</div></div>
    </div>""", unsafe_allow_html=True)
    st.dataframe(pdf.head(200), use_container_width=True)
    with st.expander("📌 Column Details"):
        info = []
        for col in st.session_state.dfs[sel].columns:
            s = st.session_state.dfs[sel][col]
            info.append({"Column": col, "Type": str(s.dtype), "Non-Null": int(s.notna().sum()), "Null": int(s.isna().sum()), "Unique": int(s.nunique()), "Sample": str(s.dropna().iloc[0]) if s.notna().any() else "N/A"})
        st.dataframe(pd.DataFrame(info), use_container_width=True)

# ═══════════════════════════════════════════════════════════
# TAB 3 — KPI SUMMARY
# ═══════════════════════════════════════════════════════════
with tab_kpi:
    # working_df_kpi = get_working_df()
    render_kpi_tab(working_df)
 

# ═══════════════════════════════════════════════════════════
# TAB 4 — AI QUERY
# ═══════════════════════════════════════════════════════════
with tab_query:
    st.subheader("⚡ AI Query")
    # working_df = get_working_df()
    if working_df is None or working_df.empty:
        st.warning("⚠️ No data available.")
        st.stop()

    with st.expander("🎛️ Quick Filters", expanded=False):
        qf = {}; qcols = st.columns(4); idx = 0
        cat_candidates = [c for c in working_df.columns if working_df[c].dtype == object and working_df[c].nunique() <= 40]
        for col in cat_candidates[:3]:
            vals = ["All"] + sorted(working_df[col].dropna().unique().tolist())
            sel_val = qcols[idx%4].selectbox(col, vals, key=f"qf_{col}")
            if sel_val != "All": qf[col] = sel_val
            idx += 1
        date_cols = [c for c in working_df.columns if pd.api.types.is_datetime64_any_dtype(working_df[c])]
        if date_cols:
            dc = date_cols[0]; mn = int(working_df[dc].dt.year.min()); mx = int(working_df[dc].dt.year.max())
            if mn < mx:
                yr = st.slider("Year Range", mn, mx, (mn, mx), key="qf_yr"); qf["__year__"] = (dc, yr)
        if qf:
            for col, val in qf.items():
                if col == "__year__":
                    dc2, (y1,y2) = val; working_df = working_df[working_df[dc2].dt.year.between(y1, y2)]
                else:
                    working_df = working_df[working_df[col].astype(str).str.strip().str.lower() == str(val).strip().lower()]
            st.success(f"✅ Filters applied — {working_df.shape[0]:,} rows")

    _wdf = working_df
    st.markdown("---")

    if st.session_state.query_history:
        st.markdown("**Recent Queries**")
        hcols = st.columns(min(len(st.session_state.query_history), 4))
        for i, hq in enumerate(st.session_state.query_history[:4]):
            if hcols[i].button(f"↩ {hq[:28]}{'…' if len(hq)>28 else ''}", key=f"h{i}"):
                st.session_state.query_input = hq; st.rerun()

    qcol, runcol, clrcol = st.columns([8,1,1])
    q = qcol.text_input("Ask anything", key="query_input", placeholder="e.g. top 10 salespersons by revenue, monthly sales trend for Ford in 2023", label_visibility="collapsed")
    run_clicked = runcol.button("▶️ Run",   use_container_width=True)
    clr_clicked = clrcol.button("🗑️ Clear", use_container_width=True)

    if clr_clicked:
        st.session_state.last_result = None; st.session_state.last_analysis  = None; st.session_state.pending_suggestion = None; st.rerun()

    if run_clicked and q.strip():
        with st.spinner("Generating SQL & fetching results…"):
            result, sql, err = run_query(_wdf, q.strip())
        st.session_state.last_result = (result, sql, err, q.strip()); st.session_state.last_analysis  = None; st.session_state.pending_suggestion = None; st.session_state.view_toggle = "📋 Table"

    if st.session_state.pending_suggestion:
        _sg = st.session_state.pending_suggestion; st.session_state.pending_suggestion = None
        with st.spinner(f"Running: {_sg}"):
            _r, _s, _e = run_query(_wdf, _sg)
        st.session_state.last_result = (_r, _s, _e, _sg); st.session_state.last_analysis  = None; st.session_state.view_toggle = "📋 Table"; st.rerun()

    st.markdown("---")

    if st.session_state.last_result is not None:
        result, sql, err, asked_q = st.session_state.last_result

        if err and result is None:
            st.error(f"❌ {err}")
            if sql:
                with st.expander("🔍 SQL Attempted"): st.code(sql, language="sql")

        elif result is not None and not result.empty:
            sql_safe_preview = html.escape((sql or "").strip().replace("\n"," "))
            first80 = sql_safe_preview[:120] + ("…" if len(sql_safe_preview)>120 else "")
            st.markdown(f"""<div class="sql-strip"><span class="badge">SQL</span><span class="sql-text">{first80}</span></div>""", unsafe_allow_html=True)

            with st.expander("✏️ View / Edit & Re-run SQL", expanded=False):
                edited_sql = st.text_area("Edit then Re-run — no LLM call, direct DuckDB execution", value=(sql or "").strip(), height=140, key="edited_sql_area")
                rcol, _ = st.columns([2,8])
                if rcol.button("▶️ Re-run SQL", key="rerun_sql_btn"):
                    safe, reason = sql_is_safe(edited_sql.strip())
                    if not safe:
                        st.error(f"🔒 Blocked: {reason}")
                    else:
                        with st.spinner("Running edited SQL…"):
                            new_result, new_err = run_sql(edited_sql.strip(), _wdf)
                        if new_err: st.error(f"SQL error: {new_err}")
                        elif new_result is not None:
                            st.session_state.last_result = (new_result, edited_sql.strip(), None, asked_q); st.session_state.last_analysis = None; st.session_state.view_toggle = "📋 Table"; st.rerun()

            num_c   = result.select_dtypes(include="number").columns.tolist()
            total_v = f"{result[num_c[0]].sum():,.1f}" if num_c else "—"
            total_l = num_c[0] if num_c else "Total"
            st.markdown(f"""<div class="stat-row">
              <div class="stat-card"><div class="sv">{result.shape[0]:,}</div><div class="sl">Rows</div></div>
              <div class="stat-card"><div class="sv">{result.shape[1]}</div><div class="sl">Columns</div></div>
              <div class="stat-card"><div class="sv">{total_v}</div><div class="sl">{total_l}</div></div>
            </div>""", unsafe_allow_html=True)

            view = st.radio("View", ["📊 Chart","📋 Table"], horizontal=True, key="view_toggle")

            if view == "📊 Chart":
                all_cols = list(result.columns)
                num_cols = result.select_dtypes(include="number").columns.tolist()
                str_cols = result.select_dtypes(exclude="number").columns.tolist()
                ctrl_col, chart_col = st.columns([2,8])
                auto_ct    = auto_chart_type(result, asked_q)
                chart_type = ctrl_col.selectbox("Chart Type", ["Bar","Line","Pie","Scatter","Area"], index=["Bar","Line","Pie","Scatter","Area"].index(auto_ct), key="ct_sel")
                default_x  = str_cols[0] if str_cols else all_cols[0]
                default_y  = num_cols[0] if num_cols else (all_cols[1] if len(all_cols)>1 else all_cols[0])
                if default_x not in all_cols: default_x = all_cols[0]
                if default_y not in all_cols: default_y = all_cols[-1]
                x_axis = ctrl_col.selectbox("X Axis", all_cols, index=all_cols.index(default_x), key="xa")
                y_default_idx = all_cols.index(default_y)
                if default_y == x_axis and len(all_cols) > 1: y_default_idx = (all_cols.index(x_axis) + 1) % len(all_cols)
                y_axis = ctrl_col.selectbox("Y Axis", all_cols, index=y_default_idx, key="ya")
                with chart_col:
                    if isinstance(x_axis, str) and isinstance(y_axis, str): build_chart(result, chart_type, x_axis, y_axis)
                    else: st.warning("⚠️ Please select valid X and Y columns.")

            else:
                st.dataframe(result, use_container_width=True)
                st.download_button("⬇️ Download CSV", data=result.to_csv(index=False).encode(), file_name="result.csv", mime="text/csv")

            # ── AI Analysis — 12 spaces indent, INSIDE elif block ──
            st.markdown("---")
            st.markdown("##### 🧠 Insights ?*")

            if st.button("✨ Generate Analysis", use_container_width=True):
                with st.spinner("Analysing data — summary, insights & recommendation…"):
                    st.session_state.last_analysis = generate_analysis(result, asked_q)

            if st.session_state.get("last_analysis"):
                ana = st.session_state.last_analysis
                sum_col, ins_col = st.columns([1, 1])

                with sum_col:
                    st.markdown("##### 📋 Summary — *What happened?*")
                    if ana.get("summary"):
                        st.markdown(
                            f"<div class='exec-box'>"
                            f"{html.escape(ana['summary'])}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

  
                with ins_col:
                    st.markdown("##### 💡 Key Facts")
                    if ana.get("facts"):
                        facts_html = "".join(
                            f"<div style='padding:6px 0;border-bottom:1px solid #e2e8f0;"
                            f"font-size:13px;'>"
                            f"<span style='color:#7c3aed;font-weight:700;'>•</span> "
                            f"{html.escape(f)}</div>"
                            for f in ana["facts"]
                        )
                        st.markdown(
                            f"<div class='exec-box' style='border-left-color:#7c3aed;"
                            f"padding:10px 16px;'>{facts_html}</div>",
                            unsafe_allow_html=True,
                        )

        elif result is not None and result.empty:
            st.warning("⚠️ Query returned no rows. Try broadening filters or rephrasing.")
            with st.expander("🔍 Generated SQL"): st.code(sql, language="sql")

    elif not run_clicked:
        st.markdown(
            "<div style='text-align:center;padding:56px;color:#8b949e;"
            "font-size:15px;'>💬 Type a question above and press <b>Run</b></div>",
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown("<div style='text-align:center;color:#555;font-size:11px;'>🚀 Capgemini AI Data Platform &nbsp;|&nbsp; DuckDB · GPT-4o · Streamlit</div>", unsafe_allow_html=True)
