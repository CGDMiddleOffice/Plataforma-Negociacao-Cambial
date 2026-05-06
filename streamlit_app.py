# cambial_dashboard_anual.py
# Dashboard Streamlit (corporativo) — Posição (Mês/Ano) + YoY (média anos anteriores) + Estimativa (apenas quando período incompleto) + Tema Light (fixo)
# Como correr:
# pip install streamlit pandas numpy altair
# streamlit run cambial_dashboard_anual.py

import io
import calendar
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from functools import lru_cache
import urllib.request
import hmac
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PERSISTENT_CSV = DATA_DIR / "report_cache.csv"

# =============================================================================
# LOGO (CGD)
# =============================================================================
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Logo_of_Caixa_Geral_de_Dep%C3%B3sitos.svg/250px-Logo_of_Caixa_Geral_de_Dep%C3%B3sitos.svg.png"

def _local_logo_candidate() -> Optional[str]:
    here = Path(__file__).resolve().parent
    for name in ("cgd_logo.png", "CGD_logo.png", "logo_cgd.png", "logo.png", "favicon.png", "icon.png"):
        p = here / name
        if p.exists() and p.is_file():
            return str(p)
    return None

@lru_cache(maxsize=2)
def _download_logo_to_local(url: str) -> Optional[str]:
    if not url:
        return None
    try:
        data = urllib.request.urlopen(url, timeout=10).read()
        p = Path(__file__).with_name("_cgd_logo.png")
        p.write_bytes(data)
        return str(p)
    except Exception:
        return None

LOGO_LOCAL = _local_logo_candidate() or _download_logo_to_local(LOGO_URL)

# =============================================================================
# Página (tab / title)
# =============================================================================
# FIX 4: page_icon usa sempre o ficheiro local (mais fiável do que URL para ícone da tab)
_page_icon = LOGO_LOCAL if LOGO_LOCAL else "🏦"

try:
    st.set_page_config(
        page_title="Plataforma Cambial — FNC",
        page_icon=_page_icon,
        layout="wide",
    )
except Exception:
    st.set_page_config(
        page_title="Plataforma Cambial — FNC",
        page_icon="🏦",
        layout="wide",
    )

# =============================================================================
# AUTENTICAÇÃO
# =============================================================================
AUTH_USER = st.secrets["auth"].get("username", "")
AUTH_PASS = st.secrets["auth"].get("password", "")

if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

def _do_login(user: str, pwd: str) -> bool:
    if user == AUTH_USER and pwd == AUTH_PASS:
        st.session_state.auth_ok = True
        st.session_state.auth_user = user
        return True
    return False

def _do_logout():
    st.session_state.auth_ok = False
    st.session_state.auth_user = None

# --- Gate de acesso ---
if not st.session_state.auth_ok:
    # FIX 5: Logo no fundo do ecrã de login via CSS + mostrar logo acima do form
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: #F6F7FB !important;
        }}
        .login-bg {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            pointer-events: none;
            z-index: 0;
        }}
        .login-bg img {{
            width: 420px;
            opacity: 0.055;
            filter: grayscale(100%);
        }}
        .login-wrap {{
            position: relative;
            z-index: 1;
        }}
        </style>
        <div class="login-bg">
            <img src="{LOGO_URL}" alt="CGD" />
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Logo visível no topo do formulário
    col_l, col_c, col_r = st.columns([1, 1.2, 1])
    with col_c:
        try:
            st.image(LOGO_URL, width=160)
        except Exception:
            if LOGO_LOCAL:
                try:
                    st.image(LOGO_LOCAL, width=160)
                except Exception:
                    pass

    st.title("Plataforma Cambial — Login")
    with st.form("login_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            user_input = st.text_input("Utilizador", autocomplete="username")
        with c2:
            pass_input = st.text_input("Palavra passe", type="password", autocomplete="current-password")
        ok = st.form_submit_button("Entrar")
        if ok:
            if _do_login(user_input, pass_input):
                st.success("Autenticado com sucesso. A carregar…")
                st.rerun()
            else:
                st.error("Credenciais inválidas. Tenta novamente.")
    st.stop()

# --- Barra de sessão + logout ---
topc1, topc2 = st.columns([0.84, 0.16])
with topc1:
    st.caption(f"✅ Sessão iniciada como **{st.session_state.auth_user}**")
with topc2:
    if st.button("Terminar sessão", use_container_width=True):
        _do_logout()
        st.rerun()

# =============================================================================
# Helpers (texto / parsing numérico robusto)
# =============================================================================
_PT_ACCENTS_SRC = "áàâãäéêëíìîïóòôõöúùûüçÁÀÂÃÄÉÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ"
_PT_ACCENTS_DST = "aaaaaeeeiiiiooooouuuucAAAAAEEEIIIIOOOOOUUUUC"

def _norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\xa0", " ")
    s = s.translate(str.maketrans(_PT_ACCENTS_SRC, _PT_ACCENTS_DST))
    s = s.strip().lower()
    s = s.replace("nº", "n").replace("n°", "n")
    for ch in [":", ";", "\t", "\n", "\r"]:
        s = s.replace(ch, " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()

def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            bio = io.BytesIO(raw)
            df = pd.read_csv(bio, sep=None, engine="python", encoding=enc)
            return df
        except Exception as e:
            last_err = e
    raise last_err

def _pick_col(cols: List[str], must_contain: List[str], occurrence: int = 0) -> Optional[str]:
    hits = []
    for c in cols:
        nc = _norm_text(c)
        ok = True
        for t in must_contain:
            if t not in nc:
                ok = False
                break
        if ok:
            hits.append(c)
    return hits[occurrence] if len(hits) > occurrence else None

def _best_ops_col(cols: List[str]) -> Optional[str]:
    best = None
    best_score = -10_000
    for c in cols:
        nc = _norm_text(c)
        if "operac" not in nc:
            continue
        score = 0
        if " num" in f" {nc} " or nc.startswith("num ") or "numero" in nc:
            score += 6
        if nc.startswith("n ") or " n " in f" {nc} ":
            score += 4
        if "n operac" in nc or "num operac" in nc or "numero operac" in nc:
            score += 8
        if "clientes" in nc:
            score -= 8
        if "ativad" in nc:
            score -= 10
        if "acesso" in nc:
            score -= 6
        if "%" in c or "percent" in nc:
            score -= 10
        if "cl " in f" {nc} ":
            score -= 2
        if len(nc) <= 18:
            score += 2
        if score > best_score:
            best_score = score
            best = c
    return best

def _parse_number_str(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).replace("\xa0", " ").strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    for token in ["€", "%", " "]:
        s = s.replace(token, "")
    mult = 1.0
    s_low = s.lower()
    if s_low.endswith("k"):
        mult = 1e3
        s = s[:-1]
    elif s_low.endswith("m"):
        mult = 1e6
        s = s[:-1]
    elif s_low.endswith("b"):
        mult = 1e9
        s = s[:-1]
    if "." in s and "," in s:
        last_dot = s.rfind(".")
        last_com = s.rfind(",")
        if last_com > last_dot:
            s = s.replace(".", "")
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if "," in s:
            parts = s.split(",")
            if len(parts[-1]) in (1, 2):
                s = s.replace(".", "")
                s = s.replace(",", ".")
            else:
                s = s.replace(",", "")
        else:
            if s.count(".") >= 2:
                s = s.replace(".", "")
    try:
        return float(s) * mult
    except Exception:
        return None

def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).map(_parse_number_str), errors="coerce")

def _to_pct_series(s: pd.Series) -> pd.Series:
    x = _to_float_series(s)
    return np.where(x > 1.0, x / 100.0, x).astype(float)

def _safe_div(a, b):
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return np.nan
    if b == 0 or np.isnan(a) or np.isnan(b):
        return np.nan
    return a / b

def _stock_last(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    return float(s2.iloc[-1]) if len(s2) else np.nan

def _flow_sum(s: pd.Series):
    s2 = pd.to_numeric(s, errors="coerce")
    return float(s2.sum()) if s2.notna().any() else np.nan

def _fmt_int(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{int(round(float(x))):,}".replace(",", " ")

def _fmt_int_compact(x, decimals=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    v = float(x)
    av = abs(v)
    if av >= 1e9:
        return f"{v/1e9:.{decimals}f} B"
    if av >= 1e6:
        return f"{v/1e6:.{decimals}f} M"
    if av >= 1e3:
        return f"{v/1e3:.{decimals}f} K"
    return f"{int(round(v))}"

def _fmt_pct(p, decimals=1):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return f"{100*float(p):.{decimals}f}%"

def _fmt_eur_compact(x, decimals=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    v = float(x)
    abs_v = abs(v)
    if abs_v >= 1e9:
        return f"€ {v/1e9:.{decimals}f} B"
    if abs_v >= 1e6:
        return f"€ {v/1e6:.{decimals}f} M"
    if abs_v >= 1e3:
        return f"€ {v/1e3:.{decimals}f} K"
    return "€ " + f"{v:,.0f}".replace(",", " ")

def _recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "volume_negocios" in out.columns and "num_operacoes" in out.columns:
        out["ticket_medio"] = out.apply(lambda r: _safe_div(r.get("volume_negocios"), r.get("num_operacoes")), axis=1)
    else:
        out["ticket_medio"] = np.nan
    if "margem_liquida" in out.columns and "num_operacoes" in out.columns:
        out["margem_por_op"] = out.apply(lambda r: _safe_div(r.get("margem_liquida"), r.get("num_operacoes")), axis=1)
    else:
        out["margem_por_op"] = np.nan
    if "margem_liquida" in out.columns and "volume_negocios" in out.columns:
        out["margem_pct_volume"] = out.apply(lambda r: _safe_div(r.get("margem_liquida"), r.get("volume_negocios")), axis=1)
    else:
        out["margem_pct_volume"] = np.nan
    if "conv_ops_s1" not in out.columns or out["conv_ops_s1"].isna().all():
        if "ativados_ops_s1" in out.columns and "clientes_acesso" in out.columns:
            out["conv_ops_s1"] = out.apply(lambda r: _safe_div(r.get("ativados_ops_s1"), r.get("clientes_acesso")), axis=1)
    if "conv_ops_s2" not in out.columns or out["conv_ops_s2"].isna().all():
        if "ativados_ops_s2" in out.columns and "clientes_acesso" in out.columns:
            out["conv_ops_s2"] = out.apply(lambda r: _safe_div(r.get("ativados_ops_s2"), r.get("clientes_acesso")), axis=1)
    return out

# =============================================================================
# Load report (cache)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_report(raw: bytes) -> pd.DataFrame:
    df = _read_csv_bytes(raw)
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]
    cols = list(df.columns)

    date_col = None
    for c in cols:
        if _norm_text(c) == "data":
            date_col = c
            break
    if date_col is None:
        date_col = cols[0]

    col_clientes = _pick_col(cols, ["clientes", "acesso"], 0)
    col_pend = _pick_col(cols, ["pedidos", "pendentes"], 0)
    col_novos = _pick_col(cols, ["novos", "pedidos"], 0)
    col_desist_total = _pick_col(cols, ["desist", "total"], 0)
    col_desist_ativ = _pick_col(cols, ["de", "ativados"], 0)
    col_desist_pend = _pick_col(cols, ["de", "pendentes"], 0)
    col_ativ1 = _pick_col(cols, ["clientes", "ativados", "operac"], 0)
    col_pct1 = _pick_col(cols, ["cl", "operac", "acesso"], 0)
    col_ativ2 = _pick_col(cols, ["clientes", "ativados", "operac"], 1)
    col_pct2 = _pick_col(cols, ["cl", "operac", "acesso"], 1)
    col_ops = _best_ops_col(cols) or _pick_col(cols, ["n", "operac"], 0) or _pick_col(cols, ["operacoes"], 0)
    col_vol = _pick_col(cols, ["volume"], 0)
    col_marg = _pick_col(cols, ["margem"], 0)

    out = pd.DataFrame()
    out["data"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    def add_num(target: str, src: Optional[str]):
        if src is None or src not in df.columns:
            out[target] = np.nan
        else:
            out[target] = _to_float_series(df[src])

    add_num("clientes_acesso", col_clientes)
    add_num("pedidos_pendentes", col_pend)
    add_num("novos_pedidos", col_novos)
    add_num("desist_total", col_desist_total)
    add_num("desist_ativados", col_desist_ativ)
    add_num("desist_pendentes", col_desist_pend)
    add_num("ativados_ops_s1", col_ativ1)
    add_num("ativados_ops_s2", col_ativ2)
    add_num("num_operacoes", col_ops)
    out["conv_ops_s1"] = _to_pct_series(df[col_pct1]) if col_pct1 and col_pct1 in df.columns else np.nan
    out["conv_ops_s2"] = _to_pct_series(df[col_pct2]) if col_pct2 and col_pct2 in df.columns else np.nan
    out["volume_negocios"] = _to_float_series(df[col_vol]) if col_vol and col_vol in df.columns else np.nan
    out["margem_liquida"] = _to_float_series(df[col_marg]) if col_marg and col_marg in df.columns else np.nan

    out = out.dropna(subset=["data"]).sort_values("data").reset_index(drop=True)
    out = _recompute_derived(out)
    return out

# =============================================================================
# Labels PT
# =============================================================================
_PT_MONTHS = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
}
_PT_MONTHS_FULL = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

# =============================================================================
# Tema (Light fixo) — CSS + Altair theme
# =============================================================================
def _apply_theme(theme: str):
    if theme == "Dark":
        bg = "#0E1117"; panel = "#111827"; card = "#0B1220"
        border = "rgba(255,255,255,0.10)"; text = "rgba(255,255,255,0.92)"
        subtle = "rgba(255,255,255,0.70)"; grid = "rgba(255,255,255,0.12)"
        accent = "#60A5FA"; accent2 = "#93C5FD"; good = "#22C55E"
        warn = "#F59E0B"; bad = "#EF4444"; df_bg = "#0B1220"
        df_text = "rgba(255,255,255,0.92)"; df_header = "rgba(255,255,255,0.08)"
        orange = "#F59E0B"; widget_shadow = "rgba(0,0,0,0.35)"
        est_bg = "rgba(255,255,255,0.06)"
    else:
        bg = "#F6F7FB"; panel = "#FFFFFF"; card = "#FFFFFF"
        border = "rgba(49,51,63,0.14)"; text = "rgba(17,24,39,0.95)"
        subtle = "rgba(17,24,39,0.65)"; grid = "rgba(17,24,39,0.10)"
        accent = "#2563EB"; accent2 = "#60A5FA"; good = "#16A34A"
        warn = "#D97706"; bad = "#DC2626"; df_bg = "#FFFFFF"
        df_text = "rgba(17,24,39,0.95)"; df_header = "rgba(17,24,39,0.06)"
        orange = "#F59E0B"; widget_shadow = "rgba(17,24,39,0.10)"
        est_bg = "rgba(17,24,39,0.04)"

    if theme == "Dark":
        switch_off_bg = "rgba(255,255,255,0.18)"; switch_off_border = "rgba(255,255,255,0.22)"
        switch_knob_bg = "rgba(255,255,255,0.92)"; switch_knob_border = "rgba(255,255,255,0.22)"
    else:
        switch_off_bg = "rgba(17,24,39,0.18)"; switch_off_border = "rgba(17,24,39,0.28)"
        switch_knob_bg = "#FFFFFF"; switch_knob_border = "rgba(17,24,39,0.18)"

    css = f"""
<style>
:root {{
    --c-bg: {bg}; --c-panel: {panel}; --c-card: {card};
    --c-border: {border}; --c-text: {text}; --c-subtle: {subtle};
    --c-accent: {accent}; --c-accent2: {accent2}; --c-good: {good};
    --c-warn: {warn}; --c-bad: {bad}; --c-orange: {orange};
    --c-shadow: {widget_shadow}; --c-switch-off-bg: {switch_off_bg};
    --c-switch-off-border: {switch_off_border}; --c-switch-knob-bg: {switch_knob_bg};
    --c-switch-knob-border: {switch_knob_border}; --c-est-bg: {est_bg};
}}
.stApp {{ background: var(--c-bg) !important; color: var(--c-text) !important; }}
html, body, [class*="css"] {{ color: var(--c-text) !important; }}
.block-container {{ padding-top: 4.5rem; padding-bottom: 2.0rem; max-width: 1500px; }}
footer, #MainMenu {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: var(--c-bg) !important; height: 0px !important; }}
[data-testid="stHeader"] * {{ display: none !important; }}
[data-testid="stToolbar"] {{ visibility: hidden !important; height: 0px !important; }}
h1, h2, h3, h4, h5, h6 {{ letter-spacing: -0.2px; color: var(--c-text) !important; }}
.subtle {{ color: var(--c-subtle) !important; }}
.panel {{ background: var(--c-panel); border: 1px solid var(--c-border); border-radius: 16px; padding: 14px 16px; box-shadow: 0 6px 18px var(--c-shadow); }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }}
@media (max-width: 1300px) {{ .kpi-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
@media (max-width: 800px) {{ .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
.kpi {{ background: var(--c-card); border: 1px solid var(--c-border); border-radius: 16px; padding: 12px 14px; min-height: 80px; }}
.kpi-title {{ font-size: 0.78rem; color: var(--c-subtle); margin-bottom: 4px; }}
.kpi-value {{ font-size: 1.25rem; font-weight: 760; line-height: 1.15; color: var(--c-text); }}
.kpi-note {{ font-size: 0.75rem; color: var(--c-subtle); margin-top: 4px; }}
.kpi-est {{ font-size: 0.70rem; color: var(--c-subtle); font-weight: 700; margin-left: 10px; padding: 2px 7px; border: 1px solid var(--c-border); border-radius: 999px; background: var(--c-est-bg); white-space: nowrap; vertical-align: middle; }}
.kpi-est::before {{ content: ""; display: inline-block; width: 6px; height: 6px; border-radius: 999px; background: var(--c-subtle); opacity: 0.45; margin-right: 6px; transform: translateY(-0.5px); }}
.delta-up {{ color: var(--c-good); font-weight: 800; }}
.delta-down {{ color: var(--c-bad); font-weight: 800; }}
.delta-flat {{ color: var(--c-subtle); font-weight: 800; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid var(--c-border); font-size: 0.72rem; color: var(--c-subtle); margin-left: 8px; }}
div[data-testid="stRadio"] label, div[data-testid="stRadio"] span {{ color: var(--c-text) !important; }}
div[data-baseweb="radio"] * {{ color: var(--c-text) !important; }}
div[data-testid="stRadio"] label *, div[data-testid="stRadio"] div[role="radiogroup"] label *, div[data-testid="stRadio"] div[role="radiogroup"] label p, div[data-testid="stRadio"] div[role="radiogroup"] label span, div[data-testid="stRadio"] div[role="radiogroup"] label div {{ color: var(--c-text) !important; opacity: 1 !important; }}
div[data-baseweb="radio"] label *, div[data-baseweb="radio"] label p, div[data-baseweb="radio"] label span, div[data-baseweb="radio"] label div {{ color: var(--c-text) !important; opacity: 1 !important; }}
div[data-baseweb="radio"] svg {{ fill: var(--c-text) !important; color: var(--c-text) !important; }}
div[data-baseweb="radio"] div[role="radio"] {{ border-color: var(--c-border) !important; background: var(--c-card) !important; }}
div[data-baseweb="radio"] div[role="radio"] > div {{ border-color: var(--c-border) !important; }}
div[data-testid="stSelectbox"] * {{ color: var(--c-text) !important; }}
div[data-baseweb="select"] > div {{ background: var(--c-card) !important; border: 1px solid var(--c-border) !important; border-radius: 12px !important; }}
div[data-baseweb="select"] span {{ color: var(--c-text) !important; }}
div[data-testid="stDateInput"] > div {{ background: var(--c-card) !important; border-radius: 12px !important; }}
div[data-testid="stDateInput"] input {{ color: var(--c-text) !important; background: var(--c-card) !important; border: 1px solid var(--c-border) !important; border-radius: 12px !important; }}
div[data-testid="stDateInput"] svg {{ fill: var(--c-subtle) !important; }}
div[data-baseweb="calendar"] * {{ color: var(--c-text) !important; }}
div[data-baseweb="calendar"] {{ background: var(--c-card) !important; }}
.st-key-upload_block details {{ background: var(--c-card) !important; border: 1px solid var(--c-border) !important; border-radius: 16px !important; overflow: hidden !important; }}
.st-key-upload_block details > summary {{ background: var(--c-panel) !important; color: var(--c-text) !important; border-bottom: 1px solid var(--c-border) !important; padding: 10px 12px !important; }}
.st-key-upload_block details > summary * {{ color: var(--c-text) !important; opacity: 1 !important; }}
.st-key-upload_block [data-testid="stExpanderDetails"] {{ background: var(--c-card) !important; }}
.st-key-upload_block [data-testid="stFileUploader"] > div {{ background: var(--c-card) !important; border: 1px dashed var(--c-border) !important; border-radius: 16px !important; padding: 12px !important; }}
.st-key-upload_block [data-testid="stFileUploaderDropzone"] {{ background: var(--c-card) !important; border: 1px dashed var(--c-border) !important; border-radius: 16px !important; }}
.st-key-upload_block [data-testid="stFileUploader"] div {{ background: var(--c-card) !important; }}
.st-key-upload_block [data-testid="stFileUploader"] *, .st-key-upload_block [data-testid="stFileUploaderDropzone"] * {{ color: var(--c-text) !important; opacity: 1 !important; }}
.st-key-upload_block [data-testid="stFileUploader"] small, .st-key-upload_block [data-testid="stFileUploaderDropzone"] small {{ color: var(--c-subtle) !important; opacity: 1 !important; }}
.st-key-upload_block [data-testid="stFileUploaderDropzone"] button, .st-key-upload_block [data-testid="stFileUploader"] button {{ background: var(--c-panel) !important; color: var(--c-text) !important; border: 1px solid var(--c-border) !important; border-radius: 12px !important; }}
.st-key-upload_block [data-testid="stFileUploaderDropzone"] button:hover, .st-key-upload_block [data-testid="stFileUploader"] button:hover {{ filter: brightness(0.98); }}
.st-key-upload_block [data-testid="stFileUploaderFile"] {{ background: var(--c-card) !important; border: 1px solid var(--c-border) !important; border-radius: 12px !important; }}
div[data-testid="stDataFrame"] {{ border-radius: 16px; border: 1px solid var(--c-border); overflow: hidden; background: {df_bg} !important; color: {df_text} !important; }}
div[data-testid="stDataFrame"] * {{ color: {df_text} !important; }}
div[data-testid="stDataFrame"] thead tr th {{ background: {df_header} !important; }}
.stDownloadButton button, .stButton button {{ border-radius: 12px; }}
.topbar {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 0.25rem; }}
.topbar-title {{ font-size: 1.45rem; font-weight: 820; line-height: 1.1; }}
.topbar-sub {{ font-size: 0.92rem; color: var(--c-subtle); }}
.ctrl {{ background: var(--c-panel); border: 1px solid var(--c-border); border-radius: 16px; padding: 12px 12px; }}
.tbl-wrap {{ overflow-x: auto; }}
table.tbl {{ width: 100%; border-collapse: separate; border-spacing: 0; }}
table.tbl thead th {{ text-align: left; font-size: 0.78rem; color: var(--c-subtle); padding: 10px 12px; border-bottom: 1px solid var(--c-border); position: sticky; top: 0; background: var(--c-panel); z-index: 1; }}
table.tbl tbody td {{ padding: 10px 12px; border-bottom: 1px solid rgba(0,0,0,0); vertical-align: middle; white-space: nowrap; color: var(--c-text); font-size: 0.92rem; }}
table.tbl tbody tr {{ background: var(--c-card); border-radius: 14px; }}
table.tbl tbody tr:nth-child(odd) td {{ background: rgba(0,0,0,0.00); }}
table.tbl tbody tr:hover td {{ background: rgba(96,165,250,0.08); }}
.arrow {{ font-weight: 900; margin-left: 8px; }}
.arrow.up {{ color: var(--c-good); }}
.arrow.down {{ color: var(--c-bad); }}
.arrow.flat {{ color: var(--c-subtle); }}
.pctcell {{ min-width: 260px; }}
.bar {{ height: 10px; width: 170px; background: rgba(0,0,0,0.06); border: 1px solid var(--c-border); border-radius: 999px; overflow: hidden; display: inline-block; vertical-align: middle; margin-right: 10px; }}
.fill {{ display: block; height: 100%; width: 0%; background: var(--c-orange); border-radius: 999px; transition: width 2.0s ease; }}
.pcttxt {{ color: var(--c-subtle); font-size: 0.82rem; vertical-align: middle; display: inline-block; }}
tr.ytd-row td {{ border-top: 1px solid var(--c-border); background: rgba(96,165,250,0.06); }}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)

    def altair_theme():
        return {
            "config": {
                "background": bg,
                "view": {"stroke": "transparent"},
                "axis": {
                    "labelColor": subtle, "titleColor": subtle,
                    "gridColor": grid, "domainColor": grid, "tickColor": grid,
                    "labelFontSize": 12, "titleFontSize": 12,
                },
                "legend": {"labelColor": subtle, "titleColor": subtle, "labelFontSize": 12, "titleFontSize": 12},
                "title": {"color": subtle, "fontSize": 13},
                "range": {"category": [accent, accent2, good, warn, bad]},
            }
        }

    alt.themes.register("corp_theme", altair_theme)
    alt.themes.enable("corp_theme")
    return {
        "bg": bg, "panel": panel, "card": card, "border": border,
        "text": text, "subtle": subtle, "grid": grid,
        "accent": accent, "accent2": accent2, "good": good,
        "warn": warn, "bad": bad, "orange": orange,
    }

# =============================================================================
# Agregações (mensal / períodos)
# =============================================================================
def build_monthly_year(df_daily: pd.DataFrame, year: int) -> pd.DataFrame:
    dfx = df_daily.copy()
    dfx = dfx[dfx["data"].dt.year == year].sort_values("data")
    if dfx.empty:
        return pd.DataFrame()
    dfx["month_end"] = (dfx["data"] + pd.offsets.MonthEnd(0)).dt.normalize()
    dfx["mes_num"] = dfx["month_end"].dt.month
    dfx["mes"] = dfx["mes_num"].map(_PT_MONTHS)
    stock_cols = ["clientes_acesso", "pedidos_pendentes", "ativados_ops_s1", "conv_ops_s1", "ativados_ops_s2", "conv_ops_s2"]
    flow_cols = ["novos_pedidos", "desist_total", "desist_ativados", "desist_pendentes", "num_operacoes", "volume_negocios", "margem_liquida"]
    keep = [c for c in stock_cols + flow_cols if c in dfx.columns]
    rows = []
    for me, g in dfx.groupby("month_end"):
        row = {"data": me, "mes_num": int(me.month), "mes": _PT_MONTHS[int(me.month)]}
        for c in stock_cols:
            if c in keep:
                row[c] = _stock_last(g[c])
        for c in flow_cols:
            if c in keep:
                row[c] = _flow_sum(g[c])
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("data").reset_index(drop=True)
    out = _recompute_derived(out)
    return out

def _month_bounds(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    d0 = pd.Timestamp(year=year, month=month, day=1)
    last_day = calendar.monthrange(year, month)[1]
    d1 = pd.Timestamp(year=year, month=month, day=last_day)
    return d0.normalize(), d1.normalize()

def _period_slice(df_daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df_daily[(df_daily["data"] >= start) & (df_daily["data"] <= end)].sort_values("data")

def _period_kpis_from_daily(df_period_daily: pd.DataFrame) -> Dict[str, float]:
    if df_period_daily is None or df_period_daily.empty:
        return {"vol": np.nan, "mar": np.nan, "ops": np.nan, "clientes_end": np.nan, "conv2_end": np.nan, "margem_pct": np.nan, "ticket": np.nan}
    vol = pd.to_numeric(df_period_daily.get("volume_negocios"), errors="coerce").sum()
    mar = pd.to_numeric(df_period_daily.get("margem_liquida"), errors="coerce").sum()
    ops = pd.to_numeric(df_period_daily.get("num_operacoes"), errors="coerce").sum()
    clientes = pd.to_numeric(df_period_daily.get("clientes_acesso"), errors="coerce").dropna()
    conv2 = pd.to_numeric(df_period_daily.get("conv_ops_s2"), errors="coerce").dropna()
    clientes_end = float(clientes.iloc[-1]) if len(clientes) else np.nan
    conv2_end = float(conv2.iloc[-1]) if len(conv2) else np.nan
    margem_pct = _safe_div(mar, vol)
    ticket = _safe_div(vol, ops)
    return {
        "vol": float(vol) if vol is not None else np.nan,
        "mar": float(mar) if mar is not None else np.nan,
        "ops": float(ops) if ops is not None else np.nan,
        "clientes_end": clientes_end, "conv2_end": conv2_end,
        "margem_pct": margem_pct, "ticket": ticket,
    }

def _forecast_month_end(df_daily: pd.DataFrame, year: int, month: int, asof: pd.Timestamp) -> Dict[str, float]:
    m_start, m_end = _month_bounds(year, month)
    asof = min(asof.normalize(), m_end)
    df_mtd = _period_slice(df_daily, m_start, asof)
    k_mtd = _period_kpis_from_daily(df_mtd)
    days_in_month = (m_end - m_start).days + 1
    days_elapsed = (asof - m_start).days + 1
    scale = _safe_div(days_in_month, days_elapsed)
    vol_f = k_mtd["vol"] * scale if not np.isnan(k_mtd["vol"]) else np.nan
    mar_f = k_mtd["mar"] * scale if not np.isnan(k_mtd["mar"]) else np.nan
    ops_f = k_mtd["ops"] * scale if not np.isnan(k_mtd["ops"]) else np.nan
    margem_pct_f = _safe_div(mar_f, vol_f)
    ticket_f = _safe_div(vol_f, ops_f)
    return {
        "vol": vol_f, "mar": mar_f, "ops": ops_f,
        "clientes_end": k_mtd["clientes_end"], "conv2_end": k_mtd["conv2_end"],
        "margem_pct": margem_pct_f, "ticket": ticket_f,
        "days_in_month": days_in_month, "days_elapsed": days_elapsed,
    }

def _ytd_slice(df_daily: pd.DataFrame, year: int, asof: pd.Timestamp) -> pd.DataFrame:
    y_start = pd.Timestamp(year=year, month=1, day=1)
    return _period_slice(df_daily, y_start, asof.normalize())

def _forecast_year_end(df_daily: pd.DataFrame, year: int, asof: pd.Timestamp) -> Dict[str, float]:
    asof = asof.normalize()
    month = asof.month
    df_ytd = _ytd_slice(df_daily, year, asof)
    k_ytd = _period_kpis_from_daily(df_ytd)
    years = sorted(df_daily["data"].dt.year.dropna().unique().tolist())
    past_full = []
    for y in years:
        if y >= year:
            continue
        df_m = build_monthly_year(df_daily, y)
        if df_m.empty:
            continue
        has_dec = (
            pd.to_numeric(df_m.loc[df_m["mes_num"] == 12, "volume_negocios"], errors="coerce").notna().any()
            or pd.to_numeric(df_m.loc[df_m["mes_num"] == 12, "margem_liquida"], errors="coerce").notna().any()
            or pd.to_numeric(df_m.loc[df_m["mes_num"] == 12, "num_operacoes"], errors="coerce").notna().any()
        )
        if not has_dec:
            continue
        past_full.append((y, df_m))
    shares = []
    for y, df_m in past_full[-5:]:
        vol_total = pd.to_numeric(df_m.get("volume_negocios"), errors="coerce").sum()
        vol_cum = pd.to_numeric(df_m.loc[df_m["mes_num"] <= month, "volume_negocios"], errors="coerce").sum()
        if vol_total and not np.isnan(vol_total) and vol_total > 0:
            shares.append(_safe_div(vol_cum, vol_total))
            continue
        mar_total = pd.to_numeric(df_m.get("margem_liquida"), errors="coerce").sum()
        mar_cum = pd.to_numeric(df_m.loc[df_m["mes_num"] <= month, "margem_liquida"], errors="coerce").sum()
        if mar_total and not np.isnan(mar_total) and mar_total > 0:
            shares.append(_safe_div(mar_cum, mar_total))
            continue
        ops_total = pd.to_numeric(df_m.get("num_operacoes"), errors="coerce").sum()
        ops_cum = pd.to_numeric(df_m.loc[df_m["mes_num"] <= month, "num_operacoes"], errors="coerce").sum()
        if ops_total and not np.isnan(ops_total) and ops_total > 0:
            shares.append(_safe_div(ops_cum, ops_total))
    share_avg = np.nan
    if len(shares):
        shares = [s for s in shares if s is not None and not np.isnan(s) and 0.05 <= s <= 0.95]
        if len(shares):
            share_avg = float(np.mean(shares))
    doy = int(asof.dayofyear)
    days_in_year = 366 if calendar.isleap(year) else 365
    share_linear = _safe_div(doy, days_in_year)
    share_used = share_avg if (share_avg is not None and not np.isnan(share_avg) and share_avg > 0) else share_linear

    def scale_flow(v):
        return _safe_div(v, share_used) if (v is not None and not np.isnan(v) and share_used and share_used > 0) else np.nan

    vol_f = scale_flow(k_ytd["vol"])
    mar_f = scale_flow(k_ytd["mar"])
    ops_f = scale_flow(k_ytd["ops"])
    margem_pct_f = _safe_div(mar_f, vol_f)
    ticket_f = _safe_div(vol_f, ops_f)
    return {
        "vol": vol_f, "mar": mar_f, "ops": ops_f,
        "clientes_end": k_ytd["clientes_end"], "conv2_end": k_ytd["conv2_end"],
        "margem_pct": margem_pct_f, "ticket": ticket_f,
        "share_used": share_used,
        "share_mode": "sazonal" if (share_used == share_avg and not np.isnan(share_avg)) else "linear",
        "history_years": [y for y, _ in past_full[-5:]],
    }

# =============================================================================
# Comparações (média anos anteriores)
# =============================================================================
def _avg_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    keys = sorted({k for d in dicts for k in d.keys()})
    out = {}
    for k in keys:
        vals = []
        for d in dicts:
            v = d.get(k, np.nan)
            if v is None:
                continue
            try:
                v = float(v)
            except Exception:
                continue
            if not np.isnan(v):
                vals.append(v)
        out[k] = float(np.mean(vals)) if len(vals) else np.nan
    return out

def _pct_change(a, b):
    if b is None or (isinstance(b, float) and np.isnan(b)) or b == 0 or a is None or (isinstance(a, float) and np.isnan(a)):
        return np.nan
    return (float(a) - float(b)) / float(b)

def _pp_change(a, b):
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return np.nan
    return float(a) - float(b)

def _same_day_year(year: int, m: int, d: int) -> pd.Timestamp:
    last = calendar.monthrange(year, m)[1]
    return pd.Timestamp(year=year, month=m, day=min(d, last)).normalize()

def _baseline_mtd(df_daily, sel_year, sel_month, asof, years_all):
    prev_years = [y for y in years_all if y < sel_year]
    dicts = []
    used = []
    for y in prev_years[-5:]:
        m_start, m_end = _month_bounds(y, sel_month)
        asof_y = _same_day_year(y, sel_month, int(asof.day))
        asof_y = min(asof_y, m_end)
        df = _period_slice(df_daily, m_start, asof_y)
        k = _period_kpis_from_daily(df)
        if any([not np.isnan(k.get(x, np.nan)) for x in ("vol", "mar", "ops", "clientes_end")]):
            dicts.append(k)
            used.append(y)
    return _avg_dict(dicts), used

def _baseline_ytd(df_daily, sel_year, asof, years_all):
    prev_years = [y for y in years_all if y < sel_year]
    dicts = []
    used = []
    for y in prev_years[-5:]:
        asof_y = _same_day_year(y, int(asof.month), int(asof.day))
        df = _ytd_slice(df_daily, y, asof_y)
        k = _period_kpis_from_daily(df)
        if any([not np.isnan(k.get(x, np.nan)) for x in ("vol", "mar", "ops", "clientes_end")]):
            dicts.append(k)
            used.append(y)
    return _avg_dict(dicts), used

def _delta_html_pct(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    cls = "delta-flat"; arrow = "→"
    if p > 0.0005:
        cls = "delta-up"; arrow = "▲"
    elif p < -0.0005:
        cls = "delta-down"; arrow = "▼"
    sign = "+" if p >= 0 else ""
    return f"<span class='{cls}'>{arrow} {sign}{p*100:.0f}%</span>"

def _delta_html_pp(pp):
    if pp is None or (isinstance(pp, float) and np.isnan(pp)):
        return ""
    cls = "delta-flat"; arrow = "→"
    if pp > 0.0005:
        cls = "delta-up"; arrow = "▲"
    elif pp < -0.0005:
        cls = "delta-down"; arrow = "▼"
    sign = "+" if pp >= 0 else ""
    return f"<span class='{cls}'>{arrow} {sign}{pp*100:.1f} p.p.</span>"

# =============================================================================
# Forecast mensal sazonal
# =============================================================================
def _fallback_month_shares_days(year: int) -> Dict[int, float]:
    days = [calendar.monthrange(year, m)[1] for m in range(1, 13)]
    tot = float(sum(days))
    return {m: (days[m-1] / tot if tot else 1/12.0) for m in range(1, 13)}

def _monthly_shares_history(df_daily, metric_col, sel_year):
    years = sorted(df_daily["data"].dt.year.dropna().unique().tolist())
    past = [y for y in years if y < sel_year]
    shares_list = []
    for y in past[-5:]:
        df_m = build_monthly_year(df_daily, y)
        if df_m is None or df_m.empty or metric_col not in df_m.columns:
            continue
        if not (df_m["mes_num"] == 12).any():
            continue
        total = pd.to_numeric(df_m[metric_col], errors="coerce").sum()
        if total is None or np.isnan(total) or total <= 0:
            continue
        s = {}
        for m in range(1, 13):
            v = pd.to_numeric(df_m.loc[df_m["mes_num"] == m, metric_col], errors="coerce")
            v = float(v.sum()) if v.notna().any() else 0.0
            s[m] = v / float(total)
        ssum = sum(s.values())
        if ssum > 0:
            s = {m: s[m]/ssum for m in s}
        shares_list.append(s)
    if not shares_list:
        return _fallback_month_shares_days(sel_year)
    out = {}
    for m in range(1, 13):
        vals = [d.get(m, np.nan) for d in shares_list]
        vals = [v for v in vals if v is not None and not np.isnan(v) and v >= 0]
        out[m] = float(np.mean(vals)) if vals else np.nan
    fb = _fallback_month_shares_days(sel_year)
    for m in range(1, 13):
        if out.get(m) is None or np.isnan(out.get(m)):
            out[m] = fb[m]
    ssum = sum(out.values())
    if ssum > 0:
        out = {m: out[m]/ssum for m in out}
    return out

def _is_month_complete(year: int, month: int, asof: pd.Timestamp) -> bool:
    _, m_end = _month_bounds(year, month)
    return asof.normalize() >= m_end

def _apply_forecast_to_month_row(df_month, df_daily, year, month, asof):
    if df_month is None or df_month.empty:
        return df_month
    if month not in df_month["mes_num"].astype(int).tolist():
        return df_month
    if _is_month_complete(year, month, asof):
        return df_month
    k_est = _forecast_month_end(df_daily, year, month, asof)
    out = df_month.copy()
    mask = out["mes_num"].astype(int) == int(month)
    if mask.any():
        if "clientes_acesso" in out.columns:
            out.loc[mask, "clientes_acesso"] = k_est.get("clientes_end", np.nan)
        if "conv_ops_s2" in out.columns:
            out.loc[mask, "conv_ops_s2"] = k_est.get("conv2_end", np.nan)
        if "num_operacoes" in out.columns:
            out.loc[mask, "num_operacoes"] = k_est.get("ops", np.nan)
        if "volume_negocios" in out.columns:
            out.loc[mask, "volume_negocios"] = k_est.get("vol", np.nan)
        if "margem_liquida" in out.columns:
            out.loc[mask, "margem_liquida"] = k_est.get("mar", np.nan)
        out = _recompute_derived(out)
    return out

def _forecast_remaining_months(df_daily, sel_year, asof, metric_col, include_current_month=False):
    asof = asof.normalize()
    curr_m = int(asof.month)
    df_m = build_monthly_year(df_daily, sel_year)
    if df_m is None or df_m.empty or metric_col not in df_m.columns:
        return pd.DataFrame(columns=["mes_num", "mes", metric_col])
    df_m = _apply_forecast_to_month_row(df_m, df_daily, sel_year, curr_m, asof)
    ytd_vals = pd.to_numeric(df_m.loc[df_m["mes_num"] <= curr_m, metric_col], errors="coerce")
    ytd_est = float(ytd_vals.sum()) if ytd_vals.notna().any() else np.nan
    shares = _monthly_shares_history(df_daily, metric_col, sel_year)
    cum_share = sum(shares[m] for m in range(1, curr_m + 1))
    if ytd_est is None or np.isnan(ytd_est) or cum_share is None or cum_share <= 0:
        return pd.DataFrame(columns=["mes_num", "mes", metric_col])
    year_total_est = ytd_est / cum_share
    residual = max(year_total_est - ytd_est, 0.0)
    rem_share = sum(shares[m] for m in range(curr_m + 1, 13))
    if rem_share <= 0:
        return pd.DataFrame(columns=["mes_num", "mes", metric_col])
    rows = []
    if include_current_month and (not _is_month_complete(sel_year, curr_m, asof)):
        v_curr = pd.to_numeric(df_m.loc[df_m["mes_num"].astype(int) == curr_m, metric_col], errors="coerce")
        v_curr = float(v_curr.sum()) if v_curr.notna().any() else np.nan
        rows.append({"mes_num": curr_m, "mes": _PT_MONTHS.get(curr_m, str(curr_m)), metric_col: v_curr})
    for m in range(curr_m + 1, 13):
        v = residual * (shares[m] / rem_share)
        rows.append({"mes_num": m, "mes": _PT_MONTHS.get(m, str(m)), metric_col: float(v)})
    return pd.DataFrame(rows)

# =============================================================================
# Charts (Altair)
# =============================================================================
def _chart_yoy_line(df_a, year_a, df_b, year_b, value_col, title, y_title, color_a, color_b, is_pct=False, df_a_forecast=None):
    if df_a is None or df_a.empty or value_col not in df_a.columns:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line().encode(x="x", y="y").properties(height=240)
    frames = []
    d1 = df_a[["mes_num", "mes", value_col]].copy()
    d1[value_col] = pd.to_numeric(d1[value_col], errors="coerce")
    d1 = d1[d1[value_col].notna()].copy()
    d1["Ano"] = str(year_a); d1["Segmento"] = "Real"
    frames.append(d1)
    if df_a_forecast is not None and (not df_a_forecast.empty) and value_col in df_a_forecast.columns:
        d1f = df_a_forecast[["mes_num", "mes", value_col]].copy()
        d1f[value_col] = pd.to_numeric(d1f[value_col], errors="coerce")
        d1f = d1f[d1f[value_col].notna()].copy()
        if len(d1) and len(d1f):
            last_real_m = int(pd.to_numeric(d1["mes_num"], errors="coerce").max())
            first_fc_m = int(pd.to_numeric(d1f["mes_num"], errors="coerce").min())
            if first_fc_m > last_real_m:
                anchor = d1.loc[d1["mes_num"].astype(int) == last_real_m, ["mes_num", "mes", value_col]].tail(1).copy()
                if not anchor.empty:
                    anchor["Ano"] = str(year_a); anchor["Segmento"] = "Forecast"
                    d1f = pd.concat([anchor, d1f], ignore_index=True)
        d1f["Ano"] = str(year_a); d1f["Segmento"] = "Forecast"
        frames.append(d1f)
    if df_b is not None and (not df_b.empty) and value_col in df_b.columns and year_b is not None:
        d2 = df_b[["mes_num", "mes", value_col]].copy()
        d2[value_col] = pd.to_numeric(d2[value_col], errors="coerce")
        d2 = d2[d2[value_col].notna()].copy()
        d2["Ano"] = str(year_b); d2["Segmento"] = "Real"
        frames.append(d2)
    d = pd.concat(frames, ignore_index=True)
    if d.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line().encode(x="x", y="y").properties(height=240)
    if year_b is not None and df_b is not None and not df_b.empty:
        dom = [str(year_b), str(year_a)]; rng = [color_b, color_a]
    else:
        dom = [str(year_a)]; rng = [color_a]
    y_enc = alt.Y(f"{value_col}:Q", title=y_title, axis=alt.Axis(grid=True))
    if is_pct:
        y_enc = alt.Y(f"{value_col}:Q", title=y_title, axis=alt.Axis(grid=True, format="%"), scale=alt.Scale(domain=[0, 1]))
    fmt = ".1%" if is_pct else ",.0f"
    base = (
        alt.Chart(d)
        .encode(
            x=alt.X("mes:N", sort=alt.SortField(field="mes_num", order="ascending"), title=None),
            y=y_enc,
            color=alt.Color("Ano:N", scale=alt.Scale(domain=dom, range=rng), legend=alt.Legend(title=None, orient="top")),
            tooltip=[
                alt.Tooltip("Ano:N", title="Ano"),
                alt.Tooltip("mes:N", title="Mês"),
                alt.Tooltip(f"{value_col}:Q", title=title, format=fmt),
                alt.Tooltip("Segmento:N", title="Segmento"),
            ],
        )
        .properties(height=240)
    )
    ch_real = base.transform_filter(alt.datum.Segmento == "Real").mark_line(point=alt.OverlayMarkDef(size=70), interpolate="monotone")
    ch_fc = base.transform_filter(alt.datum.Segmento == "Forecast").mark_line(interpolate="monotone", strokeDash=[6, 4], point=alt.OverlayMarkDef(size=55))
    return (ch_real + ch_fc)

# =============================================================================
# UI components
# =============================================================================
def _kpi_grid(items: List[Tuple[str, str, str]]):
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-grid'>", unsafe_allow_html=True)
    for title, value, note in items:
        st.markdown(
            f"""<div class="kpi">
<div class="kpi-title">{title}</div>
<div class="kpi-value">{value}</div>
<div class="kpi-note">{note}</div>
</div>""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def _with_estimativa(real_str: str, est_str: str, show_est: bool) -> str:
    """FIX 6: Só mostra a estimativa (micro-pill) quando o período está incompleto."""
    if not show_est:
        return real_str
    est_str = "—" if est_str is None else est_str
    return f"{real_str}<span class='kpi-est'>Est.: {est_str}</span>"

# =============================================================================
# Estado / Tema
# =============================================================================
SHOW_MONTH_TABLE = True

if "pos_scope" not in st.session_state:
    st.session_state.pos_scope = "Mês"
if "sel_year" not in st.session_state:
    st.session_state.sel_year = None
if "sel_month" not in st.session_state:
    st.session_state.sel_month = None
if "asof_date" not in st.session_state:
    st.session_state.asof_date = None
# FIX 2: cache do ficheiro CSV entre sessões via session_state (persiste enquanto a sessão está activa)
if "raw_report_bytes" not in st.session_state:
    st.session_state.raw_report_bytes = None
if "include_curr_month_summary" not in st.session_state:
    st.session_state.include_curr_month_summary = False
if "include_year_forecast" not in st.session_state:
    st.session_state.include_year_forecast = False


st.session_state.show_daily_detail = True

# tema fixo: Light
_theme_vars = _apply_theme("Light")

# =============================================================================
# Topo: logo + título
# =============================================================================
c_logo, c_title = st.columns([0.12, 2.88])
with c_logo:
    try:
        st.image(LOGO_URL, width=82)
    except Exception:
        if LOGO_LOCAL:
            try:
                st.image(LOGO_LOCAL, width=82)
            except Exception:
                pass
with c_title:
    st.markdown(
        """<div class='topbar'>
<div>
<div class='topbar-title'>Plataforma Cambial — Dashboard</div>
<div class='topbar-sub'>FNC</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

# =============================================================================
# Carregar dados — usa session_state para persistência (FIX 2)
# =============================================================================
raw = st.session_state.raw_report_bytes

if raw is None and PERSISTENT_CSV.exists():
    raw = PERSISTENT_CSV.read_bytes()
    st.session_state.raw_report_bytes = raw
df_daily = None
_load_err = None

if raw is not None:
    try:
        df_daily = load_report(raw)
    except Exception as e:
        _load_err = e
        df_daily = None

if df_daily is None or (isinstance(df_daily, pd.DataFrame) and df_daily.empty):
    st.info("Sem dados carregados. No final do dashboard existe o bloco para inserir o CSV.")
    if _load_err is not None:
        st.error(f"Erro ao ler o CSV atual: {_load_err}")

# =============================================================================
# Se houver dados, renderiza dashboard completo
# =============================================================================
if df_daily is not None and not df_daily.empty:
    last_date = df_daily["data"].max().normalize()
    years_all = sorted(df_daily["data"].dt.year.dropna().unique().tolist())
    last_year = int(last_date.year)
    last_month = int(last_date.month)

    # ==========================================================================
    # FIX 1: Default = último mês FINALIZADO (não o mês corrente incompleto)
    # ==========================================================================
    def _last_complete_month(df: pd.DataFrame) -> Tuple[int, int]:
        """Devolve (year, month) do último mês completo nos dados."""
        ld = df["data"].max().normalize()
        y = int(ld.year)
        m = int(ld.month)
        _, m_end = _month_bounds(y, m)
        # Se o último mês não está completo, recua um mês
        if ld.normalize() < m_end.normalize():
            if m == 1:
                y -= 1
                m = 12
            else:
                m -= 1
        return y, m

    if st.session_state.sel_year is None:
        _def_year, _def_month = _last_complete_month(df_daily)
        st.session_state.sel_year = _def_year
        st.session_state.sel_month = _def_month
        st.session_state.asof_date = _month_bounds(_def_year, _def_month)[1].date()

    if len(years_all) == 0:
        st.warning("Sem anos válidos no ficheiro.")
    else:
        # ======================================================================
        # Controlos
        # ======================================================================
        st.markdown("<div class='ctrl'>", unsafe_allow_html=True)

        def _max_date_in_month(year: int, month: int) -> pd.Timestamp:
            m0, m1 = _month_bounds(year, month)
            dfx = df_daily[(df_daily["data"] >= m0) & (df_daily["data"] <= m1)]
            if dfx.empty:
                return min(last_date, m1)
            return dfx["data"].max().normalize()

        def _max_date_in_year(year: int) -> pd.Timestamp:
            dfx = df_daily[df_daily["data"].dt.year == year]
            if dfx.empty:
                return last_date
            return dfx["data"].max().normalize()

        def _months_in_year(year: int) -> List[int]:
            ms = sorted(df_daily[df_daily["data"].dt.year == year]["data"].dt.month.unique().tolist())
            return ms if ms else list(range(1, 13))

        if "pos_scope" not in st.session_state or st.session_state.pos_scope not in ("Mês", "Ano"):
            st.session_state.pos_scope = "Mês"

        def _on_year_change():
            y = int(st.session_state.sel_year)

            if st.session_state.pos_scope == "Mês":
                ms = _months_in_year(y)
                curr_month = st.session_state.get("sel_month")

                if curr_month is None or int(curr_month) not in ms:
                    st.session_state.sel_month = ms[-1]

                st.session_state.asof_date = _max_date_in_month(
                    y, int(st.session_state.sel_month)
                ).date()
            else:
                st.session_state.asof_date = _max_date_in_year(y).date()

        def _on_scope_change():
            y = int(st.session_state.sel_year)

            if st.session_state.pos_scope == "Ano":
                st.session_state.asof_date = _max_date_in_year(y).date()
            else:
                ms = _months_in_year(y)
                curr_month = st.session_state.get("sel_month")

                if curr_month is None or int(curr_month) not in ms:
                    st.session_state.sel_month = ms[-1]

                st.session_state.asof_date = _max_date_in_month(
                    y, int(st.session_state.sel_month)
                ).date()

        def _on_month_change():
            y = int(st.session_state.sel_year)
            m = st.session_state.get("sel_month")

            if m is None:
                return

            st.session_state.asof_date = _max_date_in_month(y, int(m)).date()

        r1, r2, r3 = st.columns([1.15, 1.05, 2.35])
        with r1:
            st.radio("Posição", ["Mês", "Ano"], horizontal=True, key="pos_scope", on_change=_on_scope_change)
        with r2:
            st.selectbox("Ano", years_all, key="sel_year", on_change=_on_year_change)

        pos_scope = st.session_state.pos_scope
        sel_year = int(st.session_state.sel_year)

        if pos_scope == "Mês":
            with r3:
                ms = _months_in_year(sel_year)
                curr_month = st.session_state.get("sel_month")
                if curr_month is None or int(curr_month) not in ms:
                    st.session_state.sel_month = ms[-1]
                st.selectbox("Mês", ms, format_func=lambda m: f"{_PT_MONTHS.get(int(m), str(m))}", key="sel_month", on_change=_on_month_change)
        else:
            with r3:
                st.markdown(f"<div class='subtle'>Até: <b>{_max_date_in_year(sel_year).date()}</b></div>", unsafe_allow_html=True)

        st.markdown(f"<div class='subtle'>Última data no ficheiro: <b>{last_date.date()}</b></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        asof_date = pd.Timestamp(st.session_state.asof_date).normalize()
        sel_month = int(st.session_state.sel_month) if st.session_state.pos_scope == "Mês" else int(last_month)

        st.divider()

        # ======================================================================
        # Posição principal (Mês / Ano)
        # ======================================================================
        if pos_scope == "Mês":
            m_start, m_end = _month_bounds(sel_year, sel_month)
            asof_m = min(asof_date, m_end)
            df_mtd = _period_slice(df_daily, m_start, asof_m)
            k_real = _period_kpis_from_daily(df_mtd)
            base, used_years = _baseline_mtd(df_daily, sel_year, sel_month, asof_m, years_all)
            bench_txt = f"vs média MTD ({', '.join(map(str, used_years))})" if used_years else "vs MTD anos anteriores"

            # FIX 6: só calcula/mostra estimativa se mês incompleto
            month_is_complete = _is_month_complete(sel_year, sel_month, asof_m)
            show_est = not month_is_complete
            k_est = _forecast_month_end(df_daily, sel_year, sel_month, asof_m) if show_est else {}

            vol_d = _pct_change(k_real["vol"], base.get("vol", np.nan))
            mar_d = _pct_change(k_real["mar"], base.get("mar", np.nan))
            ops_d = _pct_change(k_real["ops"], base.get("ops", np.nan))
            cli_d = _pct_change(k_real["clientes_end"], base.get("clientes_end", np.nan))
            pp_d = _pp_change(k_real["conv2_end"], base.get("conv2_end", np.nan))

            st.markdown(f"### {_PT_MONTHS_FULL.get(sel_month, _PT_MONTHS.get(sel_month, str(sel_month)))} {sel_year}")
            if asof_m.normalize() < m_end.normalize():
                st.markdown(f"<div class='subtle'>Até <b>{asof_m.date()}</b></div>", unsafe_allow_html=True)

            _kpi_grid([
                ("Clientes com acesso",
                 _with_estimativa(_fmt_int(k_real["clientes_end"]), _fmt_int(k_est.get("clientes_end")), show_est),
                 _delta_html_pct(cli_d) + f" <span class='subtle'>{bench_txt}</span>"),
                ("Clientes com operações",
                 _with_estimativa(_fmt_pct(k_real["conv2_end"], 1), _fmt_pct(k_est.get("conv2_end"), 1), show_est),
                 _delta_html_pp(pp_d) + f" <span class='subtle'>{bench_txt}</span>"),
                ("Nº Operações",
                 _with_estimativa(_fmt_int_compact(k_real["ops"], 1), _fmt_int_compact(k_est.get("ops"), 1), show_est),
                 _delta_html_pct(ops_d) + f" <span class='subtle'>{bench_txt}</span>"),
                ("Volume",
                 _with_estimativa(_fmt_eur_compact(k_real["vol"], 1), _fmt_eur_compact(k_est.get("vol"), 1), show_est),
                 _delta_html_pct(vol_d) + f" <span class='subtle'>{bench_txt}</span>"),
                ("Margem",
                 _with_estimativa(_fmt_eur_compact(k_real["mar"], 1), _fmt_eur_compact(k_est.get("mar"), 1), show_est),
                 _delta_html_pct(mar_d) + f" <span class='subtle'>{bench_txt}</span>"),
            ])
        else:
            # Ano
            asof_y = asof_date
            df_ytd = _ytd_slice(df_daily, sel_year, asof_y)
            k_real = _period_kpis_from_daily(df_ytd)
            base, used_years = _baseline_ytd(df_daily, sel_year, asof_y, years_all)
            bench_txt = f"vs média YTD ({', '.join(map(str, used_years))})" if used_years else "vs anos anteriores"

            # FIX 6: só mostra estimativa se ano incompleto
            y_end_ts = pd.Timestamp(year=sel_year, month=12, day=31).normalize()
            year_is_complete = (asof_y.normalize() >= y_end_ts)
            show_est = not year_is_complete
            k_est = _forecast_year_end(df_daily, sel_year, asof_y) if show_est else {}

            vol_d = _pct_change(k_real["vol"], base.get("vol", np.nan))
            mar_d = _pct_change(k_real["mar"], base.get("mar", np.nan))
            ops_d = _pct_change(k_real["ops"], base.get("ops", np.nan))
            cli_d = _pct_change(k_real["clientes_end"], base.get("clientes_end", np.nan))
            pp_d = _pp_change(k_real["conv2_end"], base.get("conv2_end", np.nan))

            st.markdown(f"### {sel_year}")
            if asof_y.normalize() < y_end_ts:
                st.markdown(f"<div class='subtle'>Até <b>{asof_y.date()}</b></div>", unsafe_allow_html=True)

            _kpi_grid([
                ("Clientes com acesso",
                 _with_estimativa(_fmt_int(k_real["clientes_end"]), _fmt_int(k_est.get("clientes_end")), show_est),
                 _delta_html_pct(cli_d) + f" <span class='subtle'>{bench_txt}</span>"),
                ("Clientes com operações",
                 _with_estimativa(_fmt_pct(k_real["conv2_end"], 1), _fmt_pct(k_est.get("conv2_end"), 1), show_est),
                 _delta_html_pp(pp_d) + f" <span class='subtle'>{bench_txt}</span>"),
                ("Nº Operações",
                 _with_estimativa(_fmt_int_compact(k_real["ops"], 1), _fmt_int_compact(k_est.get("ops"), 1), show_est),
                 _delta_html_pct(ops_d) + f" <span class='subtle'>{bench_txt}</span>"),
                ("Volume",
                 _with_estimativa(_fmt_eur_compact(k_real["vol"], 1), _fmt_eur_compact(k_est.get("vol"), 1), show_est),
                 _delta_html_pct(vol_d) + f" <span class='subtle'>{bench_txt}</span>"),
                ("Margem",
                 _with_estimativa(_fmt_eur_compact(k_real["mar"], 1), _fmt_eur_compact(k_est.get("mar"), 1), show_est),
                 _delta_html_pct(mar_d) + f" <span class='subtle'>{bench_txt}</span>"),
            ])

        # ======================================================================
        # Resumo mensal (HTML)
        # ======================================================================
        if SHOW_MONTH_TABLE:
            st.divider()
            st.markdown(f"### {sel_year} — Resumo mensal")

            asof_in_year = _max_date_in_year(sel_year)
            curr_m = int(asof_in_year.month)
            is_current_year_in_file = (sel_year == int(last_date.year))
            month_in_progress = (not _is_month_complete(sel_year, curr_m, asof_in_year))

            include_curr_month = st.toggle(
                f"Incluir mês corrente ({_PT_MONTHS_FULL.get(curr_m, _PT_MONTHS.get(curr_m, str(curr_m)))})",
                value=month_in_progress,   # ✅ só true se o mês estiver incompleto
                key="include_curr_month_summary",
                disabled=not is_current_year_in_file,
            )

            is_month_estimated = bool(is_current_year_in_file and include_curr_month and month_in_progress)
            if is_month_estimated:
                st.markdown(
                    f"<div class='subtle'>*Estimando o mês de <b>{_PT_MONTHS_FULL.get(curr_m, _PT_MONTHS.get(curr_m, str(curr_m)))}</b></div>",
                    unsafe_allow_html=True,
                )

            compare_year = sel_year - 1
            has_ly = compare_year in years_all
            df_month = build_monthly_year(df_daily, sel_year)

            if df_month is None or df_month.empty:
                st.info("Sem dados para o ano selecionado.")
            else:
                if is_month_estimated:
                    df_month = _apply_forecast_to_month_row(df_month, df_daily, sel_year, curr_m, asof_in_year)

                base_num = df_month[[c for c in ["clientes_acesso", "conv_ops_s2", "num_operacoes", "volume_negocios", "margem_liquida"] if c in df_month.columns]].copy()
                keep_any = base_num.apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
                dfm = df_month.loc[keep_any].copy()

                if dfm.empty:
                    st.info("Sem dados numéricos no ano selecionado.")
                else:
                    max_month_with_data = int(dfm["mes_num"].max())
                    cutoff_month = max_month_with_data

                    if is_current_year_in_file and month_in_progress and (not include_curr_month):
                        cutoff_month = min(cutoff_month, curr_m - 1)

                    if cutoff_month < 1:
                        st.info("Mês corrente excluído — ainda não há meses completos para mostrar no resumo mensal.")
                    else:
                        dfm = dfm[dfm["mes_num"].astype(int) <= int(cutoff_month)].copy()
                        dfm["is_est"] = False
                        if is_month_estimated:
                            dfm.loc[dfm["mes_num"].astype(int) == curr_m, "is_est"] = True

                        if has_ly:
                            df_ly = build_monthly_year(df_daily, compare_year)
                            df_ly = df_ly[["mes_num", "clientes_acesso", "conv_ops_s2", "num_operacoes", "volume_negocios", "margem_liquida"]].copy() if not df_ly.empty else pd.DataFrame()
                        else:
                            df_ly = pd.DataFrame()

                        if not df_ly.empty:
                            dfm = dfm.merge(df_ly, on="mes_num", how="left", suffixes=("", "_ly"))
                        else:
                            for c in ["clientes_acesso_ly", "conv_ops_s2_ly", "num_operacoes_ly", "volume_negocios_ly", "margem_liquida_ly"]:
                                dfm[c] = np.nan

                        def _arrow_pct(delta):
                            if delta is None or (isinstance(delta, float) and np.isnan(delta)):
                                return "", "flat"
                            if delta > 0.0005:
                                return f"▲ {delta*100:.0f}%", "up"
                            if delta < -0.0005:
                                return f"▼ {delta*100:.0f}%", "down"
                            return "→ 0%", "flat"

                        def _arrow_pp(delta_pp):
                            if delta_pp is None or (isinstance(delta_pp, float) and np.isnan(delta_pp)):
                                return "", "flat"
                            if delta_pp > 0.0005:
                                return f"▲ +{delta_pp*100:.1f} p.p.", "up"
                            if delta_pp < -0.0005:
                                return f"▼ {delta_pp*100:.1f} p.p.", "down"
                            return "→ 0.0 p.p.", "flat"

                        head = f"""
<div class='panel'>
<div class='subtle'>Comparação com {compare_year}{'' if has_ly else ' (indisponível)'}</div>
<div class='tbl-wrap'>
<table class='tbl'>
<thead>
<tr>
<th>Mês</th>
<th>Clientes com acesso</th>
<th class='pctcell'>Clientes com operações</th>
<th>Nº de Operações</th>
<th>Volume</th>
<th>Margem</th>
</tr>
</thead>
<tbody>
"""
                        body = ""
                        for _, r in dfm.iterrows():
                            mnum = int(r.get("mes_num"))
                            mes = _PT_MONTHS.get(mnum, str(mnum))
                            if bool(r.get("is_est", False)):
                                mes = f"{mes}<span class='subtle'>*</span>"
                            cli = r.get("clientes_acesso"); ado = r.get("conv_ops_s2")
                            ops = r.get("num_operacoes"); vol = r.get("volume_negocios"); mar = r.get("margem_liquida")
                            cli_ly = r.get("clientes_acesso_ly"); ado_ly = r.get("conv_ops_s2_ly")
                            ops_ly = r.get("num_operacoes_ly"); vol_ly = r.get("volume_negocios_ly"); mar_ly = r.get("margem_liquida_ly")
                            d_cli = _pct_change(cli, cli_ly) if has_ly else np.nan
                            d_ops = _pct_change(ops, ops_ly) if has_ly else np.nan
                            d_vol = _pct_change(vol, vol_ly) if has_ly else np.nan
                            d_mar = _pct_change(mar, mar_ly) if has_ly else np.nan
                            d_ado_pp = _pp_change(ado, ado_ly) if has_ly else np.nan
                            a_cli_txt, a_cli_cls = _arrow_pct(d_cli); a_ops_txt, a_ops_cls = _arrow_pct(d_ops)
                            a_vol_txt, a_vol_cls = _arrow_pct(d_vol); a_mar_txt, a_mar_cls = _arrow_pct(d_mar)
                            a_ado_txt, a_ado_cls = _arrow_pp(d_ado_pp)
                            p = 0.0
                            try:
                                if ado is not None and not (isinstance(ado, float) and np.isnan(ado)):
                                    v = float(str(ado).strip().replace("%", "").replace(",", "."))
                                    if v > 1.0: v = v / 100.0
                                    p = max(0.0, min(1.0, v))
                            except Exception:
                                p = 0.0
                            w = round(p * 100, 1)
                            body += f"""
<tr>
<td><b>{mes}</b></td>
<td>{_fmt_int(cli)}{('<span class="arrow '+a_cli_cls+'">'+a_cli_txt+'</span>') if a_cli_txt else ''}</td>
<td class='pctcell'>
<span class='bar'><span class='fill' style='width:{w}%'></span></span>
<span class='pcttxt'>{_fmt_pct(ado, 1)}{('<span class="arrow '+a_ado_cls+'">'+a_ado_txt+'</span>') if a_ado_txt else ''}</span>
</td>
<td>{_fmt_int_compact(ops, 1)}{('<span class="arrow '+a_ops_cls+'">'+a_ops_txt+'</span>') if a_ops_txt else ''}</td>
<td>{_fmt_eur_compact(vol, 1)}{('<span class="arrow '+a_vol_cls+'">'+a_vol_txt+'</span>') if a_vol_txt else ''}</td>
<td>{_fmt_eur_compact(mar, 1)}{('<span class="arrow '+a_mar_cls+'">'+a_mar_txt+'</span>') if a_mar_txt else ''}</td>
</tr>
"""
                        # --- Linha YTD ---
                        ytd_end_m = int(cutoff_month)
                        df_ytdm = df_month.copy()
                        if is_month_estimated:
                            df_ytdm = _apply_forecast_to_month_row(df_ytdm, df_daily, sel_year, curr_m, asof_in_year)
                        df_ytdm = df_ytdm[df_ytdm["mes_num"].astype(int) <= ytd_end_m].copy().sort_values("mes_num")
                        last_row = df_ytdm.loc[df_ytdm["mes_num"].astype(int) == ytd_end_m].tail(1)

                        ytd_cli = float(pd.to_numeric(last_row["clientes_acesso"], errors="coerce").iloc[0]) if (not last_row.empty and "clientes_acesso" in last_row.columns) else np.nan
                        ytd_ado = float(pd.to_numeric(last_row["conv_ops_s2"], errors="coerce").iloc[0]) if (not last_row.empty and "conv_ops_s2" in last_row.columns) else np.nan
                        ytd_ops = float(pd.to_numeric(df_ytdm.get("num_operacoes"), errors="coerce").sum()) if "num_operacoes" in df_ytdm.columns else np.nan
                        ytd_vol = float(pd.to_numeric(df_ytdm.get("volume_negocios"), errors="coerce").sum()) if "volume_negocios" in df_ytdm.columns else np.nan
                        ytd_mar = float(pd.to_numeric(df_ytdm.get("margem_liquida"), errors="coerce").sum()) if "margem_liquida" in df_ytdm.columns else np.nan

                        if has_ly:
                            df_lym = build_monthly_year(df_daily, compare_year)
                            df_lym = df_lym[df_lym["mes_num"].astype(int) <= ytd_end_m].copy().sort_values("mes_num") if not df_lym.empty else pd.DataFrame()
                            last_row_ly = df_lym.loc[df_lym["mes_num"].astype(int) == ytd_end_m].tail(1) if (df_lym is not None and not df_lym.empty) else pd.DataFrame()
                            ly_cli = float(pd.to_numeric(last_row_ly["clientes_acesso"], errors="coerce").iloc[0]) if (not last_row_ly.empty and "clientes_acesso" in last_row_ly.columns) else np.nan
                            ly_ado = float(pd.to_numeric(last_row_ly["conv_ops_s2"], errors="coerce").iloc[0]) if (not last_row_ly.empty and "conv_ops_s2" in last_row_ly.columns) else np.nan
                            ly_ops = float(pd.to_numeric(df_lym.get("num_operacoes"), errors="coerce").sum()) if (df_lym is not None and not df_lym.empty and "num_operacoes" in df_lym.columns) else np.nan
                            ly_vol = float(pd.to_numeric(df_lym.get("volume_negocios"), errors="coerce").sum()) if (df_lym is not None and not df_lym.empty and "volume_negocios" in df_lym.columns) else np.nan
                            ly_mar = float(pd.to_numeric(df_lym.get("margem_liquida"), errors="coerce").sum()) if (df_lym is not None and not df_lym.empty and "margem_liquida" in df_lym.columns) else np.nan
                        else:
                            ly_cli = ly_ado = ly_ops = ly_vol = ly_mar = np.nan

                        d_ytd_cli = _pct_change(ytd_cli, ly_cli) if has_ly else np.nan
                        d_ytd_ops = _pct_change(ytd_ops, ly_ops) if has_ly else np.nan
                        d_ytd_vol = _pct_change(ytd_vol, ly_vol) if has_ly else np.nan
                        d_ytd_mar = _pct_change(ytd_mar, ly_mar) if has_ly else np.nan
                        d_ytd_ado_pp = _pp_change(ytd_ado, ly_ado) if has_ly else np.nan
                        a_cli_txt, a_cli_cls = _arrow_pct(d_ytd_cli); a_ops_txt, a_ops_cls = _arrow_pct(d_ytd_ops)
                        a_vol_txt, a_vol_cls = _arrow_pct(d_ytd_vol); a_mar_txt, a_mar_cls = _arrow_pct(d_ytd_mar)
                        a_ado_txt, a_ado_cls = _arrow_pp(d_ytd_ado_pp)
                        p = 0.0
                        try:
                            if ytd_ado is not None and not (isinstance(ytd_ado, float) and np.isnan(ytd_ado)):
                                v = float(ytd_ado)
                                if v > 1.0: v = v / 100.0
                                p = max(0.0, min(1.0, v))
                        except Exception:
                            p = 0.0
                        w = round(p * 100, 1)
                        body += f"""
<tr class='ytd-row'>
<td><b>{sel_year}</b></td>
<td><b>{_fmt_int(ytd_cli)}{('<span class="arrow '+a_cli_cls+'">'+a_cli_txt+'</span>') if a_cli_txt else ''}</b></td>
<td class='pctcell'>
<span class='bar'><span class='fill' style='width:{w}%'></span></span>
<span class='pcttxt'><b>{_fmt_pct(ytd_ado, 1)}{('<span class="arrow '+a_ado_cls+'">'+a_ado_txt+'</span>') if a_ado_txt else ''}</b></span>
</td>
<td><b>{_fmt_int_compact(ytd_ops, 1)}{('<span class="arrow '+a_ops_cls+'">'+a_ops_txt+'</span>') if a_ops_txt else ''}</b></td>
<td><b>{_fmt_eur_compact(ytd_vol, 1)}{('<span class="arrow '+a_vol_cls+'">'+a_vol_txt+'</span>') if a_vol_txt else ''}</b></td>
<td><b>{_fmt_eur_compact(ytd_mar, 1)}{('<span class="arrow '+a_mar_cls+'">'+a_mar_txt+'</span>') if a_mar_txt else ''}</b></td>
</tr>
"""
                        tail = """</tbody></table></div></div>"""
                        st.html(head + body + tail)

        # ======================================================================
        # Gráficos (YoY)
        # ======================================================================
        st.divider()
        st.markdown("### Evolução mensal")
        prev_year = sel_year - 1
        df_curr_y = build_monthly_year(df_daily, sel_year)
        asof_in_year = _max_date_in_year(sel_year)
        curr_m = int(asof_in_year.month)
        month_in_progress = not _is_month_complete(sel_year, curr_m, asof_in_year)

        if month_in_progress:
            df_curr_y = df_curr_y[df_curr_y["mes_num"].astype(int) < curr_m].copy()

        df_prev_y = build_monthly_year(df_daily, prev_year) if prev_year in years_all else pd.DataFrame()

        if df_curr_y is None or df_curr_y.empty:
            st.info("Sem dados para gráficos.")
        else:
            base_num = df_curr_y[[c for c in ["num_operacoes", "volume_negocios", "margem_liquida"] if c in df_curr_y.columns]].copy()
            keep_any = base_num.apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
            df_curr_y = df_curr_y.loc[keep_any].copy()

            curr_m = int(asof_in_year.month)
            month_in_progress = not _is_month_complete(sel_year, curr_m, asof_in_year)
            if month_in_progress:
                df_curr_y = df_curr_y[df_curr_y["mes_num"].astype(int) < curr_m].copy()

            if df_curr_y.empty:
                st.info("Sem dados para gráficos.")
            else:
                if df_prev_y is not None and not df_prev_y.empty:
                    base_num2 = df_prev_y[[c for c in ["num_operacoes", "volume_negocios", "margem_liquida"] if c in df_prev_y.columns]].copy()
                    keep_any2 = base_num2.apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
                    df_prev_y = df_prev_y.loc[keep_any2].copy()

                _, y_end = _month_bounds(sel_year, 12)
                fc_vol = _forecast_remaining_months(df_daily, sel_year, asof_in_year, "volume_negocios", include_current_month=True)
                fc_ops = _forecast_remaining_months(df_daily, sel_year, asof_in_year, "num_operacoes", include_current_month=True)
                fc_mar = _forecast_remaining_months(df_daily, sel_year, asof_in_year, "margem_liquida", include_current_month=True)

                g1, g2, g3 = st.columns(3)
                with g1:
                    st.markdown("<div class='panel'>", unsafe_allow_html=True)
                    st.markdown("**Volume de negócios**")
                    st.altair_chart(
                        _chart_yoy_line(df_curr_y, sel_year, df_prev_y if (df_prev_y is not None and not df_prev_y.empty) else None,
                                        prev_year if (df_prev_y is not None and not df_prev_y.empty) else None,
                                        "volume_negocios", "Volume", "€", _theme_vars["warn"], _theme_vars["accent2"],
                                        df_a_forecast=fc_vol if (fc_vol is not None and not fc_vol.empty) else None),
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with g2:
                    st.markdown("<div class='panel'>", unsafe_allow_html=True)
                    st.markdown("**Nº de operações**")
                    st.altair_chart(
                        _chart_yoy_line(df_curr_y, sel_year, df_prev_y if (df_prev_y is not None and not df_prev_y.empty) else None,
                                        prev_year if (df_prev_y is not None and not df_prev_y.empty) else None,
                                        "num_operacoes", "Operações", "Nº", _theme_vars["accent"], _theme_vars["accent2"],
                                        df_a_forecast=fc_ops if (fc_ops is not None and not fc_ops.empty) else None),
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                with g3:
                    st.markdown("<div class='panel'>", unsafe_allow_html=True)
                    st.markdown("**Margem líquida**")
                    st.altair_chart(
                        _chart_yoy_line(df_curr_y, sel_year, df_prev_y if (df_prev_y is not None and not df_prev_y.empty) else None,
                                        prev_year if (df_prev_y is not None and not df_prev_y.empty) else None,
                                        "margem_liquida", "Margem", "€", _theme_vars["good"], _theme_vars["accent2"],
                                        df_a_forecast=fc_mar if (fc_mar is not None and not fc_mar.empty) else None),
                        use_container_width=True,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
        # =============================================================================
        # Resumo anual (HTML) — últimos 5 anos, mais recente -> mais antigo, * para ano incompleto
        # =============================================================================
        st.divider()
        st.markdown("### Resumo anual")
        years_back = st.selectbox(
            "Número de anos a mostrar",
            options=[5, 7, 10, 15, len(years_all)],
            index=0,
            format_func=lambda x: "Todos" if x == len(years_all) else str(x),
        )
        include_year_forecast = st.toggle(
            f"Incluir estimativa {int(last_date.year)}",
            value=False,
            key="include_year_forecast",
        )
        asof_last_year = _max_date_in_year(int(last_date.year))
        year_end_ts = pd.Timestamp(year=int(last_date.year), month=12, day=31).normalize()
        year_in_progress = asof_last_year.normalize() < year_end_ts

        if include_year_forecast and year_in_progress:
            st.markdown(
                f"<div class='subtle'>*Estimando o ano de <b>{int(last_date.year)}</b></div>",
                unsafe_allow_html=True,
            )
        # --- construir mapa com TODOS os anos (para comparar YoY mesmo se só mostramos 5) ---
        years_all_sorted = sorted(years_all)
        year_rows = []
        ytd_note_date = None  # para mostrar no subtítulo

        for y in years_all_sorted:
            y_start = pd.Timestamp(year=y, month=1, day=1)
            y_end = pd.Timestamp(year=y, month=12, day=31)

            if y == int(last_date.year):
                asof_real = _max_date_in_year(y)
                is_ytd = asof_real.normalize() < y_end.normalize()

                if is_ytd:
                    ytd_note_date = asof_real.date()

                # ✅ Toggle: se ano estiver incompleto e toggle ligado -> usa forecast anual
                if is_ytd and st.session_state.include_year_forecast:
                    k = _forecast_year_end(df_daily, y, asof_real)
                    asof_y = asof_real
                else:
                    df_y = _period_slice(df_daily, y_start, asof_real)
                    k = _period_kpis_from_daily(df_y)
                    asof_y = asof_real
            else:
                asof_y = y_end
                is_ytd = False
                df_y = _period_slice(df_daily, y_start, asof_y)
                k = _period_kpis_from_daily(df_y)

            year_rows.append({
                "ano": int(y),
                "is_ytd": bool(is_ytd),   # continua True no ano incompleto (vai ficar com *)
                "asof": asof_y.date(),
                **k,
            })

        df_years_all = pd.DataFrame(year_rows)
        year_map = {int(r["ano"]): r for _, r in df_years_all.iterrows()}

        # --- últimos N anos, do mais antigo para o mais recente ---
        years_display = sorted(years_all)[-years_back:]
        df_display = df_years_all[df_years_all["ano"].isin(years_display)].copy()
        df_display = df_display.sort_values("ano", ascending=True).reset_index(drop=True)
        

        # --- helpers de setas (reutiliza os mesmos do mensal) ---
        def _arrow_pct(delta):
            if delta is None or (isinstance(delta, float) and np.isnan(delta)):
                return "", "flat"
            if delta > 0.0005:
                return f"▲ {delta*100:.0f}%", "up"
            if delta < -0.0005:
                return f"▼ {delta*100:.0f}%", "down"
            return "→ 0%", "flat"

        def _arrow_pp(delta_pp):
            if delta_pp is None or (isinstance(delta_pp, float) and np.isnan(delta_pp)):
                return "", "flat"
            if delta_pp > 0.0005:
                return f"▲ +{delta_pp*100:.1f} p.p.", "up"
            if delta_pp < -0.0005:
                return f"▼ {delta_pp*100:.1f} p.p.", "down"
            return "→ 0.0 p.p.", "flat"

        sub_note = ""
        if ytd_note_date is not None:
            sub_note = f" &nbsp; <span class='subtle'>* Dados até <b>{ytd_note_date}</b></span>"

        head = f"""
        <div class='panel'>
        <div class='subtle'>Comparação anual {sub_note}</div>
        <div class='tbl-wrap'>
        <table class='tbl'>
        <thead>
        <tr>
        <th>Ano</th>
        <th>Clientes com acesso</th>
        <th class='pctcell'>Clientes com operações</th>
        <th>Nº de operações</th>
        <th>Volume</th>
        <th>Margem</th>
        </tr>
        </thead>
        <tbody>
        """

        body = ""

        for _, r in df_display.iterrows():
            y = int(r["ano"])
            prev = year_map.get(y - 1, None)

            is_incomplete_year = bool(r.get("is_ytd", False))

            ano_label = f"{y}*" if is_incomplete_year else str(y)

            cli = r.get("clientes_end", np.nan)
            ado = r.get("conv2_end", np.nan)
            ops = r.get("ops", np.nan)
            vol = r.get("vol", np.nan)
            mar = r.get("mar", np.nan)

            # Mostrar setas:
            # - sempre em anos completos
            # - no ano incompleto APENAS se a estimativa anual estiver ligada
            allow_arrows = (not is_incomplete_year) or st.session_state.include_year_forecast

            if prev is not None and allow_arrows:
                d_cli = _pct_change(cli, prev.get("clientes_end", np.nan))
                d_ops = _pct_change(ops, prev.get("ops", np.nan))
                d_vol = _pct_change(vol, prev.get("vol", np.nan))
                d_mar = _pct_change(mar, prev.get("mar", np.nan))
                d_ado_pp = _pp_change(ado, prev.get("conv2_end", np.nan))
            else:
                d_cli = d_ops = d_vol = d_mar = d_ado_pp = np.nan

            a_cli_txt, a_cli_cls = _arrow_pct(d_cli)
            a_ops_txt, a_ops_cls = _arrow_pct(d_ops)
            a_vol_txt, a_vol_cls = _arrow_pct(d_vol)
            a_mar_txt, a_mar_cls = _arrow_pct(d_mar)
            a_ado_txt, a_ado_cls = _arrow_pp(d_ado_pp)

            # barra do % (se NaN -> 0)
            p = 0.0
            try:
                if ado is not None and not (isinstance(ado, float) and np.isnan(ado)):
                    v = float(ado)
                    if v > 1.0:
                        v = v / 100.0
                    p = max(0.0, min(1.0, v))
            except Exception:
                p = 0.0
            w = round(p * 100, 1)

            body += f"""
        <tr>
        <td><b>{ano_label}</b></td>
        <td>{_fmt_int(cli)}{(f"<span class='arrow {a_cli_cls}'>{a_cli_txt}</span>") if a_cli_txt else ""}</td>
        <td class='pctcell'>
        <span class='bar'><span class='fill' style='width:{w}%'></span></span>
        <span class='pcttxt'>{_fmt_pct(ado, 1)}{(f"<span class='arrow {a_ado_cls}'>{a_ado_txt}</span>") if a_ado_txt else ""}</span>
        </td>
        <td>{_fmt_int_compact(ops, 1)}{(f"<span class='arrow {a_ops_cls}'>{a_ops_txt}</span>") if a_ops_txt else ""}</td>
        <td>{_fmt_eur_compact(vol, 1)}{(f"<span class='arrow {a_vol_cls}'>{a_vol_txt}</span>") if a_vol_txt else ""}</td>
        <td>{_fmt_eur_compact(mar, 1)}{(f"<span class='arrow {a_mar_cls}'>{a_mar_txt}</span>") if a_mar_txt else ""}</td>
        </tr>
        """

        tail = "</tbody></table></div></div>"

        st.html(head + body + tail)
        # ======================================================================
        # Detalhe diário
        # ======================================================================
        st.divider()
        st.markdown("### Detalhe diário")

        # FIX 3: Calendário para escolher o dia de análise
        min_date = df_daily["data"].min().date()
        max_date = df_daily["data"].max().date()

        # init session_state para o dia seleccionado no detalhe diário
        if "detail_day" not in st.session_state:
            st.session_state.detail_day = max_date

        detail_col1, detail_col2 = st.columns([1, 3])
        with detail_col1:
            selected_detail_day = st.date_input(
                "Escolher dia",
                value=st.session_state.detail_day,
                min_value=min_date,
                max_value=max_date,
                key="detail_day_picker",
            )
            st.session_state.detail_day = selected_detail_day

        detail_ts = pd.Timestamp(selected_detail_day).normalize()

        # Verifica se existe dados para esse dia
        df_day_sel = df_daily[df_daily["data"] == detail_ts]
        if df_day_sel.empty:
            # Tenta o dia mais próximo anterior
            df_before = df_daily[df_daily["data"] <= detail_ts]
            if not df_before.empty:
                detail_ts = df_before["data"].max().normalize()
                df_day_sel = df_daily[df_daily["data"] == detail_ts]
                with detail_col1:
                    st.caption(f"Sem dados para {selected_detail_day}. A mostrar {detail_ts.date()}.")

        cmp_date = _same_day_year(int(detail_ts.year) - 1, int(detail_ts.month), int(detail_ts.day))
        k_last = _period_kpis_from_daily(df_daily[df_daily["data"] == detail_ts])
        k_cmp = _period_kpis_from_daily(df_daily[df_daily["data"] == cmp_date])
        cli_d = _pct_change(k_last.get("clientes_end"), k_cmp.get("clientes_end"))
        pp_d = _pp_change(k_last.get("conv2_end"), k_cmp.get("conv2_end"))
        ops_d = _pct_change(k_last.get("ops"), k_cmp.get("ops"))
        vol_d = _pct_change(k_last.get("vol"), k_cmp.get("vol"))
        mar_d = _pct_change(k_last.get("mar"), k_cmp.get("mar"))
        cmp_txt = f"vs {cmp_date.date()}"

        _kpi_grid([
            ("Data selecionada", f"{detail_ts.date()}", f"<span class='subtle'>{cmp_txt}</span>"),
            ("Clientes com acesso", _fmt_int(k_last.get("clientes_end")), _delta_html_pct(cli_d) + f" <span class='subtle'>{cmp_txt}</span>"),
            ("% clientes com operações", _fmt_pct(k_last.get("conv2_end"), 1), _delta_html_pp(pp_d) + f" <span class='subtle'>{cmp_txt}</span>"),
            ("Operações (dia)", _fmt_int_compact(k_last.get("ops"), 1), _delta_html_pct(ops_d) + f" <span class='subtle'>{cmp_txt}</span>"),
            ("Volume (dia)", _fmt_eur_compact(k_last.get("vol"), 1), _delta_html_pct(vol_d) + f" <span class='subtle'>{cmp_txt}</span>"),
            ("Margem (dia)", _fmt_eur_compact(k_last.get("mar"), 1), _delta_html_pct(mar_d) + f" <span class='subtle'>{cmp_txt}</span>"),
        ])

        # Gráficos últimos 30 dias (relativos ao dia seleccionado)
        d30 = (df_daily["data"] >= (detail_ts - pd.Timedelta(days=30))) & (df_daily["data"] <= detail_ts)
        df_30 = df_daily.loc[d30].sort_values("data").copy()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("**Volume — últimos 30 dias**")
            d = df_30[["data", "volume_negocios"]].copy()
            d["volume_negocios"] = pd.to_numeric(d["volume_negocios"], errors="coerce")
            d = d[d["volume_negocios"].notna()]
            ch = alt.Chart(d).mark_line(interpolate="monotone", color=_theme_vars["accent"]).encode(
                x=alt.X("data:T", title=None),
                y=alt.Y("volume_negocios:Q", title="€"),
                tooltip=[alt.Tooltip("data:T", title="Data"), alt.Tooltip("volume_negocios:Q", title="Volume", format=",.0f")],
            ).properties(height=240)
            st.altair_chart(ch, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='panel'>", unsafe_allow_html=True)
            st.markdown("**Operações — últimos 30 dias**")
            d = df_30[["data", "num_operacoes"]].copy()
            d["num_operacoes"] = pd.to_numeric(d["num_operacoes"], errors="coerce")
            d = d[d["num_operacoes"].notna()]
            ch = alt.Chart(d).mark_bar(color=_theme_vars["accent2"]).encode(
                x=alt.X("data:T", title=None),
                y=alt.Y("num_operacoes:Q", title="Nº"),
                tooltip=[alt.Tooltip("data:T", title="Data"), alt.Tooltip("num_operacoes:Q", title="Operações", format=",.0f")],
            ).properties(height=240)
            st.altair_chart(ch, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Ver dados diários (últimos 60 dias)", expanded=False):
            df_60 = df_daily[df_daily["data"] >= (detail_ts - pd.Timedelta(days=60))].copy().sort_values("data")
            rename = {
                "data": "Data", "clientes_acesso": "Clientes com acesso",
                "pedidos_pendentes": "Pedidos pendentes", "novos_pedidos": "Novos pedidos",
                "desist_total": "Desistências (Total)", "desist_ativados": "De Ativados",
                "desist_pendentes": "De Pendentes", "ativados_ops_s1": "Ativados c/ operações (S1)",
                "conv_ops_s1": "% ops/acesso (S1)", "ativados_ops_s2": "Ativados c/ operações (S2)",
                "conv_ops_s2": "% clientes com operações (S2)", "num_operacoes": "Nº operações",
                "volume_negocios": "Volume negócios", "margem_liquida": "Margem líquida",
            }
            show = df_60.rename(columns=rename)
            show["Data"] = pd.to_datetime(show["Data"], errors="coerce").dt.date
            st.dataframe(show, use_container_width=True, hide_index=True)

# =============================================================================
# Upload CSV (NO FIM) — FIX 2: guarda em session_state para persistência
# =============================================================================
st.divider()
with st.container(key="upload_block"):
    with st.expander("Carregar dados (CSV)", expanded=(st.session_state.raw_report_bytes is None)):
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            upl = st.file_uploader("CSV do Report", type=["csv"], label_visibility="collapsed")
        with cc2:
            fallback_path = Path(__file__).with_name("Report.csv")
            use_fallback = False
            if upl is None and fallback_path.exists():
                use_fallback = st.checkbox("Usar Report.csv local", value=False)

        new_raw = None
        if upl is not None:
            new_raw = upl.read()
        elif use_fallback and fallback_path.exists():
            new_raw = fallback_path.read_bytes()

        if new_raw is not None:
            st.session_state.raw_report_bytes = new_raw
            PERSISTENT_CSV.write_bytes(new_raw)
            st.rerun()

        if st.session_state.raw_report_bytes is not None:
            st.caption("✅ Ficheiro carregado em memória (sessão activa).")

def _clear_loaded_data():
        # 1) Remove o cache persistente do CSV (senão ele volta a carregar sozinho no próximo rerun)
        try:
            if PERSISTENT_CSV.exists():
                PERSISTENT_CSV.unlink()
        except Exception:
            pass

        # 2) Limpa caches do Streamlit (inclui st.cache_data do load_report)
        try:
            st.cache_data.clear()
        except Exception:
            pass

        # 3) Apaga keys do estado (inclui keys de widgets)
        keys_to_delete = [
            "raw_report_bytes",
            "sel_year",
            "sel_month",
            "asof_date",
            "pos_scope",
            "include_curr_month_summary",
            "include_year_forecast",
            "detail_day",
            "detail_day_picker",
        ]
        for k in keys_to_delete:
            if k in st.session_state:
                del st.session_state[k]

st.button("🗑️ Limpar dados carregados", on_click=_clear_loaded_data)
