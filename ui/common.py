"""Magic LLM – gemeinsame Helfer für alle Seiten der App: Farben, CSS, Diagramm-Stil, geteilter Zustand."""
from __future__ import annotations

import colorsys
import html
import json
import re

import plotly.graph_objects as go
import streamlit as st

from magicllm.data import PREPARED_DIR, list_prepared

PALETTE: dict[str, str] = {}        # wird von apply_theme() aus dem aktiven Theme gefüllt
COLORS: list[str] = []
T: dict[str, str] = {}              # abgeleitete Theme-Farben (bg, surface, fg, muted, grid …)
THEME = None                        # das aktive Theme-Objekt


def apply_theme(theme=None) -> None:
    """Theme-Farben in PALETTE/COLORS/T schreiben (in place, damit alle Seiten die neuen Werte sehen)."""
    global THEME
    from ui.theme import active_theme, tokens
    THEME = theme or active_theme()
    c = list(THEME.chart) + ["#737373"] * 7
    PALETTE.clear()
    PALETTE.update(fg=c[0], blue=c[1], amber=c[2], green=c[3], pink=c[4], violet=c[5], grey=c[6])
    PALETTE.update(cyan=PALETTE["blue"], orange=PALETTE["amber"], yellow=PALETTE["amber"])
    COLORS[:] = c[:7]
    T.clear()
    T.update(tokens(THEME))


def theme_css() -> str:
    names = {"bg": "bg", "surface": "surface", "surface2": "surface-2", "border": "border",
             "border_strong": "border-strong", "fg": "fg", "muted": "muted", "dim": "dim", "primary": "primary",
             "on_primary": "on-primary", "primary_hover": "primary-hover", "gen_bg": "gen-bg"}
    return "<style>:root {" + " ".join(f"--ml-{v}: {T[k]};" for k, v in names.items()) + "}</style>"


_DOT_MAP = {"#141414": "surface", "#333333": "border_strong", "#E5E5E5": "fg", "#525252": "dim", "#737373": "muted",
            "#A3A3A3": "muted", "#404040": "border_strong", "#262626": "surface2", "#FAFAFA": "primary"}
_DOT_CHART = {"#60A5FA": 1, "#F59E0B": 2, "#34D399": 3, "#A78BFA": 5}


def themed_dot(dot: str) -> str:
    """Graphviz-Diagramme sind mit den Farben des dunklen Themes geschrieben; hier werden sie auf das aktive
    Theme umgefärbt (in einem Durchgang, damit sich Ersetzungen nicht gegenseitig treffen)."""
    def repl(m):
        h = m.group(0).upper()
        if h in _DOT_MAP:
            return T[_DOT_MAP[h]]
        if h in _DOT_CHART:
            return COLORS[_DOT_CHART[h]]
        return m.group(0)
    return re.sub(r"#[0-9A-Fa-f]{6}", repl, dot)


def heat_scale() -> list:
    """Farbskala für Heatmaps: 0 = Hintergrund, 1 = Text (funktioniert hell wie dunkel)."""
    return [[0, T["bg"]], [1, T["fg"]]]


MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root { /* Werte kommen aus dem aktiven Theme (ui/theme.py), siehe theme_css() */
        
        --ml-mono:'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
html, body, [data-testid="stAppViewContainer"] {font-feature-settings: "cv11", "ss01";}
.block-container {padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px;}
h1, h2, h3, h4 {letter-spacing: -0.02em;}
h2, h3 {font-size: 1.05rem !important; font-weight: 600 !important; margin-top: .6rem;}
hr {border-color: var(--ml-border) !important;}
a {text-decoration: none;}

/* Kopfzeile einer Seite: Pfad im Terminal-Stil, Titel, Untertitel */
.ml-head {border-bottom: 1px solid var(--ml-border); padding: 0 0 18px 0; margin: 0 0 22px 0;}
.ml-crumb {font-family: var(--ml-mono); font-size: .75rem; color: var(--ml-dim); margin-bottom: 10px; letter-spacing: 0;}
.ml-crumb b {color: var(--ml-muted); font-weight: 500;}
.ml-head h1 {margin: 0 0 6px 0; padding: 0; font-size: 1.65rem; font-weight: 600; line-height: 1.2;}
.ml-head p {margin: 0; color: var(--ml-muted); font-size: .95rem; max-width: 820px;}

/* Kennzahlen: flach, Mono-Werte */
[data-testid="stMetric"] {background: var(--ml-surface); border: 1px solid var(--ml-border); border-radius: 10px;
                          padding: 12px 14px;}
[data-testid="stMetricLabel"] p {color: var(--ml-dim); font-size: .72rem; text-transform: uppercase;
                                 letter-spacing: .06em; font-weight: 500;}
[data-testid="stMetricValue"] {font-family: var(--ml-mono); font-size: 1.3rem; font-weight: 500;}
[data-testid="stMetricDelta"] {font-family: var(--ml-mono); font-size: .75rem;}

/* Knöpfe: primär = weiss auf schwarz, sekundär = Umriss */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {background: var(--ml-primary) !important; color: var(--ml-on-primary) !important; border: 1px solid var(--ml-primary) !important;
                                      font-weight: 500;}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {background: var(--ml-primary-hover) !important; border-color: var(--ml-primary-hover) !important;}
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p {color: var(--ml-on-primary) !important;}
[data-testid="stBaseButton-primary"]:disabled, [data-testid="stBaseButton-primaryFormSubmit"]:disabled {background: var(--ml-surface-2) !important; border-color: var(--ml-border) !important;}
[data-testid="stBaseButton-primary"]:disabled p, [data-testid="stBaseButton-primaryFormSubmit"]:disabled p {color: var(--ml-dim) !important;}
[data-testid="stBaseButton-secondary"] {background: transparent; border-color: var(--ml-border-strong);}
[data-testid="stBaseButton-secondary"]:hover {border-color: var(--ml-dim); background: var(--ml-surface-2); color: var(--ml-fg);}

/* Link zur nächsten Seite */
[data-testid="stPageLink"] a {background: var(--ml-primary); border-radius: 8px; padding: .5rem .95rem; display: inline-flex;}
[data-testid="stPageLink"] a p, [data-testid="stPageLink"] a span {color: var(--ml-on-primary) !important; font-weight: 500;}
[data-testid="stPageLink"] a:hover {background: var(--ml-primary-hover);}

/* Seitenleiste */
[data-testid="stSidebarHeader"] img, [data-testid="stLogo"] {height: 2rem !important; max-width: 12rem;}
[data-testid="stSidebarNav"] a span {font-size: .9rem;}
.ml-brand {display: flex; align-items: center; gap: 10px; margin: 2px 0 14px 0;}
.ml-mark {width: 26px; height: 26px; border-radius: 7px; background: var(--ml-primary); color: var(--ml-on-primary); display: grid;
          place-items: center; font-family: var(--ml-mono); font-weight: 700; font-size: .78rem; letter-spacing: -0.04em;}
.ml-brand-name {font-weight: 600; font-size: 1rem; letter-spacing: -0.01em;}
.ml-brand-ver {font-family: var(--ml-mono); font-size: .68rem; color: var(--ml-dim); border: 1px solid var(--ml-border);
               border-radius: 999px; padding: 0 7px;}
.ml-side-label {font-family: var(--ml-mono); font-size: .68rem; text-transform: uppercase; letter-spacing: .08em;
                color: var(--ml-dim); margin: 6px 0 4px 0;}
.ml-side-row {font-family: var(--ml-mono); font-size: .78rem; color: var(--ml-muted); line-height: 1.6;}

/* Status-Punkte statt Emojis */
.dot {display: inline-block; width: 7px; height: 7px; border-radius: 99px; margin-right: 7px; vertical-align: middle;
      position: relative; top: -1px; background: var(--ml-dim);}
.dot.ok {background: #34D399;} .dot.run {background: #34D399; box-shadow: 0 0 0 3px rgba(52,211,153,.18);}
.dot.warn {background: #F59E0B;} .dot.err {background: #F87171;} .dot.idle {background: var(--ml-dim);}

/* Kleine Etiketten */
.tag {display: inline-block; padding: 0 8px; margin: 0 5px 3px 0; border-radius: 6px; font-size: .7rem;
      font-family: var(--ml-mono); line-height: 1.6rem; background: var(--ml-surface-2); color: var(--ml-muted);
      border: 1px solid var(--ml-border);}
.tag.good {color: #10B981; border-color: color-mix(in srgb, #10B981 35%, var(--ml-bg)); background: color-mix(in srgb, #10B981 10%, var(--ml-bg));}
.tag.warn {color: #D97706; border-color: color-mix(in srgb, #F59E0B 35%, var(--ml-bg)); background: color-mix(in srgb, #F59E0B 10%, var(--ml-bg));}

/* Token-Chips */
.tok {display: inline-block; padding: 1px 5px; margin: 2px 2px; border-radius: 5px; font-family: var(--ml-mono);
      font-size: .85rem; color: var(--ml-fg); white-space: pre; border: 1px solid var(--ml-border);}

/* Karten, Textausgaben, Terminal-Zeilen */
.step-card {background: var(--ml-surface); border: 1px solid var(--ml-border); border-radius: 10px; padding: 14px 16px;
            height: 100%;}
.step-card .n {font-family: var(--ml-mono); font-size: .72rem; color: var(--ml-dim);}
.step-card b {color: var(--ml-fg); font-weight: 600; display: block; margin: 2px 0 6px 0;}
.sample {background: var(--ml-surface); border: 1px solid var(--ml-border); border-radius: 10px; padding: 12px 16px;
         margin: 6px 0; font-family: var(--ml-mono); font-size: .85rem; line-height: 1.6; white-space: pre-wrap;
         color: var(--ml-muted); overflow-wrap: anywhere;}
.sample .gen {color: var(--ml-fg); background: var(--ml-gen-bg); border-radius: 3px;}
.sample .cursor {display: inline-block; width: 8px; height: 1em; background: var(--ml-fg); vertical-align: text-bottom;
                 animation: ml-blink 1s steps(1) infinite;}
@keyframes ml-blink {50% {opacity: 0;}}
.ml-stats {font-family: var(--ml-mono); font-size: .75rem; color: var(--ml-dim); margin: 4px 0 10px 2px;}
.ml-stats b {color: var(--ml-muted); font-weight: 500;}
.ml-term {background: var(--ml-surface); border: 1px solid var(--ml-border); border-radius: 10px; padding: 12px 16px;
          font-family: var(--ml-mono); font-size: .8rem; color: var(--ml-muted); line-height: 1.7;}
.ml-term .p {color: var(--ml-dim);} .ml-term .c {color: var(--ml-fg);}
.muted {color: var(--ml-muted);}

/* Tabs, Expander, Radio als Segment-Schalter */
[data-testid="stExpander"] details {border-color: var(--ml-border); border-radius: 10px; background: transparent;}
[data-testid="stExpander"] summary p {font-weight: 500;}
[data-testid="stTabs"] button p {font-size: .88rem;}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] {gap: 2px; background: var(--ml-surface);
    border: 1px solid var(--ml-border); border-radius: 10px; padding: 3px; width: fit-content; flex-wrap: wrap;}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] > div {margin: 0 !important;}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"] {padding: 5px 14px;
    border-radius: 7px; margin: 0 !important; cursor: pointer;}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"] p {color: var(--ml-muted);
    font-size: .86rem;}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"][data-selected="true"] {
    background: var(--ml-surface-2);}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"][data-selected="true"] p {
    color: var(--ml-fg);}
[data-testid="stRadioGroup"][aria-orientation="horizontal"] [data-testid="stRadioOption"] > div > div:first-child {
    display: none;}
/* Auswahl-Chips (Multiselect): dunkel statt weiss-auf-weiss */
[data-baseweb="tag"] {background: var(--ml-surface-2) !important; border: 1px solid var(--ml-border-strong) !important;}
[data-baseweb="tag"] span, [data-baseweb="tag"] svg {color: var(--ml-fg) !important; fill: var(--ml-fg) !important;}
/* Häkchen in weissen Checkboxen dunkel zeichnen */
[data-testid="stCheckbox"] label[data-selected="true"] svg, [data-testid="stCheckbox"] label[data-selected="true"] svg polyline {
    stroke: var(--ml-on-primary) !important; color: var(--ml-on-primary) !important;}
[data-testid="stProgress"] > div > div > div > div {background: var(--ml-primary);}
</style>
"""


PAGES: dict = {}     # wird von app.py gefüllt: Schlüssel -> st.Page


def next_step(key: str, label: str, icon: str = ":material/arrow_forward:") -> None:
    """Knopf, der zur nächsten Seite führt (damit man nie „stecken bleibt“)."""
    if key in PAGES:
        st.page_link(PAGES[key], label=label, icon=icon)


def setup_page() -> None:
    apply_theme()
    st.markdown(theme_css() + CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, crumb: str = "") -> None:
    """Seitenkopf im Stil der llama.cpp-WebUI / eines Terminals: Pfad, Titel, Untertitel."""
    path = f"magic-llm <span>/</span> <b>{html.escape(crumb or title.lower())}</b>"
    st.markdown(f'<div class="ml-head"><div class="ml-crumb">{path}</div><h1>{html.escape(title)}</h1>'
                f'<p>{subtitle}</p></div>', unsafe_allow_html=True)


def dot(state: str) -> str:
    """Kleiner Status-Punkt (HTML). state: ok | run | warn | err | idle"""
    return f'<span class="dot {state}"></span>'


STATUS_DOT = {"läuft": "run", "fertig": "ok", "gestoppt": "warn", "fehler": "err", "bereit": "idle"}


def style_fig(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark" if (THEME is None or THEME.base == "dark") else "plotly_white", height=height,
        margin=dict(l=8, r=8, t=70 if legend else 44, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=COLORS,
        showlegend=legend, legend=dict(orientation="h", y=1.04, x=0, yanchor="bottom", font=dict(color=T["muted"])),
        font=dict(family="Inter, ui-sans-serif, system-ui, sans-serif", size=12, color=T["muted"]),
        title_font=dict(size=13, color=T["fg"]), title_x=0,
        hoverlabel=dict(font_family=MONO, bgcolor=T["surface"], font_color=T["fg"], bordercolor=T["border"]),
    )
    fig.update_xaxes(gridcolor=T["grid"], zerolinecolor=T["border"], linecolor=T["border"],
                     tickfont=dict(family=MONO, size=10))
    fig.update_yaxes(gridcolor=T["grid"], zerolinecolor=T["border"], linecolor=T["border"],
                     tickfont=dict(family=MONO, size=10))
    return fig


def fmt_int(n: float) -> str:
    n = float(n)
    for unit, div in (("Mrd", 1e9), ("Mio", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f} {unit}".replace(".00", "")
    return f"{int(n)}"


def fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def token_chips(pieces: list[tuple[int, str]]) -> str:
    """Farbige Token-Chips als HTML (jedes Token bekommt eine eigene Farbe)."""
    out = []
    for k, (tid, s) in enumerate(pieces):
        hue = (tid * 47 + k * 13) % 360
        light = THEME is not None and THEME.base == "light"
        r, g, b = colorsys.hls_to_rgb(hue / 360, 0.88 if light else 0.24, 0.55 if light else 0.38)
        out.append(f'<span class="tok" style="background:rgb({int(r*255)},{int(g*255)},{int(b*255)})" '
                   f'title="Token-ID {tid}">{html.escape(s)}</span>')
    return "".join(out)


def restore_state(name: str, defaults: dict) -> None:
    """Streamlit löscht den Zustand von Widgets, die auf einer anderen Seite waren – wir merken ihn uns selbst."""
    keep = st.session_state.setdefault(f"_keep_{name}", {})
    for k, v in defaults.items():
        keep.setdefault(k, v)
        if k not in st.session_state:
            st.session_state[k] = keep[k]


def save_state(name: str) -> None:
    keep = st.session_state.get(f"_keep_{name}", {})
    for k in keep:
        if k in st.session_state:
            keep[k] = st.session_state[k]


# ----------------------------------------------------------- geteilter Zustand
@st.cache_resource
def store() -> dict:
    """Wird von allen Browser-Sitzungen geteilt: hier lebt der laufende Trainer."""
    return {"trainer": None}


@st.cache_data(show_spinner=False)
def load_meta(name: str, _stamp: float) -> dict:
    return json.loads((PREPARED_DIR / name / "meta.json").read_text(encoding="utf-8"))


def active_data() -> dict | None:
    """Meta-Infos des in der Seitenleiste gewählten Datensatzes (oder None)."""
    name = st.session_state.get("active_data")
    if not name:
        return None
    p = PREPARED_DIR / name / "meta.json"
    return load_meta(name, p.stat().st_mtime) if p.exists() else None


def need_data() -> dict:
    meta = active_data()
    if meta is None:
        st.info("Du hast noch keinen vorbereiteten Datensatz. Gehe zuerst auf **Daten** und lade Texte.",
                icon=":material/info:")
        next_step("data", "Zu den Daten")
        st.stop()
    return meta


def prepared_names() -> list[str]:
    return [m["name"] for m in sorted(list_prepared(), key=lambda m: m["created"])]
