"""Single source of truth for the app's dark theme and typography.

Every Streamlit entry point (app.py, labelling_app.py, feedback_app.py, faculty_experiment_portal.py)
calls inject_theme() so the fonts and palette stay identical across pages. Previously
each page carried its own light "Web 1.0" stylesheet, which drifted apart and fought
the dark base declared in .streamlit/config.toml.
"""

import streamlit as st

# Shared palette. Kept as constants so Python-side f-string styling (badges,
# grade colours) can reference the same values the stylesheet uses.
BG = "#0e1117"
BG_CARD = "#161a22"
BG_ELEVATED = "#1c2129"
BORDER = "#2f3540"
BORDER_STRONG = "#454d5a"
TEXT = "#e6e8eb"
TEXT_MUTED = "#9aa3ae"
ACCENT = "#a78bfa"

GROUNDED = "#34d399"
UNSTABLE = "#fbbf24"
COLLAPSED = "#f87171"
PAUSED = "#9aa3ae"

# One font stack for every page.
FONT_STACK = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
    '"Helvetica Neue", Arial, sans-serif'
)

_CSS = f"""
<style>
    /* ---- Typography: one stack everywhere ---- */
    html, body, .stApp, button, input, textarea, select,
    h1, h2, h3, h4, h5, h6, p, label, span, li, td, th, div,
    [class*="st-"], [data-testid] {{
        font-family: {FONT_STACK} !important;
    }}
    /* Streamlit's Material Icons (expander arrows, etc.) must keep their own
    font or the ligature text renders literally instead of as a glyph. */
    [data-testid="stIconMaterial"], .material-icons,
    [class*="material-symbols"] {{
        font-family: "Material Symbols Rounded", "Material Icons" !important;
    }}
    code, pre, code * {{
        font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo,
                     Consolas, monospace !important;
    }}

    /* ---- Base surfaces ---- */
    html, body, .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {{
        background-color: {BG} !important;
        color: {TEXT} !important;
    }}
    [data-testid="stSidebar"] {{
        background-color: {BG_CARD} !important;
        border-right: 1px solid {BORDER} !important;
    }}
    h1, h2, h3, h4, h5, h6, p, label, li, td, th, span {{
        color: {TEXT} !important;
    }}
    small, caption, [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {{
        color: {TEXT_MUTED} !important;
    }}
    a, a:visited {{ color: {ACCENT} !important; }}
    hr {{ border-color: {BORDER} !important; }}

    /* ---- Controls ---- */
    button, .stButton > button {{
        background-color: {BG_ELEVATED} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER_STRONG} !important;
        border-radius: 6px !important;
        padding: 6px 14px !important;
    }}
    button:hover, .stButton > button:hover {{
        background-color: #262c36 !important;
        color: {TEXT} !important;
        border-color: {ACCENT} !important;
    }}
    button:focus, .stButton > button:focus {{
        color: {TEXT} !important;
        background-color: #262c36 !important;
        box-shadow: none !important;
    }}
    .stButton > button[kind="primary"] {{
        background-color: #6d28d9 !important;
        border-color: #7c3aed !important;
        color: #ffffff !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: #7c3aed !important;
    }}

    /* Inputs, selects, text areas, sliders */
    input, textarea, [data-baseweb="input"], [data-baseweb="textarea"],
    [data-baseweb="select"] > div, [data-testid="stNumberInput"] input {{
        background-color: {BG_ELEVATED} !important;
        color: {TEXT} !important;
        border-color: {BORDER_STRONG} !important;
    }}
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {{
        background-color: {BG_ELEVATED} !important;
        color: {TEXT} !important;
    }}
    [role="option"]:hover {{ background-color: #2b3240 !important; }}

    /* Tabs */
    [data-testid="stTabs"] button {{
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
    }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        border-bottom: 2px solid {ACCENT} !important;
        color: {ACCENT} !important;
    }}

    /* Expanders, metrics, code, tables */
    [data-testid="stExpander"] {{
        background-color: {BG_CARD} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stExpander"] details, [data-testid="stExpander"] summary {{
        background-color: {BG_CARD} !important;
        color: {TEXT} !important;
    }}
    [data-testid="stMetric"] {{
        background-color: {BG_CARD} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        padding: 12px !important;
    }}
    [data-testid="stMetricValue"] {{ color: {TEXT} !important; }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}
    pre, code {{
        background-color: {BG_ELEVATED} !important;
        color: {TEXT} !important;
    }}
    table, th, td {{ border-color: {BORDER} !important; }}
    thead tr th {{ background-color: {BG_ELEVATED} !important; }}

    /* ---- Shared card primitives ---- */
    .box {{
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        padding: 14px !important;
        background-color: {BG_CARD} !important;
        color: {TEXT} !important;
        margin-bottom: 15px !important;
    }}
    .box b, .box span, .box p {{ color: {TEXT} !important; }}
    .system-badge {{
        font-weight: bold !important;
        color: {ACCENT} !important;
    }}
    .review-card-header {{
        background-color: {BG_ELEVATED} !important;
        padding: 12px !important;
        border: 1px solid {BORDER_STRONG} !important;
        border-radius: 8px !important;
        text-align: center;
        margin-bottom: 20px;
    }}

    /* ---- app.py specific components ---- */
    .header-card {{
        border: 1px solid {BORDER_STRONG};
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 20px;
        text-align: center;
        background-color: {BG_CARD};
    }}
    .header-title {{
        color: {TEXT} !important;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 5px;
    }}
    .header-subtitle {{ color: {TEXT_MUTED} !important; font-size: 1rem; }}

    .glass-card {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        background-color: {BG_CARD};
    }}
    .claim-card {{
        border-left: 4px solid {ACCENT};
        background-color: {BG_ELEVATED};
    }}

    .badge {{
        padding: 4px 10px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        border-radius: 999px;
        border: 1px solid currentColor;
    }}
    .badge-grounded {{ background-color: rgba(52, 211, 153, 0.12); color: {GROUNDED} !important; }}
    .badge-unstable {{ background-color: rgba(251, 191, 36, 0.12); color: {UNSTABLE} !important; }}
    .badge-collapsed {{ background-color: rgba(248, 113, 113, 0.12); color: {COLLAPSED} !important; }}
    .badge-paused {{ background-color: rgba(154, 163, 174, 0.12); color: {PAUSED} !important; }}

    .chat-bubble {{
        padding: 12px 14px;
        margin-bottom: 10px;
        max-width: 90%;
        line-height: 1.5;
        border-radius: 10px;
    }}
    .assessor-bubble {{
        background-color: {BG_CARD};
        color: {TEXT} !important;
        border: 1px solid {BORDER};
        border-left: 3px solid {ACCENT};
    }}
    .student-bubble {{
        background-color: {BG_ELEVATED};
        color: {TEXT} !important;
        border: 1px solid {BORDER_STRONG};
        margin-left: auto;
    }}
    .advocate-hint-bubble {{
        background-color: rgba(167, 139, 250, 0.08);
        color: {TEXT} !important;
        border: 1px dashed {ACCENT};
        font-style: italic;
    }}

    .metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {TEXT} !important;
    }}
    .metric-label {{ font-size: 0.85rem; color: {TEXT_MUTED} !important; }}
</style>
"""


def inject_theme() -> None:
    """Apply the shared dark theme. Call once, near the top of each page."""
    st.markdown(_CSS, unsafe_allow_html=True)
