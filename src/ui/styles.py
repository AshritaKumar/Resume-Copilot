from __future__ import annotations

import streamlit as st

GLOBAL_STYLES = """
<style>
.main > div {
    padding-top: 1.25rem;
}
.app-subtle {
    color: #9aa4b2;
    font-size: 0.92rem;
}
.section-title {
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
    font-weight: 600;
    font-size: 1.02rem;
    letter-spacing: 0.2px;
}
.kpi-card {
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 10px;
    padding: 10px 12px;
    background: rgba(15, 23, 42, 0.35);
}
.kpi-label {
    color: #a6b0bf;
    font-size: 0.8rem;
    margin-bottom: 2px;
}
.kpi-value {
    font-size: 1.2rem;
    font-weight: 700;
    line-height: 1.2;
}
.chip {
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 10px;
    padding: 8px 10px;
    margin-bottom: 8px;
    background: rgba(15, 23, 42, 0.30);
}
.chip-title {
    font-size: 0.92rem;
    font-weight: 600;
    margin-bottom: 4px;
}
.chip-subtle {
    color: #a6b0bf;
    font-size: 0.82rem;
    line-height: 1.35;
}
</style>
"""


def inject_global_styles() -> None:
    st.markdown(GLOBAL_STYLES, unsafe_allow_html=True)
