from __future__ import annotations

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency guard
    st = None


def run_dashboard() -> None:
    """Render a simple Streamlit dashboard scaffold."""
    if st is None:
        raise ImportError("Streamlit is required to run the dashboard. Install the project dependencies first.")

    st.set_page_config(page_title="Loan Approval Assistant")
    st.title("Loan Approval Assistant")
    st.write("Dashboard scaffold is ready for model insights and explainability views.")
