import streamlit as st


def set_view(view: str) -> None:
    """Set the selected Canary page and begin it at the top of the main canvas."""

    st.session_state["_canary_view"] = view
    st.html(
        """
        <script>
          requestAnimationFrame(() => {
            const main = document.querySelector('[data-testid="stMain"]');
            if (main) main.scrollTo({ top: 0, left: 0, behavior: 'instant' });
          });
        </script>
        """,
        unsafe_allow_javascript=True,
    )
