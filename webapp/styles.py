"""
CSS styles for the Streamlit web application.
Returns the full CSS block as a string for injection via st.markdown.
"""


def get_css():
    """Return the custom CSS for the application."""
    return """
<style>
    /* Header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #8892A0;
        margin-bottom: 2rem;
        line-height: 1.5;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1A1F2E 0%, #232940 100%);
        border: 1px solid #2D3548;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover {
        border-color: #4A90D9;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        color: #FAFAFA;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892A0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }
    .metric-human .metric-value { color: #FF4444; }
    .metric-car .metric-value { color: #44CC00; }
    .metric-total .metric-value { color: #4A90D9; }

    /* Sidebar */
    .sidebar-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #8892A0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
    }

    /* Detection table */
    .det-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.5rem;
    }
    .det-table th {
        background: #1A1F2E;
        color: #8892A0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 0.5rem;
        text-align: left;
        border-bottom: 1px solid #2D3548;
    }
    .det-table td {
        padding: 0.4rem 0.5rem;
        border-bottom: 1px solid #1A1F2E;
        font-size: 0.9rem;
        color: #FAFAFA;
    }

    /* Utility */
    .section-divider {
        border: none;
        border-top: 1px solid #2D3548;
        margin: 2rem 0;
    }
    .placeholder-text {
        text-align: center;
        padding: 4rem 2rem;
        color: #8892A0;
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
