"""
Reusable UI components for the Streamlit web application.
Each function renders a self-contained Streamlit element.
"""

import streamlit as st


def metric_card(value, label, variant="default"):
    """
    Render a styled metric card.

    Args:
        value: The metric value to display.
        label: Label text below the value.
        variant: 'human', 'car', 'total', or 'default' for color styling.
    """
    css_class = f"metric-{variant}" if variant != "default" else ""
    st.markdown(
        f'<div class="metric-card {css_class}">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def detection_table(detections):
    """
    Render a table of individual detections.

    Args:
        detections: List of dicts with 'class', 'confidence', 'bbox' keys.
    """
    html = (
        '<table class="det-table">'
        '<tr><th>#</th><th>Class</th><th>Confidence</th><th>Bounding Box</th></tr>'
    )
    for i, det in enumerate(detections):
        bbox = det["bbox"]
        bbox_str = f"({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})"
        html += (
            f'<tr><td>{i + 1}</td><td>{det["class"]}</td>'
            f'<td>{det["confidence"]:.3f}</td><td>{bbox_str}</td></tr>'
        )
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)


def sidebar_section_title(text):
    """Render a styled sidebar section title."""
    st.markdown(f'<div class="sidebar-title">{text}</div>', unsafe_allow_html=True)


def section_divider():
    """Render a horizontal divider."""
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def placeholder_message(text):
    """Render centered placeholder text for empty states."""
    st.markdown(f'<div class="placeholder-text">{text}</div>', unsafe_allow_html=True)
