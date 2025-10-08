import streamlit as st
import plotly.express as px
import pandas as pd
from scipy import stats
import numpy as np

# Example input: annual log FX changes
# data = pd.Series(...)

st.subheader("Distribution of Annual Log FX Changes")

# Compute histogram data manually to control binning and fractions
hist_values, bin_edges = np.histogram(data, bins=20, density=False)
fractions = hist_values / len(data)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

hist_df = pd.DataFrame({
    "Bin Center": bin_centers,
    "Frequency": hist_values,
    "Fraction": fractions
})

# Create interactive histogram with Plotly
fig = px.bar(
    hist_df,
    x="Bin Center",
    y="Fraction",
    labels={"Bin Center": "Annual Log FX Change", "Fraction": "Fraction of Sample"},
    title="Histogram (Click a bar to view fraction)"
)

# Add hover tooltips and consistent styling
fig.update_traces(
    hovertemplate="Change: %{x:.3f}<br>Fraction: %{y:.3f}",
    marker_line_width=1,
    marker_line_color="black",
    opacity=0.7
)
fig.update_layout(bargap=0.05)

# Display interactive figure
clicked_bar = st.plotly_chart(fig, use_container_width=True, key="hist_click")

st.caption("Click a bar to see its corresponding fraction in the tooltip above.")
