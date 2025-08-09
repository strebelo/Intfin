import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Taylor Rule Simulator")

st.write("""
This app lets you explore how the Taylor Rule works.
Adjust the parameters using the sliders and see how the modeled policy rate
compares to the actual Federal Funds rate.
""")

# ------------------
# Load data
# ------------------
@st.cache_data
def load_data():
    # Replace with your CSV path
    # Must have: date, fed_funds_actual, inflation_used, unemployment
    df = pd.read_csv("taylor_data.csv", parse_dates=["date"])
    return df

df = load_data()

# ------------------
# Sliders for parameters
# ------------------
a = st.slider(
    "a (inflation gap coefficient)",
    min_value=-2.0,
    max_value=2.0,
    value=1.0,
    step=0.1
)

b = st.slider(
    "b (output gap coefficient)",
    min_value=-2.0,
    max_value=2.0,
    value=-0.5,
    step=0.1
)

r_star = st.slider(
    "r* (neutral real rate, %)",
    min_value=-2.0,
    max_value=5.0,
    value=2.0,
    step=0.1
)

pi_star = st.slider(
    "π* (target inflation, %)",
    min_value=0.0,
    max_value=5.0,
    value=2.0,
    step=0.1
)

rho = st.slider(
    "ρ (interest rate smoothing)",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.01
)

# ------------------
# Calculate modeled rate
# ------------------
df["output_gap"] = 4.5 - df["unemployment"]  # Example: natural rate = 4.5
df["taylor_rate"] = r_star + df["inflation_used"] + a * (df["inflation_used"] - pi_star) + b * df["output_gap"]

# Smoothing
df["fed_funds_modeled"] = np.nan
for i in range(len(df)):
    if i == 0:
        df.loc[i, "fed_funds_modeled"] = df.loc[i, "taylor_rate"]
    else:
        df.loc[i, "fed_funds_modeled"] = (
            rho * df.loc[i - 1, "fed_funds_modeled"]
            + (1 - rho) * df.loc[i, "taylor_rate"]
        )

# ------------------
# Plot
# ------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["date"], df["fed_funds_actual"], label="Actual Fed Funds", linewidth=2)
ax.plot(df["date"], df["fed_funds_modeled"], label="Taylor Rule", linewidth=2)
ax.set_ylabel("Interest Rate (%)")
ax.set_xlabel("Date")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ------------------
# Show data
# ------------------
if st.checkbox("Show data table"):
    st.dataframe(df)
