# -------------------------------
# Forecast: Spot path & 95% Normal CI by month
# -------------------------------
st.subheader("Forecast: 95% Normal-confidence interval for the spot by month")

# Inputs
default_spot = float(panel[price_col].iloc[-1]) if len(panel) else 1.0
spot_now = st.number_input("Current spot (S₀)", min_value=0.0, value=round(default_spot, 6), format="%.6f")
horizon_m = st.number_input("Horizon (months)", min_value=1, max_value=240, value=12, step=1)

# NEW: choose drift source
drift_choice = st.radio(
    "Drift used in the CI",
    options=["Historical average (μ)", "User-specified annual drift"],
    index=0,
    horizontal=True,
    help="Drift is the annual log change. If you choose a custom drift, enter it as a decimal (e.g., 0.02 ≈ 2% per year)."
)

if drift_choice == "User-specified annual drift":
    user_mu_annual = st.number_input(
        "Annual log drift μ (decimal, e.g., 0.02 ≈ 2%)",
        value=float(mu),
        format="%.6f"
    )
    mu_used = float(user_mu_annual)
else:
    mu_used = float(mu)

# Compute only if valid
if spot_now <= 0.0:
    st.warning("Please enter a positive current spot to compute the forecast.")
else:
    # Annual -> monthly scaling under Normal i.i.d. log changes assumption
    mu_month = mu_used / 12.0                      # CHANGED: use selected drift
    sigma_month = sigma / np.sqrt(12.0)

    # For month h: cumulative mean and std of log change
    months = np.arange(1, int(horizon_m) + 1, dtype=int)
    mu_h = months * mu_month
    sigma_h = sigma * np.sqrt(months / 12.0)       # same as sigma_month * sqrt(months)

    # Point forecast and 95% CI for the SPOT (level), using log-normal mapping
    point = spot_now * np.exp(mu_h)
    lower = spot_now * np.exp(mu_h - 1.96 * sigma_h)
    upper = spot_now * np.exp(mu_h + 1.96 * sigma_h)

    # Future dates (month starts) based on last data timestamp, if available
    last_date = panel.index.max() if isinstance(panel.index, pd.DatetimeIndex) and len(panel.index) else None
    if pd.notna(last_date):
        # Next month start then monthly
        future_dates = pd.date_range((last_date + pd.offsets.MonthBegin(1)).replace(day=1), periods=len(months), freq="MS")
    else:
        future_dates = pd.RangeIndex(1, len(months) + 1, name="Month")

    # Table
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "month_ahead": months,
        "spot_point": point,
        "spot_lower_95": lower,
        "spot_upper_95": upper
    })

    st.dataframe(
        forecast_df.assign(
            spot_point=lambda d: d["spot_point"].round(6),
            spot_lower_95=lambda d: d["spot_lower_95"].round(6),
            spot_upper_95=lambda d: d["spot_upper_95"].round(6),
        ),
        use_container_width=True
    )

    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download forecast table (CSV)",
        data=csv_bytes,
        file_name="fx_spot_normal_CI_forecast.csv",
        mime="text/csv"
    )

    # -------------------------------
    # Plot the forecast with CI band
    # (Legend below; extra space between plot and x-axis)
    # -------------------------------
    fig2, ax2 = plt.subplots(figsize=(8.5, 5.0))

    x_axis = np.arange(len(months))
    ax2.plot(x_axis, point, linewidth=2, label="Spot point forecast (Normal)")
    ax2.fill_between(x_axis, lower, upper, alpha=0.2, label="95% CI (Normal)")

    # Robust ticks (≈12 ticks max) — avoids slicing bugs
    step = max(1, len(x_axis) // 12)
    tick_idx = np.arange(0, len(x_axis), step)

    if isinstance(future_dates, pd.DatetimeIndex):
        ax2.set_xticks(tick_idx)
        ax2.set_xticklabels([d.strftime("%Y-%m") for d in future_dates[tick_idx]], rotation=45, ha="right")
        ax2.set_xlabel("Forecast month")
    else:
        ax2.set_xticks(tick_idx)
        ax2.set_xticklabels([str(m) for m in months[tick_idx]], rotation=0, ha="center")
        ax2.set_xlabel("Months ahead")

    ax2.set_ylabel("Spot")
    ax2.set_title("95% confidence interval implied by normal distribution")
    ax2.grid(True, linestyle=":", linewidth=0.8)

    # >>> Extra space between graph and x-axis labels <<<
    ax2.tick_params(axis="x", pad=10)   # more gap from axis line to tick labels
    ax2.xaxis.labelpad = 14            # more gap from tick labels to axis label
    ax2.margins(y=0.10)                # add vertical breathing room inside axes

    # Legend BELOW the chart, pushed well away from x-labels
    leg = ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),  # further down
        ncol=2,
        frameon=False,
        fontsize="small"
    )

    # Tight layout, then add generous bottom margin for labels + legend
    fig2.tight_layout()
    fig2.subplots_adjust(bottom=0.46)  # increase if you still see crowding

    st.pyplot(fig2)
