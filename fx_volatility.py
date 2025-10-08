# -------------------------------
# Diagnostics (temporary; safe to delete/comment out)
# -------------------------------
with st.expander("Diagnostics (temporary; safe to delete)"):
    # 1) Mass check: does KDE integrate to ~1 over the current grid?
    total_area = float(np.trapz(pdf_kde_full, grid))
    st.write(f"DEBUG — KDE total area over grid: **{total_area:.4f}**")

    # 2) Grid coverage
    st.write(f"DEBUG — grid range: **[{grid.min():.6f}, {grid.max():.6f}]**")
    st.write(f"DEBUG — μ = {mu:.6f}, σ = {sigma:.6f},  left/right 95% bounds: "
             f"[{(mu - 1.96*sigma):.6f}, {(mu + 1.96*sigma):.6f}]")

    # 3) Tail probabilities recap
    st.write(f"DEBUG — Tail (Normal): **{p_tail_norm:.4f}**")
    st.write(f"DEBUG — Tail (KDE, trapezoid on grid): **{p_tail_kde:.4f}**")
    st.write(f"DEBUG — Tail (Empirical proportion): **{p_tail_emp:.4f}**")

    # 4) Monte Carlo tail from KDE (sample on z, map back to x)
    try:
        z_samp = kde_cv.sample(100_000, random_state=0)   # (100000, 1) on z-scale
        x_samp = mu_kde + sigma_kde * z_samp.ravel()      # map back to x-scale
        p_tail_mc = float(np.mean(np.abs(x_samp - mu) > 1.96 * sigma))
        st.write(f"DEBUG — Tail (KDE Monte Carlo ~100k): **{p_tail_mc:.4f}**")
    except Exception as e:
        st.write(f"DEBUG — KDE Monte Carlo sampling failed: {e}")

    # 5) Optional: bandwidth sensitivity on z-scale (Scott/Silverman-like magnitudes)
    if st.checkbox("Run bandwidth sensitivity check (z-scale)", value=False):
        bws = [0.20, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20]
        rows = []
        z = ((x - mu) / sigma).reshape(-1, 1)
        for bw in bws:
            kde_tmp = KernelDensity(kernel="gaussian", bandwidth=bw).fit(z)
            # Evaluate on the same x-grid for apples-to-apples comparison
            pdf_tmp = evaluate_kde_pdf_on_grid(kde_tmp, grid, mu, sigma)
            tail_tmp = tail_prob_from_pdf(grid, pdf_tmp, mu, sigma, k=1.96)
            area_tmp = float(np.trapz(pdf_tmp, grid))
            rows.append({"bandwidth_z": bw, "kde_tail": tail_tmp, "total_area": area_tmp})
        st.write("Bandwidth sensitivity (z-scale):")
        st.dataframe(pd.DataFrame(rows))
