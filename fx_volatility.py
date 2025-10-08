# (excerpt showing only the modified plotting section)

# -------------------------------
# Distribution Plot (with toggles)
# -------------------------------
st.subheader("Distribution: Histogram with optional overlays")

# Overlays toggles
ocol1, ocol2, ocol3 = st.columns(3)
with ocol1:
    show_normal = st.checkbox("Show Normal distribution overlay", value=True)
with ocol2:
    show_kde = st.checkbox("Show Kernel Density Estimation (CV Gaussian) overlay", value=True)
with ocol3:
    show_ci = st.checkbox("Show 95% Normal interval (μ ± 1.96σ)", value=True)

# Grid: padding fixed to 1σ beyond min/max and ±4σ envelope around μ
pad_stds = 1.0
xmin, xmax = np.min(x), np.max(x)
lo = min(mu - 4 * sigma, xmin) - pad_stds * sigma
hi = max(mu + 4 * sigma, xmax) + pad_stds * sigma
grid = np.linspace(lo - 0.5 * sigma, hi + 0.5 * sigma, 4000)

# Evaluate overlays
pdf_norm = normal_pdf(grid, mu, sigma) if show_normal else None
pdf_kde = evaluate_kde_pdf_on_grid(kde_cv, grid, mu_kde, sigma_kde) if show_kde else None

fig1, ax1 = plt.subplots(figsize=(7, 4.25))
ax1.hist(x, bins="auto", density=True, alpha=0.6, edgecolor="black", label="Histogram")

# 95% Normal CI shading (elegant band)
if show_ci:
    left = mu - 1.96 * sigma
    right = mu + 1.96 * sigma
    ax1.axvspan(left, right, alpha=0.15, label="95% Normal interval (μ ± 1.96σ)")

if show_normal:
    ax1.plot(grid, pdf_norm, linewidth=2, label="Normal PDF")
if show_kde:
    ax1.plot(grid, pdf_kde, linewidth=2, linestyle="--", label="Kernel Density Estimation (CV Gaussian) PDF")

ax1.set_xlabel("Annual log change")
ax1.set_ylabel("Density")
ax1.set_title("Histogram with Optional Normal / KDE Overlays")
ax1.grid(True, linestyle=":", linewidth=0.8)

# Place legend **below** the figure
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

st.pyplot(fig1)
