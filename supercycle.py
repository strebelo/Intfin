# supercycle_game_matrix_discrete.py
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Helper functions
# -----------------------------
def ar1_path(T, rho, mu=1.0, sigma=0.05, seed=0, positive_bump_periods=5, bump_size=1.0):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T)
    if positive_bump_periods > 0:
        eps[:positive_bump_periods] = bump_size
    alpha = np.empty(T)
    alpha[0] = mu
    for t in range(1, T):
        alpha[t] = (1 - rho) * mu + rho * alpha[t - 1] + sigma * eps[t]
    alpha[0] = (1 - rho) * mu + rho * mu + sigma * eps[0]
    return alpha

def equilibrium_price_path(T, k, alpha_c, alpha_i, theta_c, theta_i, p_init=None):
    if p_init is None:
        p_init = np.ones(k)
    p = np.empty(T)
    p[:k] = p_init
    for t in range(k, T):
        denom = alpha_i[t - k] * (p[t - k] ** theta_i)
        denom = max(denom, 1e-10)
        p_t = (alpha_c[t] / denom) ** (1.0 / theta_c)
        p[t] = max(p_t, 1e-10)
    return p

def step_assets(a_t, r, p_t, delivered, i_t):
    return a_t * (1.0 + r) + p_t * delivered - i_t

# -----------------------------
# Streamlit layout and sidebar
# -----------------------------
st.set_page_config(page_title="Investment-Lag Model (Game Mode)", layout="wide")
st.title("Investment-Lag Commodity Model")

with st.sidebar:
    k = st.number_input("Lag k (periods)", min_value=1, max_value=24, value=4, step=1)

    theta_c = st.number_input("Demand elasticity theta_c", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    theta_i = st.number_input("Investment elasticity theta_i", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

    rho_c = st.slider("rho_c (demand persistence)", 0.0, 0.99, 0.8, 0.01)
    rho_i = st.slider("rho_i (investment persistence)", 0.0, 0.99, 0.8, 0.01)
    mu_c = st.number_input("mu_c (mean demand)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    mu_i = st.number_input("mu_i (mean investment)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    sigma_c = st.number_input("sigma_c (demand shocks)", min_value=0.0, max_value=2.0, value=0.05, step=0.01)
    sigma_i = st.number_input("sigma_i (investment shocks)", min_value=0.0, max_value=2.0, value=0.05, step=0.01)

    bump_periods = st.slider("Positive shock periods", 0, 50, 10, 1)
    bump_size = st.number_input("Bump size", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    seed = st.number_input("Random seed", min_value=0, max_value=10000, value=1234, step=1)

    a0 = st.number_input("Initial assets a0", min_value=-1_000_000.0, max_value=1_000_000.0, value=0.0, step=100.0, format="%.2f")
    p_init_val = st.number_input("Initial pre-sample price (repeated k times)", min_value=1e-6, max_value=1e6, value=1.0, step=0.1, format="%.6f")
    i_hist_val = st.number_input("Inherited pre-sample investment (repeated k times)", min_value=0.0, max_value=1e6, value=0.0, step=10.0, format="%.2f")

    r = st.number_input("Interest rate r (per period)", min_value=-0.99, max_value=10.0, value=0.01, step=0.01)

# -----------------------------
# Initialize / reset simulation
# -----------------------------
def init_sim(T_new):
    alpha_c = ar1_path(int(T_new), rho_c, mu_c, sigma_c, int(seed) + 1, int(bump_periods), bump_size)
    alpha_i = ar1_path(int(T_new), rho_i, mu_i, sigma_i, int(seed) + 2, int(bump_periods), bump_size)
    p = equilibrium_price_path(int(T_new), int(k), alpha_c, alpha_i, theta_c, theta_i, np.full(int(k), float(p_init_val)))

    st.session_state.T = int(T_new)
    st.session_state.k = int(k)
    st.session_state.r = float(r)
    st.session_state.p = p
    st.session_state.i_hist = np.full(int(k), float(i_hist_val))
    st.session_state.i = np.full(int(T_new), np.nan)
    st.session_state.a = np.empty(int(T_new) + 1)
    st.session_state.a[0] = float(a0)
    st.session_state.t = 0
    st.session_state.initialized = True

# First-time setup default T
if "initialized" not in st.session_state:
    init_sim(200)

# -----------------------------
# Left controls (T + period controls) and Right table
# -----------------------------
left, right = st.columns([1, 2])

with left:
    st.subheader("Game Controls")

    # Let user set the length T here (left side)
    T_input = st.number_input("Game length T", min_value=10, max_value=1000, value=int(st.session_state.T), step=10)
    apply_T = st.button("Apply T and Reset")
    if apply_T:
        init_sim(int(T_input))
        st.rerun()

    # Current state
    t = st.session_state.t
    T = st.session_state.T
    p = st.session_state.p
    a = st.session_state.a
    i = st.session_state.i
    i_hist = st.session_state.i_hist
    k = st.session_state.k
    r = st.session_state.r

    # Show current period number (after 10 plays, this reads 10)
    st.subheader(f"Current period: {t}")

    # Discrete slider and commit
    if t < T:
        inv_options = [0, 5, 10, 15, 20]
        default_i = 0 if np.isnan(i[t]) else int(i[t])
        i_t_choice = st.select_slider("Choose investment i_t", options=inv_options, value=default_i, key=f"i_choice_{t}")

        commit = st.button("Commit i_t and advance")
        if commit:
            # Delivered depends on i_{t-k} or pre-sample history
            if (t - k) >= 0 and not np.isnan(i[t - k]):
                delivered = float(i[t - k])
            else:
                delivered = float(i_hist[t - k]) if (t - k) < 0 else 0.0

            i[t] = float(i_t_choice)
            a[t + 1] = step_assets(a[t], r, p[t], delivered, i[t])
            st.session_state.t = t + 1
            st.rerun()
    else:
        # Game ended: show final assets and correlation
        realized_i = st.session_state.i[:T]
        realized_p = st.session_state.p[:T]
        mask = ~np.isnan(realized_i)
        corr_txt = "n/a"
        if np.sum(mask) > 1 and np.std(realized_i[mask]) > 0 and np.std(realized_p[mask]) > 0:
            corr_val = float(np.corrcoef(realized_i[mask], realized_p[mask])[0, 1])
            corr_txt = f"{corr_val:.4f}"

        st.subheader("Results")
        st.metric("Final assets", f"{st.session_state.a[T]:.4f}")
        st.metric("Corr(investment, price)", corr_txt)

with right:
    # Helper to compute delivered
    def delivered_at(j):
        if (j - k) >= 0 and not np.isnan(i[j - k]):
            return float(i[j - k])
        return float(i_hist[j - k]) if (j - k) < 0 else 0.0

    # Build matrix for realized periods ONLY (0..t-1), never index p[t]
    t = st.session_state.t
    T = st.session_state.T
    p = st.session_state.p
    a = st.session_state.a
    i = st.session_state.i

    if t > 0:
        out_hist = [delivered_at(j) for j in range(t)]
        rev_hist = [p[j] * out_hist[j] for j in range(t)]
        periods = np.arange(1, t + 1)  # 1-based period numbers

        df = pd.DataFrame(
            {
                "period": periods,
                "price": p[:t],
                "output": out_hist,
                "revenue": rev_hist,
                "assets": a[1:t+1],  # end-of-period assets
                "investment": i[:t],
            }
        )
        # Show most recent first
        df_display = df.iloc[::-1].reset_index(drop=True)
    else:
        df_display = pd.DataFrame(
            {
                "period": [1],
                "price": [p[0]],
                "output": [np.nan],
                "revenue": [np.nan],
                "assets": [np.nan],
                "investment": [np.nan],
            }
        )

    st.subheader("History (latest first)")
    st.dataframe(df_display, use_container_width=True)

    # Download
    csv = df_display.to_csv(index=False).encode()
    st.download_button("Download table (csv)", csv, file_name="simulation.csv", mime="text/csv")
