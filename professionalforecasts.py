# professionalforecasts.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Token–Context Expectations (Professional Forecasts)", layout="wide")

# ==============================
# UI
# ==============================
st.title("🧠 Token–Context Expectations for Professional Forecasts")
st.markdown(
    """
Model professional forecasts (e.g., SPF/ECB-SPF) using a **token + context window** (attention) framework inspired by LLMs.

**How it works**
- Each time-stamped macro signal is a **token** (e.g., CPI surprise, FOMC tone, oil shock, news embedding).
- A **context window** of the last *N* token dates is attended with learnable recency and salience adjustments.
- The model maps the attended context to the observed **professional forecast** for a given variable and horizon.

**What you need**
- `SPF forecasts` CSV with columns: `date, forecaster_id, variable, horizon, forecast`
- `Macro tokens` CSV with columns: `date, token_name, token_vector` *(comma-separated floats)* **or** `value`, and optional `salience`
- (Optional) `Realizations` CSV with columns: `date, variable, value`
"""
)

# ==============================
# Utilities
# ==============================
def parse_vec(s):
    """Parses 'a,b,c' → np.array([a,b,c]); supports scalar 'value' fallback."""
    if isinstance(s, (float, int)):
        return np.array([float(s)], dtype=np.float32)
    if pd.isna(s):
        return None
    return np.array([float(x) for x in str(s).split(",")], dtype=np.float32)

def build_token_panel(tokens_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses tokens to one vector per date by mean-pooling vectors and salience.
    Returns DataFrame with columns: date, salience, vec (np.array).
    """
    records = {}
    for _, row in tokens_df.iterrows():
        d = row["date"]
        vec = parse_vec(row.get("token_vector")) if "token_vector" in row else None
        if vec is None and "value" in row and not pd.isna(row["value"]):
            vec = np.array([float(row["value"])], dtype=np.float32)
        if vec is None:
            continue
        sal = float(row.get("salience", 0.0)) if not pd.isna(row.get("salience", np.nan)) else 0.0
        if d not in records:
            records[d] = {"vecs": [vec], "sal": [sal]}
        else:
            records[d]["vecs"].append(vec)
            records[d]["sal"].append(sal)

    dates = sorted(records.keys())
    if not dates:
        return pd.DataFrame(columns=["date", "salience", "vec"])

    X, S = [], []
    for d in dates:
        mat = np.stack(records[d]["vecs"], axis=0)
        X.append(mat.mean(axis=0))
        S.append(np.mean(records[d]["sal"]))
    X = np.stack(X, axis=0)
    return pd.DataFrame({"date": dates, "salience": S, "vec": list(X)})

class SingleHeadAttention(nn.Module):
    """
    Single-head attention with:
    - learnable temperature (peakedness),
    - recency penalty (down-weights older tokens),
    - salience weight (boosts tokens with higher salience).
    """
    def __init__(self, d_in, d_q=16, d_k=16, d_v=16):
        super().__init__()
        self.Wq = nn.Linear(d_in, d_q, bias=True)
        self.Wk = nn.Linear(d_in, d_k, bias=True)
        self.Wv = nn.Linear(d_in, d_v, bias=True)
        self.temp = nn.Parameter(torch.tensor(1.0))     # >=0 via ReLU in forward
        self.recency = nn.Parameter(torch.tensor(0.1))  # >=0 via ReLU in forward
        self.sal_w = nn.Parameter(torch.tensor(0.5))    # tanh for bounded effect

    def forward(self, Q, Ks, Vs, ages, sal):
        # Q: [B, d_in]; Ks,Vs: [B, N, d_in]; ages,sal: [B, N]
        q = self.Wq(Q)               # [B, d_q]
        k = self.Wk(Ks)              # [B, N, d_k]
        v = self.Wv(Vs)              # [B, N, d_v]

        scale = torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32)) + 1e-8
        att = torch.einsum("bd,bnd->bn", q, k) / scale  # dot-product attention

        # Recency penalty & salience bonus
        att = att - torch.relu(self.recency) * ages + torch.tanh(self.sal_w) * sal

        # Softmax with temperature
        att = nn.functional.softmax(att / torch.relu(self.temp), dim=1)  # [B, N]
        out = torch.einsum("bn,bnd->bd", att, v)                         # [B, d_v]
        return out, att

class TokenContextModel(nn.Module):
    def __init__(self, d_in, hidden=32):
        super().__init__()
        self.att = SingleHeadAttention(d_in, d_q=hidden, d_k=hidden, d_v=hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, Q, Ks, Vs, ages, sal):
        z, att = self.att(Q, Ks, Vs, ages, sal)
        yhat = self.head(z).squeeze(-1)
        return yhat, att

def make_batches(spf_df, tok_panel, variable, horizon, context):
    """
    For each SPF vintage date, builds a rolling window of the last `context` token dates (<= vintage),
    and uses the most recent token vector as the query.
    Returns tensors and metadata (forecaster_id, vintage).
    """
    spf = spf_df[(spf_df["variable"] == variable) & (spf_df["horizon"] == horizon)].copy()
    spf = spf.sort_values("date")
    tok_panel = tok_panel.sort_values("date").reset_index(drop=True)
    if len(tok_panel) == 0 or len(spf) == 0:
        return None

    t_dates = pd.to_datetime(tok_panel["date"]).values
    Xq, Xk, Xv, Ages, Sal, y, meta = [], [], [], [], [], [], []

    for _, row in spf.iterrows():
        d = pd.to_datetime(row["date"])
        idx = np.where(t_dates <= np.datetime64(d))[0]
        if len(idx) < context:
            continue
        use_idx = idx[-context:]
        sub = tok_panel.iloc[use_idx]
        vecs = np.stack(sub["vec"].to_list(), axis=0)             # [N, d]
        sal = sub["salience"].to_numpy(dtype=np.float32)          # [N]
        ages = np.arange(context - 1, -1, -1, dtype=np.float32)   # older→newer
        q = vecs[-1]                                              # query = most recent vector

        Xq.append(q); Xk.append(vecs); Xv.append(vecs); Ages.append(ages); Sal.append(sal)
        y.append(float(row["forecast"]))
        meta.append((row.get("forecaster_id", "AVG"), row["date"]))

    if len(y) == 0:
        return None

    Xq = torch.tensor(np.stack(Xq), dtype=torch.float32)
    Xk = torch.tensor(np.stack(Xk), dtype=torch.float32)
    Xv = torch.tensor(np.stack(Xv), dtype=torch.float32)
    Ages = torch.tensor(np.stack(Ages), dtype=torch.float32)
    Sal = torch.tensor(np.stack(Sal), dtype=torch.float32)
    y = torch.tensor(np.array(y, dtype=np.float32))
    return Xq, Xk, Xv, Ages, Sal, y, meta

def df_downloader(df, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"Download {filename}", data=csv, file_name=filename, mime="text/csv")

# ==============================
# Sidebar controls
# ==============================
st.sidebar.header("1) Upload your CSVs")
spf_file = st.sidebar.file_uploader("SPF forecasts CSV", type=["csv"], help="Required. date, forecaster_id, variable, horizon, forecast")
tokens_file = st.sidebar.file_uploader("Macro tokens CSV", type=["csv"], help="Required. date, token_name, token_vector (or value), [salience]")
real_file = st.sidebar.file_uploader("Realizations CSV (optional)", type=["csv"], help="Optional. date, variable, value")

st.sidebar.header("2) Model settings")
variable = st.sidebar.text_input("Variable", value="CPI", help="e.g., CPI, PCE, GDP, UNRATE")
horizon = st.sidebar.number_input("Horizon (periods ahead, quarters)", value=4, step=1, min_value=1, max_value=40)
context = st.sidebar.number_input("Context window length (N)", value=8, step=1, min_value=2, max_value=60)
hidden = st.sidebar.number_input("Hidden size (attention/head)", value=32, step=1, min_value=4, max_value=256)
epochs = st.sidebar.number_input("Epochs", value=200, step=10, min_value=50, max_value=3000)
lr = st.sidebar.number_input("Learning rate", value=0.01, step=0.005, format="%.3f")
seed = st.sidebar.number_input("Random seed", value=0, step=1, min_value=0)
standardize = st.sidebar.checkbox("Standardize token vectors", value=True)

run_button = st.sidebar.button("🚀 Train model")

# ==============================
# Main run
# ==============================
if run_button:
    # --- Load inputs ---
    if spf_file is None or tokens_file is None:
        st.error("Please upload both SPF and Macro Tokens CSVs.")
        st.stop()

    spf_df = pd.read_csv(spf_file)
    tok_df = pd.read_csv(tokens_file)
    real_df = pd.read_csv(real_file) if real_file is not None else None

    need_cols_spf = {"date", "variable", "horizon", "forecast"}
    if not need_cols_spf.issubset(set(spf_df.columns)):
        st.error(f"SPF file must include columns: {need_cols_spf}")
        st.stop()

    if "date" not in tok_df.columns:
        st.error("Tokens file must include at least a 'date' column plus 'token_vector' or 'value'.")
        st.stop()

    # --- Build token panel ---
    tok_panel = build_token_panel(tok_df)
    if len(tok_panel) == 0:
        st.error("No usable tokens found. Ensure 'token_vector' (comma-separated) or 'value' is present.")
        st.stop()

    V = np.stack(tok_panel["vec"].to_list(), axis=0)
    if standardize:
        scaler = StandardScaler()
        Vz = scaler.fit_transform(V).astype(np.float32)
    else:
        Vz = V.astype(np.float32)
    tok_panel["vec"] = list(Vz)

    # --- Batches ---
    batches = make_batches(spf_df, tok_panel, variable, int(horizon), int(context))
    if batches is None:
        st.error("Not enough overlapping data given the context window. Reduce N or add more tokens.")
        st.stop()

    Xq, Xk, Xv, Ages, Sal, y, meta = batches
    d_in = Xq.shape[-1]

    # --- Train ---
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    model = TokenContextModel(d_in=d_in, hidden=int(hidden))
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()

    losses = []
    prog = st.progress(0, text="Training...")
    for ep in range(int(epochs)):
        model.train(); opt.zero_grad()
        yhat, att = model(Xq, Xk, Xv, Ages, Sal)
        loss = loss_fn(yhat, y)
        loss.backward(); opt.step()
        losses.append(float(loss.item()))
        # simple progress update
        if (ep + 1) % max(1, int(epochs) // 100) == 0:
            prog.progress(min((ep + 1) / epochs, 1.0), text=f"Training... epoch {ep+1}/{int(epochs)}")

    model.eval()
    with torch.no_grad():
        yhat, att = model(Xq, Xk, Xv, Ages, Sal)
        mse = loss_fn(yhat, y).item()

    st.success(f"Training complete. In-sample MSE: {mse:.4f}")

    # --- Plots / Tables ---
    fig1 = plt.figure()
    plt.plot(losses)
    plt.title("Training MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    st.pyplot(fig1)

    out_df = pd.DataFrame({
        "forecaster_id": [m[0] for m in meta],
        "vintage": [m[1] for m in meta],
        "variable": variable,
        "horizon": int(horizon),
        "forecast_observed": y.numpy(),
        "forecast_model": yhat.numpy(),
        "squared_error": (yhat - y).numpy() ** 2
    })
    st.subheader("Observed vs Model Forecasts")
    st.dataframe(out_df)

    df_downloader(out_df, "model_vs_spf.csv")

    # Attention heatmap
    att_np = att.numpy()
    if att_np.shape[0] > 1:
        fig2 = plt.figure()
        plt.imshow(att_np, aspect="auto")
        plt.colorbar(label="Attention weight")
        plt.title("Attention Weights by Vintage (rows) and Lags (cols)")
        plt.xlabel("Lag (older → newer)")
        plt.ylabel("Vintage index")
        st.pyplot(fig2)

    att_df = pd.DataFrame(att_np, columns=[f"lag_{i}" for i in range(int(context), 0, -1)])
    att_df.insert(0, "vintage", [m[1] for m in meta])
    st.subheader("Attention Weights (table)")
    st.dataframe(att_df)
    df_downloader(att_df, "attention_weights.csv")

    # Optional: naive join to realizations on the vintage date (for a quick look)
    if real_df is not None:
        try:
            rsub = real_df[real_df["variable"] == variable].copy()
            rsub["date"] = pd.to_datetime(rsub["date"])
            out_df["vintage"] = pd.to_datetime(out_df["vintage"])
            merged = out_df.merge(rsub, left_on="vintage", right_on="date", how="left", suffixes=("", "_real")).drop(columns=["date"])
            st.subheader("With Realizations (naive merge on vintage date)")
            st.dataframe(merged)
            df_downloader(merged, "model_vs_spf_with_realizations.csv")
        except Exception as e:
            st.warning(f"Could not merge realizations for diagnostics: {e}")
