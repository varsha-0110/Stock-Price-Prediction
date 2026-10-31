import copy
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_FRAC = 0.8
SKIP = 10          # ignore ARIMA warm-up residuals
REFIT_EVERY = 50   # ARIMA parameter re-fit interval on the test period


# ------------------------------------------------------------------ model pieces
class LSTMNet(nn.Module):
    def __init__(self, hidden=32, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_windows(series, window):
    X, y, tgt = [], [], []
    for t in range(window, len(series)):
        X.append(series[t - window:t]); y.append(series[t]); tgt.append(t)
    return np.array(X)[..., None], np.array(y)[:, None], np.array(tgt)


def to_t(a):
    return torch.tensor(a, dtype=torch.float32, device=device)


def fit_lstm(X, y, tgt, train_end, epochs, lr=1e-3, batch=64, val_frac=0.15):
    m = tgt < train_end
    Xtr, ytr = X[m], y[m]
    n_val = int(len(Xtr) * val_frac)
    Xt, yt, Xv, yv = Xtr[:-n_val], ytr[:-n_val], Xtr[-n_val:], ytr[-n_val:]

    model = LSTMNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(to_t(Xt), to_t(yt)), batch_size=batch, shuffle=True)
    Xv_t, yv_t = to_t(Xv), to_t(yv)

    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv_t), yv_t).item()
        if v < best_val:
            best_val, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(to_t(X)).cpu().numpy().ravel()


def metrics(name, pred, y_true, prev, with_direction=True):
    moved = pred != prev
    if with_direction and moved.any():
        direction = np.mean(np.sign(pred[moved] - prev[moved]) == np.sign(y_true[moved] - prev[moved])) * 100
    else:
        direction = np.nan
    return {
        "Model": name,
        "RMSE ($)": np.sqrt(mean_squared_error(y_true, pred)),
        "MAE ($)": mean_absolute_error(y_true, pred),
        "MAPE (%)": np.mean(np.abs((y_true - pred) / y_true)) * 100,
        "Direction acc. (%)": direction,
    }


# ------------------------------------------------------------------ full pipeline (cached)
@st.cache_data(show_spinner=False)
def load_prices(ticker, start):
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty:
        return None
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


@st.cache_data(show_spinner=False)
def run_pipeline(ticker, start, window, epochs, auto_order):
    close = load_prices(ticker, start)
    prices = close.values.astype(float)
    dates = close.index
    n = len(prices)
    split = int(TRAIN_FRAC * n)
    train_prices = prices[:split]
    test_idx = np.arange(split, n)
    y_true = prices[test_idx]
    prev = prices[test_idx - 1]

    # naive
    naive_pred = prev.copy()

    # ARIMA order (train data only)
    if auto_order:
        best = None
        for p in range(0, 3):
            for q in range(0, 2):
                try:
                    aic = ARIMA(train_prices, order=(p, 1, q)).fit().aic
                except Exception:
                    continue
                if best is None or aic < best[0]:
                    best = (aic, (p, 1, q))
        order = best[1]
    else:
        order = (5, 1, 0)

    # ARIMA walk-forward
    res = ARIMA(train_prices, order=order).fit()
    train_fitted = res.fittedvalues
    arima_pred = np.zeros(len(test_idx))
    for k, t in enumerate(test_idx):
        arima_pred[k] = res.forecast(1)[0]
        if (k + 1) % REFIT_EVERY == 0:
            res = ARIMA(prices[:t + 1], order=order).fit()
        else:
            res = res.append(prices[t:t + 1], refit=False)
    arima_next = float(res.forecast(1)[0])

    # LSTM on prices
    sc_price = StandardScaler().fit(train_prices.reshape(-1, 1))
    p_scaled = sc_price.transform(prices.reshape(-1, 1)).ravel()
    X, y, tgt = make_windows(p_scaled, window)
    lstm_model = fit_lstm(X, y, tgt, split, epochs)
    lstm_pred = sc_price.inverse_transform(predict(lstm_model, X[tgt >= split]).reshape(-1, 1)).ravel()
    lstm_next = float(sc_price.inverse_transform(
        predict(lstm_model, p_scaled[-window:].reshape(1, window, 1)).reshape(-1, 1))[0, 0])

    # Hybrid: LSTM on ARIMA residuals
    resid = np.zeros(n)
    resid[:split] = train_prices - train_fitted
    resid[:SKIP] = 0.0
    resid[split:] = y_true - arima_pred
    sc_res = StandardScaler().fit(resid[SKIP:split].reshape(-1, 1))
    r_scaled = sc_res.transform(resid.reshape(-1, 1)).ravel()
    Xr, yr, tr = make_windows(r_scaled, window)
    keep = tr >= SKIP + window
    Xr, yr, tr = Xr[keep], yr[keep], tr[keep]
    res_model = fit_lstm(Xr, yr, tr, split, epochs)
    resid_pred = sc_res.inverse_transform(predict(res_model, Xr[tr >= split]).reshape(-1, 1)).ravel()
    hybrid_pred = arima_pred + resid_pred
    resid_next = float(sc_res.inverse_transform(
        predict(res_model, r_scaled[-window:].reshape(1, window, 1)).reshape(-1, 1))[0, 0])
    hybrid_next = arima_next + resid_next

    table = pd.DataFrame([
        metrics("Naive", naive_pred, y_true, prev, with_direction=False),
        metrics("ARIMA", arima_pred, y_true, prev),
        metrics("LSTM", lstm_pred, y_true, prev),
        metrics("ARIMA-LSTM hybrid", hybrid_pred, y_true, prev),
    ]).set_index("Model")

    chart = pd.DataFrame({
        "Actual": y_true, "Naive": naive_pred, "ARIMA": arima_pred,
        "LSTM": lstm_pred, "Hybrid": hybrid_pred,
    }, index=dates[test_idx])

    return {
        "table": table, "chart": chart, "order": order,
        "last_date": dates[-1], "last_price": float(prices[-1]),
        "next": {"Naive": float(prices[-1]), "ARIMA": arima_next, "LSTM": lstm_next, "Hybrid": hybrid_next},
        "n_train": split, "n_test": n - split,
    }


# ------------------------------------------------------------------ UI
st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")
st.title("📈 Stock Price Prediction: ARIMA vs LSTM vs Hybrid")
st.caption("Next-day closing price. Educational demo, **not financial advice**.")

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", "AAPL").strip().upper()
    start = st.date_input("Start date", pd.Timestamp("2020-01-01")).strftime("%Y-%m-%d")
    window = st.slider("LSTM window (days)", 10, 60, 30, step=5)
    epochs = st.slider("Training epochs", 10, 150, 60, step=10,
                       help="More epochs = slower. 60 is a good demo value.")
    auto_order = st.checkbox("Auto-select ARIMA order (AIC)", value=True,
                             help="If off, uses ARIMA(5,1,0).")
    run = st.button("Run prediction", type="primary")

if not run:
    st.info("Choose a ticker in the sidebar and click **Run prediction**. "
            "The first run trains the models (about a minute); repeat runs with the same settings are instant.")
    st.stop()

if load_prices(ticker, start) is None:
    st.error(f"No data found for '{ticker}'. Check the ticker symbol.")
    st.stop()

with st.spinner(f"Fitting ARIMA and training LSTMs for {ticker}..."):
    out = run_pipeline(ticker, start, window, epochs, auto_order)

st.subheader(f"{ticker}: next-day forecast")
st.write(f"Last close on **{out['last_date'].date()}**: **${out['last_price']:.2f}**")
cols = st.columns(4)
for col, (name, val) in zip(cols, out["next"].items()):
    col.metric(name, f"${val:.2f}", f"{val - out['last_price']:+.2f}")

st.subheader("Test-period performance")
st.caption(f"ARIMA order used: {out['order']}  |  Train days: {out['n_train']}  |  Test days: {out['n_test']}")
st.dataframe(out["table"].round(3), use_container_width=True)

best = out["table"]["RMSE ($)"].idxmin()
st.success(f"Lowest RMSE: **{best}**. If Naive is close to the best, that is normal: "
           "daily prices are very close to a random walk.")

st.subheader("Actual vs predicted (test period)")
models = st.multiselect("Models to show", ["Naive", "ARIMA", "LSTM", "Hybrid"], default=["LSTM", "Hybrid"])
last_n = st.radio("Range", ["Full test period", "Last 60 days"], horizontal=True)
data = out["chart"][["Actual"] + models]
if last_n == "Last 60 days":
    data = data.tail(60)
st.line_chart(data)

st.subheader("Absolute error over time")
err = out["chart"][models].sub(out["chart"]["Actual"], axis=0).abs() if models else None
if err is not None:
    st.line_chart(err)
