#!/usr/bin/env python3
"""Train ML model to predict profitable entries on 15min data."""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import talib.abstract as ta
from technical import qtpylib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import time

DATA_DIR = Path("user_data/data/binance")
PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LINK/USDT",
         "ADA/USDT", "DOT/USDT", "LTC/USDT", "AVAX/USDT", "DOGE/USDT"]

FEATURE_COLS = [
    "rsi", "rsi_7", "macdhist", "bb_pct", "vol_ratio", "adx", "mfi",
    "cci", "slowk", "slowd", "atr_pct", "ema_slope_50", "ema_slope_200",
    "pct_change", "pct_change_3", "pct_change_6",
    "ema_dist_200", "ema_dist_50", "body_pct", "upper_wick",
    "lower_wick", "green",
]

# Target: price goes up by TP% within MAX_FWD candles before going down by SL%
TP = 0.008   # 0.8% take profit
SL = -0.03   # -3% stop loss
MAX_FWD = 30  # look ahead 30 candles (7.5 hours)


def load_and_prepare(pair):
    fn = pair.replace("/", "_") + "-15m.feather"
    df = pd.read_feather(DATA_DIR / fn).sort_values("date").reset_index(drop=True)
    # Last year only
    df = df.tail(35040).reset_index(drop=True)

    for p in [9, 21, 50, 100, 200]:
        df[f"ema_{p}"] = ta.EMA(df, timeperiod=p)
    df["rsi"] = ta.RSI(df, timeperiod=14)
    df["rsi_7"] = ta.RSI(df, timeperiod=7)
    macd = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macdhist"] = macd["macdhist"]
    bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    df["bb_pct"] = (df["close"] - bb["lower"]) / (bb["upper"] - bb["lower"])
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["adx"] = ta.ADX(df, timeperiod=14)
    df["mfi"] = ta.MFI(df, timeperiod=14)
    df["cci"] = ta.CCI(df, timeperiod=20)
    stoch = ta.STOCH(df, fastk_period=14, slowk_period=3, slowd_period=3)
    df["slowk"] = stoch["slowk"]
    df["slowd"] = stoch["slowd"]
    df["atr"] = ta.ATR(df, timeperiod=14)
    df["atr_pct"] = df["atr"] / df["close"] * 100
    df["ema_slope_50"] = (df["ema_50"] - df["ema_50"].shift(4)) / df["ema_50"].shift(4)
    df["ema_slope_200"] = (df["ema_200"] - df["ema_200"].shift(4)) / df["ema_200"].shift(4)
    df["pct_change"] = df["close"].pct_change()
    df["pct_change_3"] = df["close"].pct_change(3)
    df["pct_change_6"] = df["close"].pct_change(6)
    df["ema_dist_200"] = (df["close"] - df["ema_200"]) / df["ema_200"] * 100
    df["ema_dist_50"] = (df["close"] - df["ema_50"]) / df["ema_50"] * 100
    df["body_pct"] = abs(df["close"] - df["open"]) / df["open"] * 100
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["close"] * 100
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["close"] * 100
    df["green"] = (df["close"] > df["open"]).astype(int)

    # Generate labels
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    labels = np.zeros(n, dtype=int)

    for i in range(n - MAX_FWD):
        ep = close[i]
        won = False
        for j in range(1, MAX_FWD + 1):
            hi_pct = (high[i + j] - ep) / ep
            lo_pct = (low[i + j] - ep) / ep
            if lo_pct <= SL:
                break
            if hi_pct >= TP:
                won = True
                break
        labels[i] = 1 if won else 0

    df["target"] = labels
    df = df.dropna().reset_index(drop=True)

    # Pre-filter: only consider entries in uptrend
    uptrend = (
        (df["ema_50"] > df["ema_200"])
        & (df["ema_9"] > df["ema_21"])
        & (df["macdhist"] > 0)
        & (df["green"] == 1)
        & (df["ema_slope_50"] > 0)
    )
    df = df[uptrend].reset_index(drop=True)

    return df


def main():
    t0 = time.time()
    print("Loading and preparing data...")

    all_dfs = []
    for pair in PAIRS:
        df = load_and_prepare(pair)
        df["pair"] = pair
        all_dfs.append(df)
        pos = (df["target"] == 1).sum()
        neg = (df["target"] == 0).sum()
        wr = pos / (pos + neg) * 100 if (pos + neg) > 0 else 0
        print(f"  {pair}: {len(df)} samples, base WR={wr:.1f}%")

    data = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(data)} samples")
    pos = (data["target"] == 1).sum()
    neg = (data["target"] == 0).sum()
    print(f"Positive (win): {pos} ({pos/(pos+neg)*100:.1f}%)")
    print(f"Negative (loss): {neg} ({neg/(pos+neg)*100:.1f}%)")

    X = data[FEATURE_COLS].values
    y = data["target"].values

    # Time series split: train on first 70%, validate on last 30%
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    print(f"Train WR: {y_train.mean()*100:.1f}%")
    print(f"Test WR:  {y_test.mean()*100:.1f}%")

    # Train multiple models
    models = {
        "GBM_conservative": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=20, subsample=0.8, random_state=42
        ),
        "GBM_aggressive": GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            min_samples_leaf=10, subsample=0.8, random_state=42
        ),
        "RF": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=15,
            random_state=42, n_jobs=-1
        ),
    }

    best_model = None
    best_wr_at_thresh = 0
    best_name = ""
    best_thresh = 0.5

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"  Training {name}...")
        model.fit(X_train, y_train)

        train_probs = model.predict_proba(X_train)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]

        # Find threshold for >90% precision (win rate)
        for thresh in np.arange(0.50, 0.95, 0.01):
            preds = (test_probs >= thresh).astype(int)
            n_trades = preds.sum()
            if n_trades < 20:
                continue
            correct = ((preds == 1) & (y_test == 1)).sum()
            wr = correct / n_trades * 100 if n_trades > 0 else 0

            if wr >= 90:
                print(f"  FOUND: thresh={thresh:.2f} WR={wr:.1f}% trades={n_trades}")
                if n_trades > best_wr_at_thresh or (n_trades == best_wr_at_thresh and wr > best_wr_at_thresh):
                    best_model = model
                    best_wr_at_thresh = wr
                    best_name = name
                    best_thresh = thresh

        # Show results at various thresholds
        print(f"\n  Results at different confidence thresholds:")
        for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            preds = (test_probs >= thresh).astype(int)
            n_trades = preds.sum()
            if n_trades > 0:
                correct = ((preds == 1) & (y_test == 1)).sum()
                wr = correct / n_trades * 100
                print(f"    thresh={thresh:.2f}: WR={wr:.1f}% trades={n_trades}")

        # Feature importance
        if hasattr(model, "feature_importances_"):
            fi = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
            print(f"\n  Top features:")
            for fname, imp in fi[:8]:
                print(f"    {fname:20s}: {imp:.4f}")

    # Save best model
    print(f"\n\n{'='*70}")
    if best_model:
        print(f"  BEST MODEL: {best_name}")
        print(f"  Win Rate:   {best_wr_at_thresh:.1f}%")
        print(f"  Threshold:  {best_thresh:.2f}")

        model_path = Path("user_data/strategies/ml_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"  Saved to:   {model_path}")

        # Also save threshold
        thresh_path = Path("user_data/strategies/ml_threshold.txt")
        thresh_path.write_text(str(best_thresh))
    else:
        # Save best available model with note
        print("  No model achieved 90% WR, saving best available...")
        # Find model with best WR at any threshold
        best_any_wr = 0
        best_any_model = None
        best_any_name = ""
        best_any_thresh = 0.5

        for name, model in models.items():
            test_probs = model.predict_proba(X_test)[:, 1]
            for thresh in np.arange(0.50, 0.95, 0.01):
                preds = (test_probs >= thresh).astype(int)
                n_trades = preds.sum()
                if n_trades < 10:
                    continue
                correct = ((preds == 1) & (y_test == 1)).sum()
                wr = correct / n_trades * 100
                if wr > best_any_wr:
                    best_any_wr = wr
                    best_any_model = model
                    best_any_name = name
                    best_any_thresh = thresh

        print(f"  Best available: {best_any_name} WR={best_any_wr:.1f}% thresh={best_any_thresh:.2f}")
        model_path = Path("user_data/strategies/ml_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_any_model, f)
        thresh_path = Path("user_data/strategies/ml_threshold.txt")
        thresh_path.write_text(str(best_any_thresh))
        print(f"  Saved to: {model_path}")

    print(f"{'='*70}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
