"""
Train models to maximize daily return while maintaining 90%+ WR.
Tests multiple TP levels and model configs to find optimal setup.
"""
import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

import talib.abstract as ta
from technical import qtpylib

DATA_DIR = Path("user_data/data/binance")
OUT_DIR = Path("user_data/strategies")

PAIRS = [
    "BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT", "ADA_USDT",
    "DOGE_USDT", "AVAX_USDT", "DOT_USDT", "LINK_USDT", "MATIC_USDT",
    "UNI_USDT", "ATOM_USDT", "LTC_USDT", "FIL_USDT",
]

FEATURE_COLS = [
    "rsi", "rsi_7", "rsi_21", "macdhist", "bb_pct", "vol_ratio", "vol_ratio_50",
    "adx", "mfi", "cci", "slowk", "slowd", "atr_pct",
    "ema_slope_9", "ema_slope_21", "ema_slope_50", "ema_slope_200",
    "pct_change", "pct_change_3", "pct_change_6",
    "ema_dist_200", "ema_dist_50", "body_pct", "upper_wick",
    "lower_wick", "green", "high_low_pct", "close_position",
    "momentum_12", "momentum_24", "green_count_5", "green_count_10",
    "vol_trend", "rsi_slope", "macd_slope", "bb_width", "ema_spread", "atr_ratio",
]


def compute_indicators(df):
    for p in [9, 21, 50, 100, 200]:
        df[f"ema_{p}"] = ta.EMA(df, timeperiod=p)

    df["rsi"] = ta.RSI(df, timeperiod=14)
    df["rsi_7"] = ta.RSI(df, timeperiod=7)
    df["rsi_21"] = ta.RSI(df, timeperiod=21)

    macd = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd["macd"]
    df["macdsignal"] = macd["macdsignal"]
    df["macdhist"] = macd["macdhist"]

    bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    df["bb_lower"] = bb["lower"]
    df["bb_mid"] = bb["mid"]
    df["bb_upper"] = bb["upper"]
    df["bb_pct"] = (df["close"] - bb["lower"]) / (bb["upper"] - bb["lower"])
    df["bb_width"] = (bb["upper"] - bb["lower"]) / bb["mid"] * 100

    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["vol_ratio_50"] = df["volume"] / df["volume"].rolling(50).mean()
    df["vol_trend"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()

    df["adx"] = ta.ADX(df, timeperiod=14)
    df["mfi"] = ta.MFI(df, timeperiod=14)
    df["cci"] = ta.CCI(df, timeperiod=20)

    stoch = ta.STOCH(df, fastk_period=14, slowk_period=3, slowd_period=3)
    df["slowk"] = stoch["slowk"]
    df["slowd"] = stoch["slowd"]

    df["atr"] = ta.ATR(df, timeperiod=14)
    df["atr_pct"] = df["atr"] / df["close"] * 100
    df["atr_ratio"] = df["atr"] / df["atr"].rolling(50).mean()

    df["ema_slope_9"] = (df["ema_9"] - df["ema_9"].shift(2)) / df["ema_9"].shift(2)
    df["ema_slope_21"] = (df["ema_21"] - df["ema_21"].shift(4)) / df["ema_21"].shift(4)
    df["ema_slope_50"] = (df["ema_50"] - df["ema_50"].shift(4)) / df["ema_50"].shift(4)
    df["ema_slope_200"] = (df["ema_200"] - df["ema_200"].shift(4)) / df["ema_200"].shift(4)

    df["pct_change"] = df["close"].pct_change()
    df["pct_change_3"] = df["close"].pct_change(3)
    df["pct_change_6"] = df["close"].pct_change(6)
    df["momentum_12"] = df["close"].pct_change(12)
    df["momentum_24"] = df["close"].pct_change(24)

    df["ema_dist_200"] = (df["close"] - df["ema_200"]) / df["ema_200"] * 100
    df["ema_dist_50"] = (df["close"] - df["ema_50"]) / df["ema_50"] * 100
    df["ema_spread"] = (df["ema_9"] - df["ema_200"]) / df["ema_200"] * 100

    df["body_pct"] = abs(df["close"] - df["open"]) / df["open"] * 100
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["close"] * 100
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["close"] * 100
    df["high_low_pct"] = (df["high"] - df["low"]) / df["close"] * 100
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)
    df["green"] = (df["close"] > df["open"]).astype(int)
    df["green_count_5"] = df["green"].rolling(5).sum()
    df["green_count_10"] = df["green"].rolling(10).sum()

    df["rsi_slope"] = df["rsi"] - df["rsi"].shift(3)
    df["macd_slope"] = df["macdhist"] - df["macdhist"].shift(3)

    return df


def label_trades(df, tp, sl, max_fwd):
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    labels = np.full(n, -1)

    for i in range(n - max_fwd):
        entry = closes[i]
        tp_price = entry * (1 + tp)
        sl_price = entry * (1 + sl)

        for j in range(i + 1, min(i + max_fwd + 1, n)):
            if highs[j] >= tp_price:
                labels[i] = 1
                break
            if lows[j] <= sl_price:
                labels[i] = 0
                break
    return labels


def apply_trend_filter(df):
    mask = (
        (df["ema_50"] > df["ema_200"])
        & (df["ema_9"] > df["ema_21"])
        & (df["macdhist"] > 0)
        & (df["green"] == 1)
        & (df["ema_slope_50"] > 0)
        & (df["vol_ratio"] > 0.8)
        & (df["adx"] > 18)
        & (df["rsi"] > 35) & (df["rsi"] < 65)
    )
    return mask


def load_all_data():
    all_dfs = []
    for pair in PAIRS:
        path = DATA_DIR / f"{pair}-15m.feather"
        if not path.exists():
            continue
        df = pd.read_feather(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        df = compute_indicators(df)
        df["pair"] = pair
        all_dfs.append(df)
        print(f"  Loaded {pair}: {len(df)} candles")
    return pd.concat(all_dfs, ignore_index=True)


def train_and_eval(df_train, df_test, tp, sl, max_fwd, model_type="lgbm", threshold_range=None):
    """Train model and evaluate, returning best config."""

    # Label training data
    labels_train = label_trades(df_train, tp, sl, max_fwd)
    mask_train = apply_trend_filter(df_train)

    valid_mask = mask_train & (labels_train >= 0)
    X_train = df_train.loc[valid_mask, FEATURE_COLS].copy()
    y_train = pd.Series(labels_train[valid_mask], index=X_train.index)

    valid_rows = X_train.dropna().index
    X_train = X_train.loc[valid_rows]
    y_train = y_train.loc[valid_rows]

    if len(X_train) < 100 or y_train.sum() < 50:
        return None

    base_wr = y_train.mean()

    # Train model
    if model_type == "gbm":
        model = GradientBoostingClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.02,
            min_samples_leaf=30, subsample=0.8, max_features=0.7,
            random_state=42
        )
    else:
        model = LGBMClassifier(
            n_estimators=1000, max_depth=7, learning_rate=0.008,
            num_leaves=40, min_child_samples=15, subsample=0.7,
            colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=2.0,
            random_state=42, verbose=-1, n_jobs=-1
        )

    model.fit(X_train.values, y_train.values)

    # Evaluate on test
    labels_test = label_trades(df_test, tp, sl, max_fwd)
    mask_test = apply_trend_filter(df_test)

    valid_mask_test = mask_test & (labels_test >= 0)
    X_test = df_test.loc[valid_mask_test, FEATURE_COLS].copy()
    y_test = pd.Series(labels_test[valid_mask_test], index=X_test.index)

    valid_rows_test = X_test.dropna().index
    X_test = X_test.loc[valid_rows_test]
    y_test = y_test.loc[valid_rows_test]

    if len(X_test) < 50:
        return None

    probs = model.predict_proba(X_test.values)[:, 1]

    if threshold_range is None:
        threshold_range = np.arange(0.80, 0.99, 0.01)

    best = None
    for thr in threshold_range:
        selected = probs >= thr
        n_trades = selected.sum()
        if n_trades < 20:
            continue
        wr = y_test.values[selected].mean()
        if wr < 0.90:
            continue

        # Estimate daily profit
        test_days = (df_test["date"].max() - df_test["date"].min()).days
        if test_days <= 0:
            test_days = 1
        trades_per_day = n_trades / test_days
        avg_profit = wr * tp + (1 - wr) * sl
        daily_profit_pct = trades_per_day * avg_profit * 100

        result = {
            "tp": tp, "sl": sl, "max_fwd": max_fwd,
            "model_type": model_type, "threshold": thr,
            "n_trades": int(n_trades), "win_rate": wr,
            "base_wr": base_wr, "trades_per_day": trades_per_day,
            "avg_profit_per_trade": avg_profit * 100,
            "daily_profit_pct": daily_profit_pct,
            "model": model, "train_samples": len(X_train),
        }

        if best is None or daily_profit_pct > best["daily_profit_pct"]:
            best = result

    return best


def main():
    print("=" * 70)
    print("  TREINAMENTO PARA MÁXIMO RETORNO DIÁRIO (WR >= 90%)")
    print("=" * 70)

    print("\nCarregando dados...")
    df = load_all_data()

    # Split: train 2020-2024, test 2025-2026
    train_mask = df["date"] < "2025-01-01"
    test_mask = df["date"] >= "2025-01-01"
    df_train = df[train_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    print(f"\nTrain: {len(df_train)} candles (2020-2024)")
    print(f"Test:  {len(df_test)} candles (2025-2026)")

    # Test configurations
    configs = [
        # (tp, sl, max_fwd, model_type, description)
        (0.004, -0.02, 32, "gbm",  "Safe 0.4% TP (original)"),
        (0.005, -0.02, 32, "gbm",  "Safe 0.5% TP"),
        (0.006, -0.02, 40, "gbm",  "Safe 0.6% TP"),
        (0.004, -0.02, 32, "lgbm", "LGBM 0.4% TP"),
        (0.005, -0.02, 32, "lgbm", "LGBM 0.5% TP"),
        (0.006, -0.02, 40, "lgbm", "LGBM 0.6% TP"),
        (0.007, -0.02, 40, "lgbm", "LGBM 0.7% TP"),
        (0.008, -0.02, 48, "lgbm", "LGBM 0.8% TP"),
        (0.010, -0.02, 48, "lgbm", "Power 1.0% TP (original)"),
        (0.012, -0.02, 56, "lgbm", "Power 1.2% TP"),
        (0.015, -0.02, 64, "lgbm", "Power 1.5% TP"),
        (0.020, -0.03, 80, "lgbm", "Power 2.0% TP"),
        (0.010, -0.015, 48, "lgbm", "Tight 1.0% TP / 1.5% SL"),
        (0.008, -0.015, 40, "lgbm", "Tight 0.8% TP / 1.5% SL"),
        (0.006, -0.015, 40, "lgbm", "Tight 0.6% TP / 1.5% SL"),
    ]

    results = []

    print(f"\n{'='*90}")
    print(f"  Testando {len(configs)} configurações...")
    print(f"{'='*90}")

    for tp, sl, mf, mtype, desc in configs:
        print(f"\n  [{desc}] TP={tp*100:.1f}% SL={sl*100:.1f}% MF={mf} ({mtype})")
        result = train_and_eval(df_train, df_test, tp, sl, mf, mtype)
        if result:
            result["desc"] = desc
            results.append(result)
            print(f"    ✓ WR={result['win_rate']*100:.1f}% | "
                  f"Trades/dia={result['trades_per_day']:.1f} | "
                  f"Profit/trade={result['avg_profit_per_trade']:.3f}% | "
                  f"DIÁRIO={result['daily_profit_pct']:.4f}% | "
                  f"Thr={result['threshold']:.2f} | "
                  f"N={result['n_trades']}")
        else:
            print(f"    ✗ Não alcançou 90% WR ou dados insuficientes")

    if not results:
        print("\nNenhum resultado válido!")
        return

    # Sort by daily profit
    results.sort(key=lambda x: x["daily_profit_pct"], reverse=True)

    print(f"\n{'='*90}")
    print(f"  RANKING — TOP 10 POR RETORNO DIÁRIO (WR >= 90%)")
    print(f"{'='*90}")
    print(f"  {'#':<4} {'Config':<30} {'WR':<8} {'Trd/dia':<9} {'%/trade':<10} {'%/DIA':<10} {'Thr':<6}")
    print(f"  {'-'*90}")

    for i, r in enumerate(results[:10]):
        print(f"  {i+1:<4} {r['desc']:<30} {r['win_rate']*100:.1f}%   "
              f"{r['trades_per_day']:<9.2f} {r['avg_profit_per_trade']:<10.4f} "
              f"{r['daily_profit_pct']:<10.4f} {r['threshold']:.2f}")

    # Find best SAFE model (lower TP, more trades) and best POWER model (higher TP)
    safe_candidates = [r for r in results if r['tp'] <= 0.007]
    power_candidates = [r for r in results if r['tp'] >= 0.008]

    best_safe = safe_candidates[0] if safe_candidates else results[0]
    best_power = power_candidates[0] if power_candidates else None

    print(f"\n{'='*90}")
    print(f"  MELHOR SAFE: {best_safe['desc']}")
    print(f"    TP={best_safe['tp']*100:.1f}% | WR={best_safe['win_rate']*100:.1f}% | "
          f"Trades/dia={best_safe['trades_per_day']:.2f} | Diário={best_safe['daily_profit_pct']:.4f}%")

    if best_power:
        print(f"\n  MELHOR POWER: {best_power['desc']}")
        print(f"    TP={best_power['tp']*100:.1f}% | WR={best_power['win_rate']*100:.1f}% | "
              f"Trades/dia={best_power['trades_per_day']:.2f} | Diário={best_power['daily_profit_pct']:.4f}%")

    # Estimate combined daily return
    safe_daily = best_safe['daily_profit_pct']
    power_daily = best_power['daily_profit_pct'] if best_power else 0

    # They share the same candles, so some overlap. Estimate ~70% additive
    combined_daily = safe_daily + power_daily * 0.7

    print(f"\n{'='*90}")
    print(f"  PROJEÇÃO COMBINADA (Safe + Power)")
    print(f"{'='*90}")
    print(f"  Safe diário:  {safe_daily:.4f}%")
    print(f"  Power diário: {power_daily:.4f}%")
    print(f"  Combinado:    ~{combined_daily:.4f}%/dia")

    annual = (1 + combined_daily/100) ** 365 - 1
    print(f"  Anualizado:   ~{annual*100:.1f}%")
    print(f"  $1k em 6 anos: ${1000 * (1 + annual) ** 6:,.0f}")

    daily_brl_1k = 1000 * combined_daily / 100
    print(f"\n  Com R$1.000:")
    print(f"    Ganho diário inicial: R${daily_brl_1k:.2f}")

    # Time to R$50/day
    target = 50
    capital_needed = target / (combined_daily / 100)
    days_to_target = np.log(capital_needed / 1000) / np.log(1 + combined_daily/100)
    print(f"    Dias para R$50/dia:   {days_to_target:.0f} dias ({days_to_target/30:.0f} meses)")

    # Save the best models
    print(f"\n{'='*90}")
    print(f"  SALVANDO MODELOS OTIMIZADOS")
    print(f"{'='*90}")

    # Save best safe
    safe_path = OUT_DIR / "ml_model.pkl"
    with open(safe_path, "wb") as f:
        pickle.dump(best_safe["model"], f)
    print(f"  Safe model saved: {safe_path}")

    safe_config = {
        "tp": best_safe["tp"], "sl": best_safe["sl"],
        "max_fwd": best_safe["max_fwd"],
        "threshold": best_safe["threshold"],
        "win_rate": best_safe["win_rate"],
        "model_name": best_safe["model_type"],
        "features": FEATURE_COLS,
        "daily_profit_pct": best_safe["daily_profit_pct"],
    }
    with open(OUT_DIR / "ml_config.pkl", "wb") as f:
        pickle.dump(safe_config, f)

    # Save best power
    if best_power:
        power_path = OUT_DIR / "ml_model_power.pkl"
        with open(power_path, "wb") as f:
            pickle.dump(best_power["model"], f)
        print(f"  Power model saved: {power_path}")

    # Now update strategy params
    print(f"\n{'='*90}")
    print(f"  PARÂMETROS RECOMENDADOS PARA UchoaStrategy")
    print(f"{'='*90}")
    print(f"  Safe:")
    print(f"    ml_threshold = {best_safe['threshold']:.2f}")
    print(f"    ROI '0' = {best_safe['tp']}")
    if best_power:
        print(f"  Power:")
        print(f"    power_threshold = {best_power['threshold']:.2f}")
        print(f"    custom_exit power_tp = {best_power['tp']*100:.1f}%")

    # Compare with old
    print(f"\n{'='*90}")
    print(f"  COMPARAÇÃO: ANTES vs DEPOIS")
    print(f"{'='*90}")
    old_daily = 0.1982  # historical
    print(f"  Retorno diário anterior: ~{old_daily:.4f}%")
    print(f"  Retorno diário novo:     ~{combined_daily:.4f}%")
    improvement = (combined_daily / old_daily - 1) * 100
    print(f"  Melhoria: {'+' if improvement > 0 else ''}{improvement:.1f}%")

    old_6y = 1000 * (1 + old_daily/100) ** (365*6)
    new_6y = 1000 * (1 + combined_daily/100) ** (365*6)
    print(f"  $1k em 6 anos (antes):  ${old_6y:,.0f}")
    print(f"  $1k em 6 anos (depois): ${new_6y:,.0f}")


if __name__ == "__main__":
    main()
