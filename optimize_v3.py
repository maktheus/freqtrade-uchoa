#!/usr/bin/env python3
"""V3: High win-rate scalping optimizer. Key insight: only exit at profit or stoploss."""
import numpy as np
import pandas as pd
from pathlib import Path
import talib.abstract as ta
from technical import qtpylib

DATA_DIR = Path("user_data/data/binance")
INITIAL_CAPITAL = 1000.0
MAX_OPEN_TRADES = 5
STAKE_PCT = 1.0 / MAX_OPEN_TRADES
COMMISSION = 0.001

ALL_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "UNI/USDT",
    "ATOM/USDT", "LTC/USDT", "FIL/USDT",
]


def load_pair(pair: str) -> pd.DataFrame:
    fn = pair.replace("/", "_") + "-5m.feather"
    df = pd.read_feather(DATA_DIR / fn).sort_values("date").reset_index(drop=True)

    for p in [5, 8, 9, 13, 21, 34, 50, 100, 200]:
        df[f"ema_{p}"] = ta.EMA(df, timeperiod=p)

    df["rsi"] = ta.RSI(df, timeperiod=14)
    df["rsi_7"] = ta.RSI(df, timeperiod=7)

    macd = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd["macd"]
    df["macdsignal"] = macd["macdsignal"]
    df["macdhist"] = macd["macdhist"]

    bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    df["bb_lower"] = bb["lower"]
    df["bb_mid"] = bb["mid"]
    df["bb_upper"] = bb["upper"]
    df["bb_width"] = (bb["upper"] - bb["lower"]) / bb["mid"]

    bb15 = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=1.5)
    df["bb_lower_15"] = bb15["lower"]
    df["bb_upper_15"] = bb15["upper"]

    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["adx"] = ta.ADX(df, timeperiod=14)
    df["atr"] = ta.ATR(df, timeperiod=14)
    df["cci"] = ta.CCI(df, timeperiod=20)
    df["mfi"] = ta.MFI(df, timeperiod=14)

    stoch = ta.STOCH(df, fastk_period=14, slowk_period=3, slowd_period=3)
    df["slowk"] = stoch["slowk"]
    df["slowd"] = stoch["slowd"]

    df["willr"] = ta.WILLR(df, timeperiod=14)

    df["green"] = (df["close"] > df["open"]).astype(int)
    df["prev_green"] = df["green"].shift(1)
    df["close_pct"] = df["close"].pct_change() * 100

    # Higher timeframe trend (simulated via longer EMA)
    df["trend_up"] = (df["ema_50"] > df["ema_200"]).astype(int)

    # Consecutive green candles
    df["consec_green"] = df["green"].rolling(3).sum()

    return df.dropna().reset_index(drop=True)


def run_bt(pair_data, entry_fn, params):
    capital = INITIAL_CAPITAL
    trades = []
    open_trades = {}

    sl = params.get("sl", -0.03)
    tp = params.get("tp", 0.01)  # take profit target
    roi_table = [
        (0, params.get("roi_0", tp)),
        (params.get("roi_time_1", 30), params.get("roi_1", tp * 0.75)),
        (params.get("roi_time_2", 60), params.get("roi_2", tp * 0.5)),
        (params.get("roi_time_3", 120), params.get("roi_3", tp * 0.25)),
    ]
    max_hold = params.get("max_hold_candles", 60)  # max candles to hold

    signals = {}
    for pair, df in pair_data.items():
        entry = entry_fn(df, params)
        signals[pair] = {
            "entry": entry,
            "close": df["close"].values,
            "high": df["high"].values,
            "low": df["low"].values,
        }

    n = len(next(iter(pair_data.values())))

    for i in range(n):
        to_close = []
        for pair, t in open_trades.items():
            sig = signals[pair]
            if i >= len(sig["close"]):
                continue
            price = sig["close"][i]
            high = sig["high"][i]
            low = sig["low"][i]
            ep = t["ep"]
            profit = (price - ep) / ep
            high_profit = (high - ep) / ep
            low_profit = (low - ep) / ep
            candles = i - t["idx"]
            mins = candles * 5

            # Dynamic stoploss
            if high_profit > 0.02:
                eff_sl = max(sl, -0.005)
            elif high_profit > 0.01:
                eff_sl = max(sl, -0.01)
            else:
                eff_sl = sl

            do_exit = False
            reason = ""

            # Stoploss check (using low of candle)
            if low_profit <= eff_sl:
                do_exit, reason, profit = True, "sl", eff_sl

            # ROI check (using high of candle for more accurate fill)
            if not do_exit:
                for roi_mins, roi_pct in sorted(roi_table):
                    if mins >= roi_mins and high_profit >= roi_pct:
                        do_exit, reason = True, "roi"
                        profit = roi_pct  # assume fill at ROI target
                        break

            # Force exit after max hold time
            if not do_exit and candles >= max_hold:
                do_exit, reason = True, "timeout"

            if do_exit:
                net = t["stake"] * profit - t["stake"] * COMMISSION * 2
                capital += t["stake"] + net
                trades.append({"p": profit * 100, "n": net, "r": reason, "d": mins})
                to_close.append(pair)

        for p in to_close:
            del open_trades[p]

        if len(open_trades) < MAX_OPEN_TRADES:
            for pair in signals:
                if len(open_trades) >= MAX_OPEN_TRADES or pair in open_trades:
                    continue
                sig = signals[pair]
                if i < len(sig["entry"]) and sig["entry"][i]:
                    stake = capital * STAKE_PCT
                    if stake < 1:
                        continue
                    capital -= stake
                    open_trades[pair] = {"ep": sig["close"][i], "idx": i, "stake": stake}

    for pair, t in open_trades.items():
        sig = signals[pair]
        price = sig["close"][-1]
        profit = (price - t["ep"]) / t["ep"]
        net = t["stake"] * profit - t["stake"] * COMMISSION * 2
        capital += t["stake"] + net
        trades.append({"p": profit * 100, "n": net, "r": "force", "d": 0})

    if not trades:
        return {"total": 0, "wr": 0, "pf": 0, "profit_pct": 0}

    tdf = pd.DataFrame(trades)
    w = tdf[tdf["n"] > 0]
    l = tdf[tdf["n"] <= 0]
    ls = l["n"].sum()
    return {
        "total": len(tdf), "wins": len(w), "losses": len(l),
        "wr": len(w) / len(tdf) * 100,
        "profit_pct": (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "capital": capital,
        "avg_p": tdf["p"].mean(),
        "avg_w": w["p"].mean() if len(w) else 0,
        "avg_l": l["p"].mean() if len(l) else 0,
        "pf": abs(w["n"].sum() / ls) if ls != 0 else 999,
        "avg_d": tdf["d"].mean(),
        "reasons": tdf["r"].value_counts().to_dict(),
    }


# --- Entry strategies ---

def entry_scalp_trend(df, p):
    """Scalp in direction of trend with multiple confirmations."""
    return (
        (df["close"] > df[f"ema_{p.get('trend_ema', 100)}"])
        & (df["ema_9"] > df["ema_21"])
        & (df["rsi"] > p.get("rsi_lo", 40))
        & (df["rsi"] < p.get("rsi_hi", 60))
        & (df["macdhist"] > 0)
        & (df["adx"] > p.get("adx_min", 20))
        & (df["volume"] > df["volume_ma_20"] * p.get("vol_mult", 1.0))
        & (df["close"] < df["bb_upper_15"])
        & (df["green"] == 1)
    ).values


def entry_bb_bounce(df, p):
    """Bounce from BB lower with RSI confirmation."""
    return (
        (df["close"] <= df["bb_lower"] * (1 + p.get("bb_off", 0.005)))
        & (df["rsi"] < p.get("rsi_lo", 35))
        & (df["rsi"] > p.get("rsi_floor", 15))
        & (df["close"] > df[f"ema_{p.get('trend_ema', 200)}"])
        & (df["volume"] > df["volume_ma_20"] * p.get("vol_mult", 0.8))
        & (df["green"] == 1)
        & (df["adx"] < p.get("adx_max", 40))
    ).values


def entry_ema_pullback(df, p):
    """Pullback to EMA in uptrend."""
    ema_s = p.get("ema_support", 21)
    return (
        (df["trend_up"] == 1)
        & (df["low"] <= df[f"ema_{ema_s}"] * (1 + 0.003))
        & (df["close"] > df[f"ema_{ema_s}"])
        & (df["green"] == 1)
        & (df["rsi"] > p.get("rsi_lo", 35))
        & (df["rsi"] < p.get("rsi_hi", 55))
        & (df["volume"] > df["volume_ma_20"] * p.get("vol_mult", 0.8))
        & (df["macdhist"] > df["macdhist"].shift(1))
    ).values


def entry_momentum_burst(df, p):
    """Strong momentum with volume confirmation."""
    return (
        (df["trend_up"] == 1)
        & (df["rsi"] > p.get("rsi_lo", 50))
        & (df["rsi"] < p.get("rsi_hi", 70))
        & (df["macdhist"] > 0)
        & (df["macdhist"] > df["macdhist"].shift(1))
        & (df["adx"] > p.get("adx_min", 25))
        & (df["volume"] > df["volume_ma_20"] * p.get("vol_mult", 1.2))
        & (df["green"] == 1)
        & (df["close_pct"] > p.get("min_move", 0.1))
        & (df["close_pct"] < p.get("max_move", 1.0))
    ).values


def entry_oversold_reversal(df, p):
    """RSI + Stoch oversold reversal."""
    return (
        (df["rsi"] > p.get("rsi_lo", 25))
        & (df["rsi"] < p.get("rsi_hi", 40))
        & (df["rsi"] > df["rsi"].shift(1))
        & (df["slowk"] > df["slowd"])
        & (df["slowk"].shift(1) <= df["slowd"].shift(1))
        & (df["close"] > df["ema_200"])
        & (df["green"] == 1)
        & (df["volume"] > df["volume_ma_20"] * p.get("vol_mult", 0.8))
        & (df["mfi"] < p.get("mfi_max", 40))
    ).values


def main():
    print("Loading all pairs...")
    pair_data = {}
    for pair in ALL_PAIRS:
        pair_data[pair] = load_pair(pair)
    print(f"Loaded {len(pair_data)} pairs\n")

    strategies = {
        "scalp_trend": (entry_scalp_trend, [
            {"trend_ema": te, "rsi_lo": rlo, "rsi_hi": rhi, "adx_min": am,
             "vol_mult": vm, "sl": sl, "tp": tp, "max_hold_candles": mh}
            for te in [50, 100, 200]
            for rlo in [35, 40, 45]
            for rhi in [55, 60, 65]
            for am in [20, 25]
            for vm in [0.8, 1.0, 1.2]
            for sl in [-0.02, -0.03, -0.04]
            for tp in [0.008, 0.01, 0.012, 0.015]
            for mh in [24, 36, 48]
        ]),
        "bb_bounce": (entry_bb_bounce, [
            {"bb_off": bo, "rsi_lo": rlo, "rsi_floor": rf, "trend_ema": te,
             "vol_mult": vm, "adx_max": am, "sl": sl, "tp": tp, "max_hold_candles": mh}
            for bo in [0.002, 0.005, 0.008]
            for rlo in [30, 35, 40]
            for rf in [10, 15, 20]
            for te in [100, 200]
            for vm in [0.6, 0.8, 1.0]
            for am in [35, 45]
            for sl in [-0.02, -0.03]
            for tp in [0.008, 0.01, 0.015]
            for mh in [24, 36, 48]
        ]),
        "ema_pullback": (entry_ema_pullback, [
            {"ema_support": es, "rsi_lo": rlo, "rsi_hi": rhi, "vol_mult": vm,
             "sl": sl, "tp": tp, "max_hold_candles": mh}
            for es in [21, 34, 50]
            for rlo in [30, 35, 40]
            for rhi in [50, 55, 60]
            for vm in [0.7, 0.9, 1.1]
            for sl in [-0.02, -0.03]
            for tp in [0.008, 0.01, 0.015]
            for mh in [24, 36, 48]
        ]),
        "momentum_burst": (entry_momentum_burst, [
            {"rsi_lo": rlo, "rsi_hi": rhi, "adx_min": am, "vol_mult": vm,
             "min_move": mm, "max_move": xm, "sl": sl, "tp": tp, "max_hold_candles": mh}
            for rlo in [45, 50, 55]
            for rhi in [65, 70]
            for am in [22, 28]
            for vm in [1.0, 1.3, 1.5]
            for mm in [0.05, 0.1, 0.2]
            for xm in [0.8, 1.2]
            for sl in [-0.02, -0.03]
            for tp in [0.008, 0.01, 0.015]
            for mh in [20, 36]
        ]),
        "oversold_reversal": (entry_oversold_reversal, [
            {"rsi_lo": rlo, "rsi_hi": rhi, "vol_mult": vm, "mfi_max": mm,
             "sl": sl, "tp": tp, "max_hold_candles": mh}
            for rlo in [20, 25, 30]
            for rhi in [35, 40, 45]
            for vm in [0.7, 0.9, 1.1]
            for mm in [30, 40, 50]
            for sl in [-0.02, -0.03]
            for tp in [0.008, 0.01, 0.015]
            for mh in [24, 36, 48]
        ]),
    }

    overall_best_score = 0
    overall_best = None
    results_table = []

    for name, (fn, param_list) in strategies.items():
        print(f"\n{'='*50}")
        print(f"Strategy: {name} ({len(param_list)} combos)")
        print(f"{'='*50}")
        best_score = 0
        best_res = None
        best_p = None

        for idx, p in enumerate(param_list):
            res = run_bt(pair_data, fn, p)
            if res["total"] >= 20:
                wr = res["wr"]
                pf = min(res["pf"], 10)
                # Score heavily weights win rate
                score = wr * 0.7 + pf * 2 + max(res["profit_pct"], 0) * 0.1
                if wr >= 85:
                    score += 20  # bonus for 85%+
                if wr >= 90:
                    score += 30  # big bonus for 90%+

                if score > best_score:
                    best_score = score
                    best_res = res
                    best_p = p
                    if wr >= 75:
                        print(f"  [{idx}] WR={wr:.1f}% PF={pf:.2f} T={res['total']} P={res['profit_pct']:+.1f}%")

        if best_res:
            results_table.append((name, best_res, best_p, best_score))
            print(f"\n  Best {name}: WR={best_res['wr']:.1f}% PF={best_res['pf']:.2f} T={best_res['total']} P={best_res['profit_pct']:+.1f}%")
            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best = (name, best_res, best_p)

    # Show all results sorted
    print(f"\n\n{'='*70}")
    print(f"  ALL STRATEGIES RANKED")
    print(f"{'='*70}")
    for name, res, p, score in sorted(results_table, key=lambda x: x[3], reverse=True):
        print(f"  {name:25s} WR={res['wr']:5.1f}% PF={res['pf']:5.2f} T={res['total']:4d} P={res['profit_pct']:+7.1f}% Score={score:.1f}")

    print(f"\n{'='*70}")
    print(f"  WINNER")
    print(f"{'='*70}")
    if overall_best:
        name, res, p = overall_best
        print(f"  Strategy:     {name}")
        print(f"  Win Rate:     {res['wr']:.1f}%")
        print(f"  Profit Factor:{res['pf']:.2f}")
        print(f"  Total Trades: {res['total']}")
        print(f"  Profit:       {res['profit_pct']:+.2f}%")
        print(f"  Capital:      ${res['capital']:.2f}")
        print(f"  Avg Win:      {res['avg_w']:.2f}%")
        print(f"  Avg Loss:     {res['avg_l']:.2f}%")
        print(f"  Avg Duration: {res['avg_d']:.0f} min")
        print(f"  Reasons:      {res['reasons']}")
        print(f"  Params:       {p}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
