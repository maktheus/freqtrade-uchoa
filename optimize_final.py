#!/usr/bin/env python3
"""Final optimizer: 15min timeframe, 2020-2026, target >90% win rate."""
import numpy as np
import pandas as pd
from pathlib import Path
import talib.abstract as ta
from technical import qtpylib
import itertools

DATA_DIR = Path("user_data/data/binance")
INITIAL_CAPITAL = 1000.0
MAX_OPEN_TRADES = 5
STAKE_PCT = 1.0 / MAX_OPEN_TRADES
COMMISSION = 0.001

PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
    "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "LTC/USDT",
]


def load_pair(pair):
    fn = pair.replace("/", "_") + "-15m.feather"
    df = pd.read_feather(DATA_DIR / fn).sort_values("date").reset_index(drop=True)

    for p in [8, 9, 13, 21, 34, 50, 100, 200]:
        df[f"ema_{p}"] = ta.EMA(df, timeperiod=p)

    df["rsi"] = ta.RSI(df, timeperiod=14)
    df["rsi_7"] = ta.RSI(df, timeperiod=7)

    macd = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macdhist"] = macd["macdhist"]

    bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    df["bb_lower"] = bb["lower"]
    df["bb_mid"] = bb["mid"]
    df["bb_upper"] = bb["upper"]
    df["bb_width"] = (bb["upper"] - bb["lower"]) / bb["mid"]

    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["adx"] = ta.ADX(df, timeperiod=14)
    df["atr"] = ta.ATR(df, timeperiod=14)
    df["cci"] = ta.CCI(df, timeperiod=20)
    df["mfi"] = ta.MFI(df, timeperiod=14)

    stoch = ta.STOCH(df, fastk_period=14, slowk_period=3, slowd_period=3)
    df["slowk"] = stoch["slowk"]
    df["slowd"] = stoch["slowd"]

    df["green"] = (df["close"] > df["open"]).astype(int)
    df["pct_change"] = df["close"].pct_change()
    df["trend_50_200"] = (df["ema_50"] > df["ema_200"]).astype(int)
    df["ema_slope_50"] = (df["ema_50"] - df["ema_50"].shift(5)) / df["ema_50"].shift(5)

    return df.dropna().reset_index(drop=True)


def entry_ultra_safe(df, p):
    """Ultra-conservative entry: trend + momentum + mean reversion confluence."""
    ema_t = p.get("ema_trend", 200)
    rsi_lo = p.get("rsi_lo", 35)
    rsi_hi = p.get("rsi_hi", 55)
    adx_min = p.get("adx_min", 20)
    vol_mult = p.get("vol_mult", 1.0)

    conds = (
        (df["close"] > df[f"ema_{ema_t}"])
        & (df["ema_9"] > df["ema_21"])
        & (df["rsi"] > rsi_lo) & (df["rsi"] < rsi_hi)
        & (df["macdhist"] > 0)
        & (df["adx"] > adx_min)
        & (df["volume"] > df["vol_ma"] * vol_mult)
        & (df["green"] == 1)
        & (df["close"] < df["bb_upper"] * 0.98)
        & (df["mfi"] > 20) & (df["mfi"] < 70)
        & (df["ema_slope_50"] > 0)
    )
    return conds.values


def entry_trend_scalp(df, p):
    """Scalp in strong trend."""
    rsi_lo = p.get("rsi_lo", 40)
    rsi_hi = p.get("rsi_hi", 58)

    return (
        (df["trend_50_200"] == 1)
        & (df["close"] > df["ema_21"])
        & (df["rsi"] > rsi_lo) & (df["rsi"] < rsi_hi)
        & (df["macdhist"] > 0)
        & (df["macdhist"] > df["macdhist"].shift(1))
        & (df["adx"] > p.get("adx_min", 22))
        & (df["volume"] > df["vol_ma"] * p.get("vol_mult", 1.0))
        & (df["green"] == 1)
        & (df["close"] < df["bb_upper"])
        & (df["ema_slope_50"] > 0.0001)
    ).values


def entry_bb_rsi_reversal(df, p):
    """BB touch + RSI oversold reversal in uptrend."""
    rsi_max = p.get("rsi_max", 35)

    return (
        (df["close"] > df[f"ema_{p.get('ema_trend', 200)}"])
        & (df["close"] <= df["bb_lower"] * (1 + p.get("bb_off", 0.005)))
        & (df["rsi"] < rsi_max)
        & (df["rsi"] > df["rsi"].shift(1))  # RSI turning up
        & (df["green"] == 1)
        & (df["volume"] > df["vol_ma"] * p.get("vol_mult", 0.7))
        & (df["slowk"] > df["slowd"])
    ).values


def entry_ema_support_bounce(df, p):
    """Price touches EMA support and bounces in uptrend."""
    ema_s = p.get("ema_support", 21)

    return (
        (df["trend_50_200"] == 1)
        & (df["low"] <= df[f"ema_{ema_s}"] * 1.003)
        & (df["close"] > df[f"ema_{ema_s}"])
        & (df["green"] == 1)
        & (df["rsi"] > p.get("rsi_lo", 35))
        & (df["rsi"] < p.get("rsi_hi", 55))
        & (df["macdhist"] > df["macdhist"].shift(1))
        & (df["volume"] > df["vol_ma"] * p.get("vol_mult", 0.8))
    ).values


def entry_momentum_confirm(df, p):
    """Strong momentum with all indicators aligned."""
    return (
        (df["trend_50_200"] == 1)
        & (df["ema_9"] > df["ema_21"])
        & (df["ema_21"] > df["ema_50"])
        & (df["rsi"] > p.get("rsi_lo", 45))
        & (df["rsi"] < p.get("rsi_hi", 65))
        & (df["macdhist"] > 0)
        & (df["cci"] > 0) & (df["cci"] < 150)
        & (df["mfi"] > 30) & (df["mfi"] < 70)
        & (df["adx"] > p.get("adx_min", 20))
        & (df["volume"] > df["vol_ma"] * p.get("vol_mult", 1.0))
        & (df["green"] == 1)
        & (df["close"] < df["bb_upper"] * 0.97)
        & (df["slowk"] > df["slowd"])
        & (df["ema_slope_50"] > 0.0002)
    ).values


def run_bt(pair_data, entry_fn, p):
    capital = INITIAL_CAPITAL
    trades = []
    open_trades = {}

    sl = p.get("sl", -0.03)
    tp = p.get("tp", 0.01)
    roi_table = [(0, tp), (20, tp * 0.8), (40, tp * 0.6), (80, tp * 0.3)]
    max_hold = p.get("max_hold", 40)

    signals = {}
    for pair, df in pair_data.items():
        entry = entry_fn(df, p)
        signals[pair] = {
            "entry": entry,
            "close": df["close"].values,
            "high": df["high"].values,
            "low": df["low"].values,
        }

    n = min(len(sig["close"]) for sig in signals.values())

    for i in range(n):
        to_close = []
        for pair, t in open_trades.items():
            sig = signals[pair]
            price, high, low = sig["close"][i], sig["high"][i], sig["low"][i]
            ep = t["ep"]
            profit = (price - ep) / ep
            high_p = (high - ep) / ep
            low_p = (low - ep) / ep
            candles = i - t["idx"]
            mins = candles * 15

            # Dynamic trailing stop
            best = t.get("best", 0)
            if high_p > best:
                best = high_p
                t["best"] = best

            if best > 0.02:
                eff_sl = max(sl, best * 0.5)
            elif best > 0.01:
                eff_sl = max(sl, -0.005)
            else:
                eff_sl = sl

            do_exit = False
            reason = ""
            exit_profit = profit

            if low_p <= eff_sl:
                do_exit, reason, exit_profit = True, "sl", eff_sl
            else:
                for roi_mins, roi_pct in roi_table:
                    if mins >= roi_mins and high_p >= roi_pct:
                        do_exit, reason, exit_profit = True, "roi", roi_pct
                        break

            if not do_exit and candles >= max_hold:
                do_exit, reason = True, "timeout"

            if do_exit:
                net = t["stake"] * exit_profit - t["stake"] * COMMISSION * 2
                capital += t["stake"] + net
                trades.append({"p": exit_profit * 100, "n": net, "r": reason})
                to_close.append(pair)

        for pp in to_close:
            del open_trades[pp]

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
                    open_trades[pair] = {"ep": sig["close"][i], "idx": i, "stake": stake, "best": 0}

    for pair, t in open_trades.items():
        sig = signals[pair]
        price = sig["close"][-1]
        profit = (price - t["ep"]) / t["ep"]
        net = t["stake"] * profit - t["stake"] * COMMISSION * 2
        capital += t["stake"] + net
        trades.append({"p": profit * 100, "n": net, "r": "force"})

    if not trades:
        return None

    tdf = pd.DataFrame(trades)
    w = tdf[tdf["n"] > 0]
    l = tdf[tdf["n"] <= 0]
    ls = l["n"].sum()
    return {
        "total": len(tdf), "wins": len(w), "losses": len(l),
        "wr": len(w) / len(tdf) * 100,
        "profit_pct": (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "capital": capital,
        "avg_w": w["p"].mean() if len(w) else 0,
        "avg_l": l["p"].mean() if len(l) else 0,
        "pf": abs(w["n"].sum() / ls) if ls != 0 else 999,
        "avg_d": 0,
        "reasons": tdf["r"].value_counts().to_dict(),
    }


def main():
    print("Loading 15min data for all pairs (2020-2026)...")
    pair_data = {}
    for pair in PAIRS:
        pair_data[pair] = load_pair(pair)
    print(f"Loaded {len(pair_data)} pairs\n")

    strategies = {
        "ultra_safe": (entry_ultra_safe, [
            {"ema_trend": et, "rsi_lo": rl, "rsi_hi": rh, "adx_min": am, "vol_mult": vm,
             "sl": sl, "tp": tp, "max_hold": mh}
            for et in [100, 200]
            for rl in [30, 35, 40, 45]
            for rh in [50, 55, 60]
            for am in [18, 22, 28]
            for vm in [0.8, 1.0, 1.3]
            for sl in [-0.02, -0.025, -0.03, -0.04]
            for tp in [0.006, 0.008, 0.01, 0.012, 0.015]
            for mh in [20, 30, 40]
        ]),
        "trend_scalp": (entry_trend_scalp, [
            {"rsi_lo": rl, "rsi_hi": rh, "adx_min": am, "vol_mult": vm,
             "sl": sl, "tp": tp, "max_hold": mh}
            for rl in [35, 40, 45, 50]
            for rh in [52, 56, 60]
            for am in [20, 25, 30]
            for vm in [0.8, 1.0, 1.3]
            for sl in [-0.02, -0.03, -0.04]
            for tp in [0.006, 0.008, 0.01, 0.015]
            for mh in [20, 30, 40]
        ]),
        "momentum_confirm": (entry_momentum_confirm, [
            {"rsi_lo": rl, "rsi_hi": rh, "adx_min": am, "vol_mult": vm,
             "sl": sl, "tp": tp, "max_hold": mh}
            for rl in [40, 45, 50]
            for rh in [58, 62, 66]
            for am in [18, 22, 28]
            for vm in [0.8, 1.0, 1.3]
            for sl in [-0.02, -0.03, -0.04]
            for tp in [0.006, 0.008, 0.01, 0.015]
            for mh in [20, 30, 40]
        ]),
    }

    overall_best_wr = 0
    overall_best = None

    for name, (fn, param_list) in strategies.items():
        print(f"\n{'='*55}")
        print(f"  {name} ({len(param_list)} combos)")
        print(f"{'='*55}")
        best_wr = 0
        best = None

        for idx, p in enumerate(param_list):
            res = run_bt(pair_data, fn, p)
            if res and res["total"] >= 30:
                wr = res["wr"]
                if wr > best_wr:
                    best_wr = wr
                    best = (res, p)
                    if wr >= 85:
                        print(f"  [{idx:5d}] WR={wr:.1f}% PF={res['pf']:.2f} T={res['total']:4d} P={res['profit_pct']:+.1f}%")

            if idx % 5000 == 0 and idx > 0:
                print(f"  ... {idx}/{len(param_list)} tested, best WR so far: {best_wr:.1f}%")

        if best:
            res, p = best
            print(f"\n  BEST {name}: WR={res['wr']:.1f}% PF={res['pf']:.2f} T={res['total']} P={res['profit_pct']:+.1f}%")
            print(f"  Params: {p}")
            if best_wr > overall_best_wr:
                overall_best_wr = best_wr
                overall_best = (name, res, p)

    print(f"\n\n{'='*70}")
    print(f"  FINAL WINNER (2020-2026, 15min)")
    print(f"{'='*70}")
    if overall_best:
        name, res, p = overall_best
        print(f"  Strategy:      {name}")
        print(f"  Win Rate:      {res['wr']:.1f}%")
        print(f"  Profit Factor: {res['pf']:.2f}")
        print(f"  Total Trades:  {res['total']}")
        print(f"  Wins/Losses:   {res['wins']}/{res['losses']}")
        print(f"  Total Profit:  {res['profit_pct']:+.2f}%")
        print(f"  Final Capital: ${res['capital']:.2f}")
        print(f"  Avg Win:       {res['avg_w']:.3f}%")
        print(f"  Avg Loss:      {res['avg_l']:.3f}%")
        print(f"  Exit Reasons:  {res['reasons']}")
        print(f"  Params:        {p}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
