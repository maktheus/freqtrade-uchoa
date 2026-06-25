#!/usr/bin/env python3
"""V2 Optimizer: Multiple strategy types tested and optimized."""
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

    # All indicators pre-computed
    for p in [5, 8, 9, 13, 21, 34, 50, 100, 200]:
        df[f"ema_{p}"] = ta.EMA(df, timeperiod=p)
    for p in [6, 7, 14, 21]:
        df[f"rsi_{p}"] = ta.RSI(df, timeperiod=p)

    macd = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd["macd"]
    df["macdsignal"] = macd["macdsignal"]
    df["macdhist"] = macd["macdhist"]

    for std in [1.5, 2.0, 2.5]:
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=std)
        s = str(std).replace(".", "")
        df[f"bb_lower_{s}"] = bb["lower"]
        df[f"bb_mid_{s}"] = bb["mid"]
        df[f"bb_upper_{s}"] = bb["upper"]
        df[f"bb_width_{s}"] = (bb["upper"] - bb["lower"]) / bb["mid"]

    df["volume_ma_10"] = df["volume"].rolling(10).mean()
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["adx"] = ta.ADX(df, timeperiod=14)
    df["atr"] = ta.ATR(df, timeperiod=14)
    df["cci"] = ta.CCI(df, timeperiod=20)
    df["mfi"] = ta.MFI(df, timeperiod=14)

    stoch = ta.STOCH(df, fastk_period=14, slowk_period=3, slowd_period=3)
    df["slowk"] = stoch["slowk"]
    df["slowd"] = stoch["slowd"]

    df["willr"] = ta.WILLR(df, timeperiod=14)

    # Price action
    df["green"] = df["close"] > df["open"]
    df["body_pct"] = abs(df["close"] - df["open"]) / df["open"] * 100
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["open"] * 100
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["open"] * 100

    return df.dropna().reset_index(drop=True)


def strategy_mean_reversion(df, p):
    """Buy near BB lower, sell near BB mid/upper."""
    bb_std = p.get("bb_std", "20")
    rsi_p = p.get("rsi_period", 14)
    rsi_lo = p.get("rsi_lo", 30)
    rsi_hi = p.get("rsi_hi", 70)

    entry = (
        (df["close"] <= df[f"bb_lower_{bb_std}"] * (1 + p.get("bb_offset", 0.002)))
        & (df[f"rsi_{rsi_p}"] < rsi_lo)
        & (df["volume"] > df["volume_ma_20"] * 0.5)
        & (df["adx"] < p.get("adx_max", 40))
    )

    exit_sig = (
        (df["close"] >= df[f"bb_mid_{bb_std}"])
        | (df[f"rsi_{rsi_p}"] > rsi_hi)
    )
    return entry.values, exit_sig.values


def strategy_ema_bounce(df, p):
    """Buy on EMA support bounce."""
    ema_support = p.get("ema_support", 21)
    rsi_lo = p.get("rsi_lo", 35)
    rsi_hi = p.get("rsi_hi", 70)

    ema_col = f"ema_{ema_support}"
    entry = (
        (df["low"] <= df[ema_col] * (1 + 0.002))
        & (df["close"] > df[ema_col])
        & (df["close"] > df["open"])  # green candle
        & (df[f"rsi_14"] > rsi_lo)
        & (df[f"rsi_14"] < 60)
        & (df["close"] > df["ema_200"])
        & (df["volume"] > df["volume_ma_20"] * 0.8)
    )

    exit_sig = (
        (df[f"rsi_14"] > rsi_hi)
        | (df["close"] < df[ema_col] * 0.99)
    )
    return entry.values, exit_sig.values


def strategy_momentum_pullback(df, p):
    """Enter on pullback in momentum trend."""
    rsi_lo = p.get("rsi_lo", 40)
    rsi_hi = p.get("rsi_hi", 55)
    sell_rsi = p.get("sell_rsi", 75)

    entry = (
        (df["close"] > df["ema_50"])
        & (df["ema_50"] > df["ema_200"])
        & (df["rsi_14"] > rsi_lo)
        & (df["rsi_14"] < rsi_hi)
        & (df["macdhist"] > df["macdhist"].shift(1))
        & (df["volume"] > df["volume_ma_20"] * 0.8)
        & (df["adx"] > p.get("adx_min", 20))
        & (df["close"] < df["bb_upper_20"])
    )

    exit_sig = (
        (df["rsi_14"] > sell_rsi)
        | (df["close"] < df["ema_50"])
        | (df["macdhist"] < 0)
    )
    return entry.values, exit_sig.values


def strategy_multi_confirm(df, p):
    """Multiple indicator confluence."""
    rsi_lo = p.get("rsi_lo", 30)
    rsi_hi = p.get("rsi_hi", 65)
    sell_rsi = p.get("sell_rsi", 72)
    cci_lo = p.get("cci_lo", -100)
    mfi_lo = p.get("mfi_lo", 20)

    entry = (
        (df["close"] > df["ema_200"])
        & (df["ema_9"] > df["ema_21"])
        & (df["rsi_14"] > rsi_lo) & (df["rsi_14"] < rsi_hi)
        & (df["macdhist"] > 0)
        & (df["cci"] > cci_lo) & (df["cci"] < 100)
        & (df["mfi"] > mfi_lo) & (df["mfi"] < 80)
        & (df["volume"] > df["volume_ma_20"] * 0.8)
        & (df["adx"] > p.get("adx_min", 18))
    )

    exit_sig = (
        (df["rsi_14"] > sell_rsi)
        | (
            (df["ema_9"] < df["ema_21"])
            & (df["ema_9"].shift(1) >= df["ema_21"].shift(1))
        )
        | (df["close"] > df["bb_upper_20"])
    )
    return entry.values, exit_sig.values


def strategy_stoch_rsi_reversal(df, p):
    """Stochastic + RSI reversal strategy."""
    rsi_lo = p.get("rsi_lo", 30)
    stoch_lo = p.get("stoch_lo", 20)
    sell_rsi = p.get("sell_rsi", 75)
    stoch_hi = p.get("stoch_hi", 80)

    entry = (
        (df["rsi_14"] < rsi_lo + 10)
        & (df["rsi_14"] > rsi_lo)
        & (df["slowk"] > df["slowd"])
        & (df["slowk"].shift(1) <= df["slowd"].shift(1))
        & (df["slowk"] < stoch_lo + 20)
        & (df["close"] > df["ema_200"])
        & (df["volume"] > df["volume_ma_10"] * 0.7)
    )

    exit_sig = (
        (df["rsi_14"] > sell_rsi)
        | ((df["slowk"] > stoch_hi) & (df["slowk"] < df["slowd"]))
    )
    return entry.values, exit_sig.values


def run_bt(pair_data: dict, strategy_fn, params: dict) -> dict:
    capital = INITIAL_CAPITAL
    trades = []
    open_trades = {}

    sl = params.get("stoploss", -0.04)
    roi_table = [
        (0, params.get("roi_0", 0.02)),
        (30, params.get("roi_30", 0.015)),
        (60, params.get("roi_60", 0.01)),
        (120, params.get("roi_120", 0.005)),
    ]

    signals = {}
    for pair, df in pair_data.items():
        entry, exit_sig = strategy_fn(df, params)
        signals[pair] = {
            "entry": entry, "exit": exit_sig,
            "close": df["close"].values, "low": df["low"].values,
        }

    n = len(next(iter(pair_data.values())))

    for i in range(n):
        to_close = []
        for pair, t in open_trades.items():
            sig = signals[pair]
            if i >= len(sig["close"]):
                continue
            price = sig["close"][i]
            low = sig["low"][i]
            ep = t["ep"]
            profit = (price - ep) / ep
            low_p = (low - ep) / ep
            mins = (i - t["idx"]) * 5

            if profit > 0.03:
                eff_sl = -0.005
            elif profit > 0.02:
                eff_sl = -0.01
            elif profit > 0.01:
                eff_sl = -0.015
            else:
                eff_sl = sl

            do_exit = False
            reason = ""
            if low_p <= eff_sl:
                do_exit, reason, profit = True, "sl", eff_sl
            else:
                for roi_mins, roi_pct in sorted(roi_table):
                    if mins >= roi_mins and profit >= roi_pct:
                        do_exit, reason = True, "roi"
                        break
            if not do_exit and sig["exit"][i]:
                do_exit, reason = True, "exit"

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
        return {"total": 0, "wr": 0, "pf": 0, "profit_pct": 0, "capital": capital}

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


def main():
    print("Loading all pairs...")
    pair_data = {}
    for pair in ALL_PAIRS:
        pair_data[pair] = load_pair(pair)
    print(f"Loaded {len(pair_data)} pairs\n")

    strategies = {
        "mean_reversion": (strategy_mean_reversion, [
            {"bb_std": "20", "rsi_period": 14, "rsi_lo": rlo, "rsi_hi": rhi, "adx_max": am, "bb_offset": bo,
             "stoploss": sl, "roi_0": r0, "roi_30": r30, "roi_60": r60, "roi_120": r120}
            for rlo in [25, 30, 35]
            for rhi in [65, 70, 75]
            for am in [35, 45]
            for bo in [0.001, 0.003, 0.005]
            for sl in [-0.03, -0.04]
            for r0 in [0.015, 0.02]
            for r30 in [0.01, 0.015]
            for r60 in [0.008, 0.01]
            for r120 in [0.004, 0.006]
        ]),
        "ema_bounce": (strategy_ema_bounce, [
            {"ema_support": es, "rsi_lo": rlo, "rsi_hi": rhi,
             "stoploss": sl, "roi_0": r0, "roi_30": 0.012, "roi_60": 0.008, "roi_120": 0.005}
            for es in [21, 34, 50]
            for rlo in [30, 35, 40]
            for rhi in [65, 70, 75]
            for sl in [-0.03, -0.04]
            for r0 in [0.015, 0.02, 0.025]
        ]),
        "momentum_pullback": (strategy_momentum_pullback, [
            {"rsi_lo": rlo, "rsi_hi": rhi, "sell_rsi": sr, "adx_min": am,
             "stoploss": sl, "roi_0": r0, "roi_30": 0.012, "roi_60": 0.008, "roi_120": 0.005}
            for rlo in [35, 40, 45]
            for rhi in [50, 55, 60]
            for sr in [70, 75, 80]
            for am in [18, 22, 26]
            for sl in [-0.03, -0.04]
            for r0 in [0.015, 0.02]
        ]),
        "multi_confirm": (strategy_multi_confirm, [
            {"rsi_lo": rlo, "rsi_hi": rhi, "sell_rsi": sr, "cci_lo": cl, "mfi_lo": ml, "adx_min": am,
             "stoploss": sl, "roi_0": r0, "roi_30": 0.012, "roi_60": 0.008, "roi_120": 0.005}
            for rlo in [25, 30, 35]
            for rhi in [60, 65, 70]
            for sr in [70, 75]
            for cl in [-100, -50]
            for ml in [20, 30]
            for am in [15, 20]
            for sl in [-0.03, -0.04]
            for r0 in [0.015, 0.02]
        ]),
        "stoch_rsi_reversal": (strategy_stoch_rsi_reversal, [
            {"rsi_lo": rlo, "stoch_lo": slo, "sell_rsi": sr, "stoch_hi": shi,
             "stoploss": sl, "roi_0": r0, "roi_30": 0.012, "roi_60": 0.008, "roi_120": 0.005}
            for rlo in [25, 30, 35]
            for slo in [15, 20, 25]
            for sr in [70, 75, 80]
            for shi in [75, 80, 85]
            for sl in [-0.03, -0.04]
            for r0 in [0.015, 0.02]
        ]),
    }

    overall_best_wr = 0
    overall_best = None

    for name, (fn, param_list) in strategies.items():
        print(f"\n{'='*50}")
        print(f"Strategy: {name} ({len(param_list)} combos)")
        print(f"{'='*50}")
        best_wr = 0
        best_res = None
        best_p = None

        for idx, p in enumerate(param_list):
            res = run_bt(pair_data, fn, p)
            if res["total"] >= 15 and res["wr"] > best_wr:
                best_wr = res["wr"]
                best_res = res
                best_p = p
                if best_wr > 70:
                    print(f"  [{idx}] WR={res['wr']:.1f}% PF={res['pf']:.2f} T={res['total']} P={res['profit_pct']:+.1f}%")

        if best_res:
            print(f"\n  Best {name}:")
            print(f"    WR={best_res['wr']:.1f}% PF={best_res['pf']:.2f} Trades={best_res['total']}")
            print(f"    Profit={best_res['profit_pct']:+.2f}% Capital=${best_res['capital']:.2f}")
            print(f"    Avg Win={best_res['avg_w']:.2f}% Avg Loss={best_res['avg_l']:.2f}%")
            print(f"    Reasons={best_res['reasons']}")
            print(f"    Params={best_p}")

            if best_wr > overall_best_wr:
                overall_best_wr = best_wr
                overall_best = (name, best_res, best_p)
        else:
            print(f"  No valid results with >=15 trades")

    print(f"\n\n{'='*70}")
    print(f"  OVERALL WINNER")
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
