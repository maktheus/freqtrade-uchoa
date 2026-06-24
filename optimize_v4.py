#!/usr/bin/env python3
"""V4: Fixed filters, fast vectorized optimizer for 15min 2020-2026."""
import numpy as np
import pandas as pd
from pathlib import Path
import talib.abstract as ta
from technical import qtpylib

DATA_DIR = Path("user_data/data/binance")
COMMISSION = 0.001

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LINK/USDT",
         "ADA/USDT", "DOT/USDT", "LTC/USDT", "AVAX/USDT", "DOGE/USDT"]


def load_pair(pair):
    fn = pair.replace("/", "_") + "-15m.feather"
    df = pd.read_feather(DATA_DIR / fn).sort_values("date").reset_index(drop=True)
    for p in [9, 21, 50, 100, 200]:
        df[f"ema_{p}"] = ta.EMA(df, timeperiod=p)
    df["rsi"] = ta.RSI(df, timeperiod=14)
    macd = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macdhist"] = macd["macdhist"]
    bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    df["bb_lower"] = bb["lower"]
    df["bb_mid"] = bb["mid"]
    df["bb_upper"] = bb["upper"]
    df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["adx"] = ta.ADX(df, timeperiod=14)
    df["mfi"] = ta.MFI(df, timeperiod=14)
    df["cci"] = ta.CCI(df, timeperiod=20)
    stoch = ta.STOCH(df, fastk_period=14, slowk_period=3, slowd_period=3)
    df["slowk"] = stoch["slowk"]
    df["slowd"] = stoch["slowd"]
    df["green"] = (df["close"] > df["open"]).astype(int)
    df["ema_slope"] = (df["ema_50"] - df["ema_50"].shift(4)) / df["ema_50"].shift(4)
    return df.dropna().reset_index(drop=True)


def calc_winrate(pair_data, entry_fn, tp, sl, max_hold, params):
    wins, losses = 0, 0
    for pair, df in pair_data.items():
        mask = entry_fn(df, params)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(close)
        entries = np.where(mask)[0]
        if len(entries) == 0:
            continue
        filtered = [entries[0]]
        for e in entries[1:]:
            if e - filtered[-1] >= 4:
                filtered.append(e)
        for eidx in filtered:
            ep = close[eidx]
            won = lost = False
            for j in range(1, min(max_hold + 1, n - eidx)):
                h_pct = (high[eidx + j] - ep) / ep
                l_pct = (low[eidx + j] - ep) / ep
                if l_pct <= sl:
                    lost = True
                    break
                if h_pct >= tp:
                    won = True
                    break
            if won:
                wins += 1
            elif lost:
                losses += 1
            else:
                if eidx + max_hold < n:
                    final_p = (close[min(eidx + max_hold, n-1)] - ep) / ep
                    if final_p > COMMISSION * 2:
                        wins += 1
                    else:
                        losses += 1
    total = wins + losses
    return (wins / total * 100 if total > 0 else 0), total, wins


# --- Entries with fixed BB filter ---

def entry_trend_momentum(df, p):
    """Trend + momentum, no BB upper filter."""
    return (
        (df["ema_50"] > df["ema_200"])
        & (df["ema_9"] > df["ema_21"])
        & (df["rsi"] > p["rl"]) & (df["rsi"] < p["rh"])
        & (df["macdhist"] > 0)
        & (df["adx"] > p["am"])
        & (df["volume"] > df["vol_ma"] * p["vm"])
        & (df["green"] == 1)
        & (df["bb_pct"] < p.get("bb_pct_max", 0.85))
        & (df["ema_slope"] > 0)
    ).values

def entry_triple_ema(df, p):
    """Triple EMA alignment with momentum."""
    return (
        (df["ema_50"] > df["ema_200"])
        & (df["ema_9"] > df["ema_21"])
        & (df["ema_21"] > df["ema_50"])
        & (df["rsi"] > p["rl"]) & (df["rsi"] < p["rh"])
        & (df["macdhist"] > 0)
        & (df["macdhist"] > df["macdhist"].shift(1))
        & (df["adx"] > p["am"])
        & (df["volume"] > df["vol_ma"] * p["vm"])
        & (df["green"] == 1)
        & (df["ema_slope"] > p.get("min_slope", 0.0001))
    ).values

def entry_full_confluence(df, p):
    """All indicators confirming."""
    return (
        (df["ema_50"] > df["ema_200"])
        & (df["ema_9"] > df["ema_21"])
        & (df["ema_21"] > df["ema_50"])
        & (df["rsi"] > p["rl"]) & (df["rsi"] < p["rh"])
        & (df["macdhist"] > 0)
        & (df["macdhist"] > df["macdhist"].shift(1))
        & (df["cci"] > p.get("cci_lo", 0))
        & (df["mfi"] > p.get("mfi_lo", 30)) & (df["mfi"] < p.get("mfi_hi", 60))
        & (df["adx"] > p["am"])
        & (df["volume"] > df["vol_ma"] * p["vm"])
        & (df["green"] == 1)
        & (df["slowk"] > df["slowd"])
        & (df["ema_slope"] > p.get("min_slope", 0.0001))
    ).values

def entry_rsi_stoch(df, p):
    """RSI + Stochastic alignment in trend."""
    return (
        (df["ema_50"] > df["ema_200"])
        & (df["ema_9"] > df["ema_21"])
        & (df["rsi"] > p["rl"]) & (df["rsi"] < p["rh"])
        & (df["rsi"] > df["rsi"].shift(1))
        & (df["slowk"] > df["slowd"])
        & (df["slowk"] < p.get("stoch_max", 70))
        & (df["macdhist"] > 0)
        & (df["adx"] > p["am"])
        & (df["volume"] > df["vol_ma"] * p["vm"])
        & (df["green"] == 1)
        & (df["ema_slope"] > 0)
    ).values


def main():
    print("Loading data...")
    pair_data = {}
    for pair in PAIRS:
        pair_data[pair] = load_pair(pair)
    print(f"Loaded {len(pair_data)} pairs\n")

    strategies = {
        "trend_mom": entry_trend_momentum,
        "triple_ema": entry_triple_ema,
        "full_conf": entry_full_confluence,
        "rsi_stoch": entry_rsi_stoch,
    }

    overall_best_wr = 0
    overall_best = None

    for sname, sfn in strategies.items():
        print(f"\n{'='*55}")
        print(f"  {sname}")
        print(f"{'='*55}")
        best_wr = 0
        best_total = 0
        best_p = None
        count = 0

        for rl in [30, 35, 40, 45, 50]:
            for rh in [50, 55, 58, 62, 66, 70]:
                if rh <= rl + 5:
                    continue
                for am in [18, 22, 26, 30]:
                    for vm in [0.7, 0.9, 1.1, 1.3]:
                        for tp in [0.005, 0.007, 0.009, 0.011, 0.014]:
                            for sl in [-0.015, -0.02, -0.03, -0.04, -0.05]:
                                for mh in [12, 20, 30, 48]:
                                    p = {"rl": rl, "rh": rh, "am": am, "vm": vm}
                                    wr, total, wins = calc_winrate(pair_data, sfn, tp, sl, mh, p)
                                    count += 1

                                    if total >= 50 and wr > best_wr:
                                        best_wr = wr
                                        best_total = total
                                        best_p = {**p, "tp": tp, "sl": sl, "mh": mh}
                                        if wr >= 85:
                                            print(f"  [{count:6d}] WR={wr:.1f}% T={total:5d} W={wins} | tp={tp} sl={sl} mh={mh} rl={rl} rh={rh}")

                                    if count % 5000 == 0:
                                        print(f"  ... {count} tested, best WR: {best_wr:.1f}% (T={best_total})")

        print(f"\n  BEST {sname}: WR={best_wr:.1f}% T={best_total}")
        if best_p:
            print(f"  Params: {best_p}")
            if best_wr > overall_best_wr:
                overall_best_wr = best_wr
                overall_best = (sname, best_wr, best_total, best_p)

    print(f"\n\n{'='*70}")
    print(f"  FINAL WINNER (15min, 2020-2026, 10 pairs)")
    print(f"{'='*70}")
    if overall_best:
        name, wr, total, p = overall_best
        losses = total - int(total * wr / 100)
        wins = total - losses
        pf = (wins * p["tp"]) / (losses * abs(p["sl"])) if losses > 0 else 999
        print(f"  Strategy:      {name}")
        print(f"  Win Rate:      {wr:.1f}%")
        print(f"  Total Trades:  {total}")
        print(f"  Wins/Losses:   {wins}/{losses}")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Take Profit:   {p['tp']*100:.1f}%")
        print(f"  Stop Loss:     {p['sl']*100:.1f}%")
        print(f"  Max Hold:      {p['mh']*15} min")
        print(f"  RSI Range:     {p['rl']}-{p['rh']}")
        print(f"  ADX Min:       {p['am']}")
        print(f"  Vol Mult:      {p['vm']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
