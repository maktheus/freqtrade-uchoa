#!/usr/bin/env python3
"""Fast optimizer: 15min, 2020-2026, vectorized backtest for speed."""
import numpy as np
import pandas as pd
from pathlib import Path
import talib.abstract as ta
from technical import qtpylib

DATA_DIR = Path("user_data/data/binance")
INITIAL_CAPITAL = 1000.0
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
    df["bb_upper"] = bb["upper"]
    df["bb_width"] = (bb["upper"] - bb["lower"]) / bb["mid"]
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


def vectorized_winrate(pair_data, entry_fn, tp, sl, max_hold, params):
    """Fast vectorized win rate calculation - per-trade analysis."""
    all_wins = 0
    all_losses = 0

    for pair, df in pair_data.items():
        entry_mask = entry_fn(df, params)
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(close)

        entries = np.where(entry_mask)[0]
        if len(entries) == 0:
            continue

        # Skip entries too close together (min 4 candles apart)
        filtered = [entries[0]]
        for e in entries[1:]:
            if e - filtered[-1] >= max(4, max_hold // 2):
                filtered.append(e)
        entries = filtered

        for eidx in entries:
            ep = close[eidx]
            won = False
            lost = False
            for j in range(1, min(max_hold + 1, n - eidx)):
                hi = high[eidx + j]
                lo = low[eidx + j]
                h_pct = (hi - ep) / ep
                l_pct = (lo - ep) / ep

                if l_pct <= sl:
                    lost = True
                    break
                if h_pct >= tp:
                    won = True
                    break

            if won:
                all_wins += 1
            elif lost:
                all_losses += 1
            else:
                # Timed out - check if profitable
                if eidx + max_hold < n:
                    final = close[eidx + max_hold]
                    if (final - ep) / ep > COMMISSION * 2:
                        all_wins += 1
                    else:
                        all_losses += 1

    total = all_wins + all_losses
    if total == 0:
        return 0, 0, 0
    wr = all_wins / total * 100
    return wr, total, all_wins


# --- Entry functions ---

def entry_A(df, p):
    return (
        (df["close"] > df[f"ema_{p.get('et', 200)}"])
        & (df["ema_9"] > df["ema_21"])
        & (df["rsi"] > p["rl"]) & (df["rsi"] < p["rh"])
        & (df["macdhist"] > 0)
        & (df["adx"] > p["am"])
        & (df["volume"] > df["vol_ma"] * p["vm"])
        & (df["green"] == 1)
        & (df["close"] < df["bb_upper"] * 0.98)
        & (df["mfi"] > 20) & (df["mfi"] < 70)
        & (df["ema_slope"] > 0)
    ).values

def entry_B(df, p):
    return (
        (df["ema_50"] > df["ema_200"])
        & (df["close"] > df["ema_21"])
        & (df["ema_9"] > df["ema_21"])
        & (df["ema_21"] > df["ema_50"])
        & (df["rsi"] > p["rl"]) & (df["rsi"] < p["rh"])
        & (df["macdhist"] > 0)
        & (df["macdhist"] > df["macdhist"].shift(1))
        & (df["adx"] > p["am"])
        & (df["volume"] > df["vol_ma"] * p["vm"])
        & (df["green"] == 1)
        & (df["ema_slope"] > 0.0001)
    ).values

def entry_C(df, p):
    return (
        (df["ema_50"] > df["ema_200"])
        & (df["ema_9"] > df["ema_21"])
        & (df["rsi"] > p["rl"]) & (df["rsi"] < p["rh"])
        & (df["macdhist"] > 0)
        & (df["cci"] > 0) & (df["cci"] < 150)
        & (df["mfi"] > 25) & (df["mfi"] < 65)
        & (df["adx"] > p["am"])
        & (df["volume"] > df["vol_ma"] * p["vm"])
        & (df["green"] == 1)
        & (df["close"] < df["bb_upper"] * 0.97)
        & (df["slowk"] > df["slowd"])
        & (df["ema_slope"] > 0.0002)
    ).values

def entry_D(df, p):
    """Ultra-selective: all stars aligned."""
    return (
        (df["ema_50"] > df["ema_200"])
        & (df["ema_9"] > df["ema_21"])
        & (df["ema_21"] > df["ema_50"])
        & (df["rsi"] > p["rl"]) & (df["rsi"] < p["rh"])
        & (df["rsi"] > df["rsi"].shift(1))  # RSI rising
        & (df["macdhist"] > 0)
        & (df["macdhist"] > df["macdhist"].shift(1))
        & (df["cci"] > 0) & (df["cci"] < 120)
        & (df["mfi"] > 30) & (df["mfi"] < 60)
        & (df["adx"] > p["am"])
        & (df["volume"] > df["vol_ma"] * p["vm"])
        & (df["green"] == 1)
        & (df["close"] < df["bb_upper"] * 0.96)
        & (df["slowk"] > df["slowd"])
        & (df["ema_slope"] > 0.0003)
    ).values


def main():
    print("Loading 15min data (2020-2026)...")
    pair_data = {}
    for pair in PAIRS:
        pair_data[pair] = load_pair(pair)
    print(f"Loaded {len(pair_data)} pairs\n")

    best_overall_wr = 0
    best_overall = None

    entries = {"A": entry_A, "B": entry_B, "C": entry_C, "D": entry_D}

    for name, fn in entries.items():
        print(f"\n{'='*55}")
        print(f"  Strategy {name}")
        print(f"{'='*55}")
        best_wr = 0
        best_total = 0
        best_p = None
        count = 0

        for et in ([100, 200] if name == "A" else [200]):
            for rl in [30, 35, 40, 45, 50]:
                for rh in [50, 55, 58, 62, 66]:
                    if rh <= rl:
                        continue
                    for am in [18, 22, 26, 30]:
                        for vm in [0.7, 0.9, 1.0, 1.2, 1.5]:
                            for tp in [0.005, 0.006, 0.008, 0.01, 0.012, 0.015]:
                                for sl in [-0.015, -0.02, -0.025, -0.03, -0.04]:
                                    for mh in [12, 20, 30, 40]:
                                        p = {"et": et, "rl": rl, "rh": rh, "am": am, "vm": vm}
                                        wr, total, wins = vectorized_winrate(pair_data, fn, tp, sl, mh, p)
                                        count += 1

                                        if total >= 50 and wr > best_wr:
                                            best_wr = wr
                                            best_total = total
                                            best_p = {**p, "tp": tp, "sl": sl, "mh": mh}
                                            if wr >= 88:
                                                print(f"  [{count:6d}] WR={wr:.1f}% T={total:4d} W={wins} tp={tp} sl={sl} mh={mh} rl={rl} rh={rh} am={am} vm={vm}")

                                        if count % 10000 == 0:
                                            print(f"  ... {count} tested, best WR: {best_wr:.1f}% (T={best_total})")

        print(f"\n  BEST {name}: WR={best_wr:.1f}% Trades={best_total}")
        if best_p:
            print(f"  Params: {best_p}")

        if best_wr > best_overall_wr:
            best_overall_wr = best_wr
            best_overall = (name, best_wr, best_total, best_p)

    print(f"\n\n{'='*70}")
    print(f"  FINAL WINNER (15min, 2020-2026)")
    print(f"{'='*70}")
    if best_overall:
        name, wr, total, p = best_overall
        losses = total - int(total * wr / 100)
        wins = total - losses

        # Calculate estimated profit
        avg_win = p["tp"] * 100
        avg_loss = p["sl"] * 100
        est_profit = (wins * p["tp"] - losses * abs(p["sl"]) - total * COMMISSION * 2)
        est_profit_pct = est_profit / (INITIAL_CAPITAL / 1000) * 100  # rough estimate

        print(f"  Strategy:      {name}")
        print(f"  Win Rate:      {wr:.1f}%")
        print(f"  Total Trades:  {total}")
        print(f"  Wins/Losses:   {wins}/{losses}")
        print(f"  Est. Profit:   ~{est_profit_pct:+.0f}%")
        print(f"  Take Profit:   {p['tp']*100:.1f}%")
        print(f"  Stop Loss:     {p['sl']*100:.1f}%")
        print(f"  Max Hold:      {p['mh']} candles ({p['mh']*15} min)")
        print(f"  Params:        {p}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
