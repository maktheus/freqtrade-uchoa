#!/usr/bin/env python3
"""V5: Pre-compute all indicators, then fast numpy-only win rate calc."""
import numpy as np
import pandas as pd
from pathlib import Path
import talib.abstract as ta
from technical import qtpylib
import time

DATA_DIR = Path("user_data/data/binance")
COMMISSION = 0.001

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LINK/USDT",
         "ADA/USDT", "DOT/USDT", "LTC/USDT", "AVAX/USDT", "DOGE/USDT"]


def load_all():
    result = {}
    for pair in PAIRS:
        fn = pair.replace("/", "_") + "-15m.feather"
        df = pd.read_feather(DATA_DIR / fn).sort_values("date").reset_index(drop=True)
        for p in [9, 21, 50, 100, 200]:
            df[f"ema_{p}"] = ta.EMA(df, timeperiod=p)
        df["rsi"] = ta.RSI(df, timeperiod=14)
        macd = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
        df["macdhist"] = macd["macdhist"]
        df["macdhist_prev"] = df["macdhist"].shift(1)
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
        df["bb_pct"] = (df["close"] - bb["lower"]) / (bb["upper"] - bb["lower"])
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["adx"] = ta.ADX(df, timeperiod=14)
        df["mfi"] = ta.MFI(df, timeperiod=14)
        df["cci"] = ta.CCI(df, timeperiod=20)
        stoch = ta.STOCH(df, fastk_period=14, slowk_period=3, slowd_period=3)
        df["slowk"] = stoch["slowk"]
        df["slowd"] = stoch["slowd"]
        df["green"] = (df["close"] > df["open"]).astype(np.int8)
        df["ema_slope"] = (df["ema_50"] - df["ema_50"].shift(4)) / df["ema_50"].shift(4)
        df["rsi_prev"] = df["rsi"].shift(1)

        # Pre-compute boolean arrays
        df["c_ema50_200"] = df["ema_50"] > df["ema_200"]
        df["c_ema9_21"] = df["ema_9"] > df["ema_21"]
        df["c_ema21_50"] = df["ema_21"] > df["ema_50"]
        df["c_macd_pos"] = df["macdhist"] > 0
        df["c_macd_rising"] = df["macdhist"] > df["macdhist_prev"]
        df["c_rsi_rising"] = df["rsi"] > df["rsi_prev"]
        df["c_stoch_bull"] = df["slowk"] > df["slowd"]

        df = df.dropna().reset_index(drop=True)

        # Convert to numpy for speed
        result[pair] = {
            "close": df["close"].values.astype(np.float64),
            "high": df["high"].values.astype(np.float64),
            "low": df["low"].values.astype(np.float64),
            "rsi": df["rsi"].values.astype(np.float64),
            "adx": df["adx"].values.astype(np.float64),
            "vol_ratio": df["vol_ratio"].values.astype(np.float64),
            "bb_pct": df["bb_pct"].values.astype(np.float64),
            "mfi": df["mfi"].values.astype(np.float64),
            "cci": df["cci"].values.astype(np.float64),
            "slowk": df["slowk"].values.astype(np.float64),
            "ema_slope": df["ema_slope"].values.astype(np.float64),
            # Booleans
            "c_ema50_200": df["c_ema50_200"].values,
            "c_ema9_21": df["c_ema9_21"].values,
            "c_ema21_50": df["c_ema21_50"].values,
            "c_macd_pos": df["c_macd_pos"].values,
            "c_macd_rising": df["c_macd_rising"].values,
            "c_rsi_rising": df["c_rsi_rising"].values,
            "c_stoch_bull": df["c_stoch_bull"].values,
            "green": df["green"].values.astype(bool),
        }
    return result


def fast_test(data, mask, tp, sl, max_hold):
    """Test win rate given entry mask and exit params."""
    close, high, low = data["close"], data["high"], data["low"]
    n = len(close)
    entries = np.where(mask)[0]
    if len(entries) == 0:
        return 0, 0, 0

    # Filter entries too close together
    filtered = [entries[0]]
    last = entries[0]
    for e in entries[1:]:
        if e - last >= 4:
            filtered.append(e)
            last = e
    entries = np.array(filtered)

    wins = 0
    losses = 0
    for eidx in entries:
        ep = close[eidx]
        end = min(eidx + max_hold + 1, n)
        won = lost = False
        for j in range(eidx + 1, end):
            h_pct = (high[j] - ep) / ep
            l_pct = (low[j] - ep) / ep
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
                final = (close[min(eidx + max_hold, n-1)] - ep) / ep
                if final > COMMISSION * 2:
                    wins += 1
                else:
                    losses += 1

    total = wins + losses
    return (wins / total * 100 if total else 0), total, wins


def test_strategy(all_data, entry_builder, tp, sl, max_hold, params):
    """Test across all pairs."""
    total_wins = 0
    total_losses = 0
    for pair, data in all_data.items():
        mask = entry_builder(data, params)
        wr, total, wins = fast_test(data, mask, tp, sl, max_hold)
        total_wins += wins
        total_losses += (total - wins)
    total = total_wins + total_losses
    return (total_wins / total * 100 if total else 0), total, total_wins


# --- Entry builders (work on numpy arrays) ---

def entry_trend(d, p):
    return (
        d["c_ema50_200"]
        & d["c_ema9_21"]
        & (d["rsi"] > p[0]) & (d["rsi"] < p[1])
        & d["c_macd_pos"]
        & (d["adx"] > p[2])
        & (d["vol_ratio"] > p[3])
        & d["green"]
        & (d["ema_slope"] > 0)
    )

def entry_triple(d, p):
    return (
        d["c_ema50_200"]
        & d["c_ema9_21"]
        & d["c_ema21_50"]
        & (d["rsi"] > p[0]) & (d["rsi"] < p[1])
        & d["c_macd_pos"]
        & d["c_macd_rising"]
        & (d["adx"] > p[2])
        & (d["vol_ratio"] > p[3])
        & d["green"]
        & (d["ema_slope"] > p[4])
    )

def entry_confluence(d, p):
    return (
        d["c_ema50_200"]
        & d["c_ema9_21"]
        & d["c_ema21_50"]
        & (d["rsi"] > p[0]) & (d["rsi"] < p[1])
        & d["c_macd_pos"]
        & d["c_macd_rising"]
        & (d["mfi"] > p[5]) & (d["mfi"] < p[6])
        & (d["adx"] > p[2])
        & (d["vol_ratio"] > p[3])
        & d["green"]
        & d["c_stoch_bull"]
        & (d["ema_slope"] > p[4])
    )

def entry_rsi_momentum(d, p):
    return (
        d["c_ema50_200"]
        & d["c_ema9_21"]
        & (d["rsi"] > p[0]) & (d["rsi"] < p[1])
        & d["c_rsi_rising"]
        & d["c_macd_pos"]
        & d["c_stoch_bull"]
        & (d["slowk"] < p[5])
        & (d["adx"] > p[2])
        & (d["vol_ratio"] > p[3])
        & d["green"]
        & (d["ema_slope"] > 0)
    )


def main():
    t0 = time.time()
    print("Loading data...")
    all_data = load_all()
    print(f"Loaded in {time.time()-t0:.1f}s\n")

    strategies = {
        "trend": (entry_trend, lambda: [
            (rl, rh, am, vm, 0, 0, 0)
            for rl in range(30, 52, 3)
            for rh in range(max(rl+8, 50), 72, 3)
            for am in range(16, 32, 3)
            for vm in [0.6, 0.8, 1.0, 1.2]
        ]),
        "triple": (entry_triple, lambda: [
            (rl, rh, am, vm, ms, 0, 0)
            for rl in range(30, 52, 4)
            for rh in range(max(rl+8, 50), 72, 4)
            for am in range(18, 32, 4)
            for vm in [0.7, 0.9, 1.1, 1.3]
            for ms in [0.00005, 0.0001, 0.0002, 0.0004]
        ]),
        "confluence": (entry_confluence, lambda: [
            (rl, rh, am, vm, ms, mfi_lo, mfi_hi)
            for rl in range(35, 52, 4)
            for rh in range(max(rl+8, 52), 70, 4)
            for am in range(18, 30, 4)
            for vm in [0.8, 1.0, 1.2]
            for ms in [0.0001, 0.0002, 0.0004]
            for mfi_lo in [25, 35]
            for mfi_hi in [55, 65]
        ]),
        "rsi_mom": (entry_rsi_momentum, lambda: [
            (rl, rh, am, vm, 0, sk_max, 0)
            for rl in range(30, 52, 3)
            for rh in range(max(rl+8, 50), 70, 3)
            for am in range(18, 32, 4)
            for vm in [0.7, 0.9, 1.1]
            for sk_max in [60, 70, 80]
        ]),
    }

    tp_values = [0.005, 0.007, 0.009, 0.012, 0.015]
    sl_values = [-0.015, -0.02, -0.03, -0.04, -0.05]
    mh_values = [12, 20, 30, 48]

    overall_best_wr = 0
    overall_best = None

    for sname, (sfn, param_gen) in strategies.items():
        params_list = param_gen()
        print(f"\n{'='*55}")
        combos = len(params_list) * len(tp_values) * len(sl_values) * len(mh_values)
        print(f"  {sname} ({len(params_list)} entries x {len(tp_values)*len(sl_values)*len(mh_values)} exit = {combos})")
        print(f"{'='*55}")

        best_wr = 0
        best_total = 0
        best_config = None
        count = 0
        t1 = time.time()

        for p in params_list:
            for tp in tp_values:
                for sl in sl_values:
                    for mh in mh_values:
                        wr, total, wins = test_strategy(all_data, sfn, tp, sl, mh, p)
                        count += 1

                        if total >= 50 and wr > best_wr:
                            best_wr = wr
                            best_total = total
                            best_config = (p, tp, sl, mh)
                            if wr >= 85:
                                print(f"  [{count:6d}] WR={wr:.1f}% T={total:5d} W={wins} tp={tp} sl={sl} mh={mh} rsi={p[0]}-{p[1]}")

                        if count % 2000 == 0:
                            elapsed = time.time() - t1
                            rate = count / elapsed
                            eta = (combos - count) / rate / 60
                            print(f"  ... {count}/{combos} ({rate:.0f}/s, ETA {eta:.1f}min) best={best_wr:.1f}%")

        print(f"\n  BEST {sname}: WR={best_wr:.1f}% T={best_total}")
        if best_config:
            p, tp, sl, mh = best_config
            losses = best_total - int(best_total * best_wr / 100)
            wins_n = best_total - losses
            print(f"  Params: rsi={p[0]}-{p[1]} adx>{p[2]} vol>{p[3]} tp={tp} sl={sl} mh={mh}")
            if best_wr > overall_best_wr:
                overall_best_wr = best_wr
                overall_best = (sname, best_wr, best_total, best_config)

    print(f"\n\n{'='*70}")
    print(f"  FINAL WINNER (15min, 2020-2026, {len(PAIRS)} pairs)")
    print(f"{'='*70}")
    if overall_best:
        name, wr, total, (p, tp, sl, mh) = overall_best
        losses = total - int(total * wr / 100)
        wins = total - losses
        pf = (wins * tp) / (losses * abs(sl)) if losses > 0 else 999
        print(f"  Strategy:      {name}")
        print(f"  Win Rate:      {wr:.1f}%")
        print(f"  Total Trades:  {total}")
        print(f"  Wins/Losses:   {wins}/{losses}")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Take Profit:   {tp*100:.1f}%")
        print(f"  Stop Loss:     {sl*100:.1f}%")
        print(f"  Max Hold:      {mh} candles ({mh*15} min)")
        print(f"  RSI Range:     {p[0]}-{p[1]}")
        print(f"  ADX Min:       {p[2]}")
        print(f"  Vol Mult:      {p[3]}")
        if len(p) > 4 and p[4] > 0:
            print(f"  EMA Slope Min: {p[4]}")
    print(f"{'='*70}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
