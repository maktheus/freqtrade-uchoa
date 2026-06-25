#!/usr/bin/env python3
"""V6: Pre-compute forward returns, then instant parameter testing."""
import numpy as np
import pandas as pd
from pathlib import Path
import talib.abstract as ta
from technical import qtpylib
import time

DATA_DIR = Path("user_data/data/binance")
COMMISSION = 0.001
MAX_HOLD = 48  # Pre-compute up to 48 candles ahead (12 hours at 15min)

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
        df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        df["adx"] = ta.ADX(df, timeperiod=14)
        df["mfi"] = ta.MFI(df, timeperiod=14)
        df["cci"] = ta.CCI(df, timeperiod=20)
        stoch = ta.STOCH(df, fastk_period=14, slowk_period=3, slowd_period=3)
        df["slowk"] = stoch["slowk"]
        df["slowd"] = stoch["slowd"]
        df["ema_slope"] = (df["ema_50"] - df["ema_50"].shift(4)) / df["ema_50"].shift(4)
        df = df.dropna().reset_index(drop=True)

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(close)

        # Pre-compute: for each candle, max gain & max loss over next 1..MAX_HOLD candles
        # AND the order: did max_loss happen before max_gain?
        print(f"  Pre-computing forward returns for {pair}...")
        max_gain = np.zeros(n, dtype=np.float64)
        max_loss = np.zeros(n, dtype=np.float64)
        # For each horizon, track cumulative max high / min low
        first_tp_candle = np.full(n, MAX_HOLD + 1, dtype=np.int32)  # first candle TP hit for various TPs
        first_sl_candle = np.full(n, MAX_HOLD + 1, dtype=np.int32)  # first candle SL hit

        # Build matrix: max_gain[i, h] = max (high[i+1..i+h] - close[i]) / close[i]
        # But we just need max over all horizons for each TP/SL
        # Better: for each entry i, compute forward high/low trajectory
        # Then for any TP/SL, check if TP was hit before SL

        # Pre-compute per-candle: forward max high pct and forward min low pct at each step
        # Store: fwd_max_high[i] = rolling max of high[i+1..i+MAX_HOLD] relative to close[i]
        #        fwd_min_low[i] = rolling min of low[i+1..i+MAX_HOLD] relative to close[i]
        # And the candle index where each occurs

        fwd_max = np.full((n, MAX_HOLD), np.nan)
        fwd_min = np.full((n, MAX_HOLD), np.nan)

        for h in range(MAX_HOLD):
            shift = h + 1
            if shift >= n:
                break
            # high pct at candle i+shift relative to close[i]
            hi_pct = np.empty(n)
            lo_pct = np.empty(n)
            hi_pct[:] = np.nan
            lo_pct[:] = np.nan
            hi_pct[:n-shift] = (high[shift:] - close[:n-shift]) / close[:n-shift]
            lo_pct[:n-shift] = (low[shift:] - close[:n-shift]) / close[:n-shift]

            if h == 0:
                fwd_max[:, h] = hi_pct
                fwd_min[:, h] = lo_pct
            else:
                fwd_max[:, h] = np.fmax(fwd_max[:, h-1], hi_pct)
                fwd_min[:, h] = np.fmin(fwd_min[:, h-1], lo_pct)

        result[pair] = {
            "n": n,
            "rsi": df["rsi"].values,
            "adx": df["adx"].values,
            "vol_ratio": df["vol_ratio"].values,
            "mfi": df["mfi"].values,
            "cci": df["cci"].values,
            "slowk": df["slowk"].values,
            "ema_slope": df["ema_slope"].values,
            "c_ema50_200": (df["ema_50"] > df["ema_200"]).values,
            "c_ema9_21": (df["ema_9"] > df["ema_21"]).values,
            "c_ema21_50": (df["ema_21"] > df["ema_50"]).values,
            "c_macd_pos": (df["macdhist"] > 0).values,
            "c_macd_rising": (df["macdhist"] > df["macdhist"].shift(1)).values,
            "c_rsi_rising": (df["rsi"] > df["rsi"].shift(1)).values,
            "c_stoch_bull": (df["slowk"] > df["slowd"]).values,
            "green": (df["close"] > df["open"]).values,
            "close": close,
            "fwd_max": fwd_max,
            "fwd_min": fwd_min,
        }
    return result


def instant_test(all_data, entry_mask_per_pair, tp, sl, max_hold_idx):
    """Instantly check win rate using pre-computed forward returns."""
    wins = 0
    losses = 0
    for pair, data in all_data.items():
        mask = entry_mask_per_pair[pair]
        entries = np.where(mask)[0]
        if len(entries) == 0:
            continue

        # Filter entries too close
        filtered = [entries[0]]
        last = entries[0]
        for e in entries[1:]:
            if e - last >= 4:
                filtered.append(e)
                last = e

        fwd_max = data["fwd_max"]
        fwd_min = data["fwd_min"]
        h = max_hold_idx - 1  # 0-indexed

        for eidx in filtered:
            if eidx >= data["n"] - max_hold_idx:
                continue
            # Check: did TP hit? did SL hit? which first?
            # We need to check candle by candle to see which hit first
            hit_tp = False
            hit_sl = False
            for j in range(max_hold_idx):
                if np.isnan(fwd_max[eidx, j]):
                    break
                if not hit_sl and fwd_min[eidx, j] <= sl:
                    hit_sl = True
                    if hit_tp:
                        break
                    # Check if TP also hit at same or earlier candle
                    if fwd_max[eidx, j] >= tp:
                        # Both hit - need to check which candle specifically
                        # Since fwd_max/fwd_min are cumulative, we need per-candle
                        # Approximate: if both cumulative hit at same horizon, favor SL
                        hit_tp = True
                    break  # SL hit first
                if not hit_tp and fwd_max[eidx, j] >= tp:
                    hit_tp = True
                    break

            if hit_tp and not hit_sl:
                wins += 1
            elif hit_sl:
                losses += 1
            else:
                # Timeout - check final position using last fwd_max value
                last_h = min(h, max_hold_idx - 1)
                if not np.isnan(fwd_max[eidx, last_h]):
                    if fwd_max[eidx, last_h] > COMMISSION * 2:
                        wins += 1
                    else:
                        losses += 1

    total = wins + losses
    return (wins / total * 100 if total else 0), total, wins


def build_entry_masks(all_data, entry_fn, params):
    masks = {}
    for pair, data in all_data.items():
        masks[pair] = entry_fn(data, params)
    return masks


# Entry functions
def entry_trend(d, p):
    return (d["c_ema50_200"] & d["c_ema9_21"]
            & (d["rsi"] > p[0]) & (d["rsi"] < p[1])
            & d["c_macd_pos"] & (d["adx"] > p[2])
            & (d["vol_ratio"] > p[3]) & d["green"] & (d["ema_slope"] > 0))

def entry_triple(d, p):
    return (d["c_ema50_200"] & d["c_ema9_21"] & d["c_ema21_50"]
            & (d["rsi"] > p[0]) & (d["rsi"] < p[1])
            & d["c_macd_pos"] & d["c_macd_rising"]
            & (d["adx"] > p[2]) & (d["vol_ratio"] > p[3])
            & d["green"] & (d["ema_slope"] > p[4]))

def entry_full(d, p):
    return (d["c_ema50_200"] & d["c_ema9_21"] & d["c_ema21_50"]
            & (d["rsi"] > p[0]) & (d["rsi"] < p[1])
            & d["c_macd_pos"] & d["c_macd_rising"]
            & (d["mfi"] > p[5]) & (d["mfi"] < p[6])
            & (d["adx"] > p[2]) & (d["vol_ratio"] > p[3])
            & d["green"] & d["c_stoch_bull"] & (d["ema_slope"] > p[4]))

def entry_rsi(d, p):
    return (d["c_ema50_200"] & d["c_ema9_21"]
            & (d["rsi"] > p[0]) & (d["rsi"] < p[1])
            & d["c_rsi_rising"] & d["c_macd_pos"] & d["c_stoch_bull"]
            & (d["slowk"] < p[5]) & (d["adx"] > p[2])
            & (d["vol_ratio"] > p[3]) & d["green"] & (d["ema_slope"] > 0))


def main():
    t0 = time.time()
    print("Loading and pre-computing data...")
    all_data = load_all()
    print(f"Done in {time.time()-t0:.1f}s\n")

    strategies = {
        "trend": (entry_trend, [
            (rl, rh, am, vm, 0, 0, 0)
            for rl in range(30, 52, 2)
            for rh in range(max(50, 0), 72, 2)
            if rh > rl + 5
            for am in range(16, 32, 3)
            for vm in [0.6, 0.8, 1.0, 1.2]
        ]),
        "triple": (entry_triple, [
            (rl, rh, am, vm, ms, 0, 0)
            for rl in range(30, 52, 3)
            for rh in range(50, 70, 3)
            if rh > rl + 5
            for am in range(18, 32, 4)
            for vm in [0.7, 0.9, 1.1]
            for ms in [0.00005, 0.0001, 0.0003]
        ]),
        "full": (entry_full, [
            (rl, rh, am, vm, ms, mfl, mfh)
            for rl in range(35, 50, 4)
            for rh in range(52, 68, 4)
            if rh > rl + 5
            for am in range(18, 30, 4)
            for vm in [0.8, 1.0, 1.2]
            for ms in [0.0001, 0.0003]
            for mfl in [25, 35]
            for mfh in [55, 65]
        ]),
        "rsi": (entry_rsi, [
            (rl, rh, am, vm, 0, skm, 0)
            for rl in range(30, 50, 3)
            for rh in range(50, 70, 3)
            if rh > rl + 5
            for am in range(18, 30, 3)
            for vm in [0.7, 0.9, 1.1]
            for skm in [60, 70, 80]
        ]),
    }

    tp_values = [0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.015]
    sl_values = [-0.01, -0.015, -0.02, -0.025, -0.03, -0.04, -0.05]
    mh_values = [8, 12, 16, 20, 30, 40, 48]

    overall_best_wr = 0
    overall_best = None

    for sname, (sfn, params_list) in strategies.items():
        n_exit = len(tp_values) * len(sl_values) * len(mh_values)
        total_combos = len(params_list) * n_exit
        print(f"\n{'='*55}")
        print(f"  {sname}: {len(params_list)} entry x {n_exit} exit = {total_combos}")
        print(f"{'='*55}")

        best_wr = 0
        best_total = 0
        best_config = None
        count = 0
        t1 = time.time()

        for p in params_list:
            masks = build_entry_masks(all_data, sfn, p)
            n_entries = sum(np.sum(m) for m in masks.values())
            if n_entries < 50:
                count += n_exit
                continue

            for tp in tp_values:
                for sl in sl_values:
                    for mh in mh_values:
                        wr, total, wins = instant_test(all_data, masks, tp, sl, mh)
                        count += 1

                        if total >= 50 and wr > best_wr:
                            best_wr = wr
                            best_total = total
                            best_config = (p, tp, sl, mh, total, wins)
                            if wr >= 85:
                                print(f"  [{count:7d}] WR={wr:.1f}% T={total:5d} W={wins} tp={tp} sl={sl} mh={mh} rsi={p[0]}-{p[1]}")

            if count % 10000 < n_exit:
                elapsed = time.time() - t1
                rate = count / elapsed if elapsed > 0 else 0
                eta = (total_combos - count) / rate / 60 if rate > 0 else 0
                print(f"  ... {count}/{total_combos} ({rate:.0f}/s, ETA {eta:.1f}min) best={best_wr:.1f}%")

        elapsed = time.time() - t1
        print(f"\n  BEST {sname}: WR={best_wr:.1f}% T={best_total} ({elapsed:.0f}s)")
        if best_config:
            p, tp, sl, mh, total, wins = best_config
            print(f"  rsi={p[0]}-{p[1]} adx>{p[2]} vol>{p[3]} tp={tp} sl={sl} mh={mh}")
            if best_wr > overall_best_wr:
                overall_best_wr = best_wr
                overall_best = (sname, best_wr, best_total, best_config)

    print(f"\n\n{'='*70}")
    print(f"  FINAL WINNER (15min, 2020-2026, {len(PAIRS)} pairs)")
    print(f"{'='*70}")
    if overall_best:
        name, wr, total, (p, tp, sl, mh, _, wins) = overall_best
        losses = total - wins
        pf = (wins * tp) / (losses * abs(sl)) if losses > 0 else 999
        print(f"  Strategy:      {name}")
        print(f"  Win Rate:      {wr:.1f}%")
        print(f"  Total Trades:  {total}")
        print(f"  Wins/Losses:   {wins}/{losses}")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Take Profit:   {tp*100:.2f}%")
        print(f"  Stop Loss:     {sl*100:.1f}%")
        print(f"  Max Hold:      {mh} candles ({mh*15} min)")
        print(f"  RSI Range:     {p[0]}-{p[1]}")
        print(f"  ADX Min:       {p[2]}")
        print(f"  Vol Mult:      {p[3]}")
    print(f"{'='*70}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
