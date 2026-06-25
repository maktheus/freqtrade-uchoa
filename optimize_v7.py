#!/usr/bin/env python3
"""V7: Pre-compute first_tp/first_sl candle for every row, then vectorized testing."""
import numpy as np
import pandas as pd
from pathlib import Path
import talib.abstract as ta
from technical import qtpylib
import time

DATA_DIR = Path("user_data/data/binance")
COMMISSION = 0.001
MAX_FWD = 48

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LINK/USDT",
         "ADA/USDT", "DOT/USDT", "LTC/USDT", "AVAX/USDT", "DOGE/USDT"]

TP_VALUES = np.array([0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.012, 0.015])
SL_VALUES = np.array([-0.01, -0.015, -0.02, -0.025, -0.03, -0.035, -0.04, -0.05])
MH_VALUES = np.array([8, 12, 16, 20, 24, 30, 40, 48])


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
        stoch = ta.STOCH(df, fastk_period=14, slowk_period=3, slowd_period=3)
        df["slowk"] = stoch["slowk"]
        df["slowd"] = stoch["slowd"]
        df["ema_slope"] = (df["ema_50"] - df["ema_50"].shift(4)) / df["ema_50"].shift(4)
        df = df.dropna().reset_index(drop=True)

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(close)

        print(f"  {pair}: {n} candles, pre-computing first TP/SL candles...")

        # For each candle i, compute forward high/low pct for each step
        # first_tp[i, tp_idx] = first candle offset where cumulative high >= TP
        # first_sl[i, sl_idx] = first candle offset where cumulative low <= SL
        first_tp = np.full((n, len(TP_VALUES)), MAX_FWD + 1, dtype=np.int16)
        first_sl = np.full((n, len(SL_VALUES)), MAX_FWD + 1, dtype=np.int16)
        # Also store final pct at each max_hold for timeout resolution
        final_pct = np.full((n, MAX_FWD), 0.0, dtype=np.float32)

        cum_max_high = np.full(n, -np.inf)
        cum_min_low = np.full(n, np.inf)

        for h in range(MAX_FWD):
            shift = h + 1
            if shift >= n:
                break

            # High/low pct at candle i + shift, relative to close[i]
            hi_pct = np.empty(n, dtype=np.float64)
            lo_pct = np.empty(n, dtype=np.float64)
            cl_pct = np.empty(n, dtype=np.float64)
            hi_pct[:] = -np.inf
            lo_pct[:] = np.inf
            cl_pct[:] = 0

            valid = n - shift
            hi_pct[:valid] = (high[shift:shift+valid] - close[:valid]) / close[:valid]
            lo_pct[:valid] = (low[shift:shift+valid] - close[:valid]) / close[:valid]
            cl_pct[:valid] = (close[shift:shift+valid] - close[:valid]) / close[:valid]

            cum_max_high[:valid] = np.maximum(cum_max_high[:valid], hi_pct[:valid])
            cum_min_low[:valid] = np.minimum(cum_min_low[:valid], lo_pct[:valid])
            final_pct[:valid, h] = cl_pct[:valid]

            # Check which TPs are now first reached at this step
            for ti, tp in enumerate(TP_VALUES):
                newly_hit = (first_tp[:valid, ti] > MAX_FWD) & (cum_max_high[:valid] >= tp)
                first_tp[:valid, ti] = np.where(newly_hit, h + 1, first_tp[:valid, ti])

            # Check which SLs are now first reached
            for si, sl in enumerate(SL_VALUES):
                newly_hit = (first_sl[:valid, si] > MAX_FWD) & (cum_min_low[:valid] <= sl)
                first_sl[:valid, si] = np.where(newly_hit, h + 1, first_sl[:valid, si])

        result[pair] = {
            "n": n,
            "rsi": df["rsi"].values.astype(np.float32),
            "adx": df["adx"].values.astype(np.float32),
            "vol_ratio": df["vol_ratio"].values.astype(np.float32),
            "mfi": df["mfi"].values.astype(np.float32),
            "slowk": df["slowk"].values.astype(np.float32),
            "ema_slope": df["ema_slope"].values.astype(np.float32),
            "c_ema50_200": (df["ema_50"] > df["ema_200"]).values,
            "c_ema9_21": (df["ema_9"] > df["ema_21"]).values,
            "c_ema21_50": (df["ema_21"] > df["ema_50"]).values,
            "c_macd_pos": (df["macdhist"] > 0).values,
            "c_macd_rising": (df["macdhist"] > df["macdhist"].shift(1)).values,
            "c_rsi_rising": (df["rsi"] > df["rsi"].shift(1)).values,
            "c_stoch_bull": (df["slowk"] > df["slowd"]).values,
            "green": (df["close"] > df["open"]).values,
            "first_tp": first_tp,
            "first_sl": first_sl,
            "final_pct": final_pct,
        }
    return result


def test_combo(all_data, masks, tp_idx, sl_idx, mh_idx):
    """Test one TP/SL/MH combo across all pairs. Returns (wins, losses)."""
    mh = MH_VALUES[mh_idx]
    wins = 0
    losses = 0

    for pair, data in all_data.items():
        mask = masks[pair]
        entries = np.where(mask)[0]
        if len(entries) == 0:
            continue

        # Filter entries too close
        filtered = []
        last = -10
        for e in entries:
            if e - last >= 4:
                filtered.append(e)
                last = e
        if not filtered:
            continue

        ft = data["first_tp"]
        fs = data["first_sl"]
        fp = data["final_pct"]
        n = data["n"]

        for eidx in filtered:
            if eidx >= n - mh:
                continue

            tp_hit_at = ft[eidx, tp_idx]
            sl_hit_at = fs[eidx, sl_idx]

            if tp_hit_at <= mh and (tp_hit_at <= sl_hit_at):
                wins += 1
            elif sl_hit_at <= mh:
                losses += 1
            else:
                # Timeout
                h = min(mh - 1, MAX_FWD - 1)
                if fp[eidx, h] > COMMISSION * 2:
                    wins += 1
                else:
                    losses += 1

    return wins, losses


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
    print("Loading and pre-computing...")
    all_data = load_all()
    print(f"Pre-computation done in {time.time()-t0:.1f}s\n")

    strategies = {
        "trend": (entry_trend, [
            (rl, rh, am, vm, 0, 0, 0)
            for rl in range(28, 52, 2) for rh in range(50, 72, 2)
            if rh > rl + 5
            for am in range(14, 32, 2) for vm in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3]
        ]),
        "triple": (entry_triple, [
            (rl, rh, am, vm, ms, 0, 0)
            for rl in range(28, 52, 3) for rh in range(50, 70, 3)
            if rh > rl + 5
            for am in range(16, 32, 3) for vm in [0.7, 0.9, 1.1, 1.3]
            for ms in [0.00003, 0.0001, 0.0003]
        ]),
        "full": (entry_full, [
            (rl, rh, am, vm, ms, mfl, mfh)
            for rl in range(30, 52, 3) for rh in range(50, 68, 3)
            if rh > rl + 5
            for am in range(16, 30, 3) for vm in [0.7, 0.9, 1.1]
            for ms in [0.00005, 0.0001, 0.0003]
            for mfl in [20, 30, 40] for mfh in [50, 60, 70]
        ]),
        "rsi": (entry_rsi, [
            (rl, rh, am, vm, 0, skm, 0)
            for rl in range(28, 50, 3) for rh in range(50, 70, 3)
            if rh > rl + 5
            for am in range(16, 30, 3) for vm in [0.7, 0.9, 1.1]
            for skm in [55, 65, 75, 85]
        ]),
    }

    overall_best_wr = 0
    overall_best = None
    n_tp = len(TP_VALUES)
    n_sl = len(SL_VALUES)
    n_mh = len(MH_VALUES)
    n_exit = n_tp * n_sl * n_mh

    for sname, (sfn, params_list) in strategies.items():
        total_combos = len(params_list) * n_exit
        print(f"\n{'='*55}")
        print(f"  {sname}: {len(params_list)} entry x {n_exit} exit = {total_combos}")
        print(f"{'='*55}")

        best_wr = 0
        best_total = 0
        best_config = None
        count = 0
        t1 = time.time()

        for pidx, p in enumerate(params_list):
            masks = {pair: sfn(data, p) for pair, data in all_data.items()}
            n_entries = sum(np.sum(m) for m in masks.values())
            if n_entries < 30:
                count += n_exit
                continue

            for ti in range(n_tp):
                for si in range(n_sl):
                    for mi in range(n_mh):
                        w, l = test_combo(all_data, masks, ti, si, mi)
                        total = w + l
                        count += 1
                        if total >= 50:
                            wr = w / total * 100
                            if wr > best_wr:
                                best_wr = wr
                                best_total = total
                                best_config = (p, ti, si, mi, total, w)
                                if wr >= 88:
                                    print(f"  [{count:7d}] WR={wr:.1f}% T={total:5d} tp={TP_VALUES[ti]} sl={SL_VALUES[si]} mh={MH_VALUES[mi]} rsi={p[0]}-{p[1]} adx>{p[2]} vol>{p[3]}")

            if (pidx + 1) % 50 == 0:
                elapsed = time.time() - t1
                rate = count / elapsed if elapsed > 0 else 0
                eta = (total_combos - count) / rate / 60 if rate > 0 else 0
                print(f"  ... {count}/{total_combos} ({rate:.0f}/s, ETA {eta:.1f}min) best={best_wr:.1f}%")

        elapsed = time.time() - t1
        print(f"\n  BEST {sname}: WR={best_wr:.1f}% T={best_total} ({elapsed:.0f}s)")
        if best_config:
            p, ti, si, mi, total, wins = best_config
            print(f"  rsi={p[0]}-{p[1]} adx>{p[2]} vol>{p[3]} tp={TP_VALUES[ti]} sl={SL_VALUES[si]} mh={MH_VALUES[mi]}")
            if best_wr > overall_best_wr:
                overall_best_wr = best_wr
                overall_best = (sname, best_wr, best_total, best_config)

    print(f"\n\n{'='*70}")
    print(f"  FINAL WINNER (15min, 2020-2026, {len(PAIRS)} pairs)")
    print(f"{'='*70}")
    if overall_best:
        name, wr, total, (p, ti, si, mi, _, wins) = overall_best
        tp = TP_VALUES[ti]
        sl = SL_VALUES[si]
        mh = MH_VALUES[mi]
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
