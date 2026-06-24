#!/usr/bin/env python3
"""Optimize for 90%+ win rate on last 1 year of 15min data."""
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

TP_VALS = [0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.015]
SL_VALS = [-0.01, -0.015, -0.02, -0.025, -0.03, -0.04, -0.05, -0.06]
MH_VALS = [8, 12, 16, 20, 30, 40, 48]


def load_last_year(pair):
    fn = pair.replace("/", "_") + "-15m.feather"
    df = pd.read_feather(DATA_DIR / fn).sort_values("date").reset_index(drop=True)
    # Keep last year only (~96*365 = 35040 candles)
    df = df.tail(35040).reset_index(drop=True)

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
    return df


def precompute(df):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)

    # first_tp[i, j] = first candle where cumul max high >= TP_VALS[j]
    first_tp = np.full((n, len(TP_VALS)), MAX_FWD + 1, dtype=np.int16)
    first_sl = np.full((n, len(SL_VALS)), MAX_FWD + 1, dtype=np.int16)
    final_pct = np.zeros((n, MAX_FWD), dtype=np.float32)

    cum_max = np.full(n, -np.inf)
    cum_min = np.full(n, np.inf)

    for h in range(MAX_FWD):
        s = h + 1
        if s >= n:
            break
        v = n - s
        hi = (high[s:s+v] - close[:v]) / close[:v]
        lo = (low[s:s+v] - close[:v]) / close[:v]
        cl = (close[s:s+v] - close[:v]) / close[:v]

        cum_max[:v] = np.maximum(cum_max[:v], hi)
        cum_min[:v] = np.minimum(cum_min[:v], lo)
        final_pct[:v, h] = cl

        for j, tp in enumerate(TP_VALS):
            hit = (first_tp[:v, j] > MAX_FWD) & (cum_max[:v] >= tp)
            first_tp[:v, j] = np.where(hit, s, first_tp[:v, j])

        for j, sl in enumerate(SL_VALS):
            hit = (first_sl[:v, j] > MAX_FWD) & (cum_min[:v] <= sl)
            first_sl[:v, j] = np.where(hit, s, first_sl[:v, j])

    return {
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
        "n": n,
    }


def test(all_data, masks, ti, si, mi):
    mh = MH_VALS[mi]
    w, l = 0, 0
    for pair, d in all_data.items():
        m = masks[pair]
        entries = np.where(m)[0]
        if len(entries) == 0:
            continue
        filt = [entries[0]]
        last = entries[0]
        for e in entries[1:]:
            if e - last >= 4:
                filt.append(e)
                last = e

        ft, fs, fp = d["first_tp"], d["first_sl"], d["final_pct"]
        n = d["n"]
        for ei in filt:
            if ei >= n - mh:
                continue
            tpc = ft[ei, ti]
            slc = fs[ei, si]
            if tpc <= mh and tpc <= slc:
                w += 1
            elif slc <= mh:
                l += 1
            else:
                h = min(mh - 1, MAX_FWD - 1)
                if fp[ei, h] > COMMISSION * 2:
                    w += 1
                else:
                    l += 1
    return w, l


# Entry functions
def e_trend(d, p):
    return (d["c_ema50_200"] & d["c_ema9_21"]
            & (d["rsi"] > p[0]) & (d["rsi"] < p[1])
            & d["c_macd_pos"] & (d["adx"] > p[2])
            & (d["vol_ratio"] > p[3]) & d["green"] & (d["ema_slope"] > 0))

def e_triple(d, p):
    return (d["c_ema50_200"] & d["c_ema9_21"] & d["c_ema21_50"]
            & (d["rsi"] > p[0]) & (d["rsi"] < p[1])
            & d["c_macd_pos"] & d["c_macd_rising"]
            & (d["adx"] > p[2]) & (d["vol_ratio"] > p[3])
            & d["green"] & (d["ema_slope"] > p[4]))

def e_full(d, p):
    return (d["c_ema50_200"] & d["c_ema9_21"] & d["c_ema21_50"]
            & (d["rsi"] > p[0]) & (d["rsi"] < p[1])
            & d["c_macd_pos"] & d["c_macd_rising"]
            & (d["mfi"] > p[5]) & (d["mfi"] < p[6])
            & (d["adx"] > p[2]) & (d["vol_ratio"] > p[3])
            & d["green"] & d["c_stoch_bull"] & (d["ema_slope"] > p[4]))

def e_rsi(d, p):
    return (d["c_ema50_200"] & d["c_ema9_21"]
            & (d["rsi"] > p[0]) & (d["rsi"] < p[1])
            & d["c_rsi_rising"] & d["c_macd_pos"] & d["c_stoch_bull"]
            & (d["slowk"] < p[5]) & (d["adx"] > p[2])
            & (d["vol_ratio"] > p[3]) & d["green"] & (d["ema_slope"] > 0))


def main():
    t0 = time.time()
    print("Loading last year of data and pre-computing...")
    all_data = {}
    for pair in PAIRS:
        df = load_last_year(pair)
        all_data[pair] = precompute(df)
        print(f"  {pair}: {all_data[pair]['n']} candles")
    print(f"Done in {time.time()-t0:.1f}s\n")

    strategies = {
        "trend": (e_trend, [
            (rl, rh, am, vm, 0, 0, 0)
            for rl in range(28, 54, 2) for rh in range(48, 74, 2)
            if rh > rl + 5
            for am in range(14, 34, 2) for vm in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
        ]),
        "triple": (e_triple, [
            (rl, rh, am, vm, ms, 0, 0)
            for rl in range(28, 54, 3) for rh in range(48, 72, 3)
            if rh > rl + 5
            for am in range(14, 32, 3) for vm in [0.5, 0.7, 0.9, 1.1, 1.4]
            for ms in [0.00002, 0.00005, 0.0001, 0.0003]
        ]),
        "full": (e_full, [
            (rl, rh, am, vm, ms, mfl, mfh)
            for rl in range(30, 52, 3) for rh in range(50, 70, 3)
            if rh > rl + 5
            for am in range(14, 30, 3) for vm in [0.6, 0.9, 1.2]
            for ms in [0.00005, 0.0001, 0.0003]
            for mfl in [15, 25, 35] for mfh in [50, 60, 70]
        ]),
        "rsi": (e_rsi, [
            (rl, rh, am, vm, 0, skm, 0)
            for rl in range(28, 52, 3) for rh in range(48, 72, 3)
            if rh > rl + 5
            for am in range(14, 32, 3) for vm in [0.5, 0.7, 0.9, 1.1]
            for skm in [55, 65, 75, 85]
        ]),
    }

    n_exit = len(TP_VALS) * len(SL_VALS) * len(MH_VALS)
    overall_best_wr = 0
    overall_best = None

    for sname, (sfn, plist) in strategies.items():
        tc = len(plist) * n_exit
        print(f"\n{'='*55}")
        print(f"  {sname}: {len(plist)} entry x {n_exit} exit = {tc}")
        print(f"{'='*55}")

        best_wr = 0
        best_total = 0
        best_cfg = None
        count = 0
        t1 = time.time()

        for pidx, p in enumerate(plist):
            masks = {pair: sfn(d, p) for pair, d in all_data.items()}
            ne = sum(np.sum(m) for m in masks.values())
            if ne < 20:
                count += n_exit
                continue

            for ti in range(len(TP_VALS)):
                for si in range(len(SL_VALS)):
                    for mi in range(len(MH_VALS)):
                        w, l = test(all_data, masks, ti, si, mi)
                        total = w + l
                        count += 1
                        if total >= 30:
                            wr = w / total * 100
                            if wr > best_wr:
                                best_wr = wr
                                best_total = total
                                best_cfg = (p, ti, si, mi, total, w)
                                if wr >= 90:
                                    print(f"  [{count:7d}] WR={wr:.1f}% T={total:4d} W={w} | tp={TP_VALS[ti]} sl={SL_VALS[si]} mh={MH_VALS[mi]} rsi={p[0]}-{p[1]} adx>{p[2]} vol>{p[3]}")

            if (pidx + 1) % 100 == 0:
                el = time.time() - t1
                rate = count / el if el > 0 else 1
                eta = (tc - count) / rate / 60 if rate > 0 else 0
                print(f"  ... {count}/{tc} ({rate:.0f}/s, ETA {eta:.1f}min) best={best_wr:.1f}%")

        el = time.time() - t1
        print(f"\n  BEST {sname}: WR={best_wr:.1f}% T={best_total} ({el:.0f}s)")
        if best_cfg:
            p, ti, si, mi, total, wins = best_cfg
            print(f"  Params: rsi={p[0]}-{p[1]} adx>{p[2]} vol>{p[3]} tp={TP_VALS[ti]} sl={SL_VALS[si]} mh={MH_VALS[mi]}")
            if best_wr > overall_best_wr:
                overall_best_wr = best_wr
                overall_best = (sname, best_wr, best_total, best_cfg)

    print(f"\n\n{'='*70}")
    print(f"  FINAL WINNER (15min, last 1 year, {len(PAIRS)} pairs)")
    print(f"{'='*70}")
    if overall_best:
        name, wr, total, (p, ti, si, mi, _, wins) = overall_best
        tp = TP_VALS[ti]
        sl = SL_VALS[si]
        mh = MH_VALS[mi]
        losses = total - wins
        pf = (wins * tp) / (losses * abs(sl)) if losses > 0 else 999
        est_profit = wins * tp - losses * abs(sl) - total * COMMISSION * 2
        print(f"  Strategy:      {name}")
        print(f"  Win Rate:      {wr:.1f}%")
        print(f"  Total Trades:  {total}")
        print(f"  Wins/Losses:   {wins}/{losses}")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Est. Net PnL:  ~{est_profit*100:+.1f}%")
        print(f"  Take Profit:   {tp*100:.2f}%")
        print(f"  Stop Loss:     {sl*100:.1f}%")
        print(f"  Max Hold:      {mh} candles ({mh*15} min)")
        print(f"  RSI Range:     {p[0]}-{p[1]}")
        print(f"  ADX Min:       {p[2]}")
        print(f"  Vol Mult:      {p[3]}")
        if p[4] > 0:
            print(f"  EMA Slope:     >{p[4]}")
        if name == "full" and p[5] > 0:
            print(f"  MFI Range:     {p[5]}-{p[6]}")
        if name == "rsi" and p[5] > 0:
            print(f"  Stoch Max:     {p[5]}")
    else:
        print("  No results with 30+ trades found")
    print(f"{'='*70}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
