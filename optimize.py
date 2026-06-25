#!/usr/bin/env python3
"""Fast optimizer for UchoaStrategy."""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import talib.abstract as ta
from technical import qtpylib

DATA_DIR = Path("user_data/data/binance")
INITIAL_CAPITAL = 1000.0
MAX_OPEN_TRADES = 5
STAKE_PCT = 1.0 / MAX_OPEN_TRADES
COMMISSION = 0.001

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "DOGE/USDT", "LINK/USDT"]


def load_and_prepare(pair: str) -> pd.DataFrame:
    fn = pair.replace("/", "_") + "-5m.feather"
    df = pd.read_feather(DATA_DIR / fn).sort_values("date").reset_index(drop=True)

    df["ema_9"] = ta.EMA(df, timeperiod=9)
    df["ema_21"] = ta.EMA(df, timeperiod=21)
    df["ema_50"] = ta.EMA(df, timeperiod=50)
    df["ema_200"] = ta.EMA(df, timeperiod=200)
    df["rsi"] = ta.RSI(df, timeperiod=14)

    macd = ta.MACD(df, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd["macd"]
    df["macdsignal"] = macd["macdsignal"]
    df["macdhist"] = macd["macdhist"]

    bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=20, stds=2)
    df["bb_lower"] = bb["lower"]
    df["bb_mid"] = bb["mid"]
    df["bb_upper"] = bb["upper"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    df["volume_mean_20"] = df["volume"].rolling(window=20).mean()
    df["adx"] = ta.ADX(df, timeperiod=14)
    df["atr"] = ta.ATR(df, timeperiod=14)

    return df.dropna().reset_index(drop=True)


def fast_backtest(pair_data: dict, params: dict) -> dict:
    capital = INITIAL_CAPITAL
    trades = []
    open_trades = {}

    buy_rsi_lo = params["buy_rsi"]
    buy_rsi_hi = params["buy_rsi_upper"]
    bb_w_min = params["buy_bb_width_min"]
    sell_rsi = params["sell_rsi"]
    adx_min = params.get("adx_min", 20)
    sl = params.get("stoploss", -0.05)
    roi_0 = params.get("roi_0", 0.02)
    roi_30 = params.get("roi_30", 0.015)
    roi_60 = params.get("roi_60", 0.01)
    roi_120 = params.get("roi_120", 0.005)
    ema_trend = params.get("ema_trend", True)
    use_bb_exit = params.get("use_bb_exit", True)

    # Pre-compute signals for all pairs
    signals = {}
    for pair, df in pair_data.items():
        entry = (
            (df["ema_9"] > df["ema_21"])
            & (df["rsi"] > buy_rsi_lo)
            & (df["rsi"] < buy_rsi_hi)
            & (
                (df["macdhist"] > 0)
                | (
                    (df["macdhist"] > df["macdhist"].shift(1))
                    & (df["macdhist"].shift(1) > df["macdhist"].shift(2))
                )
            )
            & (df["volume"] > df["volume_mean_20"] * 0.8)
            & (df["bb_width"] > bb_w_min)
            & (df["close"] < df["bb_upper"] * 0.98)
            & (df["adx"] > adx_min)
            & (df["volume"] > 0)
        )
        if ema_trend:
            entry = entry & (df["close"] > df["ema_200"])

        exit_sig = (
            (df["rsi"] > sell_rsi)
            | (
                (df["ema_9"] < df["ema_21"])
                & (df["ema_9"].shift(1) >= df["ema_21"].shift(1))
            )
        )
        if use_bb_exit:
            exit_sig = exit_sig | (df["close"] > df["bb_upper"])

        signals[pair] = {
            "entry": entry.values,
            "exit": exit_sig.values,
            "close": df["close"].values,
            "low": df["low"].values,
            "dates": df["date"].values,
        }

    n_candles = len(next(iter(pair_data.values())))

    for i in range(n_candles):
        # Check exits
        pairs_to_close = []
        for pair, t in open_trades.items():
            sig = signals[pair]
            if i >= len(sig["close"]):
                continue
            price = sig["close"][i]
            low = sig["low"][i]
            entry_price = t["entry_price"]
            profit = (price - entry_price) / entry_price
            low_profit = (low - entry_price) / entry_price
            candles_open = i - t["entry_idx"]
            mins = candles_open * 5

            # Custom stoploss
            if profit > 0.03:
                eff_sl = -0.005
            elif profit > 0.02:
                eff_sl = -0.01
            elif profit > 0.01:
                eff_sl = -0.015
            else:
                eff_sl = sl

            should_exit = False
            reason = ""

            if low_profit <= eff_sl:
                should_exit = True
                reason = "stoploss"
                profit = eff_sl
            elif mins >= 120 and profit >= roi_120:
                should_exit = True
                reason = "roi"
            elif mins >= 60 and profit >= roi_60:
                should_exit = True
                reason = "roi"
            elif mins >= 30 and profit >= roi_30:
                should_exit = True
                reason = "roi"
            elif profit >= roi_0:
                should_exit = True
                reason = "roi"
            elif sig["exit"][i]:
                should_exit = True
                reason = "exit_signal"

            if should_exit:
                net = t["stake"] * profit - t["stake"] * COMMISSION * 2
                capital += t["stake"] + net
                trades.append({
                    "profit_pct": profit * 100,
                    "net": net,
                    "reason": reason,
                    "duration": mins,
                })
                pairs_to_close.append(pair)

        for p in pairs_to_close:
            del open_trades[p]

        # Check entries
        if len(open_trades) < MAX_OPEN_TRADES:
            for pair, sig in signals.items():
                if len(open_trades) >= MAX_OPEN_TRADES:
                    break
                if pair in open_trades:
                    continue
                if i >= len(sig["entry"]):
                    continue
                if sig["entry"][i]:
                    stake = capital * STAKE_PCT
                    if stake < 1:
                        continue
                    capital -= stake
                    open_trades[pair] = {
                        "entry_price": sig["close"][i],
                        "entry_idx": i,
                        "stake": stake,
                    }

    # Force close remaining
    for pair, t in open_trades.items():
        sig = signals[pair]
        price = sig["close"][-1]
        profit = (price - t["entry_price"]) / t["entry_price"]
        net = t["stake"] * profit - t["stake"] * COMMISSION * 2
        capital += t["stake"] + net
        trades.append({"profit_pct": profit * 100, "net": net, "reason": "force_exit", "duration": 0})

    if not trades:
        return {"total": 0, "win_rate": 0, "profit": 0, "pf": 0, "capital": capital}

    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["net"] > 0]
    losses = tdf[tdf["net"] <= 0]
    loss_sum = losses["net"].sum()

    return {
        "total": len(tdf),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(tdf) * 100 if len(tdf) > 0 else 0,
        "profit": tdf["net"].sum(),
        "profit_pct": (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "capital": capital,
        "avg_profit": tdf["profit_pct"].mean(),
        "avg_win": wins["profit_pct"].mean() if len(wins) > 0 else 0,
        "avg_loss": losses["profit_pct"].mean() if len(losses) > 0 else 0,
        "pf": abs(wins["net"].sum() / loss_sum) if loss_sum != 0 else 999,
        "avg_dur": tdf["duration"].mean(),
        "reasons": tdf["reason"].value_counts().to_dict(),
    }


def main():
    print("Loading data...")
    pair_data = {}
    for pair in PAIRS:
        pair_data[pair] = load_and_prepare(pair)
    print(f"Loaded {len(pair_data)} pairs")

    best_score = 0
    best_params = None
    best_res = None
    iteration = 0

    # Phase 1: Grid search on key params
    print("\n--- Phase 1: Grid Search ---")
    for buy_rsi in [20, 25, 30, 35, 40]:
        for buy_rsi_upper in [50, 55, 60, 65, 70, 75]:
            for bb_w in [0.005, 0.01, 0.015, 0.02]:
                for sell_rsi in [68, 72, 76, 80, 85]:
                    for adx in [15, 20, 25]:
                        for sl in [-0.03, -0.04, -0.05]:
                            params = {
                                "buy_rsi": buy_rsi,
                                "buy_rsi_upper": buy_rsi_upper,
                                "buy_bb_width_min": bb_w,
                                "sell_rsi": sell_rsi,
                                "adx_min": adx,
                                "stoploss": sl,
                                "roi_0": 0.02,
                                "roi_30": 0.015,
                                "roi_60": 0.01,
                                "roi_120": 0.005,
                                "ema_trend": True,
                                "use_bb_exit": True,
                            }
                            res = fast_backtest(pair_data, params)
                            iteration += 1

                            if res["total"] >= 5:
                                wr = res["win_rate"]
                                pf = min(res["pf"], 10)
                                score = wr * 0.6 + pf * 3 + res["profit_pct"] * 0.1
                                if score > best_score:
                                    best_score = score
                                    best_params = params.copy()
                                    best_res = res
                                    print(f"  [{iteration}] WR={wr:.1f}% PF={pf:.2f} Trades={res['total']} Profit={res['profit_pct']:+.1f}% Score={score:.1f}")

    print(f"\nPhase 1 done: {iteration} combinations tested")

    # Phase 2: Fine-tune ROI around best params
    print("\n--- Phase 2: ROI Tuning ---")
    bp = best_params.copy()
    for roi0 in [0.01, 0.015, 0.02, 0.025, 0.03]:
        for roi30 in [0.008, 0.01, 0.012, 0.015]:
            for roi60 in [0.005, 0.008, 0.01]:
                for roi120 in [0.003, 0.005, 0.007]:
                    bp2 = bp.copy()
                    bp2["roi_0"] = roi0
                    bp2["roi_30"] = roi30
                    bp2["roi_60"] = roi60
                    bp2["roi_120"] = roi120
                    res = fast_backtest(pair_data, bp2)
                    iteration += 1
                    if res["total"] >= 5:
                        wr = res["win_rate"]
                        pf = min(res["pf"], 10)
                        score = wr * 0.6 + pf * 3 + res["profit_pct"] * 0.1
                        if score > best_score:
                            best_score = score
                            best_params = bp2.copy()
                            best_res = res
                            print(f"  [{iteration}] WR={wr:.1f}% PF={pf:.2f} Trades={res['total']} Profit={res['profit_pct']:+.1f}% ROI=[{roi0},{roi30},{roi60},{roi120}]")

    # Final results
    print("\n" + "=" * 70)
    print("  FINAL OPTIMIZED RESULTS")
    print("=" * 70)
    print(f"  Best Params: {best_params}")
    print(f"  Total Trades:  {best_res['total']}")
    print(f"  Wins/Losses:   {best_res['wins']}/{best_res['losses']}")
    print(f"  Win Rate:      {best_res['win_rate']:.1f}%")
    print(f"  Profit Factor: {best_res['pf']:.2f}")
    print(f"  Total Profit:  {best_res['profit_pct']:+.2f}%")
    print(f"  Final Capital: ${best_res['capital']:.2f}")
    print(f"  Avg Win:       {best_res['avg_win']:.2f}%")
    print(f"  Avg Loss:      {best_res['avg_loss']:.2f}%")
    print(f"  Avg Duration:  {best_res['avg_dur']:.0f} min")
    print(f"  Exit Reasons:  {best_res['reasons']}")
    print(f"  Iterations:    {iteration}")
    print("=" * 70)


if __name__ == "__main__":
    main()
