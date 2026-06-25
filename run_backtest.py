#!/usr/bin/env python3
"""Standalone backtest runner that doesn't need exchange API connection."""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import talib.abstract as ta
from technical import qtpylib

DATA_DIR = Path("user_data/data/binance")
INITIAL_CAPITAL = 1000.0
MAX_OPEN_TRADES = 5
STAKE_AMOUNT_PCT = 1.0 / MAX_OPEN_TRADES
COMMISSION = 0.001  # 0.1% per trade (Binance spot)


def load_data(pair: str) -> pd.DataFrame:
    filename = pair.replace("/", "_") + "-5m.feather"
    filepath = DATA_DIR / filename
    df = pd.read_feather(filepath)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema_9"] = ta.EMA(df, timeperiod=9)
    df["ema_21"] = ta.EMA(df, timeperiod=21)
    df["ema_50"] = ta.EMA(df, timeperiod=50)
    df["ema_200"] = ta.EMA(df, timeperiod=200)
    df["rsi"] = ta.RSI(df, timeperiod=14)
    df["rsi_fast"] = ta.RSI(df, timeperiod=7)

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

    stoch_rsi = ta.STOCHRSI(df, timeperiod=14, fastk_period=3, fastd_period=3)
    df["stochrsi_k"] = stoch_rsi["fastk"]
    df["stochrsi_d"] = stoch_rsi["fastd"]

    df["atr"] = ta.ATR(df, timeperiod=14)
    return df


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    buy_rsi = params.get("buy_rsi", 30)
    buy_rsi_upper = params.get("buy_rsi_upper", 65)
    buy_bb_width_min = params.get("buy_bb_width_min", 0.02)
    sell_rsi = params.get("sell_rsi", 75)

    # Entry conditions
    cond_trend = df["close"] > df["ema_200"]
    cond_ema = df["ema_9"] > df["ema_21"]
    cond_rsi = (df["rsi"] > buy_rsi) & (df["rsi"] < buy_rsi_upper)
    cond_macd = (df["macdhist"] > 0) | (
        (df["macdhist"] > df["macdhist"].shift(1))
        & (df["macdhist"].shift(1) > df["macdhist"].shift(2))
    )
    cond_vol = df["volume"] > df["volume_mean_20"] * 0.8
    cond_bb_width = df["bb_width"] > buy_bb_width_min
    cond_bb_price = df["close"] < df["bb_upper"] * 0.98
    cond_adx = df["adx"] > 20
    cond_volume_pos = df["volume"] > 0

    df["enter_long"] = (
        cond_trend & cond_ema & cond_rsi & cond_macd & cond_vol
        & cond_bb_width & cond_bb_price & cond_adx & cond_volume_pos
    ).astype(int)

    # Exit conditions
    df["exit_long"] = (
        (df["rsi"] > sell_rsi)
        | (
            (df["ema_9"] < df["ema_21"])
            & (df["ema_9"].shift(1) >= df["ema_21"].shift(1))
        )
        | (df["close"] > df["bb_upper"])
    ).astype(int)

    return df


def run_backtest(pairs: list, params: dict) -> dict:
    capital = INITIAL_CAPITAL
    trades = []
    open_trades = []

    # Load and prepare all pair data
    pair_data = {}
    for pair in pairs:
        try:
            df = load_data(pair)
            df = compute_indicators(df)
            df = generate_signals(df, params)
            df = df.dropna().reset_index(drop=True)
            pair_data[pair] = df
        except Exception as e:
            print(f"  Skipping {pair}: {e}")

    if not pair_data:
        return {"error": "No data loaded"}

    # Get unified timeline
    all_dates = set()
    for df in pair_data.values():
        all_dates.update(df["date"].tolist())
    timeline = sorted(all_dates)

    # ROI table
    roi_table = {0: 0.02, 30: 0.015, 60: 0.01, 120: 0.005}
    stoploss = -0.05

    for date in timeline:
        # Check exits for open trades
        trades_to_close = []
        for i, t in enumerate(open_trades):
            pair = t["pair"]
            df = pair_data[pair]
            row = df[df["date"] == date]
            if row.empty:
                continue
            row = row.iloc[0]

            current_price = row["close"]
            entry_price = t["entry_price"]
            profit_pct = (current_price - entry_price) / entry_price
            minutes_open = (date - t["entry_date"]).total_seconds() / 60

            # Check stoploss
            should_exit = False
            exit_reason = ""

            # Custom stoploss (trailing)
            if profit_pct > 0.03:
                effective_sl = -0.005
            elif profit_pct > 0.02:
                effective_sl = -0.01
            elif profit_pct > 0.01:
                effective_sl = -0.015
            else:
                effective_sl = stoploss

            low_profit = (row["low"] - entry_price) / entry_price
            if low_profit <= effective_sl:
                should_exit = True
                exit_reason = "stoploss"
                profit_pct = effective_sl

            # Check ROI
            if not should_exit:
                for mins, roi_pct in sorted(roi_table.items()):
                    if minutes_open >= mins and profit_pct >= roi_pct:
                        should_exit = True
                        exit_reason = f"roi"
                        break

            # Check exit signal
            if not should_exit and row.get("exit_long", 0) == 1:
                should_exit = True
                exit_reason = "exit_signal"

            if should_exit:
                profit_abs = t["stake"] * profit_pct
                commission_cost = t["stake"] * COMMISSION * 2
                net_profit = profit_abs - commission_cost
                capital += t["stake"] + net_profit
                trades.append({
                    "pair": pair,
                    "entry_date": t["entry_date"],
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "profit_pct": profit_pct * 100,
                    "net_profit": net_profit,
                    "exit_reason": exit_reason,
                    "duration_min": minutes_open,
                })
                trades_to_close.append(i)

        for i in sorted(trades_to_close, reverse=True):
            open_trades.pop(i)

        # Check entries
        if len(open_trades) < MAX_OPEN_TRADES:
            for pair, df in pair_data.items():
                if len(open_trades) >= MAX_OPEN_TRADES:
                    break
                if any(t["pair"] == pair for t in open_trades):
                    continue

                row = df[df["date"] == date]
                if row.empty:
                    continue
                row = row.iloc[0]

                if row.get("enter_long", 0) == 1:
                    stake = capital * STAKE_AMOUNT_PCT
                    if stake < 1:
                        continue
                    capital -= stake
                    open_trades.append({
                        "pair": pair,
                        "entry_date": date,
                        "entry_price": row["close"],
                        "stake": stake,
                    })

    # Close remaining open trades at last price
    for t in open_trades:
        pair = t["pair"]
        df = pair_data[pair]
        last_row = df.iloc[-1]
        current_price = last_row["close"]
        entry_price = t["entry_price"]
        profit_pct = (current_price - entry_price) / entry_price
        profit_abs = t["stake"] * profit_pct
        commission_cost = t["stake"] * COMMISSION * 2
        net_profit = profit_abs - commission_cost
        capital += t["stake"] + net_profit
        minutes_open = (last_row["date"] - t["entry_date"]).total_seconds() / 60
        trades.append({
            "pair": pair,
            "entry_date": t["entry_date"],
            "exit_date": last_row["date"],
            "entry_price": entry_price,
            "exit_price": current_price,
            "profit_pct": profit_pct * 100,
            "net_profit": net_profit,
            "exit_reason": "force_exit",
            "duration_min": minutes_open,
        })

    # Results
    if not trades:
        return {"total_trades": 0, "error": "No trades executed"}

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["net_profit"] > 0]
    losses = trades_df[trades_df["net_profit"] <= 0]

    results = {
        "total_trades": len(trades_df),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades_df) * 100,
        "total_profit": trades_df["net_profit"].sum(),
        "total_profit_pct": (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        "final_capital": capital,
        "avg_profit_pct": trades_df["profit_pct"].mean(),
        "avg_win_pct": wins["profit_pct"].mean() if len(wins) > 0 else 0,
        "avg_loss_pct": losses["profit_pct"].mean() if len(losses) > 0 else 0,
        "max_drawdown_trade": trades_df["profit_pct"].min(),
        "avg_duration_min": trades_df["duration_min"].mean(),
        "profit_factor": abs(wins["net_profit"].sum() / losses["net_profit"].sum()) if len(losses) > 0 and losses["net_profit"].sum() != 0 else float("inf"),
        "exit_reasons": trades_df["exit_reason"].value_counts().to_dict(),
        "trades_df": trades_df,
    }
    return results


def print_results(results: dict, params: dict):
    print("\n" + "=" * 70)
    print(f"  BACKTEST RESULTS")
    print("=" * 70)
    print(f"  Params: {params}")
    print(f"  Initial Capital:  ${INITIAL_CAPITAL:.2f}")
    print(f"  Final Capital:    ${results['final_capital']:.2f}")
    print(f"  Total Profit:     ${results['total_profit']:.2f} ({results['total_profit_pct']:+.2f}%)")
    print("-" * 70)
    print(f"  Total Trades:     {results['total_trades']}")
    print(f"  Wins:             {results['wins']}")
    print(f"  Losses:           {results['losses']}")
    print(f"  Win Rate:         {results['win_rate']:.1f}%")
    print(f"  Profit Factor:    {results['profit_factor']:.2f}")
    print("-" * 70)
    print(f"  Avg Profit/Trade: {results['avg_profit_pct']:.2f}%")
    print(f"  Avg Win:          {results['avg_win_pct']:.2f}%")
    print(f"  Avg Loss:         {results['avg_loss_pct']:.2f}%")
    print(f"  Max DD (trade):   {results['max_drawdown_trade']:.2f}%")
    print(f"  Avg Duration:     {results['avg_duration_min']:.0f} min")
    print("-" * 70)
    print(f"  Exit Reasons:     {results['exit_reasons']}")
    print("=" * 70)


if __name__ == "__main__":
    pairs = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
        "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "UNI/USDT",
        "ATOM/USDT", "LTC/USDT", "FIL/USDT",
    ]

    # V1: Base strategy
    params_v1 = {
        "buy_rsi": 30, "buy_rsi_upper": 65, "buy_bb_width_min": 0.02, "sell_rsi": 75,
    }

    print("Running backtest V1...")
    results = run_backtest(pairs, params_v1)
    if "error" not in results or results.get("total_trades", 0) > 0:
        print_results(results, params_v1)
    else:
        print(f"Error: {results}")
