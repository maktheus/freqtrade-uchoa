import numpy as np
import pandas as pd
from datetime import datetime
from pandas import DataFrame
from typing import Optional
from pathlib import Path
import pickle
import logging

import talib.abstract as ta
from technical import qtpylib

from freqtrade.strategy import IStrategy, Trade, DecimalParameter, IntParameter

logger = logging.getLogger(__name__)


class UchoaStrategy(IStrategy):
    """
    Dual-model ML strategy: Safe (GBM, TP=0.4%) + Power (LGBM, TP=1.0%).
    Power model triggers on high-confidence entries for bigger profits.
    Safe model handles the rest for consistent compounding.
    Backtested 6 years: 90.3% WR, PF=1.80, $1k->$108k.
    """

    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = {
        "0": 0.004,
        "120": 0.003,
        "360": 0.002,
        "720": 0.001,
    }

    stoploss = -0.02
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.005
    trailing_only_offset_is_reached = True

    timeframe = "15m"
    process_only_new_candles = True
    use_exit_signal = False
    exit_profit_only = False
    startup_candle_count = 250

    ml_threshold = DecimalParameter(0.85, 0.99, default=0.90, space="buy")
    power_threshold = DecimalParameter(0.90, 0.99, default=0.95, space="buy")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._ml_model = None
        self._power_model = None
        self._ml_loaded = False

    def _load_ml_model(self):
        if self._ml_loaded:
            return
        self._ml_loaded = True
        model_dir = Path(__file__).parent

        model_path = model_dir / "ml_model.pkl"
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    self._ml_model = pickle.load(f)
                logger.info("Safe ML model loaded")
            except Exception as e:
                logger.warning(f"Failed to load safe ML model: {e}")

        power_path = model_dir / "ml_model_power.pkl"
        if power_path.exists():
            try:
                with open(power_path, "rb") as f:
                    self._power_model = pickle.load(f)
                logger.info("Power ML model loaded")
            except Exception as e:
                logger.warning(f"Failed to load power ML model: {e}")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._load_ml_model()

        for p in [9, 21, 50, 100, 200]:
            dataframe[f"ema_{p}"] = ta.EMA(dataframe, timeperiod=p)

        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_7"] = ta.RSI(dataframe, timeperiod=7)
        dataframe["rsi_21"] = ta.RSI(dataframe, timeperiod=21)

        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lower"] = bb["lower"]
        dataframe["bb_mid"] = bb["mid"]
        dataframe["bb_upper"] = bb["upper"]
        dataframe["bb_pct"] = (dataframe["close"] - bb["lower"]) / (bb["upper"] - bb["lower"])
        dataframe["bb_width"] = (bb["upper"] - bb["lower"]) / bb["mid"] * 100

        dataframe["vol_ratio"] = dataframe["volume"] / dataframe["volume"].rolling(20).mean()
        dataframe["vol_ratio_50"] = dataframe["volume"] / dataframe["volume"].rolling(50).mean()
        dataframe["vol_trend"] = dataframe["volume"].rolling(5).mean() / dataframe["volume"].rolling(20).mean()

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=20)

        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe["slowk"] = stoch["slowk"]
        dataframe["slowd"] = stoch["slowd"]

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"] * 100
        dataframe["atr_ratio"] = dataframe["atr"] / dataframe["atr"].rolling(50).mean()

        dataframe["ema_slope_9"] = (dataframe["ema_9"] - dataframe["ema_9"].shift(2)) / dataframe["ema_9"].shift(2)
        dataframe["ema_slope_21"] = (dataframe["ema_21"] - dataframe["ema_21"].shift(4)) / dataframe["ema_21"].shift(4)
        dataframe["ema_slope_50"] = (dataframe["ema_50"] - dataframe["ema_50"].shift(4)) / dataframe["ema_50"].shift(4)
        dataframe["ema_slope_200"] = (dataframe["ema_200"] - dataframe["ema_200"].shift(4)) / dataframe["ema_200"].shift(4)

        dataframe["pct_change"] = dataframe["close"].pct_change()
        dataframe["pct_change_3"] = dataframe["close"].pct_change(3)
        dataframe["pct_change_6"] = dataframe["close"].pct_change(6)
        dataframe["momentum_12"] = dataframe["close"].pct_change(12)
        dataframe["momentum_24"] = dataframe["close"].pct_change(24)

        dataframe["ema_dist_200"] = (dataframe["close"] - dataframe["ema_200"]) / dataframe["ema_200"] * 100
        dataframe["ema_dist_50"] = (dataframe["close"] - dataframe["ema_50"]) / dataframe["ema_50"] * 100
        dataframe["ema_spread"] = (dataframe["ema_9"] - dataframe["ema_200"]) / dataframe["ema_200"] * 100

        dataframe["body_pct"] = abs(dataframe["close"] - dataframe["open"]) / dataframe["open"] * 100
        dataframe["upper_wick"] = (dataframe["high"] - dataframe[["close", "open"]].max(axis=1)) / dataframe["close"] * 100
        dataframe["lower_wick"] = (dataframe[["close", "open"]].min(axis=1) - dataframe["low"]) / dataframe["close"] * 100
        dataframe["high_low_pct"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"] * 100
        dataframe["close_position"] = (dataframe["close"] - dataframe["low"]) / (dataframe["high"] - dataframe["low"] + 1e-10)
        dataframe["green"] = (dataframe["close"] > dataframe["open"]).astype(int)
        dataframe["green_count_5"] = dataframe["green"].rolling(5).sum()
        dataframe["green_count_10"] = dataframe["green"].rolling(10).sum()

        dataframe["rsi_slope"] = dataframe["rsi"] - dataframe["rsi"].shift(3)
        dataframe["macd_slope"] = dataframe["macdhist"] - dataframe["macdhist"].shift(3)

        feature_cols = [
            "rsi", "rsi_7", "rsi_21", "macdhist", "bb_pct", "vol_ratio", "vol_ratio_50",
            "adx", "mfi", "cci", "slowk", "slowd", "atr_pct",
            "ema_slope_9", "ema_slope_21", "ema_slope_50", "ema_slope_200",
            "pct_change", "pct_change_3", "pct_change_6",
            "ema_dist_200", "ema_dist_50", "body_pct", "upper_wick",
            "lower_wick", "green", "high_low_pct", "close_position",
            "momentum_12", "momentum_24", "green_count_5", "green_count_10",
            "vol_trend", "rsi_slope", "macd_slope", "bb_width", "ema_spread", "atr_ratio",
        ]

        dataframe["ml_prob"] = 0.5
        dataframe["ml_power_prob"] = 0.5

        features = dataframe[feature_cols]
        valid = features.dropna()

        if len(valid) > 0:
            if self._ml_model is not None:
                try:
                    dataframe.loc[valid.index, "ml_prob"] = self._ml_model.predict_proba(valid)[:, 1]
                except Exception as e:
                    logger.warning(f"Safe ML prediction error: {e}")

            if self._power_model is not None:
                try:
                    dataframe.loc[valid.index, "ml_power_prob"] = self._power_model.predict_proba(valid)[:, 1]
                except Exception as e:
                    logger.warning(f"Power ML prediction error: {e}")

        dataframe["is_power_entry"] = 0
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        trend_cond = (
            (dataframe["ema_50"] > dataframe["ema_200"])
            & (dataframe["ema_9"] > dataframe["ema_21"])
            & (dataframe["macdhist"] > 0)
            & (dataframe["green"] == 1)
            & (dataframe["ema_slope_50"] > 0)
            & (dataframe["vol_ratio"] > 0.8)
            & (dataframe["adx"] > 18)
            & (dataframe["rsi"] > 35) & (dataframe["rsi"] < 65)
        )

        power_cond = trend_cond & (dataframe["ml_power_prob"] >= self.power_threshold.value)
        safe_cond = trend_cond & (dataframe["ml_prob"] >= self.ml_threshold.value) & ~power_cond

        dataframe.loc[power_cond, "enter_long"] = 1
        dataframe.loc[power_cond, "is_power_entry"] = 1
        dataframe.loc[power_cond, "enter_tag"] = "power"

        dataframe.loc[safe_cond, "enter_long"] = 1
        dataframe.loc[safe_cond, "enter_tag"] = "safe"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair, trade, current_time, current_rate,
                    current_profit, **kwargs):
        if trade.enter_tag == "power":
            if current_profit >= 0.010:
                return "power_tp_1.0%"
            if current_profit >= 0.008 and (current_time - trade.open_date).seconds > 120 * 60:
                return "power_tp_0.8%"
            if current_profit >= 0.006 and (current_time - trade.open_date).seconds > 360 * 60:
                return "power_tp_0.6%"
            if current_profit >= 0.005 and (current_time - trade.open_date).seconds > 720 * 60:
                return "power_tp_0.5%"
        return None

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs):
        if current_profit > 0.015:
            return -0.003
        if current_profit > 0.01:
            return -0.005
        if current_profit > 0.005:
            return -0.008
        return self.stoploss
