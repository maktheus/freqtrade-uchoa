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


class UchoaMLStrategy(IStrategy):
    """
    ML-enhanced strategy: trains a gradient boosting model on historical data
    to predict profitable entries. Falls back to indicator-based rules when
    model confidence is low.
    """

    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = {
        "0": 0.015,
        "20": 0.01,
        "40": 0.007,
        "80": 0.004,
    }

    stoploss = -0.03
    trailing_stop = True
    trailing_stop_positive = 0.008
    trailing_stop_positive_offset = 0.012
    trailing_only_offset_is_reached = True

    timeframe = "15m"
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    startup_candle_count = 250

    ml_confidence = DecimalParameter(0.55, 0.80, default=0.65, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMAs
        for p in [9, 21, 50, 100, 200]:
            dataframe[f"ema_{p}"] = ta.EMA(dataframe, timeperiod=p)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_7"] = ta.RSI(dataframe, timeperiod=7)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # Bollinger Bands
        bb = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lower"] = bb["lower"]
        dataframe["bb_mid"] = bb["mid"]
        dataframe["bb_upper"] = bb["upper"]
        dataframe["bb_pct"] = (dataframe["close"] - bb["lower"]) / (bb["upper"] - bb["lower"])

        # Volume
        dataframe["vol_ratio"] = dataframe["volume"] / dataframe["volume"].rolling(20).mean()

        # ADX
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        # MFI
        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=14)

        # CCI
        dataframe["cci"] = ta.CCI(dataframe, timeperiod=20)

        # Stochastic
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe["slowk"] = stoch["slowk"]
        dataframe["slowd"] = stoch["slowd"]

        # ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"] * 100

        # Price features
        dataframe["ema_slope_50"] = (dataframe["ema_50"] - dataframe["ema_50"].shift(4)) / dataframe["ema_50"].shift(4)
        dataframe["ema_slope_200"] = (dataframe["ema_200"] - dataframe["ema_200"].shift(4)) / dataframe["ema_200"].shift(4)
        dataframe["pct_change"] = dataframe["close"].pct_change()
        dataframe["pct_change_3"] = dataframe["close"].pct_change(3)
        dataframe["pct_change_6"] = dataframe["close"].pct_change(6)

        # Trend features
        dataframe["ema_dist_200"] = (dataframe["close"] - dataframe["ema_200"]) / dataframe["ema_200"] * 100
        dataframe["ema_dist_50"] = (dataframe["close"] - dataframe["ema_50"]) / dataframe["ema_50"] * 100

        # Candle features
        dataframe["body_pct"] = abs(dataframe["close"] - dataframe["open"]) / dataframe["open"] * 100
        dataframe["upper_wick"] = (dataframe["high"] - dataframe[["close", "open"]].max(axis=1)) / dataframe["close"] * 100
        dataframe["lower_wick"] = (dataframe[["close", "open"]].min(axis=1) - dataframe["low"]) / dataframe["close"] * 100
        dataframe["green"] = (dataframe["close"] > dataframe["open"]).astype(int)

        # ML Prediction
        dataframe["ml_signal"] = 0
        dataframe["ml_prob"] = 0.5

        try:
            features = self._get_features(dataframe)
            model_path = Path("user_data/strategies/ml_model.pkl")
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                valid = features.dropna()
                if len(valid) > 0:
                    preds = model.predict_proba(valid)[:, 1]
                    dataframe.loc[valid.index, "ml_prob"] = preds
                    dataframe.loc[valid.index, "ml_signal"] = (preds > self.ml_confidence.value).astype(int)
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")

        return dataframe

    def _get_features(self, dataframe: DataFrame) -> DataFrame:
        feature_cols = [
            "rsi", "rsi_7", "macdhist", "bb_pct", "vol_ratio", "adx", "mfi",
            "cci", "slowk", "slowd", "atr_pct", "ema_slope_50", "ema_slope_200",
            "pct_change", "pct_change_3", "pct_change_6",
            "ema_dist_200", "ema_dist_50", "body_pct", "upper_wick",
            "lower_wick", "green",
        ]
        return dataframe[feature_cols]

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Rule-based conditions (fallback and filter)
        rule_cond = (
            (dataframe["ema_50"] > dataframe["ema_200"])
            & (dataframe["ema_9"] > dataframe["ema_21"])
            & (dataframe["rsi"] > 35) & (dataframe["rsi"] < 60)
            & (dataframe["macdhist"] > 0)
            & (dataframe["adx"] > 20)
            & (dataframe["vol_ratio"] > 0.8)
            & (dataframe["green"] == 1)
            & (dataframe["ema_slope_50"] > 0)
        )

        # ML-enhanced: use ML signal when available, otherwise rule-based
        ml_available = dataframe["ml_signal"].sum() > 0

        if ml_available:
            dataframe.loc[
                rule_cond & (dataframe["ml_signal"] == 1),
                "enter_long",
            ] = 1
        else:
            dataframe.loc[rule_cond, "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["rsi"] > 75)
            | (
                (dataframe["ema_9"] < dataframe["ema_21"])
                & (dataframe["ema_9"].shift(1) >= dataframe["ema_21"].shift(1))
            ),
            "exit_long",
        ] = 1
        return dataframe

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs):
        if current_profit > 0.03:
            return -0.005
        if current_profit > 0.02:
            return -0.008
        if current_profit > 0.01:
            return -0.012
        return self.stoploss
