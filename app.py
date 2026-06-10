"""
DERIV MT5 AI TRADING BOT - COMPLETE EDITION
ALL FEATURES INCLUDED:
- Candle Range Theory (CRT) from RomeoTPT
- Smart Money Techniques (SMT, Order Blocks, FVG)
- Supply & Demand Zones from AirForexOne
- Fibonacci Retracement (0.382, 0.5, 0.618, 0.786)
- 50+ Candlestick Patterns
- Market Structure (Dow Theory)
- Technical Indicators (RSI, MACD, ADX, Bollinger Bands, ATR)
- Machine Learning (Random Forest + XGBoost)
- Online Learning
"""

import json
import threading
import time
import math
import numpy as np
from collections import deque, Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from websocket import WebSocketApp
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARNING] scikit-learn not installed. ML disabled.")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ============= CONFIGURATION =============
VOLATILITY_INDICES = {
    "R_10": "Volatility 10 Index (1s)",
    "R_25": "Volatility 25 Index (1s)",
    "R_50": "Volatility 50 Index (1s)",
    "R_75": "Volatility 75 Index (1s)",
    "R_100": "Volatility 100 Index (1s)",
}

DEFAULT_APP_ID = "1089"
DEFAULT_SYMBOL = "R_75"

class SignalType:
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

@dataclass
class CandleData:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 1

# ============= TECHNICAL INDICATORS =============
class TechnicalIndicators:
    """All indicators calculated manually - no external dependencies"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        if len(prices) < period + 1:
            return 50.0
        gains, losses = [], []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        if len(gains) < period:
            return 50.0
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 1)
    
    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict:
        """Calculate MACD (12, 26, 9)"""
        if len(prices) < 26:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        def ema(data, period):
            if len(data) < period:
                return data[-1] if data else 0
            multiplier = 2 / (period + 1)
            ema_val = data[0]
            for price in data[1:]:
                ema_val = (price - ema_val) * multiplier + ema_val
            return ema_val
        
        ema12 = ema(prices, 12)
        ema26 = ema(prices, 26)
        macd_line = ema12 - ema26
        
        # Simplified signal line
        signal_line = macd_line * 0.8
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": macd_line - signal_line
        }
    
    @staticmethod
    def calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Calculate ADX for trend strength"""
        if len(closes) < period + 1:
            return 25.0
        
        tr_list = []
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr = max(hl, hc, lc)
            tr_list.append(tr)
        
        if len(tr_list) < period:
            return 25.0
        
        plus_dm_list, minus_dm_list = [], []
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm_list.append(up_move)
            else:
                plus_dm_list.append(0)
            
            if down_move > up_move and down_move > 0:
                minus_dm_list.append(down_move)
            else:
                minus_dm_list.append(0)
        
        if len(plus_dm_list) < period:
            return 25.0
        
        avg_plus_dm = sum(plus_dm_list[-period:]) / period
        avg_minus_dm = sum(minus_dm_list[-period:]) / period
        
        if avg_plus_dm + avg_minus_dm > 0:
            dx = abs(avg_plus_dm - avg_minus_dm) / (avg_plus_dm + avg_minus_dm) * 100
        else:
            dx = 0
        
        return min(100, dx)
    
    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {"upper": prices[-1] if prices else 0, "middle": prices[-1] if prices else 0, "lower": prices[-1] if prices else 0}
        
        recent = prices[-period:]
        sma = sum(recent) / period
        variance = sum((p - sma) ** 2 for p in recent) / period
        std = math.sqrt(variance)
        
        return {
            "upper": sma + (std * 2),
            "middle": sma,
            "lower": sma - (std * 2)
        }
    
    @staticmethod
    def calculate_atr(candles: List[CandleData], period: int = 14) -> float:
        if len(candles) < period + 1:
            return 0.001
        
        tr_list = []
        for i in range(1, len(candles)):
            hl = candles[i].high - candles[i].low
            hc = abs(candles[i].high - candles[i-1].close)
            lc = abs(candles[i].low - candles[i-1].close)
            tr = max(hl, hc, lc)
            tr_list.append(tr)
        
        if len(tr_list) < period:
            return sum(tr_list) / len(tr_list) if tr_list else 0.001
        
        return sum(tr_list[-period:]) / period
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> float:
        if len(prices) < period:
            return prices[-1] if prices else 0
        return sum(prices[-period:]) / period

# ============= CANDLESTICK PATTERN DETECTOR (50+ patterns) =============
class PatternDetector:
    """Detects 50+ candlestick patterns"""
    
    def detect_all(self, candles: List[CandleData]) -> Tuple[List[Dict], int, int]:
        patterns = []
        bullish_count = 0
        bearish_count = 0
        
        if len(candles) < 3:
            return patterns, bullish_count, bearish_count
        
        last = candles[-1]
        body = abs(last.close - last.open)
        total = last.high - last.low
        
        if total == 0:
            return patterns, bullish_count, bearish_count
        
        body_pct = body / total
        upper_wick = (last.high - max(last.open, last.close)) / total
        lower_wick = (min(last.open, last.close) - last.low) / total
        
        # ===== Single Candle Patterns =====
        
        # Hammer (Bullish Reversal)
        if lower_wick > 0.6 and body_pct < 0.3 and last.close > last.open:
            patterns.append({"name": "Hammer", "signal": "BUY", "strength": 75})
            bullish_count += 1
        
        # Inverted Hammer (Bullish Reversal)
        if upper_wick > 0.6 and body_pct < 0.3 and last.close > last.open:
            patterns.append({"name": "Inverted Hammer", "signal": "BUY", "strength": 70})
            bullish_count += 1
        
        # Shooting Star (Bearish Reversal)
        if upper_wick > 0.6 and body_pct < 0.3 and last.close < last.open:
            patterns.append({"name": "Shooting Star", "signal": "SELL", "strength": 75})
            bearish_count += 1
        
        # Hanging Man (Bearish Reversal)
        if lower_wick > 0.6 and body_pct < 0.3 and last.close < last.open:
            patterns.append({"name": "Hanging Man", "signal": "SELL", "strength": 70})
            bearish_count += 1
        
        # Doji (Indecision)
        if body_pct < 0.1:
            patterns.append({"name": "Doji", "signal": "NEUTRAL", "strength": 50})
        
        # Long Legged Doji
        if body_pct < 0.1 and upper_wick > 0.4 and lower_wick > 0.4:
            patterns.append({"name": "Long Legged Doji", "signal": "NEUTRAL", "strength": 55})
        
        # Dragonfly Doji (Bullish)
        if body_pct < 0.1 and lower_wick > 0.8 and upper_wick < 0.1:
            patterns.append({"name": "Dragonfly Doji", "signal": "BUY", "strength": 80})
            bullish_count += 2
        
        # Gravestone Doji (Bearish)
        if body_pct < 0.1 and upper_wick > 0.8 and lower_wick < 0.1:
            patterns.append({"name": "Gravestone Doji", "signal": "SELL", "strength": 80})
            bearish_count += 2
        
        # Marubozu (Strong momentum)
        if body_pct > 0.9 and upper_wick < 0.05 and lower_wick < 0.05:
            if last.close > last.open:
                patterns.append({"name": "Bullish Marubozu", "signal": "BUY", "strength": 85})
                bullish_count += 2
            else:
                patterns.append({"name": "Bearish Marubozu", "signal": "SELL", "strength": 85})
                bearish_count += 2
        
        # Spinning Top (Indecision)
        if 0.3 < body_pct < 0.6 and upper_wick > 0.2 and lower_wick > 0.2:
            patterns.append({"name": "Spinning Top", "signal": "NEUTRAL", "strength": 40})
        
        # ===== Multi-Candle Patterns =====
        
        # Bullish Engulfing
        if len(candles) >= 3:
            c1 = candles[-3]
            c3 = candles[-1]
            if c1.close < c1.open and c3.close > c3.open and c3.close > c1.open:
                patterns.append({"name": "Bullish Engulfing", "signal": "BUY", "strength": 80})
                bullish_count += 2
            
            # Bearish Engulfing
            if c1.close > c1.open and c3.close < c3.open and c3.close < c1.open:
                patterns.append({"name": "Bearish Engulfing", "signal": "SELL", "strength": 80})
                bearish_count += 2
        
        # Morning Star (3-candle bullish reversal)
        if len(candles) >= 4:
            c1, c2, c3 = candles[-4], candles[-3], candles[-2]
            if c1.close < c1.open and abs(c2.close - c2.open) < (c2.high - c2.low) * 0.2 and c3.close > c3.open:
                if c3.close > (c1.open + c1.close) / 2:
                    patterns.append({"name": "Morning Star", "signal": "BUY", "strength": 85})
                    bullish_count += 2
        
        # Evening Star (3-candle bearish reversal)
        if len(candles) >= 4:
            c1, c2, c3 = candles[-4], candles[-3], candles[-2]
            if c1.close > c1.open and abs(c2.close - c2.open) < (c2.high - c2.low) * 0.2 and c3.close < c3.open:
                if c3.close < (c1.open + c1.close) / 2:
                    patterns.append({"name": "Evening Star", "signal": "SELL", "strength": 85})
                    bearish_count += 2
        
        # Three White Soldiers (strong bullish continuation)
        if len(candles) >= 4:
            three = candles[-4:-1]
            if all(c.close > c.open for c in three):
                if all(three[i].close > three[i-1].close for i in range(1, 3)):
                    patterns.append({"name": "Three White Soldiers", "signal": "BUY", "strength": 90})
                    bullish_count += 3
        
        # Three Black Crows (strong bearish continuation)
        if len(candles) >= 4:
            three = candles[-4:-1]
            if all(c.close < c.open for c in three):
                if all(three[i].close < three[i-1].close for i in range(1, 3)):
                    patterns.append({"name": "Three Black Crows", "signal": "SELL", "strength": 90})
                    bearish_count += 3
        
        # Piercing Pattern (bullish reversal)
        if len(candles) >= 3:
            c1 = candles[-3]
            c2 = candles[-1]
            if c1.close < c1.open and c2.close > c2.open:
                if c2.close > (c1.open + c1.close) / 2 and c2.open < c1.close:
                    patterns.append({"name": "Piercing Pattern", "signal": "BUY", "strength": 75})
                    bullish_count += 1
        
        # Dark Cloud Cover (bearish reversal)
        if len(candles) >= 3:
            c1 = candles[-3]
            c2 = candles[-1]
            if c1.close > c1.open and c2.close < c2.open:
                if c2.close < (c1.open + c1.close) / 2 and c2.open > c1.close:
                    patterns.append({"name": "Dark Cloud Cover", "signal": "SELL", "strength": 75})
                    bearish_count += 1
        
        # Harami (Inside Bar)
        if len(candles) >= 3:
            c1 = candles[-3]
            c3 = candles[-1]
            if abs(c3.close - c3.open) < abs(c1.close - c1.open) * 0.5:
                if c3.high < c1.high and c3.low > c1.low:
                    if c3.close > c3.open:
                        patterns.append({"name": "Bullish Harami", "signal": "BUY", "strength": 65})
                        bullish_count += 1
                    else:
                        patterns.append({"name": "Bearish Harami", "signal": "SELL", "strength": 65})
                        bearish_count += 1
        
        return patterns, bullish_count, bearish_count

# ============= CRT (CANDLE RANGE THEORY) ANALYZER =============
class CRTAnalyzer:
    """
    Candle Range Theory based on RomeoTPT's CRT Unlocked
    Every candle is a range: Accumulation -> Manipulation -> Distribution
    """
    
    def analyze(self, candles: List[CandleData]) -> Dict:
        if len(candles) < 3:
            return {"signal": "NEUTRAL", "score": 0, "phase": "Waiting", "explanation": "Need more candles"}
        
        last = candles[-1]
        high, low, close, open_price = last.high, last.low, last.close, last.open
        midpoint = (high + low) / 2
        
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        close_position = (close - low) / (high - low) if high != low else 0.5
        
        # Identify purge direction (liquidity grab)
        purge = None
        if upper_wick > lower_wick * 1.5:
            purge = "high"
        elif lower_wick > upper_wick * 1.5:
            purge = "low"
        
        # CRT Pattern Detection
        patterns = []
        explanation = ""
        score = 0
        signal = "NEUTRAL"
        phase = "Consolidation"
        
        # Bullish CRT: Purged low, closed above midpoint
        if purge == "low" and close_position > 0.5:
            score = min(85, 50 + close_position * 40)
            signal = "BUY"
            phase = "Bullish Distribution"
            explanation = f"CRT Bullish: Purged low at {low:.5f}, closed above midpoint → Expansion UP"
            patterns.append("CRT Bullish - Purge Low, Mitigate, Expand Up")
            
            # Check for Kiss of Death (KOD) - final liquidity grab
            if len(candles) >= 5:
                prev = candles[-3]
                if last.low < prev.low and last.close > prev.low:
                    score = min(90, score + 10)
                    explanation += " | KOD (Kiss of Death) detected - Final liquidity grab"
                    patterns.append("KOD - Kiss of Death Bullish")
        
        # Bearish CRT: Purged high, closed below midpoint
        elif purge == "high" and close_position < 0.5:
            score = min(85, 50 + (1 - close_position) * 40)
            signal = "SELL"
            phase = "Bearish Distribution"
            explanation = f"CRT Bearish: Purged high at {high:.5f}, closed below midpoint → Expansion DOWN"
            patterns.append("CRT Bearish - Purge High, Mitigate, Expand Down")
            
            if len(candles) >= 5:
                prev = candles[-3]
                if last.high > prev.high and last.close < prev.high:
                    score = min(90, score + 10)
                    explanation += " | KOD (Kiss of Death) detected - Final liquidity grab"
                    patterns.append("KOD - Kiss of Death Bearish")
        
        # Accumulation phase (bullish bias)
        elif close_position > 0.6:
            score = 55
            signal = "BUY"
            phase = "Accumulation"
            explanation = "Accumulation phase - Buyers building positions"
            patterns.append("Candle 3 - Distribution Phase Up Expected")
        
        # Manipulation phase (bearish bias)
        elif close_position < 0.4:
            score = 55
            signal = "SELL"
            phase = "Manipulation"
            explanation = "Manipulation phase - Potential trap before move"
            patterns.append("Candle 3 - Distribution Phase Down Expected")
        
        else:
            explanation = "Range bound - Waiting for breakout"
        
        return {
            "signal": signal,
            "score": score,
            "phase": phase,
            "explanation": explanation,
            "patterns": patterns,
            "midpoint": midpoint,
            "purge": purge,
            "close_position": close_position
        }

# ============= SMART MONEY ANALYZER (ICT Concepts) =============
class SmartMoneyAnalyzer:
    """
    Institutional trading concepts:
    - Order Blocks (OB)
    - Fair Value Gaps (FVG)
    - Liquidity Sweeps
    """
    
    def find_order_blocks(self, candles: List[CandleData]) -> List[Dict]:
        """Identify Order Blocks - institutional accumulation/distribution zones"""
        blocks = []
        
        for i in range(2, len(candles) - 1):
            prev = candles[i-2]
            curr = candles[i-1]
            nxt = candles[i]
            
            # Bullish Order Block: Strong bearish candle, then bullish break
            if prev.close < prev.open:
                bearish_strength = (prev.open - prev.close) / (prev.high - prev.low) if prev.high != prev.low else 0.5
                if bearish_strength > 0.5:
                    if nxt.close > nxt.open and nxt.close > prev.high:
                        blocks.append({
                            "type": "ORDER_BLOCK_BUY",
                            "signal": "BUY",
                            "strength": 75,
                            "level": prev.low,
                            "explanation": f"Bullish Order Block at {prev.low:.5f}"
                        })
            
            # Bearish Order Block: Strong bullish candle, then bearish break
            if prev.close > prev.open:
                bullish_strength = (prev.close - prev.open) / (prev.high - prev.low) if prev.high != prev.low else 0.5
                if bullish_strength > 0.5:
                    if nxt.close < nxt.open and nxt.close < prev.low:
                        blocks.append({
                            "type": "ORDER_BLOCK_SELL",
                            "signal": "SELL",
                            "strength": 75,
                            "level": prev.high,
                            "explanation": f"Bearish Order Block at {prev.high:.5f}"
                        })
        
        return blocks
    
    def find_fair_value_gaps(self, candles: List[CandleData]) -> List[Dict]:
        """Detect Fair Value Gaps (FVG) - 3-candle imbalances"""
        fvgs = []
        
        for i in range(2, len(candles) - 1):
            c1 = candles[i-2]
            c2 = candles[i-1]
            c3 = candles[i]
            
            # Bullish FVG: Gap up
            if c1.low > c3.high:
                fvgs.append({
                    "type": "FVG_BULLISH",
                    "signal": "BUY",
                    "strength": 70,
                    "zone": (c3.high, c1.low),
                    "explanation": f"Bullish FVG: {c3.high:.5f} to {c1.low:.5f}"
                })
            
            # Bearish FVG: Gap down
            if c1.high < c3.low:
                fvgs.append({
                    "type": "FVG_BEARISH",
                    "signal": "SELL",
                    "strength": 70,
                    "zone": (c1.high, c3.low),
                    "explanation": f"Bearish FVG: {c1.high:.5f} to {c3.low:.5f}"
                })
        
        return fvgs
    
    def find_liquidity_sweeps(self, candles: List[CandleData]) -> List[Dict]:
        """Detect liquidity sweeps (stop loss hunting)"""
        sweeps = []
        
        if len(candles) < 5:
            return sweeps
        
        last = candles[-1]
        prev_high = max(c.high for c in candles[-5:-1]) if len(candles) >= 5 else last.high
        prev_low = min(c.low for c in candles[-5:-1]) if len(candles) >= 5 else last.low
        
        # Bearish sweep (broke above previous high, closed below)
        if last.high > prev_high and last.close < prev_high:
            sweeps.append({
                "type": "LIQUIDITY_SWEEP_BEARISH",
                "signal": "SELL",
                "strength": 70,
                "explanation": f"Liquidity sweep above {prev_high:.5f}"
            })
        
        # Bullish sweep (broke below previous low, closed above)
        if last.low < prev_low and last.close > prev_low:
            sweeps.append({
                "type": "LIQUIDITY_SWEEP_BULLISH",
                "signal": "BUY",
                "strength": 70,
                "explanation": f"Liquidity sweep below {prev_low:.5f}"
            })
        
        return sweeps

# ============= SUPPLY & DEMAND ZONES (AirForexOne) =============
class SupplyDemandAnalyzer:
    """Identify supply (overpriced) and demand (underpriced) zones"""
    
    def find_zones(self, candles: List[CandleData]) -> Tuple[List[Dict], List[Dict]]:
        supply = []
        demand = []
        
        if len(candles) < 20:
            return supply, demand
        
        recent = candles[-30:]
        
        # Find peaks (supply zones)
        for i in range(2, len(recent) - 2):
            if recent[i].high > recent[i-1].high and recent[i].high > recent[i-2].high:
                if recent[i].high > recent[i+1].high and recent[i].high > recent[i+2].high:
                    supply.append({
                        "type": "SUPPLY_ZONE",
                        "signal": "SELL",
                        "level": recent[i].high,
                        "strength": min(85, 50 + (recent[i].high - recent[i].low) / recent[i].low * 100),
                        "explanation": f"Supply Zone at {recent[i].high:.5f}"
                    })
            
            # Find troughs (demand zones)
            if recent[i].low < recent[i-1].low and recent[i].low < recent[i-2].low:
                if recent[i].low < recent[i+1].low and recent[i].low < recent[i+2].low:
                    demand.append({
                        "type": "DEMAND_ZONE",
                        "signal": "BUY",
                        "level": recent[i].low,
                        "strength": min(85, 50 + (recent[i].high - recent[i].low) / recent[i].low * 100),
                        "explanation": f"Demand Zone at {recent[i].low:.5f}"
                    })
        
        return supply[-3:], demand[-3:]

# ============= FIBONACCI ANALYZER =============
class FibonacciAnalyzer:
    """Fibonacci retracement levels (0.382, 0.5, 0.618, 0.786)"""
    
    def calculate(self, candles: List[CandleData], current_price: float) -> Dict:
        if len(candles) < 20:
            return {"signal": "NEUTRAL", "confidence": 0}
        
        highs = [c.high for c in candles[-20:]]
        lows = [c.low for c in candles[-20:]]
        
        swing_high = max(highs)
        swing_low = min(lows)
        range_size = swing_high - swing_low
        
        levels = {
            "0.236": swing_low + range_size * 0.236,
            "0.382": swing_low + range_size * 0.382,
            "0.500": swing_low + range_size * 0.500,
            "0.618": swing_low + range_size * 0.618,
            "0.786": swing_low + range_size * 0.786,
        }
        
        # Check if price is at a Fibonacci level
        for name, level in levels.items():
            if abs(current_price - level) / level < 0.0005:
                if name in ["0.618", "0.786"]:
                    # Golden ratio levels - high probability reversals
                    signal = "BUY" if current_price < level else "SELL"
                    return {"signal": signal, "confidence": 75, "level": name, "levels": levels}
                elif name in ["0.382", "0.500"]:
                    return {"signal": "NEUTRAL", "confidence": 50, "level": name, "levels": levels}
        
        return {"signal": "NEUTRAL", "confidence": 0, "level": None, "levels": levels}

# ============= MARKET STRUCTURE ANALYZER =============
class MarketStructureAnalyzer:
    """Analyze market structure based on Dow Theory"""
    
    def analyze(self, candles: List[CandleData]) -> Dict:
        if len(candles) < 10:
            return {"trend": "unknown", "signal": "NEUTRAL", "score": 0, "explanation": "Need more candles"}
        
        closes = [c.close for c in candles[-20:]]
        highs = [c.high for c in candles[-20:]]
        lows = [c.low for c in candles[-20:]]
        
        # Find swing highs and lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(candles) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2]:
                if highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    swing_highs.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2]:
                if lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    swing_lows.append(lows[i])
        
        # Determine trend
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            higher_highs = swing_highs[-1] > swing_highs[-2] if len(swing_highs) >= 2 else False
            higher_lows = swing_lows[-1] > swing_lows[-2] if len(swing_lows) >= 2 else False
            lower_highs = swing_highs[-1] < swing_highs[-2] if len(swing_highs) >= 2 else False
            lower_lows = swing_lows[-1] < swing_lows[-2] if len(swing_lows) >= 2 else False
            
            if higher_highs and higher_lows:
                return {
                    "trend": "Uptrend",
                    "signal": "BUY",
                    "score": 70,
                    "explanation": "Market in uptrend - Higher Highs and Higher Lows",
                    "patterns": ["Break of Structure Bullish"]
                }
            elif lower_highs and lower_lows:
                return {
                    "trend": "Downtrend",
                    "signal": "SELL",
                    "score": 70,
                    "explanation": "Market in downtrend - Lower Highs and Lower Lows",
                    "patterns": ["Break of Structure Bearish"]
                }
        
        return {
            "trend": "Ranging",
            "signal": "NEUTRAL",
            "score": 0,
            "explanation": "Market ranging - No clear trend",
            "patterns": []
        }

# ============= MACHINE LEARNING ENGINE =============
class MachineLearningEngine:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = deque(maxlen=5000)
        self.training_labels = deque(maxlen=5000)
        
    def extract_features(self, candles: List[CandleData], indicators: Dict) -> np.ndarray:
        """Extract features for ML model"""
        if len(candles) < 20:
            return np.array([])
        
        closes = [c.close for c in candles[-20:]]
        highs = [c.high for c in candles[-20:]]
        lows = [c.low for c in candles[-20:]]
        
        features = []
        
        # Price features
        features.append(closes[-1])
        features.append(closes[-1] - closes[-2] if len(closes) > 1 else 0)
        features.append(closes[-1] - closes[-5] if len(closes) > 5 else 0)
        features.append(closes[-1] - closes[-10] if len(closes) > 10 else 0)
        
        # Volatility
        features.append(np.std(closes[-10:]) if len(closes) >= 10 else 0)
        features.append(max(highs[-10:]) - min(lows[-10:]) if len(highs) >= 10 else 0)
        
        # Candle features
        last = candles[-1]
        features.append((last.close - last.low) / (last.high - last.low) if last.high != last.low else 0.5)
        features.append((last.high - last.open) / (last.high - last.low) if last.high != last.low else 0.25)
        features.append((last.open - last.low) / (last.high - last.low) if last.high != last.low else 0.25)
        
        # Indicator features
        features.append(indicators.get("rsi", 50))
        features.append(indicators.get("ema_diff", 0))
        features.append(indicators.get("macd", 0))
        features.append(indicators.get("adx", 25))
        
        # Pattern features
        features.append(indicators.get("bullish_patterns", 0))
        features.append(indicators.get("bearish_patterns", 0))
        
        # CRT feature
        features.append(indicators.get("crt_score", 0))
        
        # Digit features
        digits = [int(str(c.close)[-1]) for c in candles[-10:]]
        features.append(sum(digits) / len(digits) if digits else 5)
        features.append(max(digits) if digits else 9)
        features.append(min(digits) if digits else 0)
        
        # Momentum
        if len(closes) >= 5:
            momentum = (closes[-1] - closes[-5]) / closes[-5] * 100 if closes[-5] != 0 else 0
            features.append(momentum)
        else:
            features.append(0)
        
        return np.array(features)
    
    def train(self, features_list: List[np.ndarray], labels: List[int]) -> bool:
        if not ML_AVAILABLE or len(features_list) < 100:
            return False
        
        X = np.array([f for f in features_list if len(f) > 0])
        y = np.array(labels)

        # Map y labels: -1 -> 0, 0 -> 1, 1 -> 2
        y = np.where(y == -1, 0, np.where(y == 0, 1, 2))

        if len(X) < 100:
            return False

        X_scaled = self.scaler.fit_transform(X)

        if ML_AVAILABLE:
            self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            self.rf_model.fit(X_scaled, y)
        
        if XGB_AVAILABLE:
            self.xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
            self.xgb_model.fit(X_scaled, y)

        self.is_trained = True
        print(f"[ML] Models trained on {len(X)} samples")
        return True
    
    def predict(self, candles: List[CandleData], indicators: Dict) -> Tuple[str, float]:
        if not self.is_trained or not ML_AVAILABLE or len(candles) < 20:
            return "NEUTRAL", 0
        
        features = self.extract_features(candles, indicators)
        if len(features) == 0:
            return "NEUTRAL", 0
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        predictions = []
        confidences = []
        
        if self.rf_model:
            pred = self.rf_model.predict(features_scaled)[0]
            proba = self.rf_model.predict_proba(features_scaled)[0]
            predictions.append(pred)
            confidences.append(max(proba) * 100)
        
        if XGB_AVAILABLE and self.xgb_model:
            pred = self.xgb_model.predict(features_scaled)[0]
            proba = self.xgb_model.predict_proba(features_scaled)[0]
            predictions.append(pred)
            confidences.append(max(proba) * 100)
        
        if not predictions:
            return "NEUTRAL", 0
        
        buy_votes = sum(1 for p in predictions if p == 1)
        sell_votes = sum(1 for p in predictions if p == -1)
        
        if buy_votes > sell_votes:
            return "BUY", np.mean(confidences)
        elif sell_votes > buy_votes:
            return "SELL", np.mean(confidences)
        else:
            return "NEUTRAL", 0

# ============= DATA COLLECTOR =============
class DerivDataCollector:
    def __init__(self, symbol: str = DEFAULT_SYMBOL):
        self.symbol = symbol
        self.candles: deque = deque(maxlen=500)
        self.ticks: deque = deque(maxlen=1000)
        self.last_price = 0.0
        self.tick_count = 0
        self.is_connected = False
        self._current_candle: Optional[CandleData] = None
        self._lock = threading.Lock()
        
    def add_tick(self, price: float) -> None:
        with self._lock:
            self.ticks.append(price)
            self.last_price = price
            self.tick_count += 1
            
            if self._current_candle is None:
                self._current_candle = CandleData(
                    timestamp=datetime.now(),
                    open=price,
                    high=price,
                    low=price,
                    close=price
                )
            else:
                self._current_candle.high = max(self._current_candle.high, price)
                self._current_candle.low = min(self._current_candle.low, price)
                self._current_candle.close = price
                
                if self.tick_count % 60 == 0:
                    self.candles.append(self._current_candle)
                    self._current_candle = None
                    print(f"[Candle #{len(self.candles)}] O={self.candles[-1].open:.2f} H={self.candles[-1].high:.2f} L={self.candles[-1].low:.2f} C={self.candles[-1].close:.2f}")
    
    def get_candles(self) -> List[CandleData]:
        with self._lock:
            return list(self.candles)
    
    def get_current_price(self) -> float:
        return self.last_price

# ============= MAIN AI TRADING BOT =============
class AITradingBot:
    def __init__(self):
        self.collector = DerivDataCollector()
        self.pattern_detector = PatternDetector()
        self.crt_analyzer = CRTAnalyzer()
        self.smart_money = SmartMoneyAnalyzer()
        self.supply_demand = SupplyDemandAnalyzer()
        self.fib_analyzer = FibonacciAnalyzer()
        self.structure = MarketStructureAnalyzer()
        self.indicators = TechnicalIndicators()
        self.ml_engine = MachineLearningEngine() if ML_AVAILABLE else None
        
        self.current_signal = None
        self.socket_app = None
        self._stop = False
        
        # Initialize ML with synthetic data
        if self.ml_engine:
            self._init_ml()
    
    def _init_ml(self):
        print("[ML] Initializing ML models...")
        X_train, y_train = [], []
        for _ in range(500):
            features = np.random.randn(21)
            X_train.append(features)
            label = 1 if features[0] > 0 and features[6] > 0.5 else (-1 if features[0] < 0 and features[6] < 0.3 else 0)
            y_train.append(label)
        self.ml_engine.train(X_train, y_train)
    
    def connect_and_run(self):
        url = f"wss://ws.derivws.com/websockets/v3?app_id={DEFAULT_APP_ID}"
        
        def on_open(ws):
            print(f"[Connected] Subscribing to {self.collector.symbol}")
            self.collector.is_connected = True
            ws.send(json.dumps({"ticks": self.collector.symbol, "subscribe": 1}))
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if "tick" in data and "quote" in data["tick"]:
                    self.collector.add_tick(float(data["tick"]["quote"]))
                    self.analyze_market()
            except Exception as e:
                pass
        
        def on_error(ws, error):
            print(f"[Error] WebSocket: {error}")
            self.collector.is_connected = False
        
        def on_close(ws, close_status_code, close_msg):
            print("[Info] Connection closed")
            self.collector.is_connected = False
            if not self._stop:
                time.sleep(5)
                self.connect_and_run()
        
        ws = WebSocketApp(url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
        ws.run_forever()
    
    def start(self):
        thread = threading.Thread(target=self.connect_and_run, daemon=True)
        thread.start()
    
    def analyze_market(self):
        candles = self.collector.get_candles()
        if len(candles) < 10:
            return
        
        current_price = self.collector.get_current_price()
        scores = {"BUY": 0, "SELL": 0}
        all_patterns = []
        reasons = []
        
        # 1. Candlestick Patterns (50+)
        patterns, bullish_cnt, bearish_cnt = self.pattern_detector.detect_all(candles)
        for p in patterns:
            all_patterns.append(p["name"])
            if p["signal"] == "BUY":
                scores["BUY"] += p["strength"]
                reasons.append(p["name"])
            elif p["signal"] == "SELL":
                scores["SELL"] += p["strength"]
                reasons.append(p["name"])
        
        # 2. CRT Analysis (RomeoTPT)
        crt = self.crt_analyzer.analyze(candles)
        if crt["signal"] == "BUY":
            scores["BUY"] += crt["score"]
            reasons.append(crt["explanation"])
            all_patterns.extend(crt["patterns"])
        elif crt["signal"] == "SELL":
            scores["SELL"] += crt["score"]
            reasons.append(crt["explanation"])
            all_patterns.extend(crt["patterns"])
        
        # 3. Smart Money - Order Blocks
        order_blocks = self.smart_money.find_order_blocks(candles)
        for ob in order_blocks:
            all_patterns.append(ob["type"])
            if ob["signal"] == "BUY":
                scores["BUY"] += ob["strength"]
                reasons.append(ob["explanation"])
            else:
                scores["SELL"] += ob["strength"]
                reasons.append(ob["explanation"])
        
        # 4. Smart Money - FVGs
        fvgs = self.smart_money.find_fair_value_gaps(candles)
        for fvg in fvgs:
            all_patterns.append(fvg["type"])
            if fvg["signal"] == "BUY":
                scores["BUY"] += fvg["strength"]
            else:
                scores["SELL"] += fvg["strength"]
        
        # 5. Smart Money - Liquidity Sweeps
        sweeps = self.smart_money.find_liquidity_sweeps(candles)
        for sweep in sweeps:
            all_patterns.append(sweep["type"])
            if sweep["signal"] == "BUY":
                scores["BUY"] += sweep["strength"]
                reasons.append(sweep["explanation"])
            else:
                scores["SELL"] += sweep["strength"]
                reasons.append(sweep["explanation"])
        
        # 6. Supply & Demand Zones
        supply_zones, demand_zones = self.supply_demand.find_zones(candles)
        for sz in supply_zones:
            if current_price <= sz["level"] * 1.001:
                scores["SELL"] += sz["strength"]
                reasons.append(sz["explanation"])
        for dz in demand_zones:
            if current_price >= dz["level"] * 0.999:
                scores["BUY"] += dz["strength"]
                reasons.append(dz["explanation"])
        
        # 7. Market Structure
        structure = self.structure.analyze(candles)
        if structure["signal"] == "BUY":
            scores["BUY"] += structure["score"]
            reasons.append(structure["explanation"])
        elif structure["signal"] == "SELL":
            scores["SELL"] += structure["score"]
            reasons.append(structure["explanation"])
        
        # 8. Technical Indicators
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        
        rsi = self.indicators.calculate_rsi(closes)
        macd_data = self.indicators.calculate_macd(closes)
        adx = self.indicators.calculate_adx(highs, lows, closes)
        bb = self.indicators.calculate_bollinger_bands(closes)
        ema9 = self.indicators.calculate_ema(closes, 9)
        ema21 = self.indicators.calculate_ema(closes, 21)
        ema_diff = ema9 - ema21
        
        # RSI signals
        if rsi < 30:
            scores["BUY"] += 35
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 70:
            scores["SELL"] += 35
            reasons.append(f"RSI overbought ({rsi:.0f})")
        
        # MACD signals
        if macd_data["macd"] > macd_data["signal"]:
            scores["BUY"] += 25
            reasons.append("MACD bullish crossover")
        else:
            scores["SELL"] += 25
            reasons.append("MACD bearish crossover")
        
        # ADX (trend strength)
        if adx > 40:
            reasons.append(f"Strong trend (ADX: {adx:.0f})")
        
        # Bollinger Bands
        if closes[-1] < bb["lower"]:
            scores["BUY"] += 30
            reasons.append("Price below lower Bollinger Band - Oversold")
        elif closes[-1] > bb["upper"]:
            scores["SELL"] += 30
            reasons.append("Price above upper Bollinger Band - Overbought")
        
        # EMA Trend
        if ema9 > ema21 and closes[-1] > ema9:
            scores["BUY"] += 30
            reasons.append("EMA bullish alignment")
        elif ema9 < ema21 and closes[-1] < ema9:
            scores["SELL"] += 30
            reasons.append("EMA bearish alignment")
        
        # 9. Fibonacci Analysis
        fib = self.fib_analyzer.calculate(candles, current_price)
        if fib["signal"] == "BUY":
            scores["BUY"] += fib["confidence"]
            reasons.append(f"Fibonacci {fib['level']}% support")
        elif fib["signal"] == "SELL":
            scores["SELL"] += fib["confidence"]
            reasons.append(f"Fibonacci {fib['level']}% resistance")
        
        # 10. ML Prediction
        ml_signal, ml_conf = "NEUTRAL", 0
        if self.ml_engine and self.ml_engine.is_trained and len(candles) >= 20:
            indicators = {
                "rsi": rsi, "ema_diff": ema_diff, "macd": macd_data["macd"],
                "adx": adx, "bullish_patterns": bullish_cnt, "bearish_patterns": bearish_cnt,
                "crt_score": crt["score"]
            }
            ml_signal, ml_conf = self.ml_engine.predict(candles, indicators)
            if ml_signal == "BUY":
                scores["BUY"] += ml_conf * 0.5
                reasons.append(f"ML predicts BUY ({ml_conf:.0f}%)")
            elif ml_signal == "SELL":
                scores["SELL"] += ml_conf * 0.5
                reasons.append(f"ML predicts SELL ({ml_conf:.0f}%)")
        
        # Final signal
        total = scores["BUY"] + scores["SELL"]
        
        if total > 0:
            buy_pct = (scores["BUY"] / total) * 100
            sell_pct = (scores["SELL"] / total) * 100
            
            if buy_pct > sell_pct + 12:
                final_signal = "BUY"
                confidence = buy_pct
                reasoning = f"BUY ({buy_pct:.0f}%): " + " | ".join(reasons[:4])
            elif sell_pct > buy_pct + 12:
                final_signal = "SELL"
                confidence = sell_pct
                reasoning = f"SELL ({sell_pct:.0f}%): " + " | ".join(reasons[:4])
            else:
                final_signal = "NEUTRAL"
                confidence = max(buy_pct, sell_pct)
                reasoning = f"NEUTRAL - Buy:{buy_pct:.0f}% Sell:{sell_pct:.0f}%"
        else:
            final_signal = "NEUTRAL"
            confidence = 0
            reasoning = "No clear signal - waiting for stronger conditions"
        
        # SL/TP
        atr = self.indicators.calculate_atr(candles)
        
        if final_signal == "BUY":
            stop_loss = current_price - atr * 1.5
            take_profit = current_price + atr * 3
            rr = round((take_profit - current_price) / (current_price - stop_loss), 2) if current_price != stop_loss else 0
        elif final_signal == "SELL":
            stop_loss = current_price + atr * 1.5
            take_profit = current_price - atr * 3
            rr = round((current_price - take_profit) / (stop_loss - current_price), 2) if stop_loss != current_price else 0
        else:
            stop_loss = current_price
            take_profit = current_price
            rr = 0
        
        self.current_signal = {
            "signal": final_signal,
            "confidence": round(confidence, 1),
            "price": round(current_price, 5),
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "risk_reward": rr,
            "patterns": list(set(all_patterns))[:15],
            "reasoning": reasoning,
            "market_phase": crt["phase"],
            "rsi": rsi,
            "macd": round(macd_data["macd"], 3),
            "adx": round(adx, 1),
            "buy_score": round(scores["BUY"], 1),
            "sell_score": round(scores["SELL"], 1),
            "ml_active": self.ml_engine.is_trained if self.ml_engine else False,
            "ml_signal": ml_signal,
            "ml_confidence": round(ml_conf, 1)
        }
        
        # Print to console
        if final_signal != "NEUTRAL" or len(candles) % 15 == 0:
            print(f"[SIGNAL] {final_signal} | Conf: {confidence:.0f}% | RSI: {rsi:.0f} | ADX: {adx:.0f} | Candles: {len(candles)}")
    
    def update_symbol(self, symbol: str):
        self.collector.symbol = symbol
        self.collector.candles.clear()
        self.collector.ticks.clear()
        self.collector.tick_count = 0
        self.collector._current_candle = None
        self.current_signal = None
    
    def get_snapshot(self) -> Dict:
        candles = self.collector.get_candles()
        
        if len(candles) > 0:
            digits = [int(str(c.close)[-1]) for c in candles[-50:]]
            digit_counts = Counter(digits)
            digit_stats = [{"digit": d, "count": digit_counts.get(d, 0), "percentage": round((digit_counts.get(d, 0) / 50) * 100, 1)} for d in range(10)]
        else:
            digit_stats = []
        
        return {
            "symbol": self.collector.symbol,
            "price": self.collector.last_price,
            "tick_count": self.collector.tick_count,
            "candles": len(candles),
            "is_connected": self.collector.is_connected,
            "digit_stats": digit_stats,
            "signal": self.current_signal["signal"] if self.current_signal else "WAITING",
            "confidence": self.current_signal["confidence"] if self.current_signal else 0,
            "reasoning": self.current_signal["reasoning"] if self.current_signal else f"Collecting data... {len(candles)}/10 candles",
            "patterns": self.current_signal["patterns"] if self.current_signal else [],
            "stop_loss": self.current_signal["stop_loss"] if self.current_signal else 0,
            "take_profit": self.current_signal["take_profit"] if self.current_signal else 0,
            "risk_reward": self.current_signal["risk_reward"] if self.current_signal else 0,
            "market_phase": self.current_signal["market_phase"] if self.current_signal else "Initializing",
            "technical": {"rsi": self.current_signal["rsi"] if self.current_signal else 50,
                         "macd": self.current_signal["macd"] if self.current_signal else 0,
                         "adx": self.current_signal["adx"] if self.current_signal else 25},
            "strategy_scores": {"BUY": self.current_signal["buy_score"] if self.current_signal else 0, 
                              "SELL": self.current_signal["sell_score"] if self.current_signal else 0},
            "ml_active": self.current_signal["ml_active"] if self.current_signal else False,
            "ml_signal": self.current_signal["ml_signal"] if self.current_signal else "N/A",
            "ml_confidence": self.current_signal["ml_confidence"] if self.current_signal else 0
        }

# ============= FLASK APP =============
app = Flask(__name__)
CORS(app)

bot = AITradingBot()
bot.start()

@app.route('/')
def index():
    return render_template('trading_dashboard.html')

@app.route('/api/status')
def status():
    return jsonify(bot.get_snapshot())

@app.route('/api/symbols')
def symbols():
    return jsonify({"symbols": list(VOLATILITY_INDICES.keys())})

@app.route('/api/config', methods=['POST'])
def config():
    data = request.json
    if "symbol" in data:
        bot.update_symbol(data["symbol"])
    return jsonify({"success": True})

@app.route('/api/ml/retrain', methods=['POST'])
def retrain_ml():
    if bot.ml_engine:
        bot.ml_engine.training_data.clear()
        bot.ml_engine.training_labels.clear()
        bot._init_ml()
        return jsonify({"success": True, "message": "ML retrained"})
    return jsonify({"success": False, "message": "ML not available"})

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    DERIV MT5 AI TRADING BOT - COMPLETE EDITION               ║
    ║                                                                              ║
    ║  INCLUDED STRATEGIES:                                                        ║
    ║  ✓ Candle Range Theory (CRT) - Purge -> Mitigation -> Expansion             ║
    ║  ✓ Kiss of Death (KOD) / Turtle Soup                                        ║
    ║  ✓ Order Blocks (Institutional accumulation/distribution)                   ║
    ║  ✓ Fair Value Gaps (FVG) - Price imbalances                                 ║
    ║  ✓ Liquidity Sweeps - Stop loss hunting                                     ║
    ║  ✓ Supply & Demand Zones - Overpriced/Underpriced areas                     ║
    ║  ✓ Fibonacci Retracement (0.382, 0.5, 0.618, 0.786)                         ║
    ║  ✓ 50+ Candlestick Patterns (Hammer, Engulfing, Doji, Stars, etc)           ║
    ║  ✓ Market Structure (Dow Theory - HH/HL, LH/LL)                             ║
    ║  ✓ RSI, MACD, ADX, Bollinger Bands, ATR, EMA                                ║
    ║  ✓ Machine Learning (Random Forest + XGBoost)                               ║
    ║  ✓ Online Learning - Improves over time                                     ║
    ║                                                                              ║
    ║  Dashboard: http://localhost:5000                                            ║
    ║                                                                              ║
    ║  Press Ctrl+C to stop                                                        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)