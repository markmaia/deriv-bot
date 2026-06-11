import os
import json
import threading
import time
import math
import numpy as np
from collections import deque, Counter
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, jsonify, request
from flask_cors import CORS
from websocket import WebSocketApp

# ============= CONFIGURATION =============
VOLATILITY_INDICES = {
    "R_10": "Volatility 10 Index",
    "R_25": "Volatility 25 Index", 
    "R_50": "Volatility 50 Index",
    "R_75": "Volatility 75 Index",
    "R_100": "Volatility 100 Index",
}
DEFAULT_APP_ID = "1089"
DEFAULT_SYMBOL = "R_75"

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
        return {"macd": macd_line, "signal": macd_line * 0.8, "histogram": macd_line * 0.2}
    
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

# ============= CRT ANALYZER =============
class CRTAnalyzer:
    def analyze(self, candles: List[CandleData]) -> Dict:
        if len(candles) < 3:
            return {"signal": "NEUTRAL", "score": 0, "phase": "Waiting"}
        last = candles[-1]
        high, low, close = last.high, last.low, last.close
        close_position = (close - low) / (high - low) if high != low else 0.5
        if close_position > 0.6:
            return {"signal": "BUY", "score": 60, "phase": "Accumulation"}
        elif close_position < 0.4:
            return {"signal": "SELL", "score": 60, "phase": "Distribution"}
        return {"signal": "NEUTRAL", "score": 30, "phase": "Consolidation"}

# ============= DATA COLLECTOR =============
class DerivDataCollector:
    def __init__(self, symbol: str = DEFAULT_SYMBOL):
        self.symbol = symbol
        self.candles: deque = deque(maxlen=500)
        self.last_price = 0.0
        self.tick_count = 0
        self.is_connected = False
        self._current_candle: Optional[CandleData] = None
        self._lock = threading.Lock()
        
    def add_tick(self, price: float) -> None:
        with self._lock:
            self.last_price = price
            self.tick_count += 1
            if self._current_candle is None:
                self._current_candle = CandleData(
                    timestamp=datetime.now(),
                    open=price, high=price, low=price, close=price
                )
            else:
                self._current_candle.high = max(self._current_candle.high, price)
                self._current_candle.low = min(self._current_candle.low, price)
                self._current_candle.close = price
                if self.tick_count % 60 == 0:
                    self.candles.append(self._current_candle)
                    self._current_candle = None
    
    def get_candles(self) -> List[CandleData]:
        with self._lock:
            return list(self.candles)
    
    def get_current_price(self) -> float:
        return self.last_price

# ============= MAIN AI TRADING BOT =============
class AITradingBot:
    def __init__(self):
        self.collector = DerivDataCollector()
        self.indicators = TechnicalIndicators()
        self.crt_analyzer = CRTAnalyzer()
        self.current_signal = None
        self._stop = False
    
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
        closes = [c.close for c in candles]
        
        rsi = self.indicators.calculate_rsi(closes)
        macd = self.indicators.calculate_macd(closes)
        atr = self.indicators.calculate_atr(candles)
        crt = self.crt_analyzer.analyze(candles)
        
        scores = {"BUY": 0, "SELL": 0}
        reasons = []
        
        if rsi < 30:
            scores["BUY"] += 35
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 70:
            scores["SELL"] += 35
            reasons.append(f"RSI overbought ({rsi:.0f})")
        
        if macd["macd"] > macd["signal"]:
            scores["BUY"] += 25
            reasons.append("MACD bullish")
        else:
            scores["SELL"] += 25
            reasons.append("MACD bearish")
        
        if crt["signal"] == "BUY":
            scores["BUY"] += crt["score"]
            reasons.append(f"CRT: {crt['phase']}")
        elif crt["signal"] == "SELL":
            scores["SELL"] += crt["score"]
            reasons.append(f"CRT: {crt['phase']}")
        
        total = scores["BUY"] + scores["SELL"]
        if total > 0:
            buy_pct = (scores["BUY"] / total) * 100
            sell_pct = (scores["SELL"] / total) * 100
            
            if buy_pct > sell_pct + 12:
                final_signal = "BUY"
                confidence = buy_pct
            elif sell_pct > buy_pct + 12:
                final_signal = "SELL"
                confidence = sell_pct
            else:
                final_signal = "NEUTRAL"
                confidence = max(buy_pct, sell_pct)
        else:
            final_signal = "NEUTRAL"
            confidence = 0
        
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
            "reasoning": " | ".join(reasons[:3]) if reasons else "Analyzing...",
            "rsi": round(rsi, 1),
            "macd": round(macd["macd"], 3),
            "buy_score": round(scores["BUY"], 1),
            "sell_score": round(scores["SELL"], 1)
        }
        
        print(f"[SIGNAL] {final_signal} | Conf: {confidence:.0f}% | RSI: {rsi:.0f}")
    
    def update_symbol(self, symbol: str):
        self.collector.symbol = symbol
        self.collector.candles.clear()
        self.collector.tick_count = 0
        self.collector._current_candle = None
        self.current_signal = None
    
    def get_snapshot(self) -> Dict:
        candles = self.collector.get_candles()
        digit_stats = []
        if len(candles) > 0:
            digits = [int(str(c.close)[-1]) for c in candles[-50:]]
            digit_counts = Counter(digits)
            digit_stats = [{"digit": d, "count": digit_counts.get(d, 0), "percentage": round((digit_counts.get(d, 0) / 50) * 100, 1)} for d in range(10)]
        
        return {
            "symbol": self.collector.symbol,
            "price": round(self.collector.last_price, 5) if self.collector.last_price else 0,
            "tick_count": self.collector.tick_count,
            "candles": len(candles),
            "is_connected": self.collector.is_connected,
            "signal": self.current_signal["signal"] if self.current_signal else "WAITING",
            "confidence": self.current_signal["confidence"] if self.current_signal else 0,
            "reasoning": self.current_signal["reasoning"] if self.current_signal else f"Collecting... {len(candles)}/10 candles",
            "stop_loss": self.current_signal["stop_loss"] if self.current_signal else 0,
            "take_profit": self.current_signal["take_profit"] if self.current_signal else 0,
            "risk_reward": self.current_signal["risk_reward"] if self.current_signal else 0,
            "patterns": ["CRT", "RSI", "MACD"],
            "technical": {"rsi": self.current_signal["rsi"] if self.current_signal else 50,
                         "macd": self.current_signal["macd"] if self.current_signal else 0},
            "strategy_scores": {"BUY": self.current_signal["buy_score"] if self.current_signal else 0, 
                              "SELL": self.current_signal["sell_score"] if self.current_signal else 0},
            "digit_stats": digit_stats
        }

# HTML dashboard embedded directly
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deriv MT5 AI Trading Bot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: rgba(30, 41, 59, 0.9); backdrop-filter: blur(10px); border-radius: 20px; padding: 20px 30px; margin-bottom: 20px; border: 1px solid rgba(71, 85, 105, 0.5); }
        .header h1 { font-size: 28px; background: linear-gradient(135deg, #34d399, #3b82f6); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .grid { display: grid; grid-template-columns: 1fr 380px; gap: 20px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        .card { background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(10px); border-radius: 20px; padding: 20px; border: 1px solid rgba(71, 85, 105, 0.5); margin-bottom: 20px; }
        .card-title { color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px; border-left: 3px solid #3b82f6; padding-left: 12px; }
        .signal-box { text-align: center; padding: 30px; border-radius: 16px; font-size: 42px; font-weight: bold; }
        .signal-buy { background: linear-gradient(135deg, #059669, #10b981); color: white; box-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }
        .signal-sell { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; box-shadow: 0 0 20px rgba(239, 68, 68, 0.3); }
        .signal-neutral { background: linear-gradient(135deg, #4b5563, #6b7280); color: white; }
        .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 15px; }
        .stat-item { text-align: center; padding: 12px; background: #0f172a; border-radius: 12px; border: 1px solid #1e293b; }
        .stat-label { color: #64748b; font-size: 11px; text-transform: uppercase; }
        .stat-value { color: white; font-size: 18px; font-weight: bold; margin-top: 5px; font-family: monospace; }
        .progress-section { margin-bottom: 15px; }
        .progress-label { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px; }
        .progress-bar { height: 8px; background: #1e293b; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; border-radius: 10px; transition: width 0.5s ease; }
        .fill-green { background: linear-gradient(90deg, #059669, #10b981); }
        .fill-red { background: linear-gradient(90deg, #dc2626, #ef4444); }
        .pattern-badge { display: inline-block; background: #1e293b; padding: 5px 12px; border-radius: 20px; font-size: 11px; color: #34d399; margin: 4px; border: 1px solid #334155; }
        .digit-grid { display: grid; grid-template-columns: repeat(10, 1fr); gap: 8px; margin-top: 15px; }
        .digit-cell { text-align: center; padding: 10px 5px; background: #0f172a; border-radius: 10px; border: 1px solid #1e293b; }
        .digit-number { font-size: 20px; font-weight: bold; color: white; }
        .digit-percent { font-size: 11px; color: #34d399; }
        .status-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }
        .status-online { background: #10b981; box-shadow: 0 0 8px #10b981; animation: pulse 2s infinite; }
        .status-offline { background: #ef4444; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .reasoning-box { background: #0f172a; border-radius: 12px; padding: 12px; margin-top: 15px; border: 1px solid #1e293b; }
        .reasoning-text { color: #94a3b8; font-size: 12px; line-height: 1.5; }
        .flex-between { display: flex; justify-content: space-between; align-items: center; }
        select, button { background: #1e293b; border: 1px solid #334155; padding: 10px 20px; border-radius: 12px; color: white; font-size: 14px; cursor: pointer; }
        select:hover, button:hover { background: #334155; border-color: #3b82f6; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="flex-between" style="flex-wrap: wrap; gap: 15px;">
                <div>
                    <h1>⚡ Deriv MT5 AI Trading Bot</h1>
                    <p>CRT + RSI + MACD + Real-time Analysis</p>
                </div>
                <div class="header-controls">
                    <select id="symbolSelect">
                        <option value="R_10">📊 Volatility 10 Index</option>
                        <option value="R_25">📊 Volatility 25 Index</option>
                        <option value="R_50">📊 Volatility 50 Index</option>
                        <option value="R_75" selected>📊 Volatility 75 Index</option>
                        <option value="R_100">📊 Volatility 100 Index</option>
                    </select>
                    <div>
                        <span class="status-dot" id="statusDot"></span>
                        <span id="statusText" style="color: #94a3b8;">Connecting...</span>
                    </div>
                </div>
            </div>
        </div>
        <div class="grid">
            <div>
                <div class="card">
                    <div class="card-title">🎯 CURRENT SIGNAL</div>
                    <div id="signalBox" class="signal-box signal-neutral">ANALYZING</div>
                    <div class="reasoning-box">
                        <div class="reasoning-text" id="reasoning">Waiting for market data...</div>
                    </div>
                    <div class="stats-grid">
                        <div class="stat-item"><div class="stat-label">🛑 STOP LOSS</div><div class="stat-value" id="stopLoss">--</div></div>
                        <div class="stat-item"><div class="stat-label">🎯 TAKE PROFIT</div><div class="stat-value" id="takeProfit">--</div></div>
                        <div class="stat-item"><div class="stat-label">📈 RISK:REWARD</div><div class="stat-value" id="riskReward">--</div></div>
                        <div class="stat-item"><div class="stat-label">💪 CONFIDENCE</div><div class="stat-value" id="confidence">--%</div></div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">⚖️ STRATEGY SCORES</div>
                    <div class="progress-section">
                        <div class="progress-label"><span style="color: #10b981;">📈 BUY Score</span><span id="buyScore" style="color: #10b981; font-weight: bold;">0</span></div>
                        <div class="progress-bar"><div id="buyBar" class="progress-fill fill-green" style="width: 0%"></div></div>
                    </div>
                    <div class="progress-section">
                        <div class="progress-label"><span style="color: #ef4444;">📉 SELL Score</span><span id="sellScore" style="color: #ef4444; font-weight: bold;">0</span></div>
                        <div class="progress-bar"><div id="sellBar" class="progress-fill fill-red" style="width: 0%"></div></div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">📊 MARKET DATA</div>
                    <div class="stats-grid">
                        <div class="stat-item"><div class="stat-label">💵 CURRENT PRICE</div><div class="stat-value" id="price">--</div></div>
                        <div class="stat-item"><div class="stat-label">📉 RSI (14)</div><div class="stat-value" id="rsi">--</div></div>
                        <div class="stat-item"><div class="stat-label">🔍 MARKET PHASE</div><div class="stat-value" id="phase">--</div></div>
                        <div class="stat-item"><div class="stat-label">🕯️ CANDLES</div><div class="stat-value" id="candleCount">--</div></div>
                    </div>
                </div>
            </div>
            <div>
                <div class="card">
                    <div class="card-title">🔍 DETECTED PATTERNS</div>
                    <div id="patternsList" style="min-height: 120px;"><span style="color: #64748b; font-size: 13px;">Waiting for data...</span></div>
                </div>
                <div class="card">
                    <div class="card-title">🔢 LAST DIGIT DISTRIBUTION</div>
                    <div class="text-sm" style="color: #64748b; margin-bottom: 10px;">Based on last 50 candles</div>
                    <div id="digitGrid" class="digit-grid"><div style="grid-column: span 10; text-align: center; color: #64748b;">Loading...</div></div>
                </div>
                <div class="card">
                    <div class="card-title">🧠 ACTIVE STRATEGIES</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                        <span class="pattern-badge">📐 Candle Range Theory</span>
                        <span class="pattern-badge">📉 RSI (14)</span>
                        <span class="pattern-badge">📊 MACD</span>
                        <span class="pattern-badge">🕯️ Real-time Analysis</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                const signal = data.signal;
                const signalBox = document.getElementById('signalBox');
                signalBox.className = 'signal-box';
                if (signal === 'BUY') { signalBox.className += ' signal-buy'; signalBox.innerHTML = '📈 BUY'; }
                else if (signal === 'SELL') { signalBox.className += ' signal-sell'; signalBox.innerHTML = '📉 SELL'; }
                else { signalBox.className += ' signal-neutral'; signalBox.innerHTML = '⏸ NEUTRAL'; }
                document.getElementById('reasoning').innerHTML = data.reasoning || 'Analyzing...';
                document.getElementById('stopLoss').innerHTML = data.stop_loss?.toFixed(5) || '--';
                document.getElementById('takeProfit').innerHTML = data.take_profit?.toFixed(5) || '--';
                document.getElementById('riskReward').innerHTML = data.risk_reward?.toFixed(2) || '--';
                document.getElementById('confidence').innerHTML = `${data.confidence?.toFixed(1) || 0}%`;
                document.getElementById('price').innerHTML = data.price?.toFixed(5) || '--';
                document.getElementById('rsi').innerHTML = data.technical?.rsi?.toFixed(1) || '--';
                document.getElementById('phase').innerHTML = data.market_phase || '--';
                document.getElementById('candleCount').innerHTML = data.candles || 0;
                const statusDot = document.getElementById('statusDot');
                const statusText = document.getElementById('statusText');
                if (data.is_connected) { statusDot.className = 'status-dot status-online'; statusText.innerHTML = 'Connected to Deriv'; }
                else { statusDot.className = 'status-dot status-offline'; statusText.innerHTML = 'Connecting...'; }
                const scores = data.strategy_scores || {};
                const buyScore = scores.BUY || 0;
                const sellScore = scores.SELL || 0;
                const total = buyScore + sellScore;
                document.getElementById('buyScore').innerHTML = buyScore.toFixed(1);
                document.getElementById('sellScore').innerHTML = sellScore.toFixed(1);
                const buyPercent = total > 0 ? (buyScore / total) * 100 : 0;
                const sellPercent = total > 0 ? (sellScore / total) * 100 : 0;
                document.getElementById('buyBar').style.width = `${buyPercent}%`;
                document.getElementById('sellBar').style.width = `${sellPercent}%`;
                const patternsList = document.getElementById('patternsList');
                if (data.patterns && data.patterns.length > 0) { patternsList.innerHTML = data.patterns.slice(0, 12).map(p => `<span class="pattern-badge">🔍 ${p}</span>`).join(''); }
                else { patternsList.innerHTML = '<span style="color: #64748b;">No patterns detected yet...</span>'; }
                const digitStats = data.digit_stats || [];
                const digitGrid = document.getElementById('digitGrid');
                if (digitStats.length > 0) { digitGrid.innerHTML = digitStats.map(stat => `<div class="digit-cell"><div class="digit-number">${stat.digit}</div><div class="digit-percent">${stat.percentage?.toFixed(1)}%</div></div>`).join(''); }
                else { digitGrid.innerHTML = '<div style="grid-column: span 10; text-align: center; color: #64748b;">Collecting data...</div>'; }
            } catch (error) { console.error('Fetch error:', error); }
        }
        async function changeSymbol() {
            const symbol = document.getElementById('symbolSelect').value;
            try {
                await fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ symbol: symbol }) });
                document.getElementById('signalBox').className = 'signal-box signal-neutral';
                document.getElementById('signalBox').innerHTML = 'ANALYZING';
                setTimeout(fetchStatus, 2000);
            } catch (error) { console.error('Symbol change error:', error); }
        }
        document.getElementById('symbolSelect').addEventListener('change', changeSymbol);
        fetchStatus();
        setInterval(fetchStatus, 2000);
    </script>
</body>
</html>
"""

# ============= FLASK APP =============
app = Flask(__name__)
CORS(app)

bot = AITradingBot()
bot.start()

@app.route('/')
def index():
    return DASHBOARD_HTML

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Deriv Trading Bot on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
