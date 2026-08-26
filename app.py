import os
import sqlite3
import json
import requests
import hmac
import hashlib
import time
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Config
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
BASE_URL = os.getenv("BYBIT_URL", "https://api.bybit.com")
DB_NAME = os.getenv("DB_NAME", "trades.db")

# Init DB
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            side TEXT,
            entry_price REAL,
            sl REAL,
            tp REAL,
            qty REAL,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Bybit v5 Signature Request Helper
def pybit_request(endpoint, method="GET", params=None):
    if params is None:
        params = {}
    
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    
    if method == "GET":
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        payload = query_string
    else:
        payload = json.dumps(params)

    param_str = timestamp + API_KEY + recv_window + payload
    signature = hmac.new(API_SECRET.encode('utf-8'), param_str.encode('utf-8'), hashlib.sha256).hexdigest()

    headers = {
        'X-BAPI-API-KEY': API_KEY,
        'X-BAPI-SIGN': signature,
        'X-BAPI-TIMESTAMP': timestamp,
        'X-BAPI-RECV-WINDOW': recv_window,
        'Content-Type': 'application/json'
    }

    url = f"{BASE_URL}{endpoint}"
    if method == "GET" and query_string:
        url += f"?{query_string}"

    try:
        if method == "GET":
            res = requests.get(url, headers=headers, timeout=10)
        else:
            res = requests.post(url, data=payload, headers=headers, timeout=10)
        return res.json()
    except Exception as e:
        return {"retCode": -1, "retMsg": str(e)}

# Get Kline Data
def fetch_klines(symbol, interval, limit=100):
    # Timeframe mapping for Bybit
    tf_map = {"5m": "5", "15m": "15", "30m": "30", "1H": "60", "4H": "240", "1D": "D"}
    bybit_tf = tf_map.get(interval, "15")
    
    res = pybit_request("/v5/market/kline", "GET", {
        "category": "linear",
        "symbol": symbol.upper(),
        "interval": bybit_tf,
        "limit": limit
    })
    
    if res.get("retCode") == 0:
        raw_data = res["result"]["list"]
        df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df.iloc[::-1].reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    return None

# Advanced SMC Analyzer Strategy
def smc_analysis(df_entry, df_htf):
    # 1. HTF Market Structure Trend (EMA / BOS simplification)
    df_htf['ema200'] = df_htf['close'].ewm(span=200, adjust=False).mean()
    htf_trend = "BULLISH" if df_htf['close'].iloc[-1] > df_htf['ema200'].iloc[-1] else "BEARISH"

    # 2. LTF FVG (Fair Value Gap) Detection
    # Bullish FVG: Low(i) > High(i-2)
    # Bearish FVG: High(i) < Low(i-2)
    fvg_type = None
    fvg_zone = None

    for i in range(len(df_entry) - 2, len(df_entry) - 10, -1):
        # Bullish FVG
        if df_entry['low'].iloc[i] > df_entry['high'].iloc[i-2]:
            fvg_type = "BULLISH"
            fvg_zone = (df_entry['high'].iloc[i-2], df_entry['low'].iloc[i])
            break
        # Bearish FVG
        elif df_entry['high'].iloc[i] < df_entry['low'].iloc[i-2]:
            fvg_type = "BEARISH"
            fvg_zone = (df_entry['high'].iloc[i], df_entry['low'].iloc[i-2])
            break

    # 3. Order Block (OB) Identification
    # Bullish OB: Last down candle before strong move up
    # Bearish OB: Last up candle before strong move down
    ob_zone = None
    if fvg_type == "BULLISH":
        ob_candle = df_entry.iloc[-5:-2][df_entry['close'] < df_entry['open']].tail(1)
        if not ob_candle.empty:
            ob_zone = (ob_candle['low'].values[0], ob_candle['high'].values[0])
    elif fvg_type == "BEARISH":
        ob_candle = df_entry.iloc[-5:-2][df_entry['close'] > df_entry['open']].tail(1)
        if not ob_candle.empty:
            ob_zone = (ob_candle['low'].values[0], ob_candle['high'].values[0])

    # Signal Logic Alignment
    signal = "NEUTRAL"
    curr_price = df_entry['close'].iloc[-1]
    
    if htf_trend == "BULLISH" and fvg_type == "BULLISH":
        signal = "BUY"
    elif htf_trend == "BEARISH" and fvg_type == "BEARISH":
        signal = "SELL"

    return {
        "signal": signal,
        "htf_trend": htf_trend,
        "fvg_type": fvg_type,
        "fvg_zone": fvg_zone,
        "ob_zone": ob_zone,
        "price": curr_price
    }

# Execute Trade
def execute_trade(symbol, side, margin, leverage, price):
    # Set Leverage
    pybit_request("/v5/position/set-leverage", "POST", {
        "category": "linear",
        "symbol": symbol,
        "buyLeverage": str(leverage),
        "sellLeverage": str(leverage)
    })
    
    qty = round((margin * leverage) / price, 2)
    sl = round(price * 0.99 if side == "Buy" else price * 1.01, 4)
    tp = round(price * 1.03 if side == "Buy" else price * 0.97, 4)

    order_payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": str(qty),
        "stopLoss": str(sl),
        "takeProfit": str(tp),
        "timeInForce": "GTC"
    }
    
    res = pybit_request("/v5/order/create", "POST", order_payload)

    if res.get("retCode") == 0:
        # Save to DB
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, side, entry_price, sl, tp, qty, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, side, price, sl, tp, qty, "OPEN"))
        conn.commit()
        conn.close()
        return True, "Sdelka üstünlikli açyldy!"
    return False, res.get("retMsg", "Ýalňyşlyk ýüze çykdy")

# API Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json
    symbol = data.get("symbol", "").upper()
    entry_tf = data.get("entry_tf", "15m")
    htf_tf = data.get("htf_tf", "1H")
    margin = float(data.get("margin", 10))
    leverage = int(data.get("leverage", 10))
    auto_trade = data.get("auto_trade", False)

    df_entry = fetch_klines(symbol, entry_tf)
    df_htf = fetch_klines(symbol, htf_tf)

    if df_entry is None or df_htf is None:
        return jsonify({"status": "error", "message": "Market maglumatlaryny alyp bolmady."})

    analysis = smc_analysis(df_entry, df_htf)
    trade_executed = False
    trade_msg = ""

    if auto_trade and analysis["signal"] in ["BUY", "SELL"]:
        side = "Buy" if analysis["signal"] == "BUY" else "Sell"
        trade_executed, trade_msg = execute_trade(symbol, side, margin, leverage, analysis["price"])

    return jsonify({
        "status": "success",
        "symbol": symbol,
        "analysis": analysis,
        "trade_executed": trade_executed,
        "trade_message": trade_msg
    })

@app.route("/api/history", methods=["GET"])
def history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, side, entry_price, sl, tp, qty, status, timestamp FROM trades ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    trades = []
    for r in rows:
        trades.append({
            "symbol": r[0], "side": r[1], "entry_price": r[2],
            "sl": r[3], "tp": r[4], "qty": r[5], "status": r[6], "timestamp": r[7]
        })
    return jsonify({"trades": trades})

if __name__ == "__main__":
    import os
    # Render berýän PORT-y alýar, tapmasa 5000 ulanýar
    port = int(os.environ.get("PORT", 5000))
    # host="0.0.0.0" Bolsa serwer daşary bilen habarlaşyp bilýär
    app.run(host="0.0.0.0", port=port)