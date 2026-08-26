import os
import sqlite3
import json
import requests
import hmac
import hashlib
import time
import threading
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BASE_URL = os.getenv("BYBIT_URL", "https://api.bybit.com")
DB_NAME = os.getenv("DB_NAME", "trades.db")

# Global Bot Ýagdaýy
bot_state = {
    "is_running": False,
    "symbol": "DOGEUSDT",
    "entry_tf": "15m",
    "htf_tf": "1H",
    "margin": 10.0,
    "leverage": 10,
    "auto_trade": False,
    "last_analysis": None,
    "logs": []
}

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

def log_event(message):
    timestamp = time.strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    bot_state["logs"].append(formatted_msg)
    if len(bot_state["logs"]) > 50:
        bot_state["logs"].pop(0)

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

def get_symbol_info(symbol):
    """Bybit-den simwolyň minimal lot we tegeleklemek takyklygyny alýar"""
    res = pybit_request("/v5/market/instruments-info", "GET", {
        "category": "linear",
        "symbol": symbol.upper()
    })
    if res.get("retCode") == 0 and res["result"]["list"]:
        info = res["result"]["list"][0]
        qty_step = float(info["lotSizeFilter"]["qtyStep"])
        min_qty = float(info["lotSizeFilter"]["minOrderQty"])
        
        precision = 0
        if "." in str(qty_step):
            precision = len(str(qty_step).split(".")[1].rstrip("0"))
            
        return qty_step, min_qty, precision
    return 1.0, 10.0, 0

def fetch_klines(symbol, interval, limit=100):
    tf_map = {"5m": "5", "15m": "15", "30m": "30", "1H": "60", "4H": "240", "1D": "D"}
    bybit_tf = tf_map.get(interval, "15")
    
    res = pybit_request("/v5/market/kline", "GET", {
        "category": "linear",
        "symbol": symbol.upper(),
        "interval": bybit_tf,
        "limit": limit
    })
    
    if res.get("retCode") == 0 and res["result"]["list"]:
        raw_data = res["result"]["list"]
        df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df = df.iloc[::-1].reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    return None

def smc_analysis(df_entry, df_htf, margin, leverage):
    # HTF Treind
    df_htf['ema200'] = df_htf['close'].ewm(span=200, adjust=False).mean()
    htf_ema = round(df_htf['ema200'].iloc[-1], 4)
    curr_price = round(df_entry['close'].iloc[-1], 4)
    htf_trend = "BULLISH" if curr_price > htf_ema else "BEARISH"

    # Fair Value Gap (FVG)
    fvg_type = "ÝOK"
    fvg_zone = None

    for i in range(len(df_entry) - 2, max(0, len(df_entry) - 15), -1):
        if df_entry['low'].iloc[i] > df_entry['high'].iloc[i-2]:
            fvg_type = "BULLISH"
            fvg_zone = [round(df_entry['high'].iloc[i-2], 4), round(df_entry['low'].iloc[i], 4)]
            break
        elif df_entry['high'].iloc[i] < df_entry['low'].iloc[i-2]:
            fvg_type = "BEARISH"
            fvg_zone = [round(df_entry['high'].iloc[i], 4), round(df_entry['low'].iloc[i-2], 4)]
            break

    # Order Block (OB)
    ob_zone = None
    if fvg_type == "BULLISH":
        ob_candle = df_entry.iloc[-8:-2][df_entry['close'] < df_entry['open']].tail(1)
        if not ob_candle.empty:
            ob_zone = [round(ob_candle['low'].values[0], 4), round(ob_candle['high'].values[0], 4)]
    elif fvg_type == "BEARISH":
        ob_candle = df_entry.iloc[-8:-2][df_entry['close'] > df_entry['open']].tail(1)
        if not ob_candle.empty:
            ob_zone = [round(ob_candle['low'].values[0], 4), round(ob_candle['high'].values[0], 4)]

    # Signal Logic
    signal = "NEUTRAL"
    if htf_trend == "BULLISH" and fvg_type == "BULLISH":
        signal = "BUY"
    elif htf_trend == "BEARISH" and fvg_type == "BEARISH":
        signal = "SELL"

    # Target Price & Risk / Reward Calculation
    sl_dist = 0.015  # 1.5% SL
    tp_dist = 0.035  # 3.5% TP
    
    sl_price = round(curr_price * (1 - sl_dist) if signal == "BUY" else curr_price * (1 + sl_dist), 4)
    tp_price = round(curr_price * (1 + tp_dist) if signal == "BUY" else curr_price * (1 - tp_dist), 4)

    est_position_vol = margin * leverage
    potential_profit = round(est_position_vol * tp_dist, 2)
    potential_loss = round(est_position_vol * sl_dist, 2)

    return {
        "signal": signal,
        "htf_trend": htf_trend,
        "htf_ema": htf_ema,
        "fvg_type": fvg_type,
        "fvg_zone": fvg_zone if fvg_zone else ["N/A", "N/A"],
        "ob_zone": ob_zone if ob_zone else ["N/A", "N/A"],
        "price": curr_price,
        "sl": sl_price,
        "tp": tp_price,
        "rr_ratio": "1:2.33",
        "est_profit": f"+${potential_profit}",
        "est_loss": f"-${potential_loss}",
        "timestamp": time.strftime("%H:%M:%S")
    }

def execute_trade(symbol, side, margin, leverage, price):
    # Set Leverage
    pybit_request("/v5/position/set-leverage", "POST", {
        "category": "linear",
        "symbol": symbol,
        "buyLeverage": str(leverage),
        "sellLeverage": str(leverage)
    })
    
    qty_step, min_qty, precision = get_symbol_info(symbol)
    
    raw_qty = (margin * leverage) / price
    
    if precision > 0:
        qty = round(raw_qty - (raw_qty % qty_step), precision)
    else:
        qty = int(raw_qty - (raw_qty % qty_step))
    
    if qty < min_qty:
        qty = min_qty

    sl = round(price * 0.985 if side == "Buy" else price * 1.015, 4)
    tp = round(price * 1.035 if side == "Buy" else price * 0.965, 4)

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
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, side, entry_price, sl, tp, qty, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, side, price, sl, tp, qty, "OPEN"))
        conn.commit()
        conn.close()
        return True, f"Sdelka açyldy! Qty: {qty}"
    
    return False, f"Bybit Ýalňyşlygy: {res.get('retMsg')} (Code: {res.get('retCode')})"

def bot_loop():
    last_trade_signal = None
    while True:
        if bot_state["is_running"]:
            try:
                symbol = bot_state["symbol"]
                entry_tf = bot_state["entry_tf"]
                htf_tf = bot_state["htf_tf"]

                df_entry = fetch_klines(symbol, entry_tf)
                df_htf = fetch_klines(symbol, htf_tf)

                if df_entry is not None and df_htf is not None:
                    res = smc_analysis(df_entry, df_htf, bot_state["margin"], bot_state["leverage"])
                    bot_state["last_analysis"] = res

                    log_event(f"[{symbol}] Baha: {res['price']} | Trend: {res['htf_trend']} | Signal: {res['signal']}")

                    # Auto Trade trigger
                    if bot_state["auto_trade"] and res["signal"] in ["BUY", "SELL"]:
                        if last_trade_signal != res["signal"]:
                            side = "Buy" if res["signal"] == "BUY" else "Sell"
                            success, msg = execute_trade(
                                symbol, side, bot_state["margin"], bot_state["leverage"], res["price"]
                            )
                            log_event(f"AUTO-TRADE: {msg}")
                            if success:
                                last_trade_signal = res["signal"]
            except Exception as e:
                log_event(f"Error Loop: {str(e)}")

        time.sleep(10)

threading.Thread(target=bot_loop, daemon=True).start()

# API Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/start", methods=["POST"])
def start_bot():
    data = request.json or {}
    bot_state["symbol"] = data.get("symbol", "DOGEUSDT").upper()
    bot_state["entry_tf"] = data.get("entry_tf", "15m")
    bot_state["htf_tf"] = data.get("htf_tf", "1H")
    bot_state["margin"] = float(data.get("margin", 10))
    bot_state["leverage"] = int(data.get("leverage", 10))
    bot_state["auto_trade"] = bool(data.get("auto_trade", False))
    bot_state["is_running"] = True
    log_event(f"Bot Başlatyldy: {bot_state['symbol']} ({bot_state['entry_tf']}/{bot_state['htf_tf']}) AutoTrade: {bot_state['auto_trade']}")
    return jsonify({"status": "started", "message": "Bot üstünlikli işledildi!"})

@app.route("/api/stop", methods=["POST"])
def stop_bot():
    bot_state["is_running"] = False
    log_event("Bot Togtadyldy!")
    return jsonify({"status": "stopped", "message": "Bot togtadyldy!"})

@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "is_running": bot_state["is_running"],
        "config": {
            "symbol": bot_state["symbol"],
            "margin": bot_state["margin"],
            "leverage": bot_state["leverage"],
            "auto_trade": bot_state["auto_trade"]
        },
        "analysis": bot_state["last_analysis"],
        "logs": bot_state["logs"][-15:]
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)