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

bot_state = {
    "is_running": False,
    "symbol": "DOGEUSDT",
    "entry_tf": "3m",
    "htf_tf": "15m",
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
        query_string = ""

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
    res = pybit_request("/v5/market/instruments-info", "GET", {
        "category": "linear",
        "symbol": symbol.upper()
    })
    if res.get("retCode") == 0 and res["result"]["list"]:
        info = res["result"]["list"][0]
        qty_step = float(info["lotSizeFilter"]["qtyStep"])
        min_qty = float(info["lotSizeFilter"]["minOrderQty"])
        price_tick = str(info["priceFilter"]["tickSize"])
        
        qty_precision = len(str(qty_step).split(".")[1].rstrip("0")) if "." in str(qty_step) else 0
        price_precision = len(price_tick.split(".")[1]) if "." in price_tick else 4
            
        return qty_step, min_qty, qty_precision, price_precision
    return 1.0, 10.0, 0, 4

def fetch_klines(symbol, interval, limit=100):
    tf_map = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "1H": "60"}
    bybit_tf = tf_map.get(interval, "3")
    
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

def calculate_atr(df, period=7):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean().iloc[-1]

def calculate_rsi(df, period=7):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

def smc_advanced_analysis(df_entry, df_htf, margin, leverage, price_prec):
    curr_price = df_entry['close'].iloc[-1]
    atr = calculate_atr(df_entry, 7)
    rsi = calculate_rsi(df_entry, 7)

    if pd.isna(atr) or atr == 0:
        atr = curr_price * 0.003

    # Fast EMA for Scalping
    df_htf['ema50'] = df_htf['close'].ewm(span=50, adjust=False).mean()
    htf_bullish = bool(df_htf['close'].iloc[-1] > df_htf['ema50'].iloc[-1])
    htf_bearish = bool(df_htf['close'].iloc[-1] < df_htf['ema50'].iloc[-1])

    # Micro Liquidity Sweep (10 Candles)
    recent_low = df_entry['low'].iloc[-12:-2].min()
    recent_high = df_entry['high'].iloc[-12:-2].max()
    
    swept_low = bool(df_entry['low'].iloc[-2:].min() <= recent_low and curr_price > recent_low)
    swept_high = bool(df_entry['high'].iloc[-2:].max() >= recent_high and curr_price < recent_high)

    # Micro MSS
    mss_bullish = bool(curr_price > df_entry['high'].iloc[-4:-1].max())
    mss_bearish = bool(curr_price < df_entry['low'].iloc[-4:-1].min())

    # Fast FVG Check
    has_bullish_fvg = False
    has_bearish_fvg = False
    fvg_zone = [0.0, 0.0]

    for i in range(len(df_entry) - 1, max(2, len(df_entry) - 10), -1):
        if df_entry['low'].iloc[i] > df_entry['high'].iloc[i-2]:
            has_bullish_fvg = True
            fvg_zone = [df_entry['high'].iloc[i-2], df_entry['low'].iloc[i]]
            break
        elif df_entry['high'].iloc[i] < df_entry['low'].iloc[i-2]:
            has_bearish_fvg = True
            fvg_zone = [df_entry['high'].iloc[i], df_entry['low'].iloc[i-2]]
            break

    bullish_checks = {
        "HTF Trend Up (EMA50)": htf_bullish,
        "Liquidity Sweep Low": swept_low,
        "Micro Shift (MSS)": mss_bullish,
        "Fair Value Gap (FVG)": has_bullish_fvg
    }

    bearish_checks = {
        "HTF Trend Down (EMA50)": htf_bearish,
        "Liquidity Sweep High": swept_high,
        "Micro Shift (MSS)": mss_bearish,
        "Fair Value Gap (FVG)": has_bearish_fvg
    }

    bull_score = sum(bullish_checks.values())
    bear_score = sum(bearish_checks.values())

    signal = "NEUTRAL"
    sl_price = 0.0
    tp_price = 0.0

    # Scalping SL/TP Logic (0.6x ATR SL & 1.2x ATR TP -> R:R 1:2 Fast Scalp)
    if bull_score >= 2 and htf_bullish and rsi < 70:
        signal = "BULLISH"
        sl_price = curr_price - (atr * 0.6)
        tp_price = curr_price + (atr * 1.2)

    elif bear_score >= 2 and htf_bearish and rsi > 30:
        signal = "BEARISH"
        sl_price = curr_price + (atr * 0.6)
        tp_price = curr_price - (atr * 1.2)

    fmt = f"{{:.{price_prec}f}}"
    
    pos_size = (margin * leverage) / curr_price if curr_price > 0 else 0
    p_profit = abs(tp_price - curr_price) * pos_size if signal != "NEUTRAL" else 0
    p_loss = abs(curr_price - sl_price) * pos_size if signal != "NEUTRAL" else 0

    return {
        "signal": signal,
        "curr_price": fmt.format(curr_price),
        "bullish_checks": bullish_checks,
        "bearish_checks": bearish_checks,
        "fvg_zone": [fmt.format(fvg_zone[0]), fmt.format(fvg_zone[1])],
        "sl": fmt.format(sl_price),
        "tp": fmt.format(tp_price),
        "rsi": round(rsi, 2),
        "rr_ratio": "1 : 2.0 (Scalp)",
        "est_profit": f"+${round(p_profit, 2)}",
        "est_loss": f"-${round(p_loss, 2)}",
        "timestamp": time.strftime("%H:%M:%S")
    }

def execute_trade(symbol, side, margin, leverage, price, sl, tp):
    pybit_request("/v5/position/set-leverage", "POST", {
        "category": "linear",
        "symbol": symbol,
        "buyLeverage": str(leverage),
        "sellLeverage": str(leverage)
    })
    
    qty_step, min_qty, qty_prec, price_prec = get_symbol_info(symbol)
    raw_qty = (margin * leverage) / float(price)
    
    qty = round(raw_qty - (raw_qty % qty_step), qty_prec) if qty_prec > 0 else int(raw_qty)
    if qty < min_qty:
        qty = min_qty

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
        return True, f"Scalp Trade Açyldy! Qty: {qty} | SL: {sl} | TP: {tp}"
    
    return False, f"Bybit Error: {res.get('retMsg')} (Code: {res.get('retCode')})"

def bot_loop():
    last_trade_signal = None
    while True:
        if bot_state["is_running"]:
            try:
                symbol = bot_state["symbol"]
                entry_tf = bot_state["entry_tf"]
                htf_tf = bot_state["htf_tf"]

                _, _, _, price_prec = get_symbol_info(symbol)
                df_entry = fetch_klines(symbol, entry_tf)
                df_htf = fetch_klines(symbol, htf_tf)

                if df_entry is not None and df_htf is not None:
                    res = smc_advanced_analysis(df_entry, df_htf, bot_state["margin"], bot_state["leverage"], price_prec)
                    bot_state["last_analysis"] = res

                    log_event(f"[{symbol}] Baha: {res['curr_price']} | Signal: {res['signal']} | RSI: {res['rsi']}")

                    if bot_state["auto_trade"] and res["signal"] in ["BULLISH", "BEARISH"]:
                        if last_trade_signal != res["signal"]:
                            side = "Buy" if res["signal"] == "BULLISH" else "Sell"
                            success, msg = execute_trade(
                                symbol, side, bot_state["margin"], bot_state["leverage"],
                                res["curr_price"], res["sl"], res["tp"]
                            )
                            log_event(f"AUTO-SCALP-ORDER: {msg}")
                            if success:
                                last_trade_signal = res["signal"]
            except Exception as e:
                log_event(f"Loop Error: {str(e)}")

        time.sleep(2)

threading.Thread(target=bot_loop, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/start", methods=["POST"])
def start_bot():
    data = request.json or {}
    bot_state["symbol"] = data.get("symbol", "DOGEUSDT").upper()
    bot_state["entry_tf"] = data.get("entry_tf", "3m")
    bot_state["htf_tf"] = data.get("htf_tf", "15m")
    bot_state["margin"] = float(data.get("margin", 10))
    bot_state["leverage"] = int(data.get("leverage", 10))
    bot_state["auto_trade"] = bool(data.get("auto_trade", False))
    bot_state["is_running"] = True
    
    symbol = bot_state["symbol"]
    _, _, _, price_prec = get_symbol_info(symbol)
    df_entry = fetch_klines(symbol, bot_state["entry_tf"])
    df_htf = fetch_klines(symbol, bot_state["htf_tf"])
    if df_entry is not None and df_htf is not None:
        bot_state["last_analysis"] = smc_advanced_analysis(df_entry, df_htf, bot_state["margin"], bot_state["leverage"], price_prec)
        
    log_event(f"Scalp Bot Başlatyldy: {bot_state['symbol']}")
    return jsonify({"status": "started"})

@app.route("/api/stop", methods=["POST"])
def stop_bot():
    bot_state["is_running"] = False
    log_event("Bot Togtadyldy!")
    return jsonify({"status": "stopped"})

@app.route("/api/status", methods=["GET"])
def get_status():
    return jsonify({
        "is_running": bot_state["is_running"],
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
    app.run(host="0.0.0.0", port=port)