import os
import time
import hmac
import hashlib
import json
import logging
import sqlite3
import threading
from decimal import Decimal, ROUND_DOWN
from datetime import datetime

import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIG
# ============================================================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
BYBIT_URL = os.getenv("BYBIT_URL", "https://api.bybit.com").rstrip("/")
DB_NAME = os.getenv("DB_NAME", "trades.db")

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("SMC")

bot_lock = threading.Lock()
bot_thread = None

bot_state = {
    "running": False,
    "symbol": "PEPEUSDT",
    "leverage": 5,
    "amount": 1.0,
    "timeframe": "15",
    "htf_timeframe": "60",
    "analysis": None,
    "last_signal": None,
    "last_error": None,
    "last_scan": None,
    "logs": [],
    "positions": []
}

# ============================================================
# DATABASE
# ============================================================
def get_db():
    conn = sqlite3.connect(DB_NAME, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trade_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL,
                sl REAL,
                tp REAL,
                pnl REAL DEFAULT 0,
                status TEXT DEFAULT 'OPEN',
                signal_score INTEGER DEFAULT 0,
                reason TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                event TEXT NOT NULL
            )
        """)

init_db()

# ============================================================
# LOGGING
# ============================================================
def log_terminal(message):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    logger.info(message)
    bot_state["logs"].insert(0, line)
    bot_state["logs"] = bot_state["logs"][:250]

# ============================================================
# BYBIT V5 API
# ============================================================
session = requests.Session()
session.headers.update({"Content-Type": "application/json"})

def _json_compact(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def bybit_request(method, endpoint, params=None, payload=None):
    if not BYBIT_API_KEY or not BYBIT_API_SECRET:
        raise RuntimeError("BYBIT_API_KEY / BYBIT_API_SECRET .env-da tapylmady.")

    method = method.upper()
    
    # Bybit serwer wagtyny al
    try:
        time_resp = session.get(BYBIT_URL + "/v5/market/time", timeout=5)
        time_data = time_resp.json()
        server_time = int(time_data["result"]["timeSecond"]) * 1000
        timestamp = str(server_time)
    except:
        timestamp = str(int((time.time() - 1.05) * 1000))
    
    recv_window = "60000"

    if method == "GET":
        params = params or {}
        query = "&".join(
            f"{k}={str(params[k])}"
            for k in sorted(params)
            if params[k] is not None
        )
        sign_payload = timestamp + BYBIT_API_KEY + recv_window + query
        request_kwargs = {"params": params}
    else:
        body = _json_compact(payload or {})
        sign_payload = timestamp + BYBIT_API_KEY + recv_window + body
        request_kwargs = {"data": body}

    signature = hmac.new(
        BYBIT_API_SECRET.encode(),
        sign_payload.encode(),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-SIGN": signature,
        "X-BAPI-SIGN-TYPE": "2",
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json",
    }

    url = BYBIT_URL + endpoint

    try:
        response = session.request(
            method,
            url,
            headers=headers,
            timeout=15,
            **request_kwargs
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"HTTP/API error: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Bybit JSON jogaby nädogry.") from exc

    if data.get("retCode") != 0:
        raise RuntimeError(
            f"Bybit retCode={data.get('retCode')}: {data.get('retMsg')}"
        )
    return data

def public_request(endpoint, params=None):
    try:
        r = session.get(BYBIT_URL + endpoint, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("retCode") != 0:
            raise RuntimeError(
                f"Bybit retCode={data.get('retCode')}: {data.get('retMsg')}"
            )
        return data
    except Exception as exc:
        raise RuntimeError(f"Public API error: {exc}") from exc

def get_wallet_balance():
    try:
        data = bybit_request(
            "GET",
            "/v5/account/wallet-balance",
            {"accountType": "UNIFIED", "coin": "USDT"}
        )
        coins = data.get("result", {}).get("list", [{}])[0].get("coin", [])
        for coin in coins:
            if coin.get("coin") == "USDT":
                return float(coin.get("equity", 0) or 0)
        return 0.0
    except Exception:
        return None

def get_klines(symbol, interval="15", limit=300):
    """Islendik symbol üçin bahalary alýar"""
    data = public_request(
        "/v5/market/kline",
        {
            "category": "linear",
            "symbol": symbol.upper(),
            "interval": str(interval),
            "limit": min(int(limit), 500)
        }
    )
    rows = data["result"]["list"]

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp", "open", "high", "low",
            "close", "volume", "turnover"
        ]
    )
    numeric = ["open", "high", "low", "close", "volume", "turnover"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    return df

def get_instrument_info(symbol):
    """Islendik symbol üçin instrument maglumatlaryny alýar"""
    data = public_request(
        "/v5/market/instruments-info",
        {"category": "linear", "symbol": symbol.upper()}
    )
    item = data["result"]["list"][0]
    return {
        "tickSize": float(item["priceFilter"]["tickSize"]),
        "qtyStep": float(item["lotSizeFilter"]["qtyStep"]),
        "minOrderQty": float(item["lotSizeFilter"]["minOrderQty"]),
        "maxOrderQty": float(item["lotSizeFilter"]["maxOrderQty"]),
    }

def validate_symbol(symbol):
    """Bybit-de symbol-yň barlygyny barlaýar"""
    try:
        data = public_request(
            "/v5/market/instruments-info",
            {"category": "linear", "symbol": symbol.upper()}
        )
        if data.get("result", {}).get("list", []):
            return True
        return False
    except Exception:
        return False

def round_step(value, step):
    value_d = Decimal(str(value))
    step_d = Decimal(str(step))
    if step_d <= 0:
        return float(value_d)
    rounded = (value_d / step_d).quantize(
        Decimal("1"), rounding=ROUND_DOWN
    ) * step_d
    return float(rounded)

def get_open_positions(symbol):
    try:
        data = bybit_request(
            "GET",
            "/v5/position/list",
            {"category": "linear", "symbol": symbol.upper()}
        )
        positions = []
        for p in data.get("result", {}).get("list", []):
            size = float(p.get("size", 0) or 0)
            if size > 0:
                positions.append({
                    "symbol": p.get("symbol"),
                    "side": p.get("side"),
                    "size": size,
                    "entry_price": float(p.get("avgPrice", 0) or 0),
                    "leverage": float(p.get("leverage", 0) or 0),
                    "unrealized_pnl": float(p.get("unrealisedPnl", 0) or 0)
                })
        return positions
    except Exception as e:
        return []

# ============================================================
# SMC ENGINE - DOLY WE DOGRY
# ============================================================

def true_range(df):
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

def atr(df, length=14):
    tr = true_range(df)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

def rsi(df, length=14):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))

def find_swings(df, left=2, right=2):
    """Swing high we low nokatlaryny tapýar"""
    highs = []
    lows = []
    high_values = df["high"].values
    low_values = df["low"].values
    
    for i in range(left, len(df) - right):
        h = high_values[i]
        l = low_values[i]
        
        # Swing high
        is_high = True
        for j in range(i-left, i):
            if high_values[j] >= h:
                is_high = False
                break
        if is_high:
            for j in range(i+1, i+right+1):
                if high_values[j] >= h:
                    is_high = False
                    break
        if is_high:
            highs.append(i)
        
        # Swing low
        is_low = True
        for j in range(i-left, i):
            if low_values[j] <= l:
                is_low = False
                break
        if is_low:
            for j in range(i+1, i+right+1):
                if low_values[j] <= l:
                    is_low = False
                    break
        if is_low:
            lows.append(i)
    
    return highs, lows

def structure_bias(df):
    """Structure bias - BULLISH, BEARISH, RANGE"""
    sh, sl = find_swings(df)
    
    if len(sh) < 2 or len(sl) < 2:
        return "RANGE", None
    
    h1 = float(df["high"].iloc[sh[-2]])
    h2 = float(df["high"].iloc[sh[-1]])
    l1 = float(df["low"].iloc[sl[-2]])
    l2 = float(df["low"].iloc[sl[-1]])
    
    if h2 > h1 and l2 > l1:
        return "BULLISH", {
            "last_swing_high": h2,
            "previous_swing_high": h1,
            "last_swing_low": l2,
            "previous_swing_low": l1,
        }
    
    if h2 < h1 and l2 < l1:
        return "BEARISH", {
            "last_swing_high": h2,
            "previous_swing_high": h1,
            "last_swing_low": l2,
            "previous_swing_low": l1,
        }
    
    return "RANGE", {
        "last_swing_high": h2,
        "previous_swing_high": h1,
        "last_swing_low": l2,
        "previous_swing_low": l1,
    }

def detect_bos_choch(df, lookback=120):
    """BOS (Break of Structure) we CHoCH (Change of Character) detection"""
    work = df.tail(lookback).reset_index(drop=True)
    sh, sl = find_swings(work)
    
    events = []
    if len(sh) == 0 and len(sl) == 0:
        return events
    
    current_bias, _ = structure_bias(work)
    last = work.iloc[-1]
    price = float(last["close"])
    
    if len(sh) > 0:
        last_high_idx = sh[-1]
        level = float(work["high"].iloc[last_high_idx])
        if price > level:
            events.append({
                "type": "BOS_UP",
                "level": level,
                "index": last_high_idx
            })
    
    if len(sl) > 0:
        last_low_idx = sl[-1]
        level = float(work["low"].iloc[last_low_idx])
        if price < level:
            events.append({
                "type": "BOS_DOWN",
                "level": level,
                "index": last_low_idx
            })
    
    if current_bias == "BEARISH":
        for event in events:
            if event["type"] == "BOS_UP":
                events.append({
                    "type": "CHOCH_UP",
                    "level": event["level"],
                    "index": event["index"]
                })
    elif current_bias == "BULLISH":
        for event in events:
            if event["type"] == "BOS_DOWN":
                events.append({
                    "type": "CHOCH_DOWN",
                    "level": event["level"],
                    "index": event["index"]
                })
    
    return events

def detect_fvg(df, lookback=50):
    """FVG (Fair Value Gap) detection"""
    fvgs = []
    start = max(3, len(df) - lookback)
    
    for i in range(start, len(df) - 1):
        # Bullish FVG
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            fvgs.append({
                "type": "BULLISH",
                "low": float(df["high"].iloc[i - 2]),
                "high": float(df["low"].iloc[i]),
                "index": i,
                "timestamp": int(df["timestamp"].iloc[i])
            })
        
        # Bearish FVG
        if df["high"].iloc[i] < df["low"].iloc[i - 2]:
            fvgs.append({
                "type": "BEARISH",
                "low": float(df["high"].iloc[i]),
                "high": float(df["low"].iloc[i - 2]),
                "index": i,
                "timestamp": int(df["timestamp"].iloc[i])
            })
    
    return fvgs

def detect_order_blocks(df, lookback=80):
    """Order Block detection"""
    blocks = []
    start = max(2, len(df) - lookback)
    
    for i in range(start, len(df) - 3):
        if i + 3 >= len(df):
            break
        
        rng = df["high"].iloc[i] - df["low"].iloc[i]
        if rng <= 0:
            continue
        
        next3_high = df["high"].iloc[i+1:i+4].max()
        next3_low = df["low"].iloc[i+1:i+4].min()
        
        # Bullish OB
        if df["close"].iloc[i] < df["open"].iloc[i]:
            if next3_high > df["high"].iloc[i] + rng * 0.6:
                blocks.append({
                    "type": "BULLISH",
                    "low": float(df["low"].iloc[i]),
                    "high": float(df["open"].iloc[i]),
                    "index": i,
                    "timestamp": int(df["timestamp"].iloc[i])
                })
        
        # Bearish OB
        if df["close"].iloc[i] > df["open"].iloc[i]:
            if next3_low < df["low"].iloc[i] - rng * 0.6:
                blocks.append({
                    "type": "BEARISH",
                    "low": float(df["open"].iloc[i]),
                    "high": float(df["high"].iloc[i]),
                    "index": i,
                    "timestamp": int(df["timestamp"].iloc[i])
                })
    
    return blocks

def detect_liquidity_sweeps(df, lookback=15):
    """Liquidity sweep detection"""
    if len(df) < lookback + 2:
        return {
            "buy_side_sweep": False,
            "sell_side_sweep": False,
            "buy_side_level": None,
            "sell_side_level": None
        }
    
    recent = df.iloc[-lookback-1:-1]
    last = df.iloc[-1]
    
    prev_high = float(recent["high"].max())
    prev_low = float(recent["low"].min())
    
    current_high = float(last["high"])
    current_low = float(last["low"])
    current_close = float(last["close"])
    
    buy_sweep = current_high > prev_high and current_close < prev_high
    sell_sweep = current_low < prev_low and current_close > prev_low
    
    return {
        "buy_side_sweep": buy_sweep,
        "sell_side_sweep": sell_sweep,
        "buy_side_level": prev_high,
        "sell_side_level": prev_low
    }

def get_support_resistance(df, lookback=50):
    """Support and resistance levels"""
    if len(df) < lookback:
        return None, None
    
    recent = df.tail(lookback)
    sh, sl = find_swings(recent)
    
    resistance_levels = []
    support_levels = []
    
    for idx in sh[-5:]:
        resistance_levels.append(float(recent["high"].iloc[idx]))
    
    for idx in sl[-5:]:
        support_levels.append(float(recent["low"].iloc[idx]))
    
    current_price = float(df["close"].iloc[-1])
    
    support = None
    resistance = None
    
    if support_levels:
        supports_below = [s for s in support_levels if s < current_price]
        if supports_below:
            support = max(supports_below)
    
    if resistance_levels:
        resistances_above = [r for r in resistance_levels if r > current_price]
        if resistances_above:
            resistance = min(resistances_above)
    
    return support, resistance

def analyze_smc(df, htf_df=None):
    """Main SMC analysis - ähli alamatlar"""
    
    if len(df) < 50:
        raise ValueError("SMC analiz üçin azyndan 50 ýapylan candle gerek.")
    
    x = df.copy()
    x["atr"] = atr(x, 14)
    x["rsi"] = rsi(x, 14)
    
    last = x.iloc[-1]
    price = float(last["close"])
    atr_value = float(last["atr"]) if pd.notna(last["atr"]) else 0.0
    rsi_value = float(last["rsi"]) if pd.notna(last["rsi"]) else 50.0
    
    # Structure
    ltf_bias, ltf_structure = structure_bias(x)
    
    htf_bias = "RANGE"
    if htf_df is not None and len(htf_df) >= 50:
        htf_bias, _ = structure_bias(htf_df)
    
    # BOS/CHoCH
    events = detect_bos_choch(x, lookback=120)
    bos_up = any(e["type"] == "BOS_UP" for e in events)
    bos_down = any(e["type"] == "BOS_DOWN" for e in events)
    choch_up = any(e["type"] == "CHOCH_UP" for e in events)
    choch_down = any(e["type"] == "CHOCH_DOWN" for e in events)
    
    # FVG
    fvgs = detect_fvg(x, lookback=50)
    bullish_fvg = None
    bearish_fvg = None
    
    for fvg in fvgs:
        if fvg["type"] == "BULLISH" and fvg["low"] <= price <= fvg["high"]:
            bullish_fvg = fvg
            break
    
    for fvg in fvgs:
        if fvg["type"] == "BEARISH" and fvg["low"] <= price <= fvg["high"]:
            bearish_fvg = fvg
            break
    
    # Order Blocks
    obs = detect_order_blocks(x, lookback=80)
    bullish_ob = None
    bearish_ob = None
    
    for ob in obs:
        if ob["type"] == "BULLISH" and ob["low"] <= price <= ob["high"]:
            bullish_ob = ob
            break
    
    for ob in obs:
        if ob["type"] == "BEARISH" and ob["low"] <= price <= ob["high"]:
            bearish_ob = ob
            break
    
    # Liquidity
    liquidity = detect_liquidity_sweeps(x, lookback=15)
    
    # Support/Resistance
    support, resistance = get_support_resistance(x, lookback=50)
    
    # Premium/Discount
    if ltf_structure:
        range_high = ltf_structure["last_swing_high"]
        range_low = ltf_structure["last_swing_low"]
    else:
        range_high = float(x["high"].tail(50).max())
        range_low = float(x["low"].tail(50).min())
    
    equilibrium = (range_high + range_low) / 2
    location = "DISCOUNT" if price < equilibrium else "PREMIUM"
    
    # ============================================================
    # SCORING - GÜÝJENDIRILEN
    # ============================================================
    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []
    
    # LTF Structure (3 bal)
    if ltf_bias == "BULLISH":
        buy_score += 3
        buy_reasons.append("✅ LTF structure bullish (+3)")
    elif ltf_bias == "BEARISH":
        sell_score += 3
        sell_reasons.append("✅ LTF structure bearish (+3)")
    
    # HTF Structure (2 bal)
    if htf_bias == "BULLISH":
        buy_score += 2
        buy_reasons.append("✅ HTF structure bullish (+2)")
    elif htf_bias == "BEARISH":
        sell_score += 2
        sell_reasons.append("✅ HTF structure bearish (+2)")
    
    # BOS/CHoCH (2 bal)
    if bos_up:
        buy_score += 2
        buy_reasons.append("✅ BOS/CHoCH UP (+2)")
    if bos_down:
        sell_score += 2
        sell_reasons.append("✅ BOS/CHoCH DOWN (+2)")
    if choch_up:
        buy_score += 1
        buy_reasons.append("✅ CHoCH UP (+1)")
    if choch_down:
        sell_score += 1
        sell_reasons.append("✅ CHoCH DOWN (+1)")
    
    # Liquidity Sweeps (2 bal)
    if liquidity["sell_side_sweep"]:
        buy_score += 2
        buy_reasons.append("✅ Sell-side liquidity swept (+2)")
    if liquidity["buy_side_sweep"]:
        sell_score += 2
        sell_reasons.append("✅ Buy-side liquidity swept (+2)")
    
    # FVG (1 bal)
    if bullish_fvg:
        buy_score += 1
        buy_reasons.append(f"✅ Bullish FVG (+1)")
    if bearish_fvg:
        sell_score += 1
        sell_reasons.append(f"✅ Bearish FVG (+1)")
    
    # Order Blocks (1 bal)
    if bullish_ob:
        buy_score += 1
        buy_reasons.append(f"✅ Bullish OB (+1)")
    if bearish_ob:
        sell_score += 1
        sell_reasons.append(f"✅ Bearish OB (+1)")
    
    # Premium/Discount (1 bal)
    if location == "DISCOUNT":
        buy_score += 1
        buy_reasons.append("✅ Discount area (+1)")
    else:
        sell_score += 1
        sell_reasons.append("✅ Premium area (+1)")
    
    # Support/Resistance (1 bal)
    if support and price <= support * 1.005:
        buy_score += 1
        buy_reasons.append(f"✅ Near support (+1)")
    if resistance and price >= resistance * 0.995:
        sell_score += 1
        sell_reasons.append(f"✅ Near resistance (+1)")
    
    # RSI (1 bal)
    if rsi_value < 30:
        buy_score += 1
        buy_reasons.append(f"✅ RSI oversold ({rsi_value:.1f}) (+1)")
    elif rsi_value > 70:
        sell_score += 1
        sell_reasons.append(f"✅ RSI overbought ({rsi_value:.1f}) (+1)")
    
    # ============================================================
    # SIGNAL - 4-den başlaýar
    # ============================================================
    signal = "WAIT"
    sl = None
    tp = None
    rr = None
    
    if buy_score >= 4 and buy_score >= sell_score + 1:
        signal = "BUY"
        if atr_value > 0:
            sl = price - atr_value * 1.0
            if support and sl < support:
                sl = support - atr_value * 0.05
            tp = price + (price - sl) * 2.0
            rr = 2.0
    
    elif sell_score >= 4 and sell_score >= buy_score + 1:
        signal = "SELL"
        if atr_value > 0:
            sl = price + atr_value * 1.0
            if resistance and sl > resistance:
                sl = resistance + atr_value * 0.05
            tp = price - (sl - price) * 2.0
            rr = 2.0
    
    # Weak signal
    elif buy_score >= 3 and buy_score > sell_score:
        signal = "BUY_WEAK"
    elif sell_score >= 3 and sell_score > buy_score:
        signal = "SELL_WEAK"
    
    return {
        "symbol": bot_state["symbol"],
        "timeframe": bot_state["timeframe"],
        "htf_timeframe": bot_state["htf_timeframe"],
        "price": price,
        "atr": atr_value,
        "rsi": rsi_value,
        "ltf_bias": ltf_bias,
        "htf_bias": htf_bias,
        "location": location,
        "equilibrium": equilibrium,
        "range_high": range_high,
        "range_low": range_low,
        "support": support,
        "resistance": resistance,
        "events": events,
        "liquidity": liquidity,
        "bullish_fvg": bullish_fvg,
        "bearish_fvg": bearish_fvg,
        "bullish_ob": bullish_ob,
        "bearish_ob": bearish_ob,
        "buy_score": round(buy_score, 1),
        "sell_score": round(sell_score, 1),
        "confidence": max(round(buy_score, 1), round(sell_score, 1)),
        "signal": signal,
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "buy_reasons": buy_reasons,
        "sell_reasons": sell_reasons,
        "candle_time": int(last["timestamp"]),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# BOT WORKER
# ============================================================
def run_analysis():
    symbol = bot_state["symbol"]
    tf = bot_state["timeframe"]
    htf = bot_state["htf_timeframe"]
    
    # Symbol-y barla
    if not validate_symbol(symbol):
        bot_state["last_error"] = f"Symbol {symbol} nädogry!"
        log_terminal(f"[ERROR] Symbol {symbol} nädogry!")
        return None
    
    ltf_df = get_klines(symbol, tf, 300)
    htf_df = get_klines(symbol, htf, 200)
    
    analysis = analyze_smc(ltf_df, htf_df)
    bot_state["analysis"] = analysis
    bot_state["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bot_state["last_error"] = None
    bot_state["last_signal"] = analysis["signal"]
    
    return analysis

def save_trade(trade):
    with get_db() as conn:
        conn.execute("""
            INSERT INTO trade_history
            (symbol, side, entry_price, sl, tp, pnl, status, signal_score, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade["symbol"], trade["side"], trade["entry_price"],
            trade["sl"], trade["tp"], trade.get("pnl", 0),
            trade.get("status", "OPEN"),
            trade.get("signal_score", 0),
            trade.get("reason", "")
        ))

def get_trades(limit=50):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT symbol, side, entry_price, sl, tp, pnl, status,
                   signal_score, reason, created_at
            FROM trade_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]

def bot_worker():
    global bot_thread
    
    log_terminal("[SYSTEM] SMC bot başlady.")
    bot_state["running"] = True
    
    while bot_state["running"]:
        try:
            bot_state["positions"] = get_open_positions(bot_state["symbol"])
            
            analysis = run_analysis()
            
            if analysis:
                log_terminal(
                    f"[SMC] {analysis['symbol']} {analysis['timeframe']} | "
                    f"LTF={analysis['ltf_bias']} HTF={analysis['htf_bias']} | "
                    f"BUY={analysis['buy_score']} SELL={analysis['sell_score']} | "
                    f"signal={analysis['signal']}"
                )
                
                # Trade execution
                if analysis["signal"] in ("BUY", "SELL"):
                    positions = get_open_positions(bot_state["symbol"])
                    if len(positions) == 0:
                        log_terminal(f"🚨 {analysis['signal']} SIGNAL DETECTED!")
                        try:
                            save_trade({
                                "symbol": bot_state["symbol"],
                                "side": analysis["signal"],
                                "entry_price": analysis["price"],
                                "sl": analysis["sl"],
                                "tp": analysis["tp"],
                                "signal_score": analysis["confidence"],
                                "reason": "; ".join(analysis["buy_reasons"] if analysis["signal"] == "BUY" else analysis["sell_reasons"])
                            })
                            log_terminal(f"✅ Trade saved: {analysis['signal']} at {analysis['price']}")
                        except Exception as e:
                            log_terminal(f"[ERROR] Trade save: {e}")
            
        except Exception as exc:
            bot_state["last_error"] = str(exc)
            log_terminal(f"[BOT ERROR] {exc}")
        
        for _ in range(2):
            if not bot_state["running"]:
                break
            time.sleep(1)
    
    bot_state["running"] = False
    log_terminal("[SYSTEM] SMC bot saklandy.")
    bot_thread = None

# ============================================================
# FLASK ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/start", methods=["POST"])
def start_bot():
    global bot_thread
    
    data = request.get_json(silent=True) or {}
    
    symbol = str(data.get("symbol", "PEPEUSDT")).upper().strip()
    timeframe = str(data.get("timeframe", "15"))
    htf_timeframe = str(data.get("htf_timeframe", "60"))
    
    try:
        leverage = int(data.get("leverage", 5))
        amount = float(data.get("amount", 1.0))
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Leverage we amount san bolmaly."}), 400
    
    # Symbol-y barla
    if not validate_symbol(symbol):
        return jsonify({
            "status": "error",
            "message": f"❌ Symbol {symbol} Bybit-de ýok! BTCUSDT, PEPEUSDT, DOGEUSDT ýaly dogry symbol saýlaň."
        }), 400
    
    with bot_lock:
        bot_state["symbol"] = symbol
        bot_state["timeframe"] = timeframe
        bot_state["htf_timeframe"] = htf_timeframe
        bot_state["leverage"] = leverage
        bot_state["amount"] = amount
        bot_state["last_error"] = None
        
        if bot_state["running"]:
            return jsonify({"status": "error", "message": "Bot eýýäm işleýär."}), 409
        
        bot_state["running"] = True
        bot_thread = threading.Thread(target=bot_worker, daemon=True)
        bot_thread.start()
    
    return jsonify({"status": "success", "message": f"✅ SMC bot {symbol} üçin başlady."})

@app.route("/api/stop", methods=["POST"])
def stop_bot():
    bot_state["running"] = False
    log_terminal("[SYSTEM] Stop signal berildi.")
    return jsonify({"status": "success", "message": "Bot saklanýar."})

@app.route("/api/analyze", methods=["POST"])
def manual_analyze():
    try:
        data = request.get_json(silent=True) or {}
        if data.get("symbol"):
            symbol = str(data["symbol"]).upper().strip()
            if not validate_symbol(symbol):
                return jsonify({"status": "error", "message": f"Symbol {symbol} nädogry!"}), 400
            bot_state["symbol"] = symbol
        if data.get("timeframe"):
            bot_state["timeframe"] = str(data["timeframe"])
        if data.get("htf_timeframe"):
            bot_state["htf_timeframe"] = str(data["htf_timeframe"])
        
        analysis = run_analysis()
        if analysis:
            return jsonify({"status": "success", "analysis": analysis})
        else:
            return jsonify({"status": "error", "message": "Analiz başarısız"}), 500
    except Exception as exc:
        bot_state["last_error"] = str(exc)
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route("/api/dashboard")
def dashboard():
    balance = None
    balance_error = None
    
    try:
        balance = get_wallet_balance()
    except Exception as exc:
        balance_error = str(exc)
    
    return jsonify({
        "balance": balance,
        "balance_error": balance_error,
        "bot_state": {
            **bot_state,
            "logs": bot_state["logs"][:80]
        },
        "trades": get_trades(50)
    })

@app.route("/api/sync-time", methods=["POST"])
def sync_time():
    try:
        resp = requests.get(BYBIT_URL + "/v5/market/time", timeout=5)
        data = resp.json()
        server_time = int(data["result"]["timeSecond"]) * 1000
        local_time = int(time.time() * 1000)
        diff = local_time - server_time
        
        return jsonify({
            "status": "success",
            "message": f"Bybit time: {server_time}, Local: {local_time}, Diff: {diff}ms",
            "diff": diff
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "api_configured": bool(BYBIT_API_KEY and BYBIT_API_SECRET),
        "mode": "LIVE" if "testnet" not in BYBIT_URL.lower() else "TESTNET",
        "running": bot_state.get("running", False),
        "symbol": bot_state.get("symbol", "—")
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)