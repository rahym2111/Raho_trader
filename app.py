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
from flask import Flask, render_template, request, jsonify, send_from_directory
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
    "symbol": "BTCUSDT",
    "leverage": 5,
    "amount": 10.0,
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
    data = public_request(
        "/v5/market/kline",
        {
            "category": "linear",
            "symbol": symbol.upper(),
            "interval": str(interval),
            "limit": min(int(limit), 1000)
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
        log_terminal(f"Error getting positions: {e}")
        return []

# ============================================================
# SMC ENGINE - IMPROVED
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
    sh, sl = find_swings(df)
    
    if len(sh) < 2 or len(sl) < 2:
        return "NEUTRAL", None
    
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

def detect_fvg(df, lookback=50):
    fvgs = []
    start = max(2, len(df) - lookback)
    
    for i in range(start, len(df) - 1):
        # Bullish FVG: current low > high two candles earlier
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            fvgs.append({
                "type": "BULLISH",
                "low": float(df["high"].iloc[i - 2]),
                "high": float(df["low"].iloc[i]),
                "index": i,
                "timestamp": int(df["timestamp"].iloc[i])
            })
        
        # Bearish FVG: current high < low two candles earlier
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
    blocks = []
    start = max(2, len(df) - lookback)
    
    for i in range(start, len(df) - 3):
        if i + 3 >= len(df):
            break
            
        body = abs(df["close"].iloc[i] - df["open"].iloc[i])
        rng = df["high"].iloc[i] - df["low"].iloc[i]
        
        if rng <= 0:
            continue
        
        next3_high = df["high"].iloc[i+1:i+4].max()
        next3_low = df["low"].iloc[i+1:i+4].min()
        
        if df["close"].iloc[i] < df["open"].iloc[i]:
            if next3_high > df["high"].iloc[i] + rng * 0.7:
                blocks.append({
                    "type": "BULLISH",
                    "low": float(df["low"].iloc[i]),
                    "high": float(df["open"].iloc[i]),
                    "index": i,
                    "timestamp": int(df["timestamp"].iloc[i])
                })
        
        if df["close"].iloc[i] > df["open"].iloc[i]:
            if next3_low < df["low"].iloc[i] - rng * 0.7:
                blocks.append({
                    "type": "BEARISH",
                    "low": float(df["open"].iloc[i]),
                    "high": float(df["high"].iloc[i]),
                    "index": i,
                    "timestamp": int(df["timestamp"].iloc[i])
                })
    
    return blocks

def detect_liquidity_sweeps(df, lookback=20):
    if len(df) < lookback + 2:
        return {"buy_side_sweep": False, "sell_side_sweep": False}
    
    recent = df.iloc[-lookback-1:-1]
    last = df.iloc[-1]
    
    prev_high = float(recent["high"].max())
    prev_low = float(recent["low"].min())
    
    buy_side = (
        float(last["high"]) > prev_high and
        float(last["close"]) < prev_high
    )
    
    sell_side = (
        float(last["low"]) < prev_low and
        float(last["close"]) > prev_low
    )
    
    return {
        "buy_side_sweep": buy_side,
        "sell_side_sweep": sell_side,
        "buy_side_level": prev_high,
        "sell_side_level": prev_low,
        "current_high": float(last["high"]),
        "current_low": float(last["low"])
    }

def detect_bos_choch(df, lookback=120):
    work = df.tail(lookback).reset_index(drop=True)
    sh, sl = find_swings(work)
    
    events = []
    if not sh and not sl:
        return events
    
    current_bias, _ = structure_bias(work)
    last = work.iloc[-1]
    
    if sh and len(sh) > 0:
        last_high_idx = sh[-1]
        level = float(work["high"].iloc[last_high_idx])
        if float(last["close"]) > level:
            events.append({
                "type": "BOS_UP",
                "level": level,
                "index": last_high_idx,
                "timestamp": int(work["timestamp"].iloc[last_high_idx])
            })
    
    if sl and len(sl) > 0:
        last_low_idx = sl[-1]
        level = float(work["low"].iloc[last_low_idx])
        if float(last["close"]) < level:
            events.append({
                "type": "BOS_DOWN",
                "level": level,
                "index": last_low_idx,
                "timestamp": int(work["timestamp"].iloc[last_low_idx])
            })
    
    if current_bias == "BEARISH":
        for event in events:
            if event["type"] == "BOS_UP":
                events.append({
                    "type": "CHOCH_UP",
                    "level": event["level"],
                    "index": event["index"],
                    "timestamp": event["timestamp"]
                })
    elif current_bias == "BULLISH":
        for event in events:
            if event["type"] == "BOS_DOWN":
                events.append({
                    "type": "CHOCH_DOWN",
                    "level": event["level"],
                    "index": event["index"],
                    "timestamp": event["timestamp"]
                })
    
    return events

def get_support_resistance(df, lookback=100):
    if len(df) < lookback:
        return None, None
    
    recent = df.tail(lookback)
    sh, sl = find_swings(recent)
    
    resistance_levels = []
    support_levels = []
    
    for idx in sh:
        level = float(recent["high"].iloc[idx])
        resistance_levels.append(level)
    
    for idx in sl:
        level = float(recent["low"].iloc[idx])
        support_levels.append(level)
    
    resistance = None
    support = None
    current_price = float(df["close"].iloc[-1])
    
    if resistance_levels:
        resistances_above = [r for r in resistance_levels if r > current_price]
        if resistances_above:
            resistance = min(resistances_above)
    
    if support_levels:
        supports_below = [s for s in support_levels if s < current_price]
        if supports_below:
            support = max(supports_below)
    
    return support, resistance

def analyze_smc(df, htf_df=None):
    if len(df) < 50:
        raise ValueError("SMC analiz üçin azyndan 50 ýapylan candle gerek.")
    
    x = df.copy()
    x["atr"] = atr(x)
    x["rsi"] = rsi(x)
    
    last = x.iloc[-1]
    price = float(last["close"])
    atr_value = float(last["atr"]) if pd.notna(last["atr"]) else 0.0
    rsi_value = float(last["rsi"]) if pd.notna(last["rsi"]) else 50.0
    
    bias, structure = structure_bias(x)
    
    htf_bias = "UNKNOWN"
    htf_structure = None
    if htf_df is not None and len(htf_df) >= 50:
        htf_bias, htf_structure = structure_bias(htf_df)
    
    events = detect_bos_choch(x)
    fvgs = detect_fvg(x)
    obs = detect_order_blocks(x)
    sweep = detect_liquidity_sweeps(x)
    support, resistance = get_support_resistance(x)
    
    latest_bull_fvg = None
    latest_bear_fvg = None
    latest_bull_ob = None
    latest_bear_ob = None
    
    for fvg in reversed(fvgs):
        if fvg["type"] == "BULLISH" and fvg["low"] <= price <= fvg["high"]:
            latest_bull_fvg = fvg
            break
    
    for fvg in reversed(fvgs):
        if fvg["type"] == "BEARISH" and fvg["low"] <= price <= fvg["high"]:
            latest_bear_fvg = fvg
            break
    
    for ob in reversed(obs):
        if ob["type"] == "BULLISH" and ob["low"] <= price <= ob["high"]:
            latest_bull_ob = ob
            break
    
    for ob in reversed(obs):
        if ob["type"] == "BEARISH" and ob["low"] <= price <= ob["high"]:
            latest_bear_ob = ob
            break
    
    if structure:
        range_high = structure["last_swing_high"]
        range_low = structure["last_swing_low"]
    else:
        range_high = float(x["high"].tail(50).max())
        range_low = float(x["low"].tail(50).min())
    
    equilibrium = (range_high + range_low) / 2
    location = "DISCOUNT" if price < equilibrium else "PREMIUM"
    
    bos_up = any(e["type"] in ("BOS_UP", "CHOCH_UP") for e in events)
    bos_down = any(e["type"] in ("BOS_DOWN", "CHOCH_DOWN") for e in events)
    choch_up = any(e["type"] == "CHOCH_UP" for e in events)
    choch_down = any(e["type"] == "CHOCH_DOWN" for e in events)
    
    # ============================================================
    # SCORING SYSTEM
    # ============================================================
    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []
    
    # 1. STRUCTURE (MAIN FACTOR - 3 points)
    if bias == "BULLISH":
        buy_score += 3
        buy_reasons.append("✅ LTF structure bullish")
    elif bias == "BEARISH":
        sell_score += 3
        sell_reasons.append("✅ LTF structure bearish")
    
    if htf_bias == "BULLISH":
        buy_score += 3
        buy_reasons.append("✅ HTF structure bullish")
    elif htf_bias == "BEARISH":
        sell_score += 3
        sell_reasons.append("✅ HTF structure bearish")
    
    # 2. BOS/CHoCH (2 points)
    if bos_up:
        buy_score += 2
        buy_reasons.append("✅ BOS/CHoCH up confirmed")
    if bos_down:
        sell_score += 2
        sell_reasons.append("✅ BOS/CHoCH down confirmed")
    
    if choch_up:
        buy_score += 1
        buy_reasons.append("✅ CHoCH up (trend change)")
    if choch_down:
        sell_score += 1
        sell_reasons.append("✅ CHoCH down (trend change)")
    
    # 3. LIQUIDITY SWEEPS (2 points)
    if sweep["sell_side_sweep"]:
        buy_score += 2
        buy_reasons.append("✅ Sell-side liquidity swept")
    if sweep["buy_side_sweep"]:
        sell_score += 2
        sell_reasons.append("✅ Buy-side liquidity swept")
    
    # 4. FVGs (1 point if price inside)
    if latest_bull_fvg:
        buy_score += 1
        buy_reasons.append(f"✅ Price in bullish FVG")
    if latest_bear_fvg:
        sell_score += 1
        sell_reasons.append(f"✅ Price in bearish FVG")
    
    # 5. ORDER BLOCKS (1 point if price inside)
    if latest_bull_ob:
        buy_score += 1
        buy_reasons.append("✅ Price in bullish OB")
    if latest_bear_ob:
        sell_score += 1
        sell_reasons.append("✅ Price in bearish OB")
    
    # 6. PREMIUM/DISCOUNT (1 point)
    if location == "DISCOUNT":
        buy_score += 1
        buy_reasons.append("✅ Discount area (buy zone)")
    else:
        sell_score += 1
        sell_reasons.append("✅ Premium area (sell zone)")
    
    # 7. SUPPORT/RESISTANCE (1 point)
    if support and price <= support * 1.01:
        buy_score += 1
        buy_reasons.append(f"✅ Near support")
    if resistance and price >= resistance * 0.99:
        sell_score += 1
        sell_reasons.append(f"✅ Near resistance")
    
    # 8. RSI (1 point - filter only)
    if rsi_value < 30:
        buy_score += 1
        buy_reasons.append("✅ RSI oversold (<30)")
    elif rsi_value > 70:
        sell_score += 1
        sell_reasons.append("✅ RSI overbought (>70)")
    
    # 9. HTF conflict penalty (-2)
    if htf_bias == "BEARISH":
        buy_score -= 2
        buy_reasons.append("⚠️ HTF bearish conflict")
    if htf_bias == "BULLISH":
        sell_score -= 2
        sell_reasons.append("⚠️ HTF bullish conflict")
    
    # 10. Momentum check
    if len(x) > 5:
        recent_high = float(x["high"].iloc[-5:].max())
        recent_low = float(x["low"].iloc[-5:].min())
        if price > recent_low + (recent_high - recent_low) * 0.6:
            buy_score += 0.5
            buy_reasons.append("✅ Upward momentum")
        if price < recent_high - (recent_high - recent_low) * 0.6:
            sell_score += 0.5
            sell_reasons.append("✅ Downward momentum")
    
    # ============================================================
    # SIGNAL GENERATION
    # ============================================================
    signal = None
    confidence = max(buy_score, sell_score)
    
    if buy_score >= 7 and buy_score >= sell_score + 2:
        signal = "BUY"
    elif sell_score >= 7 and sell_score >= buy_score + 2:
        signal = "SELL"
    elif buy_score >= 6 and buy_score > sell_score:
        signal = "BUY_WEAK"
    elif sell_score >= 6 and sell_score > buy_score:
        signal = "SELL_WEAK"
    
    # ============================================================
    # RISK MANAGEMENT
    # ============================================================
    sl = None
    tp = None
    rr = None
    
    if signal in ("BUY", "BUY_WEAK"):
        sl_options = []
        if atr_value > 0:
            sl_options.append(price - atr_value * 1.2)
        if latest_bull_fvg:
            sl_options.append(latest_bull_fvg["low"] - atr_value * 0.1)
        if support:
            sl_options.append(support - atr_value * 0.1)
        if sweep["sell_side_level"]:
            sl_options.append(sweep["sell_side_level"] - atr_value * 0.1)
        
        if sl_options:
            sl = max(sl_options) if sl_options else price - atr_value * 1.0
            risk = price - sl
            if risk > 0:
                tp = price + risk * 2.5
                rr = 2.5
    
    elif signal in ("SELL", "SELL_WEAK"):
        sl_options = []
        if atr_value > 0:
            sl_options.append(price + atr_value * 1.2)
        if latest_bear_fvg:
            sl_options.append(latest_bear_fvg["high"] + atr_value * 0.1)
        if resistance:
            sl_options.append(resistance + atr_value * 0.1)
        if sweep["buy_side_level"]:
            sl_options.append(sweep["buy_side_level"] + atr_value * 0.1)
        
        if sl_options:
            sl = min(sl_options) if sl_options else price + atr_value * 1.0
            risk = sl - price
            if risk > 0:
                tp = price - risk * 2.5
                rr = 2.5
    
    if signal and (sl is None or tp is None or
                   (signal in ("BUY", "BUY_WEAK") and not (sl < price < tp)) or
                   (signal in ("SELL", "SELL_WEAK") and not (tp < price < sl))):
        signal = "WAIT"
        sl = tp = rr = None
    
    if signal in ("BUY_WEAK", "SELL_WEAK") and rr is None:
        signal = "WAIT"
    
    return {
        "symbol": bot_state["symbol"],
        "timeframe": bot_state["timeframe"],
        "htf_timeframe": bot_state["htf_timeframe"],
        "price": price,
        "atr": atr_value,
        "rsi": rsi_value,
        "ltf_bias": bias,
        "htf_bias": htf_bias,
        "location": location,
        "equilibrium": equilibrium,
        "range_high": range_high,
        "range_low": range_low,
        "support": support,
        "resistance": resistance,
        "events": events[-8:] if events else [],
        "liquidity": sweep,
        "bullish_fvg": latest_bull_fvg,
        "bearish_fvg": latest_bear_fvg,
        "bullish_ob": latest_bull_ob,
        "bearish_ob": latest_bear_ob,
        "buy_score": round(buy_score, 1),
        "sell_score": round(sell_score, 1),
        "confidence": round(confidence, 1),
        "signal": signal or "WAIT",
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "buy_reasons": buy_reasons,
        "sell_reasons": sell_reasons,
        "candle_time": int(last["timestamp"]),
        "timestamp": datetime.now().isoformat()
    }

# ============================================================
# ORDER EXECUTION
# ============================================================
def set_leverage(symbol, leverage):
    return bybit_request(
        "POST",
        "/v5/position/set-leverage",
        payload={
            "category": "linear",
            "symbol": symbol.upper(),
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }
    )

def execute_trade(symbol, signal, margin_usdt, leverage, sl, tp, price):
    info = get_instrument_info(symbol)
    qty = (margin_usdt * leverage) / price
    qty = round_step(qty, info["qtyStep"])
    
    if qty < info["minOrderQty"]:
        raise RuntimeError(f"Qty {qty} minimum qty {info['minOrderQty']}-dan kiçi.")
    
    qty = min(qty, info["maxOrderQty"])
    sl = round_step(sl, info["tickSize"])
    tp = round_step(tp, info["tickSize"])
    
    side = "Buy" if signal in ("BUY", "BUY_WEAK") else "Sell"
    
    set_leverage(symbol, leverage)
    
    payload = {
        "category": "linear",
        "symbol": symbol.upper(),
        "side": side,
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "GTC",
        "positionIdx": 0,
        "stopLoss": str(sl),
        "takeProfit": str(tp),
        "tpslMode": "Full",
        "slTriggerBy": "MarkPrice",
        "tpTriggerBy": "MarkPrice",
    }
    
    result = bybit_request("POST", "/v5/order/create", payload=payload)
    return result, qty, sl, tp

# ============================================================
# BOT WORKER
# ============================================================
def run_analysis():
    symbol = bot_state["symbol"]
    tf = bot_state["timeframe"]
    htf = bot_state["htf_timeframe"]
    
    ltf_df = get_klines(symbol, tf, 350)
    htf_df = get_klines(symbol, htf, 250)
    
    analysis = analyze_smc(ltf_df, htf_df)
    bot_state["analysis"] = analysis
    bot_state["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bot_state["last_error"] = None
    bot_state["last_signal"] = analysis["signal"]
    
    return analysis

def can_open_new_position(symbol):
    try:
        positions = get_open_positions(symbol)
        return len(positions) == 0
    except Exception as exc:
        log_terminal(f"[POSITION CHECK ERROR] {exc}")
        return False

def bot_worker():
    global bot_thread
    
    log_terminal("[SYSTEM] SMC bot başlady.")
    bot_state["running"] = True
    
    while bot_state["running"]:
        try:
            try:
                bot_state["positions"] = get_open_positions(bot_state["symbol"])
            except Exception as e:
                bot_state["positions"] = []
            
            analysis = run_analysis()
            
            log_terminal(
                f"[SMC] {analysis['symbol']} {analysis['timeframe']} | "
                f"LTF={analysis['ltf_bias']} HTF={analysis['htf_bias']} | "
                f"BUY={analysis['buy_score']} SELL={analysis['sell_score']} | "
                f"signal={analysis['signal']}"
            )
            
            if analysis["signal"] in ("BUY", "BUY_WEAK", "SELL", "SELL_WEAK"):
                symbol = bot_state["symbol"]
                margin = bot_state["amount"]
                leverage = bot_state["leverage"]
                
                if not can_open_new_position(symbol):
                    log_terminal("[SKIP] Açyk position bar. Täze position açylmady.")
                elif analysis["sl"] and analysis["tp"]:
                    try:
                        result, qty, sl, tp = execute_trade(
                            symbol=symbol,
                            signal=analysis["signal"],
                            margin_usdt=margin,
                            leverage=leverage,
                            sl=analysis["sl"],
                            tp=analysis["tp"],
                            price=analysis["price"]
                        )
                        
                        log_terminal(
                            f"[ORDER] {analysis['signal']} qty={qty} entry≈{analysis['price']} "
                            f"SL={sl} TP={tp}"
                        )
                        
                        save_trade({
                            "symbol": symbol,
                            "side": analysis["signal"],
                            "entry_price": analysis["price"],
                            "sl": sl,
                            "tp": tp,
                            "signal_score": analysis["confidence"],
                            "reason": "; ".join(
                                analysis["buy_reasons"] if "BUY" in analysis["signal"] 
                                else analysis["sell_reasons"]
                            )
                        })
                        
                        time.sleep(60)
                    except Exception as e:
                        log_terminal(f"[ORDER ERROR] {e}")
                        bot_state["last_error"] = str(e)
            
            try:
                bot_state["positions"] = get_open_positions(bot_state["symbol"])
            except Exception as e:
                bot_state["positions"] = []
            
        except Exception as exc:
            bot_state["last_error"] = str(exc)
            log_terminal(f"[BOT ERROR] {exc}")
        
        for _ in range(5):
            if not bot_state["running"]:
                break
            time.sleep(1.5)
    
    bot_state["running"] = False
    log_terminal("[SYSTEM] SMC bot saklandy.")
    bot_thread = None

# ============================================================
# DATABASE HELPERS
# ============================================================
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
    
    symbol = str(data.get("symbol", "BTCUSDT")).upper().strip()
    timeframe = str(data.get("timeframe", "15"))
    htf_timeframe = str(data.get("htf_timeframe", "60"))
    
    try:
        leverage = int(data.get("leverage", 5))
        amount = float(data.get("amount", 10))
    except (TypeError, ValueError):
        return jsonify({
            "status": "error",
            "message": "Leverage we amount san bolmaly."
        }), 400
    
    allowed_tf = {"1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"}
    
    if timeframe not in allowed_tf or htf_timeframe not in allowed_tf:
        return jsonify({
            "status": "error",
            "message": "Timeframe nädogry."
        }), 400
    
    if leverage < 1 or leverage > 100:
        return jsonify({
            "status": "error",
            "message": "Leverage 1-100 aralygynda bolmaly."
        }), 400
    
    if amount <= 0:
        return jsonify({
            "status": "error",
            "message": "Amount 0-dan uly bolmaly."
        }), 400
    
    with bot_lock:
        bot_state["symbol"] = symbol
        bot_state["timeframe"] = timeframe
        bot_state["htf_timeframe"] = htf_timeframe
        bot_state["leverage"] = leverage
        bot_state["amount"] = amount
        bot_state["last_error"] = None
        
        if bot_state["running"]:
            return jsonify({
                "status": "error",
                "message": "Bot eýýäm işleýär."
            }), 409
        
        bot_state["running"] = True
        bot_thread = threading.Thread(target=bot_worker, daemon=True)
        bot_thread.start()
    
    return jsonify({
        "status": "success",
        "message": "SMC bot başlady."
    })

@app.route("/api/stop", methods=["POST"])
def stop_bot():
    bot_state["running"] = False
    log_terminal("[SYSTEM] Stop signal berildi.")
    return jsonify({
        "status": "success",
        "message": "Bot saklanýar."
    })

@app.route("/api/analyze", methods=["POST"])
def manual_analyze():
    try:
        data = request.get_json(silent=True) or {}
        if data.get("symbol"):
            bot_state["symbol"] = str(data["symbol"]).upper().strip()
        if data.get("timeframe"):
            bot_state["timeframe"] = str(data["timeframe"])
        if data.get("htf_timeframe"):
            bot_state["htf_timeframe"] = str(data["htf_timeframe"])
        
        analysis = run_analysis()
        return jsonify({
            "status": "success",
            "analysis": analysis
        })
    except Exception as exc:
        bot_state["last_error"] = str(exc)
        return jsonify({
            "status": "error",
            "message": str(exc)
        }), 500

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

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "api_configured": bool(BYBIT_API_KEY and BYBIT_API_SECRET),
        "mode": "LIVE" if "testnet" not in BYBIT_URL.lower() else "TESTNET",
        "running": bot_state.get("running", False)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)