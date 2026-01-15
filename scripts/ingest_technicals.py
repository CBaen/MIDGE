#!/usr/bin/env python3
"""
ingest_technicals.py - Technical Indicator Signal Worker

Fetches price data, calculates ALL technical indicators, generates signals,
and stores them to Qdrant for MIDGE to find patterns.

Each indicator generates its own signal with direction and confidence.
Pattern emerges when multiple signals align.

Usage:
    python scripts/ingest_technicals.py                    # Run once
    python scripts/ingest_technicals.py --continuous       # Run forever
    python scripts/ingest_technicals.py --interval 600     # Every 10 minutes
"""

import sys
import time
import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading.technical.indicators import TechnicalIndicators, OHLCV

QDRANT_URL = "http://localhost:6333"
COLLECTION = "midge_signals"
OLLAMA_URL = "http://localhost:11434"

# Symbols to analyze
TRACKED_SYMBOLS = [
    "LMT", "RTX", "BA", "NOC", "GD",           # Defense
    "MSFT", "AAPL", "GOOGL", "AMZN", "NVDA",   # Tech
    "JPM", "GS", "BAC",                         # Finance
    "SPY", "QQQ",                               # ETFs
]


@dataclass
class TechnicalSignal:
    """A signal generated from technical analysis."""
    signal_id: str
    signal_source: str      # e.g., "rsi", "macd", "bollinger"
    signal_type: str        # "technical"
    symbol: str
    timestamp: str
    direction: str          # "bullish" or "bearish"
    confidence: float
    indicator_value: float  # The actual indicator reading
    threshold: str          # What triggered the signal
    details: dict


def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("embedding")
    except:
        pass
    return None


def store_signal(signal: TechnicalSignal) -> bool:
    """Store a signal to Qdrant."""
    text = f"Technical {signal.signal_source} signal: {signal.symbol} {signal.direction} "
    text += f"(value: {signal.indicator_value:.2f}, threshold: {signal.threshold})"

    embedding = get_embedding(text)
    if not embedding:
        return False

    payload = {
        "signal_id": signal.signal_id,
        "signal_source": signal.signal_source,
        "signal_type": signal.signal_type,
        "symbol": signal.symbol,
        "timestamp": signal.timestamp,
        "direction": signal.direction,
        "confidence": signal.confidence,
        "indicator_value": signal.indicator_value,
        "threshold": signal.threshold,
        "details": signal.details,
        "text": text,
        "ingested_at": datetime.now().isoformat(),
        "decayed": False
    }

    try:
        response = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION}/points",
            json={
                "points": [{
                    "id": abs(hash(signal.signal_id)) % (10**18),
                    "vector": embedding,
                    "payload": payload
                }]
            }
        )
        return response.status_code in (200, 201)
    except Exception as e:
        print(f"  [ERROR] Failed to store: {e}")
        return False


def fetch_ohlcv(symbol: str, days: int = 100) -> List[OHLCV]:
    """Fetch OHLCV data using yfinance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{days}d")

        data = []
        for idx, row in hist.iterrows():
            data.append(OHLCV(
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume'],
                timestamp=idx.isoformat()
            ))
        return data
    except Exception as e:
        print(f"  [ERROR] Failed to fetch {symbol}: {e}")
        return []


def generate_signals_from_indicators(symbol: str, ohlcv: List[OHLCV]) -> List[TechnicalSignal]:
    """Generate trading signals from technical indicators."""
    if len(ohlcv) < 50:  # Need enough data for indicators
        return []

    closes = [bar.close for bar in ohlcv]
    highs = [bar.high for bar in ohlcv]
    lows = [bar.low for bar in ohlcv]
    volumes = [bar.volume for bar in ohlcv]

    ti = TechnicalIndicators(closes, highs, lows, volumes)
    signals = []
    timestamp = datetime.now().isoformat()

    # Latest values (most recent)
    latest_idx = -1

    # =========================================================================
    # RSI SIGNALS
    # =========================================================================
    rsi_values = ti.rsi()
    if rsi_values[latest_idx] is not None:
        rsi_val = rsi_values[latest_idx]
        if rsi_val < 30:
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_rsi_{timestamp}".encode()).hexdigest(),
                signal_source="rsi",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction="bullish",
                confidence=0.7 + (30 - rsi_val) / 100,  # More oversold = higher confidence
                indicator_value=rsi_val,
                threshold="< 30 (oversold)",
                details={"interpretation": "Oversold - potential bounce"}
            ))
        elif rsi_val > 70:
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_rsi_{timestamp}".encode()).hexdigest(),
                signal_source="rsi",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction="bearish",
                confidence=0.7 + (rsi_val - 70) / 100,
                indicator_value=rsi_val,
                threshold="> 70 (overbought)",
                details={"interpretation": "Overbought - potential pullback"}
            ))

    # =========================================================================
    # MACD SIGNALS
    # =========================================================================
    macd_line, signal_line, histogram = ti.macd()
    if macd_line[latest_idx] is not None and signal_line[latest_idx] is not None:
        macd_val = macd_line[latest_idx]
        signal_val = signal_line[latest_idx]
        hist_val = histogram[latest_idx] if histogram[latest_idx] else 0

        # Check for crossover (compare current and previous)
        if len(macd_line) > 2 and macd_line[-2] is not None and signal_line[-2] is not None:
            prev_diff = macd_line[-2] - signal_line[-2]
            curr_diff = macd_val - signal_val

            # Bullish crossover
            if prev_diff < 0 and curr_diff > 0:
                signals.append(TechnicalSignal(
                    signal_id=hashlib.md5(f"{symbol}_macd_{timestamp}".encode()).hexdigest(),
                    signal_source="macd",
                    signal_type="technical",
                    symbol=symbol,
                    timestamp=timestamp,
                    direction="bullish",
                    confidence=0.75,
                    indicator_value=macd_val,
                    threshold="MACD crossed above signal",
                    details={"histogram": hist_val, "interpretation": "Bullish momentum crossover"}
                ))
            # Bearish crossover
            elif prev_diff > 0 and curr_diff < 0:
                signals.append(TechnicalSignal(
                    signal_id=hashlib.md5(f"{symbol}_macd_{timestamp}".encode()).hexdigest(),
                    signal_source="macd",
                    signal_type="technical",
                    symbol=symbol,
                    timestamp=timestamp,
                    direction="bearish",
                    confidence=0.75,
                    indicator_value=macd_val,
                    threshold="MACD crossed below signal",
                    details={"histogram": hist_val, "interpretation": "Bearish momentum crossover"}
                ))

    # =========================================================================
    # BOLLINGER BAND SIGNALS
    # =========================================================================
    upper_bb, middle_bb, lower_bb = ti.bollinger_bands()
    if upper_bb[latest_idx] is not None:
        current_price = closes[latest_idx]
        upper = upper_bb[latest_idx]
        lower = lower_bb[latest_idx]

        # Price at lower band = potential bounce
        if current_price <= lower * 1.01:  # Within 1% of lower band
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_bb_{timestamp}".encode()).hexdigest(),
                signal_source="bollinger",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction="bullish",
                confidence=0.65,
                indicator_value=current_price,
                threshold=f"Price at lower band ({lower:.2f})",
                details={"upper": upper, "lower": lower, "interpretation": "Potential bounce from support"}
            ))
        # Price at upper band = potential pullback
        elif current_price >= upper * 0.99:
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_bb_{timestamp}".encode()).hexdigest(),
                signal_source="bollinger",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction="bearish",
                confidence=0.65,
                indicator_value=current_price,
                threshold=f"Price at upper band ({upper:.2f})",
                details={"upper": upper, "lower": lower, "interpretation": "Potential pullback from resistance"}
            ))

    # =========================================================================
    # STOCHASTIC SIGNALS
    # =========================================================================
    stoch_k, stoch_d = ti.stochastic()
    if stoch_k[latest_idx] is not None:
        k_val = stoch_k[latest_idx]

        if k_val < 20:
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_stoch_{timestamp}".encode()).hexdigest(),
                signal_source="stochastic",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction="bullish",
                confidence=0.65,
                indicator_value=k_val,
                threshold="< 20 (oversold)",
                details={"interpretation": "Stochastic oversold"}
            ))
        elif k_val > 80:
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_stoch_{timestamp}".encode()).hexdigest(),
                signal_source="stochastic",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction="bearish",
                confidence=0.65,
                indicator_value=k_val,
                threshold="> 80 (overbought)",
                details={"interpretation": "Stochastic overbought"}
            ))

    # =========================================================================
    # ADX TREND STRENGTH
    # =========================================================================
    adx_val, plus_di, minus_di = ti.adx()
    if adx_val[latest_idx] is not None and plus_di[latest_idx] is not None:
        adx = adx_val[latest_idx]
        pdi = plus_di[latest_idx]
        mdi = minus_di[latest_idx]

        # Strong trend with direction
        if adx > 25:
            direction = "bullish" if pdi > mdi else "bearish"
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_adx_{timestamp}".encode()).hexdigest(),
                signal_source="adx",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction=direction,
                confidence=0.6 + (adx - 25) / 100,
                indicator_value=adx,
                threshold=f"ADX > 25 (strong trend), +DI={pdi:.1f}, -DI={mdi:.1f}",
                details={"+DI": pdi, "-DI": mdi, "interpretation": f"Strong {direction} trend"}
            ))

    # =========================================================================
    # WILLIAMS %R SIGNALS
    # =========================================================================
    williams = ti.williams_r()
    if williams[latest_idx] is not None:
        wr_val = williams[latest_idx]

        if wr_val < -80:
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_williams_{timestamp}".encode()).hexdigest(),
                signal_source="williams_r",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction="bullish",
                confidence=0.6,
                indicator_value=wr_val,
                threshold="< -80 (oversold)",
                details={"interpretation": "Williams %R oversold"}
            ))
        elif wr_val > -20:
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_williams_{timestamp}".encode()).hexdigest(),
                signal_source="williams_r",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction="bearish",
                confidence=0.6,
                indicator_value=wr_val,
                threshold="> -20 (overbought)",
                details={"interpretation": "Williams %R overbought"}
            ))

    # =========================================================================
    # CCI SIGNALS
    # =========================================================================
    cci_values = ti.cci()
    if cci_values[latest_idx] is not None:
        cci_val = cci_values[latest_idx]

        if cci_val < -100:
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_cci_{timestamp}".encode()).hexdigest(),
                signal_source="cci",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction="bullish",
                confidence=0.6,
                indicator_value=cci_val,
                threshold="< -100 (oversold)",
                details={"interpretation": "CCI indicates potential reversal up"}
            ))
        elif cci_val > 100:
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_cci_{timestamp}".encode()).hexdigest(),
                signal_source="cci",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction="bearish",
                confidence=0.6,
                indicator_value=cci_val,
                threshold="> 100 (overbought)",
                details={"interpretation": "CCI indicates potential reversal down"}
            ))

    # =========================================================================
    # CHAIKIN MONEY FLOW
    # =========================================================================
    cmf_values = ti.chaikin_money_flow()
    if cmf_values[latest_idx] is not None:
        cmf_val = cmf_values[latest_idx]

        if cmf_val > 0.25:
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_cmf_{timestamp}".encode()).hexdigest(),
                signal_source="cmf",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction="bullish",
                confidence=0.7,
                indicator_value=cmf_val,
                threshold="> 0.25 (strong buying)",
                details={"interpretation": "Strong accumulation"}
            ))
        elif cmf_val < -0.25:
            signals.append(TechnicalSignal(
                signal_id=hashlib.md5(f"{symbol}_cmf_{timestamp}".encode()).hexdigest(),
                signal_source="cmf",
                signal_type="technical",
                symbol=symbol,
                timestamp=timestamp,
                direction="bearish",
                confidence=0.7,
                indicator_value=cmf_val,
                threshold="< -0.25 (strong selling)",
                details={"interpretation": "Strong distribution"}
            ))

    # =========================================================================
    # SMA CROSSOVERS (Golden Cross / Death Cross)
    # =========================================================================
    sma_50 = ti.sma(50)
    sma_200 = ti.sma(200)

    if sma_50[latest_idx] is not None and sma_200[latest_idx] is not None:
        if len(sma_50) > 2 and sma_50[-2] is not None and sma_200[-2] is not None:
            prev_50 = sma_50[-2]
            prev_200 = sma_200[-2]
            curr_50 = sma_50[latest_idx]
            curr_200 = sma_200[latest_idx]

            # Golden Cross (50 crosses above 200)
            if prev_50 < prev_200 and curr_50 > curr_200:
                signals.append(TechnicalSignal(
                    signal_id=hashlib.md5(f"{symbol}_goldencross_{timestamp}".encode()).hexdigest(),
                    signal_source="sma_cross",
                    signal_type="technical",
                    symbol=symbol,
                    timestamp=timestamp,
                    direction="bullish",
                    confidence=0.8,  # Golden cross is strong signal
                    indicator_value=curr_50,
                    threshold="Golden Cross (50 SMA > 200 SMA)",
                    details={"sma_50": curr_50, "sma_200": curr_200, "interpretation": "Major bullish trend signal"}
                ))
            # Death Cross (50 crosses below 200)
            elif prev_50 > prev_200 and curr_50 < curr_200:
                signals.append(TechnicalSignal(
                    signal_id=hashlib.md5(f"{symbol}_deathcross_{timestamp}".encode()).hexdigest(),
                    signal_source="sma_cross",
                    signal_type="technical",
                    symbol=symbol,
                    timestamp=timestamp,
                    direction="bearish",
                    confidence=0.8,
                    indicator_value=curr_50,
                    threshold="Death Cross (50 SMA < 200 SMA)",
                    details={"sma_50": curr_50, "sma_200": curr_200, "interpretation": "Major bearish trend signal"}
                ))

    return signals


def run_ingestion_cycle():
    """Run one complete technical analysis cycle."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting technical analysis cycle...")

    total_signals = 0
    total_symbols = len(TRACKED_SYMBOLS)

    for i, symbol in enumerate(TRACKED_SYMBOLS, 1):
        print(f"  [{i}/{total_symbols}] Analyzing {symbol}...", end=" ")

        # Fetch price data
        ohlcv = fetch_ohlcv(symbol, days=100)
        if not ohlcv:
            print("no data")
            continue

        # Generate signals
        signals = generate_signals_from_indicators(symbol, ohlcv)

        # Store each signal
        stored = 0
        for signal in signals:
            if store_signal(signal):
                stored += 1

        print(f"{len(signals)} signals, {stored} stored")
        total_signals += stored

        time.sleep(0.5)  # Brief pause between symbols

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle complete. Stored {total_signals} signals.")
    return total_signals


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Technical Analysis Signal Worker")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between cycles (default: 600 = 10 min)")
    parser.add_argument("--symbols", nargs="+", help="Override symbols to track")

    args = parser.parse_args()

    if args.symbols:
        global TRACKED_SYMBOLS
        TRACKED_SYMBOLS = args.symbols

    print("="*60)
    print("MIDGE Technical Analysis Worker")
    print("="*60)
    print(f"Qdrant: {QDRANT_URL}")
    print(f"Collection: {COLLECTION}")
    print(f"Tracking {len(TRACKED_SYMBOLS)} symbols")
    print(f"Indicators: RSI, MACD, Bollinger, Stochastic, ADX, Williams%R, CCI, CMF, SMA Cross")
    print(f"Mode: {'Continuous' if args.continuous else 'Single run'}")
    if args.continuous:
        print(f"Interval: {args.interval} seconds ({args.interval/60:.1f} minutes)")

    if args.continuous:
        print("\nPress Ctrl+C to stop.\n")
        try:
            while True:
                run_ingestion_cycle()
                print(f"Sleeping {args.interval}s until next cycle...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped by user.")
    else:
        run_ingestion_cycle()


if __name__ == "__main__":
    main()
