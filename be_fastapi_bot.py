"""FastAPI-based autonomous trading bot using Alpaca API.

This script fetches historical and realtime market data, applies a simple
moving-average strategy, and manages risk via position limits and stop-losses.
It exposes a FastAPI application so it can run persistently on a server and
provides logging and optional email alerts.

Environment variables:
- APCA_API_KEY_ID / APCA_API_SECRET_KEY: Alpaca credentials
- APCA_BASE_URL: Optional; defaults to paper trading URL
- TRADE_SYMBOL: Ticker to trade (default SPY)
- MAX_SHARES: Maximum number of shares to hold (default 10)
- STOP_LOSS_PCT: Stop-loss percentage (default 0.02)
- POLL_INTERVAL: Seconds between strategy evaluations (default 60)
- ALERT_EMAIL, SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD: optional
  settings for email alerts
"""

from __future__ import annotations

import asyncio
import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Optional

from fastapi import FastAPI

# The newer `alpaca-py` library is used instead of the deprecated
# `alpaca-trade-api`.  It provides separate clients for historical data and
# for trading operations.
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Position
from alpaca.trading.requests import MarketOrderRequest

logger = logging.getLogger("autotrader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_BASE_URL", "https://paper-api.alpaca.markets")
PAPER = "paper" in BASE_URL

SYMBOL = os.getenv("TRADE_SYMBOL", "SPY")
MAX_SHARES = int(os.getenv("MAX_SHARES", "10"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "60"))

app = FastAPI()

# Clients are created lazily because environment variables may not be set at
# import time when the module is used by uvicorn.
trading_client: Optional[TradingClient] = None
data_client: Optional[StockHistoricalDataClient] = None


def send_alert(subject: str, body: str) -> None:
    """Send an email alert if SMTP settings are configured."""
    to_addr = os.getenv("ALERT_EMAIL")
    smtp_server = os.getenv("SMTP_SERVER")
    if not (to_addr and smtp_server):
        logger.warning("Alert requested but email settings not configured: %s", subject)
        return

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.getenv("SMTP_USER", "alert@example.com")
    msg["To"] = to_addr
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_server, int(os.getenv("SMTP_PORT", "587"))) as server:
            server.starttls()
            user = os.getenv("SMTP_USER")
            password = os.getenv("SMTP_PASSWORD")
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        logger.info("Sent alert: %s", subject)
    except Exception as exc:  # pragma: no cover - best effort
        logger.exception("Failed to send alert: %s", exc)


def get_clients() -> tuple[TradingClient, StockHistoricalDataClient]:
    """Return trading and data clients, creating them on first use."""
    global trading_client, data_client
    if trading_client is None:
        trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
    if data_client is None:
        data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    return trading_client, data_client


def get_price_data() -> tuple[list[float], float]:
    """Fetch recent bars and return their closes and the latest price."""
    trading_client, data_client = get_clients()
    request = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Minute,
        limit=50,
    )
    bars = data_client.get_stock_bars(request)
    symbol_bars = bars[SYMBOL]
    closes = [bar.close for bar in symbol_bars]
    return closes, closes[-1]


def evaluate_position(current_price: float) -> Optional[Position]:
    trading_client, _ = get_clients()
    positions = {p.symbol: p for p in trading_client.get_all_positions()}
    return positions.get(SYMBOL)


def run_strategy() -> None:
    """Execute a single iteration of the trading strategy."""
    closes, price = get_price_data()
    avg_price = sum(closes) / len(closes)

    trading_client, _ = get_clients()
    position = evaluate_position(price)

    if position is None and price > avg_price:
        qty = min(1, MAX_SHARES)
        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        trading_client.submit_order(order_data=order)
        logger.info("Bought %s %s at %.2f", qty, SYMBOL, price)
        send_alert("Bought position", f"Bought {qty} {SYMBOL} at {price:.2f}")
        return

    if position is not None:
        qty = int(position.qty)
        entry = float(position.avg_entry_price)
        if price < entry * (1 - STOP_LOSS_PCT):
            order = MarketOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            trading_client.submit_order(order_data=order)
            logger.info("Stop loss triggered; sold %s %s", qty, SYMBOL)
            send_alert("Stop loss", f"Sold {qty} {SYMBOL} via stop loss")
            return
        if price < avg_price:
            order = MarketOrderRequest(
                symbol=SYMBOL,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            trading_client.submit_order(order_data=order)
            logger.info("Price below average; sold %s %s", qty, SYMBOL)
            send_alert("Exit position", f"Sold {qty} {SYMBOL} at {price:.2f}")


async def trader_loop() -> None:
    """Continuously run the strategy at the configured interval."""
    while True:
        try:
            run_strategy()
        except Exception as exc:  # pragma: no cover - unexpected errors
            logger.exception("Strategy iteration failed: %s", exc)
            send_alert("Strategy error", str(exc))
        await asyncio.sleep(POLL_INTERVAL)


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(trader_loop())
    logger.info("Autotrader started for %s", SYMBOL)


@app.get("/status")
async def status() -> dict[str, str]:
    return {"status": "ok", "symbol": SYMBOL}


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    import uvicorn

    uvicorn.run("be_fastapi_bot:app", host="0.0.0.0", port=8000)

