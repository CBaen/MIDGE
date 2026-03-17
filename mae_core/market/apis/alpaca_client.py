"""
alpaca_client.py — Alpaca Trading Bridge

Connects MIDGE's convergence alerts to live/paper trading via Alpaca.
Paper trading by default. Bracket orders with stop-loss for risk management.
Kelly position sizing from the existing kelly_position_sizer.

Requires: ALPACA_API_KEY + ALPACA_SECRET_KEY in .env
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        GetOrdersRequest,
        TakeProfitRequest,
        StopLossRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        OrderClass,
        QueryOrderStatus,
    )
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.debug("alpaca-py not installed — AlpacaClient disabled")


@dataclass
class TradeResult:
    """Result of a trade submission."""
    order_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    status: str
    submitted_at: datetime
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current position snapshot."""
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    side: str


@dataclass
class AccountInfo:
    """Account snapshot."""
    equity: float
    buying_power: float
    cash: float
    portfolio_value: float
    pattern_day_trader: bool
    daytrade_count: int


class AlpacaClient:
    """
    MIDGE's execution bridge to Alpaca Markets.

    Paper trading by default. Supports:
    - Market orders (immediate execution)
    - Bracket orders (entry + take-profit + stop-loss)
    - Position tracking
    - Account info
    - Order history
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        self._paper = paper
        self._client: Optional[Any] = None

        if not ALPACA_AVAILABLE:
            logger.warning("AlpacaClient: alpaca-py not installed")
            return

        if not api_key or not secret_key:
            logger.info("AlpacaClient: no API keys — disabled")
            return

        try:
            self._client = TradingClient(api_key, secret_key, paper=paper)
            mode = "PAPER" if paper else "LIVE"
            logger.info("AlpacaClient: connected (%s mode)", mode)
        except Exception as e:
            logger.error("AlpacaClient: connection failed: %s", e)

    @property
    def connected(self) -> bool:
        return self._client is not None

    def get_account(self) -> Optional[AccountInfo]:
        """Get account balance, equity, buying power."""
        if not self._client:
            return None
        try:
            acc = self._client.get_account()
            return AccountInfo(
                equity=float(acc.equity),
                buying_power=float(acc.buying_power),
                cash=float(acc.cash),
                portfolio_value=float(acc.portfolio_value),
                pattern_day_trader=acc.pattern_day_trader,
                daytrade_count=int(acc.daytrade_count),
            )
        except Exception as e:
            logger.error("AlpacaClient.get_account failed: %s", e)
            return None

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        if not self._client:
            return []
        try:
            positions = self._client.get_all_positions()
            return [
                Position(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_plpc=float(p.unrealized_plpc),
                    side=str(p.side),
                )
                for p in positions
            ]
        except Exception as e:
            logger.error("AlpacaClient.get_positions failed: %s", e)
            return []

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str = "buy",
        take_profit_price: Optional[float] = None,
        stop_loss_price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[TradeResult]:
        """Submit a market order, optionally as a bracket with TP/SL.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            qty: Number of shares (supports fractional).
            side: "buy" or "sell".
            take_profit_price: If set, creates bracket order with take-profit.
            stop_loss_price: If set, creates bracket order with stop-loss.
            metadata: Extra context from convergence alert.

        Returns:
            TradeResult on success, None on failure.
        """
        if not self._client:
            logger.warning("AlpacaClient: not connected — order rejected")
            return None

        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        try:
            # Fractional qty requires DAY time-in-force. Bracket orders
            # require GTC. Use bracket only when qty is whole shares.
            _is_fractional = qty != int(qty)
            _is_crypto = "/" in symbol  # BTC/USD format

            if take_profit_price and stop_loss_price and not _is_fractional and not _is_crypto:
                # Bracket order — entry + TP + SL (whole shares, non-crypto only)
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.GTC,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=take_profit_price),
                    stop_loss=StopLossRequest(stop_price=stop_loss_price),
                )
            else:
                # Simple market order
                order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                )

            order = self._client.submit_order(order_data=order_data)

            result = TradeResult(
                order_id=str(order.id),
                symbol=symbol,
                side=side.lower(),
                qty=qty,
                order_type="bracket" if take_profit_price else "market",
                status=str(order.status),
                submitted_at=datetime.now(),
                take_profit=take_profit_price,
                stop_loss=stop_loss_price,
                metadata=metadata or {},
            )
            logger.info(
                "AlpacaClient: %s %s %s shares of %s (order %s)",
                "PAPER" if self._paper else "LIVE",
                side.upper(), qty, symbol, order.id,
            )
            return result

        except Exception as e:
            logger.error("AlpacaClient.submit_market_order failed: %s", e)
            return None

    def get_orders(self, status: str = "open") -> list:
        """Get orders by status (open, closed, all)."""
        if not self._client:
            return []
        try:
            status_map = {
                "open": QueryOrderStatus.OPEN,
                "closed": QueryOrderStatus.CLOSED,
                "all": QueryOrderStatus.ALL,
            }
            params = GetOrdersRequest(status=status_map.get(status, QueryOrderStatus.OPEN))
            return self._client.get_orders(filter=params)
        except Exception as e:
            logger.error("AlpacaClient.get_orders failed: %s", e)
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        if not self._client:
            return False
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info("AlpacaClient: cancelled order %s", order_id)
            return True
        except Exception as e:
            logger.error("AlpacaClient.cancel_order failed: %s", e)
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        if not self._client:
            return False
        try:
            self._client.cancel_orders()
            logger.info("AlpacaClient: cancelled all open orders")
            return True
        except Exception as e:
            logger.error("AlpacaClient.cancel_all_orders failed: %s", e)
            return False

    def close_position(self, symbol: str) -> bool:
        """Close an entire position for a symbol."""
        if not self._client:
            return False
        try:
            self._client.close_position(symbol)
            logger.info("AlpacaClient: closed position in %s", symbol)
            return True
        except Exception as e:
            logger.error("AlpacaClient.close_position failed: %s", e)
            return False
