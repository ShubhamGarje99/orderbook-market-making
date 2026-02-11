"""
Limit Order Book Data Structure
================================
Core data structure for representing and manipulating a Limit Order Book (LOB).
Supports adding/cancelling orders, querying mid-price, spread, and depth.
"""

from typing import Dict, List, Optional, Tuple
import time


class LimitOrderBook:
    """
    A Limit Order Book maintaining sorted bid and ask price levels.

    Attributes:
        bids (dict): Mapping of price → quantity for buy orders (descending).
        asks (dict): Mapping of price → quantity for sell orders (ascending).
        timestamp (float | None): Unix timestamp of the last update.
    """

    def __init__(self):
        self.bids: Dict[float, float] = {}   # price -> quantity
        self.asks: Dict[float, float] = {}   # price -> quantity
        self.timestamp: Optional[float] = None

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def add_order(self, side: str, price: float, quantity: float) -> None:
        """
        Add an order to the book.

        Args:
            side: 'bid' or 'ask'.
            price: Limit price of the order.
            quantity: Size of the order (must be > 0).

        Raises:
            ValueError: If side is invalid or quantity <= 0.
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        book = self._get_book(side)
        book[price] = book.get(price, 0.0) + quantity
        self.timestamp = time.time()

    def cancel_order(self, side: str, price: float, quantity: float) -> None:
        """
        Cancel (reduce) an order at a given price level.

        If the remaining quantity at the level drops to zero or below, the
        level is removed entirely.

        Args:
            side: 'bid' or 'ask'.
            price: Price level to cancel from.
            quantity: Amount to cancel.

        Raises:
            ValueError: If side is invalid or price level does not exist.
        """
        book = self._get_book(side)
        if price not in book:
            raise ValueError(f"No {side} orders at price {price}")

        book[price] -= quantity
        if book[price] <= 0:
            del book[price]
        self.timestamp = time.time()

    # ------------------------------------------------------------------
    # Market data queries
    # ------------------------------------------------------------------

    def get_best_bid(self) -> Optional[float]:
        """Return the highest bid price, or None if no bids."""
        return max(self.bids.keys()) if self.bids else None

    def get_best_ask(self) -> Optional[float]:
        """Return the lowest ask price, or None if no asks."""
        return min(self.asks.keys()) if self.asks else None

    def get_mid_price(self) -> Optional[float]:
        """
        Calculate the mid-price: (best_bid + best_ask) / 2.

        Returns:
            Mid-price as a float, or None if either side is empty.
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0

    def get_spread(self) -> Optional[float]:
        """
        Calculate the bid-ask spread: best_ask - best_bid.

        Returns:
            Spread as a float, or None if either side is empty.
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid is None or best_ask is None:
            return None
        return best_ask - best_bid

    def get_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """
        Return the top *levels* price levels on each side.

        Args:
            levels: Number of price levels to return per side.

        Returns:
            Dictionary with keys ``'bids'`` and ``'asks'``, each containing a
            list of ``(price, quantity)`` tuples sorted best-to-worst.
        """
        sorted_bids = sorted(self.bids.items(), key=lambda x: -x[0])[:levels]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:levels]
        return {"bids": sorted_bids, "asks": sorted_asks}

    def get_total_bid_volume(self, levels: Optional[int] = None) -> float:
        """Total quantity on the bid side (optionally limited to top *levels*)."""
        if levels is None:
            return sum(self.bids.values())
        depth = self.get_depth(levels)
        return sum(qty for _, qty in depth["bids"])

    def get_total_ask_volume(self, levels: Optional[int] = None) -> float:
        """Total quantity on the ask side (optionally limited to top *levels*)."""
        if levels is None:
            return sum(self.asks.values())
        depth = self.get_depth(levels)
        return sum(qty for _, qty in depth["asks"])

    def get_snapshot(self) -> dict:
        """
        Return a full snapshot of the current book state.

        Returns:
            Dictionary containing timestamp, bids, asks, mid_price, spread,
            best_bid, best_ask, and depth (top 10 levels).
        """
        depth = self.get_depth(levels=10)
        return {
            "timestamp": self.timestamp,
            "bids": dict(self.bids),
            "asks": dict(self.asks),
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "mid_price": self.get_mid_price(),
            "spread": self.get_spread(),
            "depth": depth,
            "bid_volume": self.get_total_bid_volume(),
            "ask_volume": self.get_total_ask_volume(),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the entire order book."""
        self.bids.clear()
        self.asks.clear()
        self.timestamp = None

    def _get_book(self, side: str) -> Dict[float, float]:
        """Return the correct side dict, raising on invalid input."""
        side = side.lower()
        if side in ("bid", "bids", "buy"):
            return self.bids
        elif side in ("ask", "asks", "sell"):
            return self.asks
        else:
            raise ValueError(f"Invalid side '{side}'. Use 'bid' or 'ask'.")

    def __repr__(self) -> str:
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        spread = self.get_spread()
        return (
            f"LimitOrderBook(best_bid={best_bid}, best_ask={best_ask}, "
            f"spread={spread}, bid_levels={len(self.bids)}, "
            f"ask_levels={len(self.asks)})"
        )
