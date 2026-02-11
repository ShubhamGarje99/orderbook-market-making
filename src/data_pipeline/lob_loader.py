"""
LOB Data Loader & Synthetic Generator
=======================================
Functions to load real tick/LOB data from CSV files and to generate
synthetic order book data using a zero-intelligence model when real
data is unavailable.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Optional

from src.data_pipeline.lob_structure import LimitOrderBook


# ------------------------------------------------------------------
# CSV / tick-data loader
# ------------------------------------------------------------------

def load_nse_data(filepath: str) -> pd.DataFrame:
    """
    Load NSE (or generic) tick data from a CSV file and return a
    DataFrame of LOB snapshots.

    Expected CSV columns (flexible — will auto-detect):
        timestamp, bid_price_1..N, bid_qty_1..N,
        ask_price_1..N, ask_qty_1..N

    If the file uses a simpler format (timestamp, side, price, qty),
    the function reconstructs LOB snapshots incrementally.

    Args:
        filepath: Path to the CSV file.

    Returns:
        pd.DataFrame with columns:
            timestamp, mid_price, spread, best_bid, best_ask,
            bid_prices, bid_quantities, ask_prices, ask_quantities
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ------ Format A: already-structured LOB snapshots ------
    if "mid_price" in df.columns and "spread" in df.columns:
        return df

    # ------ Format B: per-level bid/ask columns ------
    bid_price_cols = sorted([c for c in df.columns if c.startswith("bid_price")])
    ask_price_cols = sorted([c for c in df.columns if c.startswith("ask_price")])

    if bid_price_cols and ask_price_cols:
        bid_qty_cols = sorted([c for c in df.columns if c.startswith("bid_qty") or c.startswith("bid_quantity")])
        ask_qty_cols = sorted([c for c in df.columns if c.startswith("ask_qty") or c.startswith("ask_quantity")])

        records = []
        for _, row in df.iterrows():
            bp = [row[c] for c in bid_price_cols if pd.notna(row[c])]
            bq = [row[c] for c in bid_qty_cols if pd.notna(row[c])]
            ap = [row[c] for c in ask_price_cols if pd.notna(row[c])]
            aq = [row[c] for c in ask_qty_cols if pd.notna(row[c])]
            best_bid = bp[0] if bp else np.nan
            best_ask = ap[0] if ap else np.nan
            records.append({
                "timestamp": row.get("timestamp", row.get("time", None)),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": (best_bid + best_ask) / 2 if bp and ap else np.nan,
                "spread": best_ask - best_bid if bp and ap else np.nan,
                "bid_prices": bp,
                "bid_quantities": bq,
                "ask_prices": ap,
                "ask_quantities": aq,
            })
        return pd.DataFrame(records)

    # ------ Format C: event-level (timestamp, side, price, qty) ------
    required = {"side", "price"}
    if required.issubset(set(df.columns)):
        return _reconstruct_lob_from_events(df)

    raise ValueError(
        "Unrecognised CSV format. Expected columns like "
        "'bid_price_1,ask_price_1,...' or 'side,price,quantity'."
    )


def _reconstruct_lob_from_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Replay event-level data through a LimitOrderBook and extract snapshots."""
    lob = LimitOrderBook()
    snapshots = []

    for _, row in events_df.iterrows():
        side = str(row["side"]).lower()
        price = float(row["price"])
        qty = float(row.get("quantity", row.get("qty", 1)))
        ts = row.get("timestamp", row.get("time", None))

        if side in ("bid", "buy", "b"):
            lob.add_order("bid", price, qty)
        elif side in ("ask", "sell", "a", "s"):
            lob.add_order("ask", price, qty)

        snap = lob.get_snapshot()
        snap["timestamp"] = ts
        depth = snap.pop("depth")
        snap["bid_prices"] = [p for p, _ in depth["bids"]]
        snap["bid_quantities"] = [q for _, q in depth["bids"]]
        snap["ask_prices"] = [p for p, _ in depth["asks"]]
        snap["ask_quantities"] = [q for _, q in depth["asks"]]
        snapshots.append(snap)

    return pd.DataFrame(snapshots)


# ------------------------------------------------------------------
# Synthetic data generator (zero-intelligence model)
# ------------------------------------------------------------------

def simulate_lob_data(
    n_ticks: int = 10_000,
    n_levels: int = 10,
    base_price: float = 1000.0,
    tick_size: float = 0.05,
    seed: Optional[int] = 42,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate synthetic LOB snapshots using a **zero-intelligence** model.

    The model works as follows for each tick:
    1. Draw ``n_levels`` bid prices below a drifting mid-price and
       ``n_levels`` ask prices above it.
    2. Quantities at each level are drawn from an exponential distribution
       (thinner at the top of the book, thicker further away).
    3. The mid-price follows a discrete random walk with small drift.

    Args:
        n_ticks: Number of snapshots to generate.
        n_levels: Number of price levels per side.
        base_price: Starting mid-price.
        tick_size: Minimum price increment.
        seed: Random seed for reproducibility.
        save_path: If provided, save the DataFrame as CSV at this path.

    Returns:
        pd.DataFrame with columns:
            timestamp, mid_price, spread, best_bid, best_ask,
            bid_prices (list), bid_quantities (list),
            ask_prices (list), ask_quantities (list),
            bid_volume, ask_volume
    """
    rng = np.random.default_rng(seed)
    records: List[dict] = []

    mid = base_price
    t0 = 1_700_000_000.0  # synthetic epoch start

    for i in range(n_ticks):
        # --- mid-price random walk ---
        mid += rng.normal(0, tick_size * 0.5)

        # --- spread: half-spread drawn from exponential ---
        half_spread = max(tick_size, rng.exponential(tick_size * 2))

        best_bid = round(mid - half_spread, 2)
        best_ask = round(mid + half_spread, 2)

        # --- generate price levels ---
        bid_prices = [round(best_bid - j * tick_size, 2) for j in range(n_levels)]
        ask_prices = [round(best_ask + j * tick_size, 2) for j in range(n_levels)]

        # --- quantities: exponential, increasing away from mid ---
        bid_qtys = [round(rng.exponential(50) * (1 + 0.3 * j), 0) for j in range(n_levels)]
        ask_qtys = [round(rng.exponential(50) * (1 + 0.3 * j), 0) for j in range(n_levels)]

        # Ensure no zero quantities
        bid_qtys = [max(q, 1.0) for q in bid_qtys]
        ask_qtys = [max(q, 1.0) for q in ask_qtys]

        ts = t0 + i * 0.1  # 100ms between ticks

        records.append({
            "timestamp": ts,
            "mid_price": round(mid, 4),
            "spread": round(best_ask - best_bid, 4),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_prices": bid_prices,
            "bid_quantities": bid_qtys,
            "ask_prices": ask_prices,
            "ask_quantities": ask_qtys,
            "bid_volume": sum(bid_qtys),
            "ask_volume": sum(ask_qtys),
        })

    df = pd.DataFrame(records)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Saved {len(df)} synthetic LOB snapshots → {save_path}")

    return df


def simulate_order_events(
    n_events: int = 5000,
    base_price: float = 1000.0,
    tick_size: float = 0.05,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate a stream of individual order *events* (arrivals).

    Useful for feeding into `_reconstruct_lob_from_events` or for
    Hawkes-process analysis.

    Returns:
        pd.DataFrame with columns: timestamp, side, price, quantity
    """
    rng = np.random.default_rng(seed)
    events = []
    t = 0.0
    mid = base_price

    for _ in range(n_events):
        # inter-arrival from exponential
        dt = rng.exponential(0.05)
        t += dt
        mid += rng.normal(0, tick_size * 0.3)

        side = rng.choice(["bid", "ask"])
        # price offset from mid
        if side == "bid":
            offset = rng.exponential(tick_size * 3)
            price = round(mid - offset, 2)
        else:
            offset = rng.exponential(tick_size * 3)
            price = round(mid + offset, 2)

        qty = max(1, round(rng.exponential(30), 0))

        events.append({
            "timestamp": round(t, 6),
            "side": side,
            "price": price,
            "quantity": qty,
        })

    return pd.DataFrame(events)


# ------------------------------------------------------------------
# Convenience: build LOB object from a snapshot row
# ------------------------------------------------------------------

def snapshot_to_lob(row: pd.Series) -> LimitOrderBook:
    """
    Convert a single DataFrame row (from ``simulate_lob_data``) into a
    populated ``LimitOrderBook`` instance.
    """
    lob = LimitOrderBook()
    bid_prices = row["bid_prices"] if isinstance(row["bid_prices"], list) else eval(row["bid_prices"])
    ask_prices = row["ask_prices"] if isinstance(row["ask_prices"], list) else eval(row["ask_prices"])
    bid_qtys = row["bid_quantities"] if isinstance(row["bid_quantities"], list) else eval(row["bid_quantities"])
    ask_qtys = row["ask_quantities"] if isinstance(row["ask_quantities"], list) else eval(row["ask_quantities"])

    for p, q in zip(bid_prices, bid_qtys):
        lob.add_order("bid", p, q)
    for p, q in zip(ask_prices, ask_qtys):
        lob.add_order("ask", p, q)

    lob.timestamp = row.get("timestamp", None)
    return lob


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating synthetic LOB data …")
    sim_df = simulate_lob_data(
        n_ticks=1000,
        save_path="data/simulated/synthetic_lob_1000.csv",
    )
    print(sim_df.head())
    print(f"\nShape: {sim_df.shape}")
    print(f"Mid-price range: {sim_df['mid_price'].min():.2f} – {sim_df['mid_price'].max():.2f}")
    print(f"Avg spread: {sim_df['spread'].mean():.4f}")

    # Build a single LOB from the first snapshot
    lob = snapshot_to_lob(sim_df.iloc[0])
    print(f"\n{lob}")
    print(f"Depth: {lob.get_depth(3)}")
