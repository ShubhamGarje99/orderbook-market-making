# High-Frequency Limit Order Book Dynamics

An end-to-end framework for developing, backtesting, and simulating high-frequency market-making strategies. This project implements statistical models for order flow (Hawkes Processes) and optimal execution strategies (Avellaneda-Stoikov) on high-frequency Limit Order Book (LOB) data.

---

## Overview

This repository contains a modular system for:
1.  **Parsing and reconstructing** Limit Order Books from raw tick data.
2.  **Visualizing** market depth and liquidity evolution in real-time.
3.  **Modeling** order arrival intensities using self-exciting point processes (Hawkes).
4.  **Simulating** market-making strategies with inventory risk management.
5.  **Backtesting** performance using an event-driven engine.

##  Key Features

### 1. Data Pipeline & Visualization
- [cite_start]**LOB Reconstruction**: Efficient structures to manage bids/asks and derive metrics like mid-price, spread, and depth [cite: 41-56].
- [cite_start]**Interactive Dashboard**: A Streamlit-based GUI for real-time order book visualization, spread evolution, and order flow imbalance (OFI) monitoring [cite: 66-86].
- [cite_start]**Synthetic Generator**: Tools to simulate LOB dynamics when real NSE data is unavailable[cite: 60].

### 2. Statistical Modeling (Day 2 Scope)
- [cite_start]**Hawkes Processes**: MLE estimation to model the arrival rates of buy/sell orders and clustering of market events [cite: 112-152].
- [cite_start]**Price Impact**: Linear regression models (Almgren-Chriss style) to estimate permanent and temporary price impact coefficients [cite: 159-182].
- [cite_start]**Flow Toxicity**: Calculation of VPIN (Volume-Synchronized Probability of Informed Trading)[cite: 158].

### 3. Strategy & Backtesting (Day 3 Scope)
- **Avellaneda-Stoikov Strategy**: Implementation of the classic market-making model optimizing bid-ask spreads based on:
  - Inventory Risk ($\gamma$)
  - Order Arrival Intensity ($k$)
  - [cite_start]Volatility ($\sigma$) [cite: 192-223].
- [cite_start]**Event-Driven Engine**: A backtester that handles order latency, realistic fills, and calculates Sharpe Ratio and Maximum Drawdown [cite: 224-259].

---

## Directory Structure

```text
orderbook-market-making/
├── data/
│   ├── raw/            # Unprocessed NSE tick data
│   ├── processed/      # Cleaned LOB snapshots
│   └── simulated/      # Synthetic Hawkes process data
├── src/
│   ├── data_pipeline/  # LOB reconstruction & loaders
│   ├── models/         # Hawkes, Price Impact, VPIN
│   ├── strategy/       # Avellaneda-Stoikov logic
│   ├── backtesting/    # Event-driven engine & metrics
│   └── visualization/  # Plotly & Streamlit plotting utils
├── dashboard/          # Streamlit app entry point
├── notebooks/          # EDA and parameter fitting (MLE)
├── tests/              # Pytest suite
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
