# P2-ETF-LSTM-ATTENTION

**Bidirectional LSTM + Multi‑Head Self‑Attention – Interpretable ETF Forecasting**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-LSTM-ATTENTION/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-LSTM-ATTENTION/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--lstm--attention--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-lstm-attention-results)

## Overview

`P2-ETF-LSTM-ATTENTION` applies a **Bidirectional LSTM** followed by **multi‑head self‑attention** to a 60‑day window of ETF returns and macro features. The attention mechanism reveals *which days* drive each forecast, providing interpretability. ETFs are ranked by predicted next‑day return.

## Methodology

1. **Input**: 60‑day sequences of all ETF returns + macro features (VIX, DXY, T10Y2Y, TBILL_3M).
2. **BiLSTM**: 2‑layer bidirectional LSTM (hidden=128) processes the sequence.
3. **Multi‑Head Self‑Attention**: 4 heads attend over time steps; residual connection + layer norm.
4. **Prediction**: Global mean pooling → linear head to forecast all ETFs simultaneously.
5. **Three Training Modes**: Daily, Global, Shrinking Windows Consensus.

## Interpretability

Attention weights from the last time step are stored and displayed as a heatmap, showing which historical days most influenced the current prediction.

## Universe

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |
