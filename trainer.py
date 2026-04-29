"""
Main training script – Daily, Global, and Shrinking‑Window modes.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from lstm_attention_model import LSTMAttentionTrainer
import push_results


def run_lstm_mode(returns, macro, tickers, mode_name, epochs):
    """Train LSTM‑Attention and produce forecasts + attention weights."""
    if len(returns) < config.MIN_OBSERVATIONS:
        return None

    X, y, _ = data_manager.build_sequences(returns, macro)
    if len(X) < config.MIN_OBSERVATIONS:
        return None

    input_dim = X.shape[2]
    output_dim = len(tickers)

    trainer = LSTMAttentionTrainer(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=config.LSTM_HIDDEN_SIZE,
        num_layers=config.LSTM_NUM_LAYERS,
        bidirectional=config.LSTM_BIDIRECTIONAL,
        dropout=config.LSTM_DROPOUT,
        num_heads=config.ATTENTION_HEADS,
        lr=config.LEARNING_RATE,
        seed=config.RANDOM_SEED
    )

    print(f"  Training LSTM‑Attention on {len(X)} samples...")
    trainer.fit(X, y, epochs=epochs, batch_size=config.BATCH_SIZE, patience=config.PATIENCE)

    # Latest window
    latest_X = X[-1:]  # keep batch dim
    preds, attn_weights = trainer.predict(latest_X[0])

    sorted_indices = np.argsort(preds)[::-1]
    top3 = [{'ticker': tickers[i], 'predicted_return': float(preds[i])} for i in sorted_indices[:3]]
    all_scores = [{'ticker': tickers[i], 'predicted_return': float(preds[i])} for i in range(len(tickers))]

    # Attention matrix (seq_len × seq_len) – we'll store the last row (attention from last step to all steps)
    last_step_attention = attn_weights[-1].tolist() if attn_weights is not None else []

    return {
        'top_picks': top3,
        'all_scores': all_scores,
        'attention_last_step': last_step_attention,
        'training_start': str(returns.index[0].date()),
        'training_end': str(returns.index[-1].date()),
        'n_observations': len(returns)
    }


def run_shrinking_windows(df_master, macro, tickers, epochs):
    """Fixed shrinking windows with consensus on top ETF."""
    windows = []
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        sd = pd.Timestamp(f"{start_year}-01-01")
        ed = pd.Timestamp(f"{start_year+2}-12-31")
        mask = (df_master['Date'] >= sd) & (df_master['Date'] <= ed)
        window_df = df_master[mask].copy()
        if len(window_df) < config.MIN_OBSERVATIONS:
            continue

        returns = data_manager.prepare_returns_matrix(window_df, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        m = macro.loc[returns.index]
        mode_out = run_lstm_mode(returns, m, tickers, f"Shrinking {start_year}", epochs)
        if mode_out:
            top_ticker = mode_out['top_picks'][0]['ticker']
            top_return = mode_out['top_picks'][0]['predicted_return']
            windows.append({
                'window_start': start_year,
                'window_end': start_year + 2,
                'ticker': top_ticker,
                'predicted_return': top_return
            })

    if not windows:
        return None

    vote = {}
    for w in windows:
        vote[w['ticker']] = vote.get(w['ticker'], 0) + 1
    pick = max(vote, key=vote.get)
    conviction = vote[pick] / len(windows) * 100
    return {'ticker': pick, 'conviction': conviction, 'num_windows': len(windows), 'windows': windows}


def main():
    import os
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set")
        return

    df_master = data_manager.load_master_data()
    df_master['Date'] = pd.to_datetime(df_master['Date'])
    macro = data_manager.prepare_macro_features(df_master)

    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== {universe_name} ===")
        returns_all = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns_all) < config.MIN_OBSERVATIONS:
            continue

        m = macro.loc[returns_all.index].dropna()
        returns_all = returns_all.loc[m.index]
        m = m.loc[returns_all.index]

        universe_out = {}

        # Daily
        daily_ret = returns_all.iloc[-config.DAILY_LOOKBACK:]
        daily_macro = m.iloc[-config.DAILY_LOOKBACK:]
        daily_out = run_lstm_mode(daily_ret, daily_macro, tickers, "Daily", epochs=80)
        if daily_out:
            universe_out['daily'] = daily_out
            print(f"  Daily top: {daily_out['top_picks'][0]['ticker']}")

        # Global
        global_out = run_lstm_mode(returns_all, m, tickers, "Global", epochs=config.EPOCHS)
        if global_out:
            universe_out['global'] = global_out
            print(f"  Global top: {global_out['top_picks'][0]['ticker']}")

        # Shrinking Windows
        shrinking = run_shrinking_windows(df_master, macro, tickers, epochs=60)
        if shrinking:
            universe_out['shrinking'] = shrinking
            print(f"  Shrinking consensus: {shrinking['ticker']} ({shrinking['conviction']:.0f}%)")

        all_results[universe_name] = universe_out

    push_results.push_daily_result({"run_date": config.TODAY, "universes": all_results})
    print("\n=== Run Complete ===")


if __name__ == "__main__":
    main()
