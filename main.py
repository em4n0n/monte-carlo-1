import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import itertools

# --- Monte Carlo Simulation Function (with break-even trades) ---
def monte_carlo_simulation(initial_balance, risk_percent, win_rate, breakeven_rate,
                           reward_to_risk, num_trades, num_simulations):
    sims = []
    for _ in range(num_simulations):
        balance = initial_balance
        curve   = [balance]
        returns = []
        for _ in range(num_trades):
            risk_amt = balance * (risk_percent / 100)
            r = np.random.rand()
            if r < win_rate:
                pnl =  risk_amt * reward_to_risk
            elif r < win_rate + breakeven_rate:
                pnl = 0.0
            else:
                pnl = -risk_amt
            balance += pnl
            returns.append(pnl)
            curve.append(balance)
        sims.append({
            'curve':   np.array(curve),
            'ending':  balance,
            'returns': np.array(returns)
        })
    return sims

st.title("Trading Monte Carlo Simulator")

# ——— Inputs (above charts) ———
initial_balance = st.number_input(
    "Initial Balance ($)", min_value=1_000.0, max_value=10_000_000.0,
    value=50_000.0, step=1_000.0
)
risk_percent = st.slider(
    "Risk % per Trade", min_value=0.01, max_value=10.0,
    value=0.125, step=0.01
)
win_rate = st.slider(
    "Winning Trades %", min_value=0, max_value=100,
    value=50, step=1
) / 100.0
breakeven_rate = st.slider(
    "Break-even Trades %", min_value=0, max_value=100,
    value=0, step=1
) / 100.0

avg_trades_month = st.number_input(
    "Avg Trades per Month", min_value=1, max_value=1000,
    value=100, step=10
)
total_months = st.number_input(
    "Total Months", min_value=1, max_value=120,
    value=6, step=1
)
reward_to_risk = st.slider(
    "TakeProfit/StopLoss Ratio", min_value=0.1, max_value=10.0,
    value=2.0, step=0.1
)
num_simulations = st.slider(
    "Number of Simulations", min_value=100, max_value=20000,
    value=5000, step=100
)

# derive total trades
num_trades = int(avg_trades_month * total_months)

if st.button("Run Simulation"):
    sims = monte_carlo_simulation(
        initial_balance, risk_percent, win_rate, breakeven_rate,
        reward_to_risk, num_trades, num_simulations
    )

    # extract endings, curves, and returns
    endings = np.array([s['ending'] for s in sims])
    curves  = [s['curve'] for s in sims]
    returns_list = [s['returns'] for s in sims]

    # Basic metrics
    result_balance = np.mean(endings)
    return_pct = (result_balance - initial_balance) / initial_balance * 100

    # Maximum drawdown across all sims
    drawdowns = []
    for curve in curves:
        peak = np.maximum.accumulate(curve)
        dd = (peak - curve) / peak
        drawdowns.append(np.max(dd))
    max_drawdown = np.max(drawdowns)

    # Max consecutive wins/losses and win trade percentage
    max_consec_losses = 0
    max_consec_wins = 0
    total_wins = 0
    total_trades = 0
    for ret in returns_list:
        total_wins += np.sum(ret > 0)
        total_trades += len(ret)
        # losses
        runs = [len(list(g)) for k, g in itertools.groupby(ret < 0) if k]
        if runs:
            max_consec_losses = max(max_consec_losses, max(runs))
        # wins
        runs = [len(list(g)) for k, g in itertools.groupby(ret > 0) if k]
        if runs:
            max_consec_wins = max(max_consec_wins, max(runs))
    win_trade_pct = (total_wins / total_trades * 100) if total_trades else 0

    # Worst, Most Probable, Best outcomes
    worst = endings.min()
    best  = endings.max()
    counts, bins = np.histogram(endings, bins=50)
    mode_idx = np.argmax(counts)
    most_probable = (bins[mode_idx] + bins[mode_idx+1]) / 2

    # Results Summary
    st.subheader("Results Summary")
    st.write(f"Initial Balance:            ${initial_balance:,.2f}")
    st.write(f"Result Balance (Avg):       ${result_balance:,.2f}")
    st.write(f"Return % Over Period:       {return_pct:.2f}%")
    st.write(f"Maximum Drawdown:           {max_drawdown:.2%}")
    st.write(f"Max Consecutive Losses:     {max_consec_losses}")
    st.write(f"Max Consecutive Wins:       {max_consec_wins}")
    st.write(f"Win Trades Percentage:      {win_trade_pct:.2f}%")
    st.write(f"Worst Outcome:              ${worst:,.2f}")
    st.write(f"Most Probable Outcome:      ${most_probable:,.2f}")
    st.write(f"Best Outcome:               ${best:,.2f}")

    # Histogram of endings
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.hist(endings, bins=50, color='skyblue', edgecolor='black')
    ax1.axvline(worst, color='red', linestyle='--', label='Worst')
    ax1.axvline(most_probable, color='orange', linestyle='--', label='Mode')
    ax1.axvline(best, color='green', linestyle='--', label='Best')
    ax1.set(title="Ending Balance Distribution", xlabel="Balance", ylabel="Frequency")
    ax1.legend()
    st.pyplot(fig1)

    # Equity curves with 10–90% probability cone
    max_len = max(len(c) for c in curves)
    padded = np.array([np.pad(c, (0, max_len-len(c)), 'edge') for c in curves])
    mean_curve = padded.mean(axis=0)
    p10 = np.percentile(padded, 10, axis=0)
    p90 = np.percentile(padded, 90, axis=0)

    fig2, ax2 = plt.subplots(figsize=(10,4))
    x = np.arange(max_len)
    ax2.plot(x, mean_curve, color='black', label='Mean Equity')
    ax2.fill_between(x, p10, p90, color='gray', alpha=0.3, label='10–90% Cone')
    ax2.set(title="Equity Curve with 10–90% Cone", xlabel="Trade #", ylabel="Balance")
    ax2.legend()
    st.pyplot(fig2)

