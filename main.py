import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
import itertools

# --- Professional Risk Metrics ---
def calculate_expectancy(win_rate, reward_to_risk):
    return (win_rate * reward_to_risk) - (1 - win_rate)

def calculate_profit_factor(endings, risk_amount, reward_amount, win_rate, num_trades):
    wins    = win_rate * num_trades
    losses  = num_trades - wins
    total_gain = wins * reward_amount
    total_loss = losses * risk_amount
    return total_gain / abs(total_loss) if total_loss != 0 else np.nan

def calculate_sharpe(returns):
    if len(returns) < 2:
        return np.nan
    mean_ret = np.mean(returns)
    std_ret  = np.std(returns, ddof=1)
    return (mean_ret / (std_ret + 1e-9)) * np.sqrt(len(returns))

# --- Monte Carlo Simulation Function ---
def monte_carlo_simulation(
    initial_balance, risk_percent, win_rate, reward_to_risk,
    num_trades, num_simulations, max_daily_loss, commission
):
    sims = []
    for _ in range(num_simulations):
        balance    = initial_balance
        peak       = balance
        daily_loss = 0
        drawdowns  = []
        curve      = [balance]
        returns    = []
        for i in range(num_trades):
            if i % 10 == 0:  # reset daily loss every ~10 trades
                daily_loss = 0
            risk_amt   = balance * (risk_percent / 100)
            reward_amt = risk_amt * reward_to_risk
            pnl        = reward_amt if np.random.rand() < win_rate else -risk_amt
            if pnl < 0:
                daily_loss += abs(pnl)
            balance   += pnl - commission
            returns.append(pnl)
            curve.append(balance)
            peak    = max(peak, balance)
            drawdowns.append((peak - balance) / peak)
            if daily_loss > max_daily_loss or balance < 0:
                break
        sims.append({
            'curve':   np.array(curve),
            'ending':  balance,
            'max_dd':  np.max(drawdowns) if drawdowns else 0,
            'returns': np.array(returns)
        })
    return sims

# --- Streamlit App ---
st.title("Professional Futures Monte Carlo Simulator")

# Inputs above the charts
initial_balance = st.number_input(
    "Initial Balance ($)",
    min_value=1_000.0, max_value=1_000_000.0,
    value=50_000.0, step=1_000.0
)
risk_percent   = st.slider("Risk per Trade (%)", 0.01, 10.0, 0.125, 0.01)
win_rate       = st.slider("Win Rate (%)", 1, 99, 50, 1) / 100
reward_to_risk = st.slider("TakeProfit/StopLoss Ratio", 1.0, 5.0, 2.0, 0.1)
trades         = st.slider("Trades per Simulation", 50, 1000, 200, 10)
sims_count     = st.slider("Number of Simulations", 100, 10000, 5000, 100)
max_daily_loss = st.number_input(
    "Max Daily Loss ($)",
    min_value=0.0, max_value=initial_balance,
    value=1000.0, step=100.0
)
commission     = st.number_input(
    "Commission per Trade ($)",
    min_value=0.0, max_value=50.0,
    value=2.0, step=0.5
)
bust_threshold = st.number_input(
    "Bust Threshold ($)",
    min_value=0.0, max_value=initial_balance,
    value=0.0, step=100.0
)
target_balance = st.number_input(
    "Success Target ($)",
    min_value=initial_balance, max_value=initial_balance*2,
    value=initial_balance+3000.0, step=100.0
)

if st.button("Run Simulation"):
    sims    = monte_carlo_simulation(
        initial_balance, risk_percent, win_rate, reward_to_risk,
        trades, sims_count, max_daily_loss, commission
    )

    # extract arrays
    endings    = np.array([s['ending']   for s in sims])
    max_dds    = np.array([s['max_dd']    for s in sims])
    curves_list= [s['curve']             for s in sims]
    returns_list = [s['returns']        for s in sims]

    # Basic metrics
    mean_ending = np.mean(endings)
    result_balance = mean_ending
    return_pct = (mean_ending - initial_balance) / initial_balance * 100
    worst_dd = np.max(max_dds)

    # Consecutive wins/losses & win %
    max_consec_losses = 0
    max_consec_wins   = 0
    total_wins        = 0
    total_trades_ex  = 0
    for ret in returns_list:
        total_wins += np.sum(ret > 0)
        total_trades_ex += len(ret)
        # longest loss run
        runs = [len(list(g)) for val,g in itertools.groupby(ret < 0) if val]
        if runs:
            max_consec_losses = max(max_consec_losses, max(runs))
        # longest win run
        runs = [len(list(g)) for val,g in itertools.groupby(ret > 0) if val]
        if runs:
            max_consec_wins = max(max_consec_wins, max(runs))

    win_trade_pct = total_wins / total_trades_ex * 100 if total_trades_ex else 0

    # Other risk metrics
    expectancy = calculate_expectancy(win_rate, reward_to_risk)
    pfactor    = calculate_profit_factor(
        endings,
        initial_balance * (risk_percent/100),
        initial_balance * (risk_percent/100) * reward_to_risk,
        win_rate, trades
    )
    sharpe_vals = np.array([calculate_sharpe(r) for r in returns_list])

    # Summary Metrics
    st.subheader("Results Summary")
    st.write(f"Initial Balance:               ${initial_balance:,.2f}")
    st.write(f"Result (Avg Ending) Balance:   ${result_balance:,.2f}")
    st.write(f"Return % Over Period:          {return_pct:.2f}%")
    st.write(f"Maximum Drawdown (Worst):      {worst_dd:.2%}")
    st.write(f"Max Consecutive Losses:        {max_consec_losses}")
    st.write(f"Max Consecutive Wins:          {max_consec_wins}")
    st.write(f"Win Trades Percentage:         {win_trade_pct:.2f}%")
    st.write(f"Expectancy (R):                {expectancy:.2f}")
    st.write(f"Profit Factor:                 {pfactor:.2f}")
    st.write(f"Average Sharpe Ratio:          {np.nanmean(sharpe_vals):.2f}")

    # Outcome Distribution
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.hist(endings, bins=50, color='skyblue', edgecolor='black')
    ax1.axvline(target_balance, color='green', linestyle='--', label='Target')
    ax1.axvline(bust_threshold, color='red', linestyle='--', label='Bust')
    ax1.set(title='Ending Balance Distribution',
            xlabel='Balance', ylabel='Frequency')
    ax1.legend()
    st.pyplot(fig1)

    # Equity Curve with Probability Cone
    max_len = max(len(c) for c in curves_list)
    padded  = np.array([np.pad(c, (0, max_len-len(c)), 'edge')
                        for c in curves_list])
    mean_c  = np.mean(padded, axis=0)
    p10     = np.percentile(padded, 10, axis=0)
    p90     = np.percentile(padded, 90, axis=0)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x = np.arange(max_len)
    ax2.plot(x, mean_c, color='black', label='Mean Equity')
    ax2.fill_between(x, p10, p90, color='gray', alpha=0.3,
                     label='10–90% Cone')
    ax2.set(title='Equity Curve with Probability Cone',
            xlabel='Trade #', ylabel='Balance')
    ax2.legend()
    st.pyplot(fig2)
