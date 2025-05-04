import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def monte_carlo(start_cap, win_rate, risk_pct, rr, n_trades, n_sims):
    endings = []
    for _ in range(n_sims):
        bal = start_cap
        for _ in range(n_trades):
            risk = bal * risk_pct
            if np.random.rand() < win_rate:
                bal += risk * rr
            else:
                bal -= risk
            if bal <= 0:
                break
        endings.append(bal)
    return np.array(endings)

def plot_histogram(endings, target=None, bust=None):
    plt.figure(figsize=(8,4))
    plt.hist(endings, bins=50, color='skyblue', edgecolor='black')
    if target: plt.axvline(target, color='green', ls='--', label='Target')
    if bust:   plt.axvline(bust,   color='red',   ls='--', label='Bust')
    plt.xlabel('Ending Balance'); plt.ylabel('Frequency')
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    # --- User parameters ---
    START    = 50_000
    WIN_RATE = 0.5         # 50%
    RISK_PCT = 0.00125     # 0.125%
    RR       = 2.0         # 2:1
    TRADES   = 200
    SIMS     = 5000
    TARGET   = 53_000
    BUST     = 48_000

    endings = monte_carlo(START, WIN_RATE, RISK_PCT, RR, TRADES, SIMS)
    plot_histogram(endings, target=TARGET, bust=BUST)

endings = np.array(endings)

# Worst and best
worst = endings.min()
best  = endings.max()

# Most probable: find the histogram bin with highest count
counts, bins = np.histogram(endings, bins=50)
idx_mode = np.argmax(counts)
most_probable = (bins[idx_mode] + bins[idx_mode+1]) / 2

# Display them
st.write(f"**Worst Outcome:** ${worst:,.2f}")
st.write(f"**Most Probable Outcome:** ${most_probable:,.2f}")
st.write(f"**Best Outcome:** ${best:,.2f}")