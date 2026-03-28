#!/usr/bin/env python
"""
Combined HMM + Kalman Filter Visualization.

Overlays regime states with filtered volatility to show model agreement.

Usage:
    python scripts/visualize_combined.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Output directory
FIG_DIR = PROJECT_ROOT / "output" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Colors
STATE_COLORS = {
    0: '#27ae60',  # Stable - green
    1: '#f39c12',  # Elevated - orange
    2: '#e74c3c',  # Crisis - red
    3: '#8e44ad',  # Catastrophic - purple
}
STATE_LABELS = {
    0: 'Stable',
    1: 'Elevated', 
    2: 'Crisis',
    3: 'Catastrophic'
}


def load_all_data():
    """Load features, HMM states, and Kalman results."""
    data = {}
    
    # Features
    for path in [
        PROJECT_ROOT / "data" / "processed" / "features_all.parquet",
        PROJECT_ROOT / "data" / "clean" / "features_all.parquet",
    ]:
        if path.exists():
            data["features"] = pd.read_parquet(path)
            break
    
    # HMM states
    for path in [
        PROJECT_ROOT / "models" / "hmm" / "state_assignments.parquet",
        PROJECT_ROOT / "src" / "models" / "hmm" / "state_assignments.parquet",
    ]:
        if path.exists():
            data["hmm_states"] = pd.read_parquet(path)
            break
    
    # Kalman results
    for path in [
        PROJECT_ROOT / "models" / "kalman" / "kalman_results.parquet",
        PROJECT_ROOT / "src" / "models" / "kalman" / "kalman_results.parquet",
    ]:
        if path.exists():
            data["kalman"] = pd.read_parquet(path)
            break
    
    return data


def plot_combined_regimes_volatility(features, hmm_states, kalman=None):
    """
    Main visualization: HMM regimes + Kalman volatility + Price.
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True, 
                              gridspec_kw={'height_ratios': [1.5, 1, 1.5, 1]})
    
    # Prepare HMM states
    states_df = hmm_states.copy()
    if "timestamp" in states_df.columns:
        states_df["timestamp"] = pd.to_datetime(states_df["timestamp"])
        states_df = states_df.set_index("timestamp")
    
    # =========================================================================
    # Panel 1: Price with regime background shading
    # =========================================================================
    ax1 = axes[0]
    
    # Plot price for each stablecoin
    symbols = features["symbol"].unique()
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#1abc9c']
    
    for i, symbol in enumerate(symbols):
        asset = features[features["symbol"] == symbol]
        ax1.plot(asset.index, asset["price"], label=symbol, 
                color=colors[i % len(colors)], alpha=0.8, linewidth=1)
    
    # Add regime shading
    add_regime_shading(ax1, states_df)
    
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.set_title("Stablecoin Prices with HMM Regime Detection", fontsize=13, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.set_ylim(0, 1.15)
    
    # =========================================================================
    # Panel 2: HMM State Probabilities
    # =========================================================================
    ax2 = axes[1]
    
    prob_cols = sorted([c for c in states_df.columns if c.startswith("state_prob_")])
    
    if prob_cols:
        # Stacked area chart
        bottoms = np.zeros(len(states_df))
        for i, col in enumerate(prob_cols):
            state_num = int(col.split("_")[-1])
            ax2.fill_between(states_df.index, bottoms, bottoms + states_df[col],
                           color=STATE_COLORS.get(state_num, '#95a5a6'),
                           alpha=0.7, label=STATE_LABELS.get(state_num, f'State {state_num}'))
            bottoms += states_df[col].values
    
    ax2.set_ylabel("State Probability", fontsize=11)
    ax2.set_title("HMM Regime Probabilities", fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, ncol=4)
    ax2.set_ylim(0, 1)
    
    # =========================================================================
    # Panel 3: Volatility (Kalman filtered if available, else rolling)
    # =========================================================================
    ax3 = axes[2]
    
    # Use Kalman filtered volatility if available
    if kalman is not None and "filtered_volatility" in kalman.columns:
        kalman_df = kalman.copy()
        if "timestamp" in kalman_df.columns:
            kalman_df["timestamp"] = pd.to_datetime(kalman_df["timestamp"])
            kalman_df = kalman_df.set_index("timestamp")
        
        ax3.plot(kalman_df.index, kalman_df["filtered_volatility"], 
                color='#e74c3c', linewidth=1.5, label='Kalman Filtered')
        
        if "smoothed_volatility" in kalman_df.columns:
            ax3.plot(kalman_df.index, kalman_df["smoothed_volatility"],
                    color='#3498db', linewidth=1.5, alpha=0.7, label='Kalman Smoothed')
    
    # Also plot rolling volatility from features
    if "volatility_30d" in features.columns:
        # Average across assets
        vol_pivot = features.pivot_table(values="volatility_30d", index=features.index, 
                                         columns="symbol", aggfunc="mean")
        avg_vol = vol_pivot.mean(axis=1)
        ax3.plot(avg_vol.index, avg_vol, color='#95a5a6', linewidth=1, 
                alpha=0.5, label='30d Rolling (avg)')
    
    add_regime_shading(ax3, states_df)
    
    ax3.set_ylabel("Volatility", fontsize=11)
    ax3.set_title("Volatility with Regime Overlay", fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    
    # =========================================================================
    # Panel 4: Deviation from peg
    # =========================================================================
    ax4 = axes[3]
    
    for i, symbol in enumerate(symbols):
        asset = features[features["symbol"] == symbol]
        if "pct_deviation" in asset.columns:
            ax4.plot(asset.index, asset["pct_deviation"] * 100, label=symbol,
                    color=colors[i % len(colors)], alpha=0.7, linewidth=1)
    
    add_regime_shading(ax4, states_df)
    
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.axhline(y=-1, color='#e74c3c', linestyle=':', alpha=0.5, label='1% threshold')
    ax4.axhline(y=1, color='#e74c3c', linestyle=':', alpha=0.5)
    ax4.set_ylabel("Deviation (%)", fontsize=11)
    ax4.set_xlabel("Date", fontsize=11)
    ax4.set_title("Peg Deviation with Regime Overlay", fontsize=13, fontweight='bold')
    ax4.legend(loc='lower left', fontsize=9)
    
    # Format x-axis
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "combined_regimes_volatility.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: combined_regimes_volatility.png")


def add_regime_shading(ax, states_df):
    """Add background shading based on HMM state."""
    if "state" not in states_df.columns:
        return
    
    # Get state changes
    states = states_df["state"].values
    dates = states_df.index
    
    # Find contiguous blocks of same state
    i = 0
    while i < len(states):
        state = int(states[i])
        start_idx = i
        
        # Find end of this state block
        while i < len(states) and int(states[i]) == state:
            i += 1
        
        end_idx = i - 1
        
        # Only shade non-stable states (state > 0)
        if state > 0:
            ax.axvspan(dates[start_idx], dates[end_idx], 
                      alpha=0.2, color=STATE_COLORS.get(state, '#95a5a6'))


def plot_ust_collapse_detailed(features, hmm_states):
    """Detailed view of UST collapse with regime overlay."""
    
    # Filter to UST and collapse period
    ust = features[(features["symbol"] == "UST") & 
                   (features.index >= "2022-04-01") & 
                   (features.index <= "2022-06-15")].copy()
    
    if len(ust) == 0:
        print("No UST data in collapse period")
        return
    
    # Get HMM states for same period
    states_df = hmm_states.copy()
    if "timestamp" in states_df.columns:
        states_df["timestamp"] = pd.to_datetime(states_df["timestamp"])
        states_df = states_df.set_index("timestamp")
    
    states_period = states_df[(states_df.index >= "2022-04-01") & 
                              (states_df.index <= "2022-06-15")]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Panel 1: Price
    ax1 = axes[0]
    ax1.plot(ust.index, ust["price"], color='#e74c3c', linewidth=2)
    ax1.fill_between(ust.index, ust["price"], 1, alpha=0.3, color='red', 
                     where=ust["price"] < 1)
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    add_regime_shading(ax1, states_period)
    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.set_title("UST Collapse - May 2022 (with HMM Regime Detection)", 
                  fontsize=13, fontweight='bold')
    
    # Add key dates
    key_dates = [
        ("2022-05-07", "Initial depeg"),
        ("2022-05-09", "Death spiral begins"),
        ("2022-05-12", "Below $0.30"),
    ]
    for date, label in key_dates:
        try:
            ax1.axvline(x=pd.to_datetime(date), color='black', linestyle=':', alpha=0.5)
            ax1.text(pd.to_datetime(date), ax1.get_ylim()[1] * 0.95, label, 
                    rotation=90, va='top', fontsize=8)
        except:
            pass
    
    # Panel 2: Volatility
    ax2 = axes[1]
    if "volatility_30d" in ust.columns:
        ax2.plot(ust.index, ust["volatility_30d"], color='#9b59b6', linewidth=2)
        ax2.fill_between(ust.index, 0, ust["volatility_30d"], alpha=0.3, color='purple')
    add_regime_shading(ax2, states_period)
    ax2.set_ylabel("30d Volatility", fontsize=11)
    ax2.set_title("Volatility Explosion", fontsize=13, fontweight='bold')
    
    # Panel 3: HMM State
    ax3 = axes[2]
    if len(states_period) > 0:
        for state in sorted(states_period["state"].unique()):
            mask = states_period["state"] == state
            ax3.scatter(states_period.index[mask], states_period["state"][mask],
                       c=STATE_COLORS.get(int(state), '#95a5a6'), 
                       label=STATE_LABELS.get(int(state), f'State {state}'),
                       s=50, alpha=0.8)
    
    ax3.set_ylabel("HMM State", fontsize=11)
    ax3.set_xlabel("Date", fontsize=11)
    ax3.set_title("Regime Classification", fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['Stable', 'Elevated', 'Crisis', 'Catastrophic'])
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ust_collapse_detailed.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: ust_collapse_detailed.png")


def plot_regime_transition_analysis(hmm_states):
    """Analyze regime transitions and durations."""
    states_df = hmm_states.copy()
    if "timestamp" in states_df.columns:
        states_df["timestamp"] = pd.to_datetime(states_df["timestamp"])
        states_df = states_df.set_index("timestamp")
    
    states = states_df["state"].values
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Empirical transition matrix
    ax1 = axes[0]
    n_states = int(states.max()) + 1
    trans_matrix = np.zeros((n_states, n_states))
    
    for i in range(len(states) - 1):
        trans_matrix[int(states[i]), int(states[i+1])] += 1
    
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    trans_matrix = np.divide(trans_matrix, row_sums, where=row_sums != 0)
    
    im = ax1.imshow(trans_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    ax1.set_xticks(range(n_states))
    ax1.set_yticks(range(n_states))
    labels = [STATE_LABELS.get(i, f'S{i}') for i in range(n_states)]
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel("To State")
    ax1.set_ylabel("From State")
    ax1.set_title("Transition Probabilities", fontweight='bold')
    
    for i in range(n_states):
        for j in range(n_states):
            ax1.text(j, i, f'{trans_matrix[i,j]:.2f}', ha='center', va='center',
                    color='white' if trans_matrix[i,j] > 0.5 else 'black', fontsize=10)
    
    plt.colorbar(im, ax=ax1)
    
    # 2. State duration distribution
    ax2 = axes[1]
    
    # Calculate durations
    durations = {s: [] for s in range(n_states)}
    current_state = int(states[0])
    current_duration = 1
    
    for i in range(1, len(states)):
        if int(states[i]) == current_state:
            current_duration += 1
        else:
            durations[current_state].append(current_duration)
            current_state = int(states[i])
            current_duration = 1
    durations[current_state].append(current_duration)
    
    # Box plot
    duration_data = [durations[s] for s in range(n_states) if durations[s]]
    positions = [s for s in range(n_states) if durations[s]]
    
    bp = ax2.boxplot(duration_data, positions=positions, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(STATE_COLORS.get(positions[i], '#95a5a6'))
        patch.set_alpha(0.7)
    
    ax2.set_xticks(range(n_states))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Duration (days)")
    ax2.set_title("Regime Duration Distribution", fontweight='bold')
    
    # Add median labels
    for i, pos in enumerate(positions):
        median = np.median(durations[pos])
        ax2.text(pos, median, f'{median:.0f}d', ha='center', va='bottom', fontsize=9)
    
    # 3. Time spent in each regime
    ax3 = axes[2]
    
    state_counts = pd.Series(states).value_counts().sort_index()
    colors = [STATE_COLORS.get(int(s), '#95a5a6') for s in state_counts.index]
    
    wedges, texts, autotexts = ax3.pie(
        state_counts.values, 
        labels=[STATE_LABELS.get(int(s), f'S{s}') for s in state_counts.index],
        colors=colors,
        autopct='%1.1f%%',
        explode=[0.02] * len(state_counts)
    )
    ax3.set_title("Time in Each Regime", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "regime_transition_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: regime_transition_analysis.png")


def plot_early_warning_signals(features, hmm_states):
    """
    Identify potential early warning signals before regime transitions.
    """
    states_df = hmm_states.copy()
    if "timestamp" in states_df.columns:
        states_df["timestamp"] = pd.to_datetime(states_df["timestamp"])
        states_df = states_df.set_index("timestamp")
    
    # Find transitions to crisis/catastrophic states
    states = states_df["state"].values
    transition_dates = []
    
    for i in range(1, len(states)):
        # Transition from stable/elevated (0,1) to crisis/catastrophic (2,3)
        if states[i-1] in [0, 1] and states[i] in [2, 3]:
            transition_dates.append(states_df.index[i])
    
    if not transition_dates:
        print("No crisis transitions found")
        return
    
    print(f"Found {len(transition_dates)} transitions to crisis states")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Analyze features around transitions
    lookback = 30  # days before transition
    lookahead = 10  # days after
    
    feature_cols = ["volatility_30d", "abs_pct_deviation", "log_return"]
    feature_cols = [f for f in feature_cols if f in features.columns]
    
    for idx, feat in enumerate(feature_cols[:4]):
        ax = axes[idx // 2, idx % 2]
        
        all_windows = []
        
        for trans_date in transition_dates[:20]:  # Limit to first 20
            # Get window around transition
            start = trans_date - pd.Timedelta(days=lookback)
            end = trans_date + pd.Timedelta(days=lookahead)
            
            window = features[(features.index >= start) & (features.index <= end)][feat]
            
            if len(window) > 0:
                # Normalize to days relative to transition
                window_df = pd.DataFrame({
                    'value': window.values,
                    'days': [(d - trans_date).days for d in window.index]
                })
                all_windows.append(window_df)
                
                ax.plot(window_df['days'], window_df['value'], alpha=0.3, color='gray')
        
        # Plot average
        if all_windows:
            combined = pd.concat(all_windows)
            avg = combined.groupby('days')['value'].mean()
            ax.plot(avg.index, avg.values, color='#e74c3c', linewidth=3, label='Average')
        
        ax.axvline(x=0, color='black', linestyle='--', label='Transition')
        ax.set_xlabel("Days relative to crisis transition")
        ax.set_ylabel(feat)
        ax.set_title(f"Early Warning: {feat}", fontweight='bold')
        ax.legend()
    
    plt.suptitle("Feature Behavior Around Crisis Transitions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "early_warning_signals.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: early_warning_signals.png")


def main():
    print("=" * 60)
    print("Combined HMM + Kalman Visualization")
    print("=" * 60)
    
    data = load_all_data()
    
    if "features" not in data:
        print("ERROR: Features not found")
        return 1
    
    if "hmm_states" not in data:
        print("ERROR: HMM states not found")
        return 1
    
    features = data["features"]
    hmm_states = data["hmm_states"]
    kalman = data.get("kalman")
    
    # Ensure datetime index
    if not isinstance(features.index, pd.DatetimeIndex):
        if "timestamp" in features.columns:
            features["timestamp"] = pd.to_datetime(features["timestamp"])
            features = features.set_index("timestamp")
    
    print(f"Features: {len(features)} rows")
    print(f"HMM states: {len(hmm_states)} rows")
    print(f"Kalman: {'Available' if kalman is not None else 'Not found'}")
    
    # Generate visualizations
    print("\n[1] Combined Regimes + Volatility")
    plot_combined_regimes_volatility(features, hmm_states, kalman)
    
    print("\n[2] UST Collapse Detailed")
    plot_ust_collapse_detailed(features, hmm_states)
    
    print("\n[3] Regime Transition Analysis")
    plot_regime_transition_analysis(hmm_states)
    
    print("\n[4] Early Warning Signals")
    plot_early_warning_signals(features, hmm_states)
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIG_DIR}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())