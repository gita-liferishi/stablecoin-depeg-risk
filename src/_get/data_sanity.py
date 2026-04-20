"""
sanity_check_panel.py
---------------------------------------------------------------
Run AFTER add_features.py produces data/pomp_panel.csv.
Run BEFORE updating the R pomp script.

Checks:
  1. Zero/near-zero fraction per covariate (identification risk)
  2. Warmup periods where covariates are structurally missing
  3. Scale parity across covariates (rw.sd effective magnitude)
  4. Temporal coverage of non-zero events
  5. Correlation structure among covariates (collinearity)
  6. Per-asset breakdown for dy_inflow (DY is computed jointly)
"""

import numpy as np
import pandas as pd
from pathlib import Path

PANEL = Path("data/pomp_panel.csv")

p = pd.read_csv(PANEL, parse_dates=["timestamp"])
print(f"Loaded panel: {p.shape}  date range {p['timestamp'].min()} to {p['timestamp'].max()}")
print(f"Assets: {sorted(p['symbol'].unique())}\n")

covars = ["delta_fgi_scaled", "eth_drawdown", "btc_drawdown", "dy_inflow"]
available = [c for c in covars if c in p.columns]


# -----------------------------------------------------------------------------
# 1. Zero fraction and distribution summary
# -----------------------------------------------------------------------------
print("=" * 72)
print("1. COVARIATE DISTRIBUTION SUMMARY")
print("=" * 72)

summary = []
for c in available:
    v = p[c]
    eps = v.abs().max() * 1e-6 if v.abs().max() > 0 else 1e-12
    summary.append({
        "covariate": c,
        "mean":        v.mean(),
        "std":         v.std(),
        "min":         v.min(),
        "max":         v.max(),
        "frac_zero":   (v.abs() < eps).mean(),
        "frac_near_0": (v.abs() < v.abs().std() * 0.01).mean() if v.std() > 0 else np.nan,
        "p50":         v.quantile(0.50),
        "p90":         v.quantile(0.90),
        "p99":         v.quantile(0.99),
    })
summary_df = pd.DataFrame(summary).set_index("covariate")
print(summary_df.round(4).to_string())

print("\n[interpretation]")
for c in available:
    fz = summary_df.loc[c, "frac_zero"]
    if fz > 0.95:
        print(f"  ⚠ {c}: {fz:.1%} zero — coefficient likely UNIDENTIFIABLE")
    elif fz > 0.7:
        print(f"  ! {c}: {fz:.1%} zero — coefficient weakly identified, wide CI expected")
    elif fz > 0.3:
        print(f"  ~ {c}: {fz:.1%} zero — acceptable but verify signal days")
    else:
        print(f"  ✓ {c}: {fz:.1%} zero — well-populated")


# -----------------------------------------------------------------------------
# 2. Warmup detection — when did each covariate START having non-zero values?
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("2. WARMUP ANALYSIS — first non-zero date per covariate")
print("=" * 72)

for c in available:
    nonzero = p[p[c].abs() > 1e-10]
    if len(nonzero) == 0:
        print(f"  {c}: ALL ZEROS — broken data pipeline")
        continue
    first_nz = nonzero["timestamp"].min()
    last_nz  = nonzero["timestamp"].max()
    n_nz     = len(nonzero)
    print(f"  {c:22s}  first_nonzero={first_nz.date()}  "
          f"last_nonzero={last_nz.date()}  n_nonzero={n_nz}")


# -----------------------------------------------------------------------------
# 3. Scale parity across covariates
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("3. SCALE PARITY (for IF2 rw.sd calibration)")
print("=" * 72)
print("If scales differ by >10x, use a larger rw.sd for the small-scale covariate")
print("or rescale the covariate in R before passing to pomp.\n")

scales = {c: p[c].abs().quantile(0.95) for c in available}
scales_df = pd.Series(scales).sort_values(ascending=False)
print("95th percentile of |covariate|:")
print(scales_df.round(4).to_string())

print("\nRatio of largest/smallest 95th percentile scale:")
print(f"  {scales_df.max() / max(scales_df.min(), 1e-10):.1f}x")
if scales_df.max() / max(scales_df.min(), 1e-10) > 10:
    print("  ⚠ Scale mismatch >10x — rescale before IF2")


# -----------------------------------------------------------------------------
# 4. DY inflow per asset — joint computation but should vary per asset
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("4. DY_INFLOW PER-ASSET BREAKDOWN")
print("=" * 72)
print("DY inflow is computed jointly — each asset should have its own series.")
print("If all assets have identical inflow, the computation is broken.\n")

if "dy_inflow" in available:
    print(p.groupby("symbol")["dy_inflow"].agg(
        ["mean", "std", "min", "max",
         lambda x: (x.abs() < 1e-10).mean()]
    ).rename(columns={"<lambda_0>": "frac_zero"}).round(3).to_string())

    # Are the series actually different across assets?
    wide = p.pivot_table(index="timestamp", columns="symbol", values="dy_inflow")
    if wide.shape[1] > 1:
        corr = wide.corr()
        print(f"\nCorrelation of dy_inflow across assets (should be < 1.0 off-diagonal):")
        print(corr.round(3).to_string())
        max_off_diag = corr.values[~np.eye(len(corr), dtype=bool)].max()
        if max_off_diag > 0.99:
            print(f"  ⚠ Max off-diagonal correlation {max_off_diag:.3f} — "
                  f"assets have near-identical inflow, verify DY computation")


# -----------------------------------------------------------------------------
# 5. Event days — does the covariate light up when it should?
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("5. KNOWN STRESS DAYS — covariate values on event dates")
print("=" * 72)

stress_days = {
    "2020-03-12 Black Thursday":           "2020-03-12",
    "2020-03-13 COVID crash":              "2020-03-13",
    "2021-01-11 DAI congestion":           "2021-01-11",
    "2022-05-09 UST day -1":               "2022-05-09",
    "2022-05-10 UST collapse":             "2022-05-10",
    "2022-05-12 UST trough":               "2022-05-12",
    "2022-11-08 FTX collapse":             "2022-11-08",
    "2023-03-10 SVB announcement":         "2023-03-10",
    "2023-03-11 USDC depeg":               "2023-03-11",
    "2023-03-13 SVB rescue":               "2023-03-13",
}
for label, dstr in stress_days.items():
    d = pd.Timestamp(dstr)
    row = p[p["timestamp"] == d]
    if len(row) == 0:
        print(f"  {label:35s}  [no data for date]")
        continue
    bits = []
    for c in available:
        vals = row[c].values
        if len(vals) > 0:
            # Show max across assets on that day (for per-asset covariates like dy_inflow)
            bits.append(f"{c}={vals.max():.3f}")
    print(f"  {label:35s}  " + "  ".join(bits))


# -----------------------------------------------------------------------------
# 6. Collinearity check — correlation among covariates
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("6. COVARIATE CORRELATION MATRIX (check for collinearity)")
print("=" * 72)
print("If any pair > 0.7, consider orthogonalization or dropping one.\n")

cov_corr = p[available].corr()
print(cov_corr.round(3).to_string())

high_corr = []
for i, c1 in enumerate(available):
    for c2 in available[i+1:]:
        r = cov_corr.loc[c1, c2]
        if abs(r) > 0.7:
            high_corr.append((c1, c2, r))
if high_corr:
    print("\n⚠ High-correlation pairs (potential collinearity):")
    for c1, c2, r in high_corr:
        print(f"  {c1} × {c2}: r = {r:+.3f}")
else:
    print("\n✓ No covariate pair exceeds |r| = 0.7")


# -----------------------------------------------------------------------------
# 7. Suggested actions summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("7. RECOMMENDED ACTIONS")
print("=" * 72)

actions = []

# Check eth/btc drawdown zero fraction
for c in ["eth_drawdown", "btc_drawdown"]:
    if c in available:
        fz = (p[c].abs() < 1e-10).mean()
        if fz > 0.5:
            actions.append(
                f"  • {c} is {fz:.1%} zero (structurally — ETH/BTC up days). "
                f"This is NORMAL for downside-only covariates. Coefficient is identified\n"
                f"    from the ~{100*(1-fz):.0f}% down-days. Expect wide CIs. OK to keep."
            )

# DY inflow warmup
if "dy_inflow" in available:
    fz = (p["dy_inflow"].abs() < 1e-10).mean()
    if fz > 0.1:
        n_zero = (p["dy_inflow"].abs() < 1e-10).sum()
        actions.append(
            f"  • dy_inflow has {fz:.1%} ({n_zero} rows) zeros — likely warmup.\n"
            f"    OPTION A: Truncate panel to after DY warmup (drop first ~400 rows).\n"
            f"    OPTION B: Keep zeros — early period just has no spillover info.\n"
            f"    RECOMMEND: Check if zeros are all at the start (warmup) vs scattered."
        )
        # Diagnose: are the zeros clustered at the start?
        z = p[p["dy_inflow"].abs() < 1e-10]
        if len(z):
            earliest_nonzero = p[p["dy_inflow"].abs() > 1e-10]["timestamp"].min()
            print(f"\n  dy_inflow first non-zero: {earliest_nonzero.date()}")
            print(f"  Suggest truncating panel to dates >= {earliest_nonzero.date()}")

# Scale suggestion
if len(scales_df) > 1 and scales_df.max() / max(scales_df.min(), 1e-10) > 10:
    actions.append(
        f"  • Rescale dy_inflow by /100 (percent → fraction) before passing to pomp\n"
        f"    so all covariates live in [-1, 1] range. Then rw.sd=0.01 is fair."
    )

if actions:
    print()
    for a in actions:
        print(a)
else:
    print("\n  ✓ No major issues detected. Proceed to POMP.")

print("\n" + "=" * 72)
print("DONE")
print("=" * 72)