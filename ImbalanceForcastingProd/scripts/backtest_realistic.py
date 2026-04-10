"""
Realistic P&L Backtest Using Actual Bid/Ask Execution Prices
=============================================================

Compares three execution scenarios:
  A: VWAP execution with Lead 4 (old optimistic backtest)
  B: Bid/ask at 65min before delivery with Lead 5 (realistic)
  C: Same as B but skip trades where our side has no quote (conservative)

Execution timing: 65min before delivery (5min before gate closure).
Model: Lead 5 (75min) predictions — uses only data available at T-75min,
       which is safely before our T-65min execution time.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPO_ROOT = BASE_DIR.parent


def load_data():
    """Load predictions, order book, and market prices."""
    # Lead 5 predictions (75min ahead — safe for 65min execution)
    pred5 = pd.read_csv(DATA_DIR / "predictions_lead5.csv",
                        parse_dates=['datetime'], index_col='datetime')
    print(f"[+] Lead 5 predictions: {len(pred5)} rows, {pred5.index.min()} to {pred5.index.max()}")

    # Lead 4 predictions for comparison
    pred4 = pd.read_csv(DATA_DIR / "predictions_lead4.csv",
                        parse_dates=['datetime'], index_col='datetime')

    # QH order book at 65min (execution prices)
    ob = pd.read_csv(DATA_DIR / "orderbook_qh_features.csv", parse_dates=['delivery_start'])
    ob_exec = ob[ob['lead_minutes'] == 65].set_index('delivery_start')
    ob_exec = ob_exec[~ob_exec.index.duplicated(keep='last')]
    print(f"[+] Order book (65min): {len(ob_exec)} rows")

    # Market prices (VWAP + imbalance settlement)
    mkt = pd.read_csv(REPO_ROOT / "MarketPriceGap" / "data" / "processed" / "hourly_market_prices.csv",
                       parse_dates=['timestamp_hour'], index_col='timestamp_hour')
    mkt = mkt[~mkt.index.duplicated(keep='last')]
    print(f"[+] Market prices: {len(mkt)} rows")

    return pred4, pred5, ob_exec, mkt


def build_trades(pred_df, label):
    """Filter to tradeable predictions (|pred| > 2)."""
    t = pred_df[pred_df['pred_median'].abs() > 2].copy()
    t['size'] = t['pred_median'].abs().clip(upper=5)
    t['direction'] = np.where(t['pred_median'] > 0, 'surplus', 'deficit')
    t['label'] = label
    return t


def compute_pnl(trades, exec_col_sell, exec_col_buy, imb_col='imb_settlement_price'):
    """Compute P&L given execution price columns."""
    t = trades.copy()
    t['pnl'] = np.nan

    surplus = t['direction'] == 'surplus'
    deficit = t['direction'] == 'deficit'

    # Surplus: sell at exec_col_sell, settle at imbalance
    sell_price = t[exec_col_sell]
    t.loc[surplus, 'pnl'] = (sell_price[surplus] - t.loc[surplus, imb_col]) * t.loc[surplus, 'size'] / 4

    # Deficit: buy at exec_col_buy, settle at imbalance
    buy_price = t[exec_col_buy]
    t.loc[deficit, 'pnl'] = (t.loc[deficit, imb_col] - buy_price[deficit]) * t.loc[deficit, 'size'] / 4

    return t


def report(t, name):
    """Print backtest results."""
    valid = t.dropna(subset=['pnl'])
    if len(valid) == 0:
        print(f"\n--- {name}: No valid trades ---")
        return

    n_days = valid.index.normalize().nunique()
    total = valid['pnl'].sum()
    wr = (valid['pnl'] > 0).mean()
    daily = valid.groupby(valid.index.date)['pnl'].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    prof = (daily > 0).sum()

    print(f"\n--- {name} ---")
    print(f"  Trades: {len(valid)} ({len(valid)/n_days:.0f}/day)")
    print(f"  Win rate: {wr:.1%}")
    print(f"  P&L: {total:>+,.0f} EUR | {total/n_days:>+,.0f}/day | Sharpe: {sharpe:.1f}")
    print(f"  Days: {prof}/{len(daily)} profitable ({prof/len(daily):.0%})")
    print(f"  Daily: mean={daily.mean():>+,.0f}, median={daily.median():>+,.0f}, worst={daily.min():>+,.0f}")

    # Monthly
    valid['month'] = valid.index.to_period('M')
    for period, grp in valid.groupby('month'):
        nd = grp.index.normalize().nunique()
        mpnl = grp['pnl'].sum()
        mwr = (grp['pnl'] > 0).mean()
        print(f"    {period}: win={mwr:.0%}, P&L={mpnl:>+,.0f} EUR ({mpnl/nd:>+,.0f}/day)")

    return {'total': total, 'daily': total/n_days, 'wr': wr, 'sharpe': sharpe}


def main():
    print("=" * 70)
    print("REALISTIC P&L BACKTEST — Bid/Ask vs VWAP Execution")
    print("=" * 70)

    pred4, pred5, ob_exec, mkt = load_data()

    # --- Scenario A: VWAP execution, Lead 4 (old optimistic) ---
    trades_a = build_trades(pred4, 'Lead4_VWAP')
    trades_a['hour_ts'] = trades_a.index.floor('h')
    trades_a = trades_a.join(mkt[['idm_vwap', 'imb_settlement_price']], on='hour_ts', how='left')
    trades_a = trades_a.dropna(subset=['idm_vwap', 'imb_settlement_price'])
    trades_a = trades_a[trades_a['imb_settlement_price'].abs() <= 5000]
    trades_a = compute_pnl(trades_a, 'idm_vwap', 'idm_vwap')

    # --- Scenario B: Bid/ask at 65min, Lead 5 (realistic) ---
    trades_b = build_trades(pred5, 'Lead5_BidAsk')
    trades_b['hour_ts'] = trades_b.index.floor('h')
    trades_b = trades_b.join(mkt[['imb_settlement_price']], on='hour_ts', how='left')
    # Join order book bid/ask (QH resolution — join on 15-min index)
    trades_b = trades_b.join(ob_exec[['bid', 'ask', 'mid', 'spread']], how='left')
    trades_b = trades_b.dropna(subset=['imb_settlement_price'])
    trades_b = trades_b[trades_b['imb_settlement_price'].abs() <= 5000]
    trades_b = compute_pnl(trades_b, 'bid', 'ask')

    # --- Scenario C: Same as B but skip if our side has no quote ---
    trades_c = trades_b.copy()
    surplus_no_bid = (trades_c['direction'] == 'surplus') & trades_c['bid'].isna()
    deficit_no_ask = (trades_c['direction'] == 'deficit') & trades_c['ask'].isna()
    trades_c.loc[surplus_no_bid | deficit_no_ask, 'pnl'] = np.nan

    print("\n" + "=" * 70)
    print("RESULTS (Feb-Mar 2026)")
    print("=" * 70)

    res_a = report(trades_a, "A: Lead 4 + VWAP (optimistic baseline)")
    res_b = report(trades_b, "B: Lead 5 + Bid/Ask at 65min (realistic)")
    res_c = report(trades_c, "C: Lead 5 + Bid/Ask, skip if no quote (conservative)")

    # Spread cost analysis
    print("\n" + "=" * 70)
    print("SPREAD COST ANALYSIS")
    print("=" * 70)

    valid_b = trades_b.dropna(subset=['pnl', 'spread'])
    print(f"\n  Trades with spread data: {len(valid_b)}")
    print(f"  Spread distribution (EUR/MWh):")
    print(f"    Median: {valid_b['spread'].median():.1f}")
    print(f"    Mean:   {valid_b['spread'].mean():.1f}")
    print(f"    P90:    {valid_b['spread'].quantile(0.9):.1f}")

    # Half-spread cost (what we actually pay)
    half_spread = valid_b['spread'].abs() / 2
    avg_size = valid_b['size'].mean()
    spread_cost_per_trade = (half_spread * avg_size / 4).mean()
    print(f"\n  Average half-spread cost per trade: {spread_cost_per_trade:.1f} EUR")
    print(f"  Total spread cost estimate: {spread_cost_per_trade * len(valid_b):,.0f} EUR")

    # How many trades had no quote on our side?
    surplus_trades = trades_b[trades_b['direction'] == 'surplus']
    deficit_trades = trades_b[trades_b['direction'] == 'deficit']
    print(f"\n  Surplus trades: {len(surplus_trades)}, bid available: {surplus_trades['bid'].notna().mean():.0%}")
    print(f"  Deficit trades: {len(deficit_trades)}, ask available: {deficit_trades['ask'].notna().mean():.0%}")

    # Save trade-level results
    out = trades_b[['target', 'pred_median', 'direction', 'size',
                     'bid', 'ask', 'mid', 'spread',
                     'imb_settlement_price', 'pnl']].copy()
    out.to_csv(DATA_DIR / "backtest_realistic.csv")
    print(f"\n[+] Saved: {DATA_DIR / 'backtest_realistic.csv'}")


if __name__ == "__main__":
    main()
