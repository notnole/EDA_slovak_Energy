"""Test IDM VWAP with reduced shift (simulating OKTE API freshness)."""
import pandas as pd, numpy as np, lightgbm as lgb, sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
LEAD = 8

LGB_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_child_samples=200,
                  subsample=0.5, colsample_bytree=0.5, reg_alpha=1.0,
                  reg_lambda=10.0, n_estimators=200, verbose=-1)

FOLDS = [
    ("2025-04-01", "2025-04-01", "2025-07-01"),
    ("2025-07-01", "2025-07-01", "2025-10-01"),
    ("2025-10-01", "2025-10-01", "2026-01-01"),
    ("2026-01-01", "2026-01-01", "2026-02-01"),
    ("2026-02-01", "2026-02-01", "2026-03-01"),
    ("2026-03-01", "2026-03-01", "2026-04-10"),
]

FEATS = [
    "da_price_qh", "temp_national_change6h", "da_demand", "nowcast_momentum_h2h3",
    "nowcast_trend_h2_h5", "idm_vwap_lag", "da_flow_cz", "prod_rmean8",
    "spread_da_idm_lag", "nowcast_h5", "damas_fe_rmean4", "reg_rmean8",
    "nowcast_momentum_h3h4", "da_price_qh_dev_hourly", "proxy_lag9",
    "proxy_lag96_diff", "da_price_qh_diff_next", "dow_sin", "wind_national",
    "temp_surprise_lag", "proxy_lag15", "prod_momentum", "nowcast_pred_rmean4",
    "hour_sin", "da_net_import", "is_weekend", "temp_bratislava", "da_supply",
    "xborder_vol", "xborder_deviation",
]


def run_backtest(df_input, feats):
    all_oof = []
    for train_end, pred_start, pred_end in FOLDS:
        train = df_input[df_input.index < train_end].dropna(
            subset=["target", f"proxy_lag{LEAD+1}"])
        pred = df_input[
            (df_input.index >= pred_start) & (df_input.index < pred_end)
        ].dropna(subset=[f"proxy_lag{LEAD+1}"])
        sp_train = train.dropna(subset=["spread_target"])
        sp_train = sp_train[sp_train["imb_settlement_price"].abs() <= 5000]
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **LGB_PARAMS)
        m.fit(sp_train[feats].values, sp_train["spread_target"].values)
        pred = pred.copy()
        pred["spread_pred"] = m.predict(pred[feats].values)
        all_oof.append(pred)

    test = pd.concat(all_oof[-2:])
    test = test[test["exec_spread"].notna() & (test["exec_spread"] <= 10)]
    surplus = test["spread_pred"] <= -3
    deficit = test["spread_pred"] >= 3
    trades = test[surplus | deficit].copy()
    trades["size"] = trades["spread_pred"].abs().clip(upper=5)
    s = surplus.reindex(trades.index, fill_value=False)
    d = deficit.reindex(trades.index, fill_value=False)
    trades["pnl"] = 0.0
    trades.loc[s, "pnl"] = (
        (trades.loc[s, "exec_bid"] - trades.loc[s, "imb_settlement_price"])
        * trades.loc[s, "size"] / 4
    )
    trades.loc[d, "pnl"] = (
        (trades.loc[d, "imb_settlement_price"] - trades.loc[d, "exec_ask"])
        * trades.loc[d, "size"] / 4
    )
    nd = trades.index.normalize().nunique()
    daily = trades.groupby(trades.index.date)["pnl"].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    dd = (daily.cumsum() - daily.cumsum().cummax()).min()

    months = {}
    trades["month"] = trades.index.to_period("M")
    for mo, g in trades.groupby("month"):
        mnd = g.index.normalize().nunique()
        months[str(mo)] = g.pnl.sum() / mnd

    return trades.pnl.sum() / nd, sharpe, (trades.pnl > 0).mean(), dd, months


def main():
    data = load_all_data()

    ob_exec = pd.read_csv(
        DATA_DIR / "features" / "orderbook_qh_features.csv",
        parse_dates=["delivery_start"],
    )
    ob_120 = ob_exec[ob_exec["lead_minutes"] == 120].set_index("delivery_start")[
        ["bid", "ask", "spread", "mid"]
    ]
    ob_120 = ob_120[~ob_120.index.duplicated(keep="last")]
    ob_120.columns = ["exec_bid", "exec_ask", "exec_spread", "exec_mid"]

    tml.TRAIN_END = "2026-04-01"
    tml.TEST_START = "2026-04-01"
    df_base, all_fc = build_features(data, LEAD)
    df_base = df_base.join(ob_120, how="left")
    df_base["imb_settlement_price"] = df_base["imb_settle_price"]
    df_base["spread_target"] = df_base["imb_settlement_price"] - df_base["exec_mid"]

    # Timing:
    # We trade at T-120min for delivery at T.
    # IDM gate closes at T-30min.
    # At trade time (T-120min), latest gate-closed period delivers at T-90min.
    # In QH: shift=6 from target. But gate JUST closed — might not be in API yet.
    # Safe: shift=8 (period delivering 120min before target, gate closed 150min before target,
    #        = 30min before trade time). Definitely available.
    # Current: shift=12 (hourly CSV, 3h stale)

    print("=" * 90)
    print("IDM VWAP SHIFT TEST")
    print("Current: shift=12 (3h stale, hourly CSV)")
    print("OKTE API enables QH-granularity at gate close (T-30min)")
    print("=" * 90)
    print()
    print("Timing at trade time (T-120min before delivery):")
    print("  shift=12: VWAP for period 180min before delivery (current)")
    print("  shift=10: VWAP for period 150min before delivery")
    print("  shift=8:  VWAP for period 120min before delivery (safe with API)")
    print("  shift=6:  VWAP for period 90min before delivery (borderline)")
    print("  shift=4:  VWAP for period 60min before delivery (LEAKY - gate not closed)")
    print("  shift=2:  VWAP for period 30min before delivery (LEAKY)")
    print()

    header = (f"  {'Shift':>7s}  {'Stale':>6s}  {'Status':>12s}  "
              f"{'EUR/d':>7s}  {'Sharpe':>7s}  {'Win':>5s}  {'DD':>7s}")
    print(header)
    print("-" * len(header))

    for idm_shift in [12, 10, 8, 6, 4, 2]:
        df_mod = df_base.copy()
        if "idm_vwap" in df_mod.columns:
            df_mod["idm_vwap_lag"] = df_mod["idm_vwap"].shift(idm_shift)
        if "spread_da_idm" in df_mod.columns:
            df_mod["spread_da_idm_lag"] = df_mod["spread_da_idm"].shift(idm_shift)

        pnl, sharpe, wr, dd, months = run_backtest(df_mod, FEATS)

        if idm_shift == 12:
            status = "CURRENT"
        elif idm_shift >= 8:
            status = "SAFE (API)"
        elif idm_shift == 6:
            status = "BORDERLINE"
        else:
            status = "LEAKY"

        stale = f"{idm_shift * 15}min"
        mo_str = "  ".join(f"{k}: {v:+.0f}" for k, v in sorted(months.items()))

        print(f"  {idm_shift:>7d}  {stale:>6s}  {status:>12s}  "
              f"{pnl:>+7.0f}  {sharpe:>7.1f}  {wr:>5.0%}  {dd:>+7.0f}  | {mo_str}")


if __name__ == "__main__":
    main()
