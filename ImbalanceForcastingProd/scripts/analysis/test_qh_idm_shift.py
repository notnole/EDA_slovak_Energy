"""Test QH-resolution IDM VWAP with reduced shift (OKTE API simulation)."""
import pandas as pd, numpy as np, lightgbm as lgb, sys
from pathlib import Path
from glob import glob

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_multi_lead import load_all_data, build_features
import train_multi_lead as tml

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REPO_ROOT = Path(__file__).resolve().parents[3]
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


def load_qh_idm():
    """Load all 15-min IDM CSVs into a single DataFrame with QH resolution."""
    idm_dir = REPO_ROOT / "RawData" / "IDM_MarketData"
    folders = sorted(idm_dir.glob("IDM_total_results_*"))

    dfs = []
    for folder in folders:
        f = folder / "15 min.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, sep=";", decimal=",")
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)

    # Parse datetime: "1.2.2026" + period 1 = 00:00
    raw["date"] = pd.to_datetime(raw["Delivery day"], format="%d.%m.%Y")
    raw["hour"] = (raw["Period number"] - 1) // 4
    raw["minute"] = ((raw["Period number"] - 1) % 4) * 15
    raw["datetime"] = raw["date"] + pd.to_timedelta(raw["hour"], unit="h") + pd.to_timedelta(raw["minute"], unit="m")

    raw = raw.set_index("datetime").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]

    # Key columns
    result = pd.DataFrame(index=raw.index)
    result["idm_vwap_qh"] = pd.to_numeric(
        raw["Weighted average price of all trades (EUR/MWh)"], errors="coerce"
    )
    result["idm_volume_qh"] = pd.to_numeric(
        raw["Total Traded Quantity (MW)"], errors="coerce"
    )

    print(f"[+] QH IDM: {len(result)} rows, {result.index.min().date()} to {result.index.max().date()}")
    print(f"    VWAP non-null: {result['idm_vwap_qh'].notna().sum()}")
    return result


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
    # Load base features
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

    # Load QH IDM
    qh_idm = load_qh_idm()
    df_base = df_base.join(qh_idm, how="left")

    # Also compute QH spread: DA QH price - IDM QH VWAP
    if "da_price_qh" in df_base.columns and "idm_vwap_qh" in df_base.columns:
        df_base["spread_da_idm_qh"] = df_base["da_price_qh"] - df_base["idm_vwap_qh"]

    print()
    print("=" * 95)
    print("QH IDM VWAP SHIFT TEST")
    print("Using 15-min resolution IDM VWAP (not hourly aggregation)")
    print("=" * 95)
    print()

    # Baseline: current hourly shift=12
    print("--- Baseline (current hourly IDM, shift=12) ---")
    pnl, sharpe, wr, dd, months = run_backtest(df_base, FEATS)
    mo_str = "  ".join(f"{k}: {v:+.0f}" for k, v in sorted(months.items()))
    print(f"  Hourly shift=12:  {pnl:>+6.0f}/d  Sh={sharpe:5.1f}  win={wr:.0%}  DD={dd:>+6.0f}  | {mo_str}")
    print()

    # Test QH IDM with various shifts
    print("--- QH IDM VWAP with different shifts ---")
    for idm_shift in [12, 10, 8, 6, 4, 2]:
        df_mod = df_base.copy()
        # Replace hourly idm features with QH versions
        df_mod["idm_vwap_lag"] = df_mod["idm_vwap_qh"].shift(idm_shift)
        df_mod["spread_da_idm_lag"] = df_mod["spread_da_idm_qh"].shift(idm_shift)

        pnl, sharpe, wr, dd, months = run_backtest(df_mod, FEATS)

        if idm_shift >= 8:
            status = "SAFE"
        elif idm_shift == 6:
            status = "BORDERLINE"
        else:
            status = "LEAKY"

        mo_str = "  ".join(f"{k}: {v:+.0f}" for k, v in sorted(months.items()))
        print(f"  QH shift={idm_shift:2d} ({idm_shift*15:3d}min) [{status:10s}]  "
              f"{pnl:>+6.0f}/d  Sh={sharpe:5.1f}  win={wr:.0%}  DD={dd:>+6.0f}  | {mo_str}")


if __name__ == "__main__":
    main()
