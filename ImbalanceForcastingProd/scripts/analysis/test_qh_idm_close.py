"""Test QH IDM: add closest 2 gate-closed periods as features."""
import pandas as pd, numpy as np, lightgbm as lgb, sys
from pathlib import Path

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

BASE_FEATS = [
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
    idm_dir = REPO_ROOT / "RawData" / "IDM_MarketData"
    dfs = []
    for folder in sorted(idm_dir.glob("IDM_total_results_*")):
        f = folder / "15 min.csv"
        if f.exists():
            dfs.append(pd.read_csv(f, sep=";", decimal=","))
    raw = pd.concat(dfs, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["Delivery day"], format="%d.%m.%Y")
    raw["hour"] = (raw["Period number"] - 1) // 4
    raw["minute"] = ((raw["Period number"] - 1) % 4) * 15
    raw["datetime"] = (raw["date"]
                       + pd.to_timedelta(raw["hour"], unit="h")
                       + pd.to_timedelta(raw["minute"], unit="m"))
    raw = raw.set_index("datetime").sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    result = pd.DataFrame(index=raw.index)
    result["idm_vwap_qh"] = pd.to_numeric(
        raw["Weighted average price of all trades (EUR/MWh)"], errors="coerce")
    result["idm_volume_qh"] = pd.to_numeric(
        raw["Total Traded Quantity (MW)"], errors="coerce")
    return result


def run_bt(df_input, feats):
    all_oof = []
    for te, ps, pe in FOLDS:
        tr = df_input[df_input.index < te].dropna(subset=["target", f"proxy_lag{LEAD+1}"])
        pr = df_input[(df_input.index >= ps) & (df_input.index < pe)].dropna(
            subset=[f"proxy_lag{LEAD+1}"])
        sp = tr.dropna(subset=["spread_target"])
        sp = sp[sp["imb_settlement_price"].abs() <= 5000]
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **LGB_PARAMS)
        m.fit(sp[feats].values, sp["spread_target"].values)
        pr = pr.copy()
        pr["spread_pred"] = m.predict(pr[feats].values)
        all_oof.append(pr)
    test = pd.concat(all_oof[-2:])
    test = test[test["exec_spread"].notna() & (test["exec_spread"] <= 10)]
    surplus = test["spread_pred"] <= -3
    deficit = test["spread_pred"] >= 3
    trades = test[surplus | deficit].copy()
    if len(trades) == 0:
        return 0, 0, 0, 0, {}
    trades["size"] = trades["spread_pred"].abs().clip(upper=5)
    s = surplus.reindex(trades.index, fill_value=False)
    d = deficit.reindex(trades.index, fill_value=False)
    trades["pnl"] = 0.0
    trades.loc[s, "pnl"] = (
        (trades.loc[s, "exec_bid"] - trades.loc[s, "imb_settlement_price"])
        * trades.loc[s, "size"] / 4)
    trades.loc[d, "pnl"] = (
        (trades.loc[d, "imb_settlement_price"] - trades.loc[d, "exec_ask"])
        * trades.loc[d, "size"] / 4)
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


def rpt(label, r):
    pnl, sharpe, wr, dd, months = r
    mo = "  ".join(f"{k}: {v:+.0f}" for k, v in sorted(months.items()))
    print(f"  {label:55s}  {pnl:>+5.0f}/d  Sh={sharpe:5.1f}  win={wr:.0%}  DD={dd:>+6.0f}  | {mo}")


def main():
    data = load_all_data()
    ob_exec = pd.read_csv(DATA_DIR / "features" / "orderbook_qh_features.csv",
                           parse_dates=["delivery_start"])
    ob_120 = ob_exec[ob_exec["lead_minutes"] == 120].set_index("delivery_start")[
        ["bid", "ask", "spread", "mid"]]
    ob_120 = ob_120[~ob_120.index.duplicated(keep="last")]
    ob_120.columns = ["exec_bid", "exec_ask", "exec_spread", "exec_mid"]

    tml.TRAIN_END = "2026-04-01"
    tml.TEST_START = "2026-04-01"
    df_base, _ = build_features(data, LEAD)
    df_base = df_base.join(ob_120, how="left")
    df_base["imb_settlement_price"] = df_base["imb_settle_price"]
    df_base["spread_target"] = df_base["imb_settlement_price"] - df_base["exec_mid"]

    qh = load_qh_idm()
    df_base = df_base.join(qh, how="left")
    df_base["spread_da_idm_qh"] = df_base["da_price_qh"] - df_base["idm_vwap_qh"]
    print()

    print("=" * 100)
    print("QH IDM: CLOSEST PERIODS TEST")
    print("=" * 100)
    print()

    # Baseline
    rpt("Baseline (hourly IDM, shift=12)", run_bt(df_base, BASE_FEATS))
    print()

    # Just replace with QH shift=8
    df_mod = df_base.copy()
    df_mod["idm_vwap_lag"] = df_mod["idm_vwap_qh"].shift(8)
    df_mod["spread_da_idm_lag"] = df_mod["spread_da_idm_qh"].shift(8)
    rpt("QH replace shift=8 only", run_bt(df_mod, BASE_FEATS))
    print()

    # Add closest 2: shift=8 (close) and shift=9 (close-1)
    print("--- Add 2 closest gate-closed periods ---")
    for close, close2 in [(8, 9), (8, 10), (6, 7), (6, 8)]:
        df_mod = df_base.copy()
        # Keep original idm_vwap_lag as the "far" reference
        df_mod["idm_vwap_lag"] = df_mod["idm_vwap_qh"].shift(12)
        df_mod["spread_da_idm_lag"] = df_mod["spread_da_idm_qh"].shift(12)
        # Add 2 close periods
        df_mod["idm_vwap_close1"] = df_mod["idm_vwap_qh"].shift(close)
        df_mod["idm_vwap_close2"] = df_mod["idm_vwap_qh"].shift(close2)
        feats = BASE_FEATS + ["idm_vwap_close1", "idm_vwap_close2"]
        safe = "SAFE" if close >= 8 else "BORDERLINE"
        rpt(f"Far=12 + close1={close} close2={close2} [{safe}]", run_bt(df_mod, feats))
    print()

    # Add close + momentum + volume
    print("--- Add close + momentum + volume ---")
    for close in [8, 6]:
        df_mod = df_base.copy()
        df_mod["idm_vwap_lag"] = df_mod["idm_vwap_qh"].shift(12)
        df_mod["spread_da_idm_lag"] = df_mod["spread_da_idm_qh"].shift(12)
        df_mod["idm_vwap_close"] = df_mod["idm_vwap_qh"].shift(close)
        df_mod["idm_spread_close"] = df_mod["spread_da_idm_qh"].shift(close)
        df_mod["idm_momentum"] = df_mod["idm_vwap_close"] - df_mod["idm_vwap_lag"]
        df_mod["idm_volume_close"] = df_mod["idm_volume_qh"].shift(close)
        feats = BASE_FEATS + ["idm_vwap_close", "idm_spread_close",
                              "idm_momentum", "idm_volume_close"]
        safe = "SAFE" if close >= 8 else "BORDERLINE"
        rpt(f"Full package close={close} [{safe}]", run_bt(df_mod, feats))
    print()

    # Also try: close + close-1 + momentum + spread_close
    print("--- Kitchen sink: 2 close + momentum + spread + volume ---")
    for close in [8, 6]:
        df_mod = df_base.copy()
        df_mod["idm_vwap_lag"] = df_mod["idm_vwap_qh"].shift(12)
        df_mod["spread_da_idm_lag"] = df_mod["spread_da_idm_qh"].shift(12)
        df_mod["idm_vwap_close1"] = df_mod["idm_vwap_qh"].shift(close)
        df_mod["idm_vwap_close2"] = df_mod["idm_vwap_qh"].shift(close + 1)
        df_mod["idm_spread_close"] = df_mod["spread_da_idm_qh"].shift(close)
        df_mod["idm_momentum"] = df_mod["idm_vwap_close1"] - df_mod["idm_vwap_lag"]
        df_mod["idm_volume_close"] = df_mod["idm_volume_qh"].shift(close)
        df_mod["idm_vwap_rmean4"] = df_mod["idm_vwap_qh"].shift(close).rolling(4).mean()
        feats = BASE_FEATS + ["idm_vwap_close1", "idm_vwap_close2",
                              "idm_spread_close", "idm_momentum",
                              "idm_volume_close", "idm_vwap_rmean4"]
        safe = "SAFE" if close >= 8 else "BORDERLINE"
        rpt(f"Kitchen sink close={close} [{safe}]", run_bt(df_mod, feats))


if __name__ == "__main__":
    main()
