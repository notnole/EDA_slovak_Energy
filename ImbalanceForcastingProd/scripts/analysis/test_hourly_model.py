"""Hourly target model with QH execution."""
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

FOLDS = [
    ("2025-04-01", "2025-04-01", "2025-07-01"),
    ("2025-07-01", "2025-07-01", "2025-10-01"),
    ("2025-10-01", "2025-10-01", "2026-01-01"),
    ("2026-01-01", "2026-01-01", "2026-02-01"),
    ("2026-02-01", "2026-02-01", "2026-03-01"),
    ("2026-03-01", "2026-03-01", "2026-04-10"),
]


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
    df_qh, all_fc = build_features(data, LEAD)
    df_qh = df_qh.join(ob_120, how="left")
    df_qh["imb_settlement_price"] = df_qh["imb_settle_price"]
    df_qh["hour"] = df_qh.index.floor("h")

    valid_qh = df_qh[
        df_qh["exec_spread"].notna() & (df_qh["exec_spread"] <= 10)
    ].copy()

    # --- Hourly target: avg(settlement) - avg(exec_mid) ---
    hourly_agg = valid_qh.groupby("hour").agg(
        settle_h=("imb_settlement_price", "mean"),
        exec_mid_h=("exec_mid", "mean"),
        n_qh=("exec_mid", "count"),
    )
    hourly_agg["spread_target_h"] = hourly_agg["settle_h"] - hourly_agg["exec_mid_h"]

    # Features: first QH per hour
    hourly_feats = df_qh.groupby("hour").first()[all_fc]
    hourly_feats = hourly_feats.join(hourly_agg)

    usable = hourly_feats[
        hourly_feats["spread_target_h"].notna()
        & hourly_feats["proxy_lag9"].notna()
        & (hourly_feats["n_qh"] >= 3)
    ].copy()

    feats_h = [f for f in FEATS if f in usable.columns]

    # QH execution data for P&L
    qh_exec = valid_qh[
        ["exec_bid", "exec_ask", "imb_settlement_price", "hour"]
    ].copy()

    print(f"Hourly dataset: {len(usable):,} rows, {len(feats_h)} features")
    print(f"Target std: {usable.spread_target_h.std():.1f}")
    print()

    # ================================================================
    # HOURLY MODEL
    # ================================================================
    print("=" * 80)
    print("HOURLY MODEL (target=hourly avg spread, execution=QH bid/ask)")
    print("=" * 80)

    all_oof_h = []
    for te, ps, pe in FOLDS:
        tr = usable[usable.index < te].dropna(subset=["spread_target_h"])
        pr = usable[(usable.index >= ps) & (usable.index < pe)]
        pr = pr[pr["proxy_lag9"].notna()]
        if len(tr) < 100 or len(pr) < 20:
            continue
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **LGB_PARAMS)
        m.fit(tr[feats_h].values, tr["spread_target_h"].values)
        pr = pr.copy()
        pr["spread_pred"] = m.predict(pr[feats_h].values)
        all_oof_h.append(pr)

    test_h = pd.concat(all_oof_h[-2:])

    for thresh in [3, 5, 8]:
        surplus_h = test_h["spread_pred"] <= -thresh
        deficit_h = test_h["spread_pred"] >= thresh
        traded = test_h[surplus_h | deficit_h].copy()
        if len(traded) == 0:
            continue

        traded["direction"] = np.where(traded["spread_pred"] > 0, "deficit", "surplus")
        traded["size"] = traded["spread_pred"].abs().clip(upper=5)

        # Execute at QH level
        qh_t = qh_exec[qh_exec["hour"].isin(traded.index)].copy()
        qh_t["direction"] = qh_t["hour"].map(traded["direction"])
        qh_t["size"] = qh_t["hour"].map(traded["size"])
        qh_t["pnl"] = 0.0
        s = qh_t["direction"] == "surplus"
        d = qh_t["direction"] == "deficit"
        qh_t.loc[s, "pnl"] = (
            (qh_t.loc[s, "exec_bid"] - qh_t.loc[s, "imb_settlement_price"])
            * qh_t.loc[s, "size"] / 4
        )
        qh_t.loc[d, "pnl"] = (
            (qh_t.loc[d, "imb_settlement_price"] - qh_t.loc[d, "exec_ask"])
            * qh_t.loc[d, "size"] / 4
        )

        nd = qh_t.index.normalize().nunique()
        daily = qh_t.groupby(qh_t.index.date)["pnl"].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        dd = (daily.cumsum() - daily.cumsum().cummax()).min()

        print(f"\n  Hourly |pred|>={thresh}:  {qh_t.pnl.sum()/nd:>+5.0f}/d  "
              f"Sh={sharpe:5.1f}  win={(qh_t.pnl > 0).mean():.0%}  "
              f"DD={dd:>+6.0f}  {len(traded)}h / {len(qh_t)}qh")

        qh_t["month"] = qh_t.index.to_period("M")
        for mo, g in qh_t.groupby("month"):
            mnd = g.index.normalize().nunique()
            print(f"    {mo}: {g.pnl.sum()/mnd:>+5.0f}/d  win={(g.pnl > 0).mean():.0%}  "
                  f"{len(g)}qh")

    # ================================================================
    # QH MODEL BASELINE
    # ================================================================
    print()
    print("=" * 80)
    print("QH MODEL BASELINE (for comparison)")
    print("=" * 80)

    df_qh["spread_target"] = df_qh["imb_settlement_price"] - df_qh["exec_mid"]
    all_oof_qh = []
    for te, ps, pe in FOLDS:
        tr = df_qh[df_qh.index < te].dropna(subset=["target", f"proxy_lag{LEAD+1}"])
        pr = df_qh[(df_qh.index >= ps) & (df_qh.index < pe)].dropna(
            subset=[f"proxy_lag{LEAD+1}"]
        )
        sp = tr.dropna(subset=["spread_target"])
        sp = sp[sp["imb_settlement_price"].abs() <= 5000]
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **LGB_PARAMS)
        m.fit(sp[feats_h].values, sp["spread_target"].values)
        pr = pr.copy()
        pr["spread_pred"] = m.predict(pr[feats_h].values)
        all_oof_qh.append(pr)

    test_qh = pd.concat(all_oof_qh[-2:])
    test_qh = test_qh[test_qh["exec_spread"].notna() & (test_qh["exec_spread"] <= 10)]

    for thresh in [3, 5, 8]:
        surplus = test_qh["spread_pred"] <= -thresh
        deficit = test_qh["spread_pred"] >= thresh
        trades = test_qh[surplus | deficit].copy()
        if len(trades) == 0:
            continue
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

        print(f"\n  QH |pred|>={thresh}:      {trades.pnl.sum()/nd:>+5.0f}/d  "
              f"Sh={sharpe:5.1f}  win={(trades.pnl > 0).mean():.0%}  "
              f"DD={dd:>+6.0f}  {len(trades)}qh")

        trades["month"] = trades.index.to_period("M")
        for mo, g in trades.groupby("month"):
            mnd = g.index.normalize().nunique()
            print(f"    {mo}: {g.pnl.sum()/mnd:>+5.0f}/d  win={(g.pnl > 0).mean():.0%}  "
                  f"{len(g)}qh")


if __name__ == "__main__":
    main()
