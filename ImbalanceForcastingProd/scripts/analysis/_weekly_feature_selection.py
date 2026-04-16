"""Weekly adaptive feature selection comparison."""
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

LEAKY = {"spread_da_imb_lag", "imb_price_rmean4"}

FIXED_30 = [
    "da_price_qh", "temp_national_change6h", "da_demand", "nowcast_momentum_h2h3",
    "nowcast_trend_h2_h5", "idm_vwap_lag", "da_flow_cz", "prod_rmean8",
    "spread_da_idm_lag", "nowcast_h5", "damas_fe_rmean4", "reg_rmean8",
    "nowcast_momentum_h3h4", "da_price_qh_dev_hourly", "proxy_lag9",
    "proxy_lag96_diff", "da_price_qh_diff_next", "dow_sin", "wind_national",
    "temp_surprise_lag", "proxy_lag15", "prod_momentum", "nowcast_pred_rmean4",
    "hour_sin", "da_net_import", "is_weekend", "temp_bratislava", "da_supply",
    "xborder_vol", "xborder_deviation",
]


def trade_pnl(test, pred_col="spread_pred"):
    surplus = test[pred_col] <= -3
    deficit = test[pred_col] >= 3
    trades = test[surplus | deficit].copy()
    if len(trades) == 0:
        return 0.0
    trades["size"] = trades[pred_col].abs().clip(upper=5)
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
    return trades["pnl"].sum()


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

    clean_features = [f for f in all_fc if f not in LEAKY]

    weeks = pd.date_range("2026-02-02", "2026-04-06", freq="W-MON")
    configs = ["Fixed 30", "Weekly top-20", "Weekly top-30", "Weekly top-40", "All 121"]
    results = {k: [] for k in configs}
    week_labels = []

    print("=== WEEKLY ADAPTIVE FEATURE SELECTION ===\n")

    prev_set = None
    for wi, week_start in enumerate(weeks):
        week_end = min(week_start + pd.Timedelta(days=7), pd.Timestamp("2026-04-10"))
        train_end = str(week_start.date())

        train = df_base[df_base.index < train_end].dropna(
            subset=["target", f"proxy_lag{LEAD+1}"]
        )
        test = df_base[
            (df_base.index >= str(week_start.date()))
            & (df_base.index < str(week_end.date()))
        ].dropna(subset=[f"proxy_lag{LEAD+1}"])
        test = test[test["exec_spread"].notna() & (test["exec_spread"] <= 10)]
        if len(test) < 20:
            continue

        sp_train = train.dropna(subset=["spread_target"])
        sp_train = sp_train[sp_train["imb_settlement_price"].abs() <= 5000]

        # Full model for feature importance
        m_full = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **LGB_PARAMS)
        m_full.fit(sp_train[clean_features].values, sp_train["spread_target"].values)
        imp = pd.Series(
            m_full.feature_importances_, index=clean_features
        ).sort_values(ascending=False)

        top20 = imp.head(20).index.tolist()
        top30 = imp.head(30).index.tolist()
        top40 = imp.head(40).index.tolist()

        feature_sets = {
            "Fixed 30": [f for f in FIXED_30 if f in all_fc],
            "Weekly top-20": top20,
            "Weekly top-30": top30,
            "Weekly top-40": top40,
            "All 121": clean_features,
        }

        label = f"{week_start.strftime('%b %d')}-{week_end.strftime('%b %d')}"
        week_labels.append(label)

        for name in configs:
            feats = feature_sets[name]
            m = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **LGB_PARAMS)
            m.fit(sp_train[feats].values, sp_train["spread_target"].values)
            test_c = test.copy()
            test_c["spread_pred"] = m.predict(test_c[feats].values)
            pnl = trade_pnl(test_c)
            results[name].append(pnl)

        # Feature stability
        cur_set = set(top30)
        if prev_set is not None:
            overlap = len(cur_set & prev_set)
            print(
                f"  {label}: top-5 = {', '.join(imp.head(5).index)}"
                f"  | overlap={overlap}/30"
            )
        else:
            print(f"  {label}: top-5 = {', '.join(imp.head(5).index)}")
        prev_set = cur_set

    # Summary table
    print("\n=== WEEKLY P&L ===")
    hdr = f"{'Week':>20s}"
    for name in configs:
        hdr += f"  {name:>14s}"
    print(hdr)
    print("-" * len(hdr))

    for wi, label in enumerate(week_labels):
        row = f"{label:>20s}"
        for name in configs:
            row += f"  {results[name][wi]:>+14,.0f}"
        print(row)

    n_weeks = len(week_labels)
    days = n_weeks * 7

    print("-" * len(hdr))
    row = f"{'TOTAL':>20s}"
    for name in configs:
        row += f"  {sum(results[name]):>+14,.0f}"
    print(row)

    row = f"{'EUR/day':>20s}"
    for name in configs:
        row += f"  {sum(results[name])/days:>+14,.0f}"
    print(row)

    row = f"{'Losing weeks':>20s}"
    for name in configs:
        lw = sum(1 for v in results[name] if v < 0)
        row += f"  {lw:>10d}/{n_weeks:<3d}"
    print(row)

    row = f"{'Sharpe (weekly)':>20s}"
    for name in configs:
        s = pd.Series(results[name])
        sh = s.mean() / s.std() * np.sqrt(52) if s.std() > 0 else 0
        row += f"  {sh:>14.1f}"
    print(row)

    row = f"{'Worst week':>20s}"
    for name in configs:
        row += f"  {min(results[name]):>+14,.0f}"
    print(row)


if __name__ == "__main__":
    main()
