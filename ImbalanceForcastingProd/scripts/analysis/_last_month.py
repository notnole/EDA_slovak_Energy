"""Evaluate all model configs on the last month of data."""
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


def trade_pnl(test):
    surplus = test["spread_pred"] <= -3
    deficit = test["spread_pred"] >= 3
    trades = test[surplus | deficit].copy()
    if len(trades) == 0:
        return pd.Series(dtype=float), pd.DataFrame()
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
    daily = trades.groupby(trades.index.date)["pnl"].sum()
    return daily, trades


def report(daily, trades, label):
    if len(daily) == 0:
        print(f"  {label}: no trades")
        return
    nd = len(daily)
    total = daily.sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    dd = (daily.cumsum() - daily.cumsum().cummax()).min()
    prof = (daily > 0).sum()
    wr = (trades["pnl"] > 0).mean()
    print(f"  {label:30s}  {total/nd:>+6.0f}/d  Sh={sharpe:5.1f}  "
          f"win={wr:.0%}  DD={dd:>+6.0f}  prof={prof}/{nd}  {len(trades)}t")


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

    TEST_START = "2026-03-10"
    TEST_END = "2026-04-10"
    TRAIN_END = TEST_START

    train = df_base[df_base.index < TRAIN_END].dropna(
        subset=["target", f"proxy_lag{LEAD+1}"]
    )
    test = df_base[
        (df_base.index >= TEST_START) & (df_base.index < TEST_END)
    ].dropna(subset=[f"proxy_lag{LEAD+1}"])
    test = test[test["exec_spread"].notna() & (test["exec_spread"] <= 10)]
    sp_train = train.dropna(subset=["spread_target"])
    sp_train = sp_train[sp_train["imb_settlement_price"].abs() <= 5000]

    print(f"Test: {TEST_START} to {TEST_END}, {test.index.normalize().nunique()} days")
    print(f"Train: {len(sp_train)} rows\n")

    # Adaptive top-30
    m_full = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **LGB_PARAMS)
    m_full.fit(sp_train[clean_features].values, sp_train["spread_target"].values)
    imp = pd.Series(
        m_full.feature_importances_, index=clean_features
    ).sort_values(ascending=False)

    configs = {
        "Fixed 30": [f for f in FIXED_30 if f in all_fc],
        "Adaptive top-20": imp.head(20).index.tolist(),
        "Adaptive top-30": imp.head(30).index.tolist(),
        "All 121": clean_features,
    }

    print("=== LAST MONTH (Mar 10 - Apr 9) ===\n")
    best_label = None
    best_daily = None
    best_trades = None
    best_pnl = -1e9

    for name, feats in configs.items():
        m = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **LGB_PARAMS)
        m.fit(sp_train[feats].values, sp_train["spread_target"].values)
        t = test.copy()
        t["spread_pred"] = m.predict(t[feats].values)
        daily, trades = trade_pnl(t)
        report(daily, trades, name)
        if daily.sum() > best_pnl:
            best_pnl = daily.sum()
            best_label = name
            best_daily = daily
            best_trades = trades

    # Random
    rng = np.random.RandomState(42)
    df_rand = df_base.copy()
    for f in clean_features:
        df_rand[f] = rng.normal(0, 1, len(df_rand))
    train_r = df_rand[df_rand.index < TRAIN_END].dropna(
        subset=["target", f"proxy_lag{LEAD+1}"]
    )
    sp_r = train_r.dropna(subset=["spread_target"])
    sp_r = sp_r[sp_r["imb_settlement_price"].abs() <= 5000]
    m_r = lgb.LGBMRegressor(objective="quantile", alpha=0.50, **LGB_PARAMS)
    m_r.fit(sp_r[clean_features].values, sp_r["spread_target"].values)
    t_r = test.copy()
    t_r["spread_pred"] = m_r.predict(t_r[clean_features].values)
    daily_r, trades_r = trade_pnl(t_r)
    report(daily_r, trades_r, "Random baseline")

    # Daily breakdown for best
    print(f"\n=== DAILY BREAKDOWN: {best_label} ===")
    for date, pnl in best_daily.items():
        dow = pd.Timestamp(date).strftime("%a")
        day_trades = best_trades[best_trades.index.date == date]
        nt = len(day_trades)
        wr = (day_trades["pnl"] > 0).mean() if nt > 0 else 0
        print(f"  {date} ({dow}): {pnl:>+7,.0f}  {nt}t  {wr:.0%} win")


if __name__ == "__main__":
    main()
