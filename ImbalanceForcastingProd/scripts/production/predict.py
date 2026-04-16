"""
Production Prediction Module
==============================

Accepts pre-loaded pandas DataFrames for each data source, engineers
features using the SAME code as the training pipeline (build_features),
and returns a spread prediction.

NO file I/O happens here. The caller is responsible for loading data.

Usage:
    from predict import SpreadPredictor

    predictor = SpreadPredictor(model_path="path/to/model.joblib")
    result = predictor.predict(
        regulation_3min=df_reg,     # datetime index, column: regulation_mw
        load_3min=df_load,          # datetime index, column: load_mw
        production_3min=df_prod,    # datetime index, column: production_mw (optional)
        export_import_3min=df_xi,   # datetime index, column: export_import_mw (optional)
        solar_hourly=df_solar,      # datetime index, columns: solar_surprise_mw
        damas_load=df_damas,        # datetime index, columns: forecast_error_mw, forecast_load_mw
        da_prices=df_da,            # datetime index, columns: price_eur_mwh, demand_mw, ...
        market_spreads=df_mkt,      # datetime index, columns: idm_vwap, imb_settlement_price
        weather=df_weather,         # datetime index, columns: temperature, temp_national_mean, ...
        nowcast_oos=df_nowcast,     # datetime index, columns: nowcast_pred_error, nowcast_h3_pred, ...
        imbalance=df_imb,           # datetime index, columns: imbalance_mwh, imb_settle_price
    )
    # result: dict with 'prediction', 'timestamp', 'signal', 'features'
"""

import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# Import the EXACT training pipeline functions
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import train_multi_lead as tml
from train_multi_lead import build_features


# Lead = 8 (2h ahead). Fixed for the spread model.
LEAD = 8
SCADA_SHIFT = LEAD + 1

# 30 features: Top-30 permutation importance (P&L-driven) on full 64k dataset,
# after removing 2 leaky features (imb_price_rmean4, spread_da_imb_lag) and
# correlation pruning (|r|>0.95). Must match train_stacked_model.py exactly.
SELECTED_FEATURES = [
    'da_price_qh', 'temp_national_change6h', 'da_demand', 'nowcast_momentum_h2h3',
    'nowcast_trend_h2_h5', 'idm_vwap_lag', 'da_flow_cz', 'prod_rmean8',
    'spread_da_idm_lag', 'nowcast_h5', 'damas_fe_rmean4', 'reg_rmean8',
    'nowcast_momentum_h3h4', 'da_price_qh_dev_hourly', 'proxy_lag9',
    'proxy_lag96_diff', 'da_price_qh_diff_next', 'dow_sin', 'wind_national',
    'temp_surprise_lag', 'proxy_lag15', 'prod_momentum', 'nowcast_pred_rmean4',
    'hour_sin', 'da_net_import', 'is_weekend', 'temp_bratislava', 'da_supply',
    'xborder_vol', 'xborder_deviation',
]


def _dedup_index(df):
    """Remove duplicate index entries, keeping the last."""
    return df[~df.index.duplicated(keep='last')]


class SpreadPredictor:
    """Production spread predictor. Uses the exact same feature engineering as training."""

    def __init__(self, model_path, train_end=None):
        """
        Args:
            model_path: path to trained LightGBM .joblib file
            train_end: date string for the train/test boundary. Controls the
                       frozen baselines (proxy hourly mean, etc.) used in
                       build_features. If None, reads from metadata JSON
                       next to the model file. For production models trained
                       on all data, this is set far in the future ('2099-01-01').
        """
        self.model = joblib.load(model_path)

        if train_end is None:
            # Try to load from metadata
            meta_path = Path(model_path).with_name(
                Path(model_path).stem + '_metadata.json')
            if meta_path.exists():
                import json
                with open(meta_path) as f:
                    meta = json.load(f)
                # Production model: use far-future so all data is "training"
                self.train_end = '2099-01-01'
            else:
                self.train_end = '2099-01-01'
        else:
            self.train_end = train_end

    def predict(self, regulation_3min, load_3min, production_3min=None,
                export_import_3min=None, solar_hourly=None, damas_load=None,
                da_prices=None, market_spreads=None, weather=None,
                nowcast_oos=None, imbalance=None):
        """
        Build features from raw DataFrames and return prediction for the latest row.

        All DataFrames must have a datetime index sorted chronologically.
        Provide enough history for rolling windows (48h+ recommended).

        Returns:
            dict with keys:
              - prediction: float, the spread prediction (EUR/MWh)
              - timestamp: the prediction timestamp (15-min period)
              - signal: 'surplus' | 'deficit' | 'no_trade'
              - features: pd.Series of the 30 feature values used
        """
        # Build the data dict in the same format as load_all_data() returns
        data = self._prepare_data_dict(
            regulation_3min, load_3min, production_3min, export_import_3min,
            solar_hourly, damas_load, da_prices, market_spreads,
            weather, nowcast_oos, imbalance,
        )

        # Call the EXACT same build_features from the training pipeline
        # Set the module-level train/test boundary to match training
        tml.TRAIN_END = self.train_end
        tml.TEST_START = self.train_end
        feat_df, feature_cols = build_features(data, LEAD)

        # Validate selected features exist
        missing = [f for f in SELECTED_FEATURES if f not in feat_df.columns]
        if missing:
            raise ValueError(f"Missing features after engineering: {missing}")

        # Get the latest valid row
        last_valid = feat_df.dropna(subset=[f'proxy_lag{SCADA_SHIFT}'])
        if len(last_valid) == 0:
            raise ValueError("No valid rows after feature engineering — check input data coverage")

        row = last_valid.iloc[[-1]]
        ts = row.index[0]
        X = row[SELECTED_FEATURES].values
        pred = float(self.model.predict(X)[0])

        if pred <= -3:
            signal = 'surplus'
        elif pred >= 3:
            signal = 'deficit'
        else:
            signal = 'no_trade'

        return {
            'prediction': pred,
            'timestamp': ts,
            'signal': signal,
            'features': row[SELECTED_FEATURES].iloc[0],
        }

    def predict_batch(self, regulation_3min, load_3min, production_3min=None,
                      export_import_3min=None, solar_hourly=None, damas_load=None,
                      da_prices=None, market_spreads=None, weather=None,
                      nowcast_oos=None, imbalance=None):
        """
        Build features and return predictions for ALL valid rows.
        Useful for backtesting or batch prediction.

        Returns:
            pd.DataFrame with columns: prediction, signal + all 52 features
        """
        data = self._prepare_data_dict(
            regulation_3min, load_3min, production_3min, export_import_3min,
            solar_hourly, damas_load, da_prices, market_spreads,
            weather, nowcast_oos, imbalance,
        )

        tml.TRAIN_END = self.train_end
        tml.TEST_START = self.train_end
        feat_df, feature_cols = build_features(data, LEAD)

        valid = feat_df.dropna(subset=[f'proxy_lag{SCADA_SHIFT}'])
        if len(valid) == 0:
            raise ValueError("No valid rows after feature engineering")

        X = valid[SELECTED_FEATURES].values
        preds = self.model.predict(X)

        result = valid[SELECTED_FEATURES].copy()
        result['prediction'] = preds
        result['signal'] = np.where(preds <= -3, 'surplus',
                           np.where(preds >= 3, 'deficit', 'no_trade'))
        return result

    # ------------------------------------------------------------------
    # INTERNAL: Build the data dict matching load_all_data() format
    # ------------------------------------------------------------------

    def _prepare_data_dict(self, regulation_3min, load_3min, production_3min,
                           export_import_3min, solar_hourly, damas_load,
                           da_prices, market_spreads, weather, nowcast_oos,
                           imbalance):
        """
        Aggregate raw DataFrames into the same format that load_all_data() returns.
        This dict is then passed directly to build_features().
        """
        # --- Regulation -> 15-min ---
        reg_15 = regulation_3min['regulation_mw'].resample('15min').agg(
            ['mean', 'std', 'min', 'max', 'count'])
        reg_15.columns = ['reg_mean', 'reg_std', 'reg_min', 'reg_max', 'reg_count']
        reg_15 = reg_15[reg_15['reg_count'] >= 2]
        reg_15['proxy'] = -0.25 * reg_15['reg_mean']

        # --- Load -> 15-min ---
        load_15 = load_3min['load_mw'].resample('15min').agg(['mean', 'std', 'min', 'max'])
        load_15.columns = ['load_mean', 'load_std', 'load_min', 'load_max']

        # --- Production -> 15-min (optional) ---
        prod_15 = None
        if production_3min is not None and len(production_3min) > 0:
            prod_15 = production_3min['production_mw'].resample('15min').agg(['mean', 'std'])
            prod_15.columns = ['prod_mean', 'prod_std']

        # --- Export/Import -> 15-min (optional) ---
        xi_15 = None
        if export_import_3min is not None and len(export_import_3min) > 0:
            xi_15 = export_import_3min['export_import_mw'].resample('15min').agg(['mean', 'std'])
            xi_15.columns = ['xborder_mean', 'xborder_std']

        # --- Hourly sources -> 15-min forward-fill ---
        solar_15 = None
        if solar_hourly is not None and len(solar_hourly) > 0:
            s = _dedup_index(solar_hourly)
            solar_15 = s.resample('15min').ffill()

        damas_15 = None
        if damas_load is not None and len(damas_load) > 0:
            d = _dedup_index(damas_load)
            damas_15 = d[['forecast_error_mw', 'forecast_error_pct', 'forecast_load_mw']].resample('15min').ffill()

        da_15 = None
        if da_prices is not None and len(da_prices) > 0:
            da = _dedup_index(da_prices)
            keep = [c for c in ['price_eur_mwh', 'demand_mw', 'supply_mw',
                    'net_flow_cz', 'net_flow_pl', 'net_flow_hu', 'net_import',
                    'price_lag24', 'price_change_24h'] if c in da.columns]
            da_15 = da[keep].resample('15min').ffill()

        mkt_15 = None
        if market_spreads is not None and len(market_spreads) > 0:
            mkt = _dedup_index(market_spreads)
            keep = [c for c in ['idm_vwap', 'idm_volume_mwh', 'imb_settlement_price',
                    'spread_da_idm', 'spread_idm_imb', 'spread_da_imb'] if c in mkt.columns]
            mkt_15 = mkt[keep].resample('15min').ffill()

        weather_15 = None
        if weather is not None and len(weather) > 0:
            weather_15 = _dedup_index(weather).resample('15min').ffill()

        # Nowcast: each horizon must be deduped and resampled independently
        # before joining — matches load_all_data() which loads each file separately.
        nowcast_15 = None
        if nowcast_oos is not None and len(nowcast_oos) > 0:
            base_cols = [c for c in ['nowcast_pred_error', 'nowcast_actual_error']
                         if c in nowcast_oos.columns]
            horizon_cols = [c for c in nowcast_oos.columns if c.startswith('nowcast_h')]
            # Resample base (H+2) independently
            base = _dedup_index(nowcast_oos[base_cols])
            nowcast_15 = base.resample('15min').ffill()
            # Resample each horizon independently, then join
            for col in horizon_cols:
                h_series = _dedup_index(nowcast_oos[[col]].dropna())
                h_15 = h_series.resample('15min').ffill()
                nowcast_15 = nowcast_15.join(h_15, how='left')

        # --- Imbalance labels ---
        imb_df = None
        if imbalance is not None and len(imbalance) > 0:
            imb_df = imbalance[['imbalance_mwh', 'imb_settle_price']].copy()

        return {
            'reg': reg_15, 'load': load_15, 'prod': prod_15, 'xborder': xi_15,
            'solar': solar_15, 'damas': damas_15, 'da': da_15, 'mkt': mkt_15,
            'weather': weather_15, 'nowcast': nowcast_15, 'imb': imb_df,
        }
