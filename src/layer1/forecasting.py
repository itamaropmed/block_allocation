"""
Layer 1 — Forecasting  (standalone, no src imports)
====================================================
Fix: changed  from src.layer1.feature_engineering import feature_columns
          to  from feature_engineering import feature_columns
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from feature_engineering import feature_columns   # same-folder import


def _make_model(config: dict):
    """
    Build the point-forecast model.
    Uses XGBoost when available; falls back to Ridge on sklearn pipeline.
    Both are wrapped in a SimpleImputer so NaN features don't crash the fit.
    """
    seed = int(config.get('layer1', {}).get('random_seed', 42))
    try:
        from xgboost import XGBRegressor
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                reg_alpha=0.01, reg_lambda=1.2, gamma=0.2,
                random_state=seed, verbosity=0,
            )),
        ])
        return model, 'xgboost'
    except ImportError:
        pass
    # Fallback: Ridge — fast and stable
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', Ridge(alpha=1.0)),
    ])
    return model, 'sklearn_ridge'


def train_point_forecasts(features: pd.DataFrame, out_dir, config: dict) -> dict:
    out  = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = feature_columns(features)
    if not cols:
        raise ValueError('No numeric feature columns were generated.')

    f = features[features['weeks_observed'].fillna(0) >= 1].copy()
    min_rows = int(config.get('layer1', {}).get('min_history_rows_for_model', 30))
    if len(f) < min_rows:
        warnings.warn(f'Only {len(f)} feature rows (< {min_rows}); metrics may be unstable.')

    max_week = f['week_index'].max()
    train = f[f['week_index'] < max_week].copy()
    valid = f[f['week_index'] == max_week].copy()
    if train.empty or valid.empty:
        rng   = int(config.get('layer1', {}).get('random_seed', 42))
        train = f.sample(frac=0.8, random_state=rng)
        valid = f.drop(train.index)

    metrics = {'feature_columns': cols}
    predictions = valid[[
        'provider_id', 'service_line', 'week_start', 'week_index',
        'day_of_week', 'casetime_min', 'turnover_min',
    ]].copy()
    models = {}

    for target in ['casetime_min', 'turnover_min']:
        model, model_name = _make_model(config)
        model.fit(train[cols], train[target])
        pred = np.maximum(0.0, model.predict(valid[cols]))
        predictions[f'pred_{target}'] = pred
        rmse = float(mean_squared_error(valid[target], pred) ** 0.5)
        mae  = float(mean_absolute_error(valid[target], pred))
        metrics[target] = {
            'model':   model_name,
            'rmse':    rmse,
            'mae':     mae,
            'n_train': int(len(train)),
            'n_valid': int(len(valid)),
        }
        print(f'  {target}: model={model_name}  RMSE={rmse:.1f}  MAE={mae:.1f}')
        joblib.dump(model, out / f'{target}_model.joblib')
        models[target] = model

    predictions.to_csv(out / 'validation_predictions.csv', index=False)
    with open(out / 'model_metrics.json', 'w', encoding='utf-8') as fp:
        json.dump(metrics, fp, indent=2)

    return {
        'models': models, 'metrics': metrics,
        'predictions': predictions, 'feature_columns': cols,
    }


def make_next_week_forecasts(
    features: pd.DataFrame,
    models: dict,
    feature_cols: list[str],
    out_dir,
) -> pd.DataFrame:
    latest = (
        features.sort_values('week_index')
        .groupby(['provider_id', 'day_of_week'], as_index=False)
        .tail(1)
        .copy()
    )
    latest['forecast_week_index'] = latest['week_index'] + 1

    for target in ['casetime_min', 'turnover_min']:
        latest[f'mu_{target}'] = np.maximum(
            0.0, models[target].predict(latest[feature_cols])
        )

    keep = [
        'provider_id', 'service_line', 'day_of_week',
        'forecast_week_index', 'mu_casetime_min', 'mu_turnover_min',
        'allocated_min', 'source_mode',
    ]
    out_df = latest[[c for c in keep if c in latest.columns]].copy()
    out_df.to_csv(Path(out_dir) / 'point_forecasts.csv', index=False)
    return out_df
