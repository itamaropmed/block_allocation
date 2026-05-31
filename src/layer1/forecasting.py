from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer

from src.layer1.feature_engineering import feature_columns


def _make_model(random_seed: int = 42):
    # Very fast baseline that is stable on every machine.
    # You can swap this to XGBRegressor after installing xgboost.
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', Ridge(alpha=1.0, random_state=random_seed)),
    ]), 'sklearn_ridge_baseline'


def train_point_forecasts(features: pd.DataFrame, out_dir: str | Path, config: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = feature_columns(features)
    if not cols:
        raise ValueError('No numeric feature columns were generated.')
    f = features.copy()
    f = f[f['weeks_observed'].fillna(0) >= 1].copy()
    if len(f) < int(config['layer1'].get('min_history_rows_for_model', 30)):
        warnings.warn('Very small feature table; model metrics may be unstable.')
    max_week = f['week_index'].max()
    train = f[f['week_index'] < max_week].copy()
    valid = f[f['week_index'] == max_week].copy()
    if train.empty or valid.empty:
        train = f.sample(frac=0.8, random_state=config['layer1'].get('random_seed', 42))
        valid = f.drop(train.index)
    metrics = {'feature_columns': cols}
    predictions = valid[['provider_id', 'service_line', 'week_start', 'week_index', 'day_of_week', 'casetime_min', 'turnover_min']].copy()
    models = {}
    for target in ['casetime_min', 'turnover_min']:
        model, model_name = _make_model(config['layer1'].get('random_seed', 42))
        model.fit(train[cols], train[target])
        pred = np.maximum(0.0, model.predict(valid[cols]))
        predictions[f'pred_{target}'] = pred
        rmse = mean_squared_error(valid[target], pred) ** 0.5
        mae = mean_absolute_error(valid[target], pred)
        metrics[target] = {'model': model_name, 'rmse': float(rmse), 'mae': float(mae), 'n_train': int(len(train)), 'n_valid': int(len(valid))}
        joblib.dump(model, out / f'{target}_model.joblib')
        models[target] = model
    predictions.to_csv(out / 'validation_predictions.csv', index=False)
    with open(out / 'model_metrics.json', 'w', encoding='utf-8') as fp:
        json.dump(metrics, fp, indent=2)
    return {'models': models, 'metrics': metrics, 'predictions': predictions, 'feature_columns': cols}


def make_next_week_forecasts(features: pd.DataFrame, models: dict, feature_cols: list[str], out_dir: str | Path) -> pd.DataFrame:
    latest = features.sort_values('week_index').groupby(['provider_id', 'day_of_week'], as_index=False).tail(1).copy()
    latest['forecast_week_index'] = latest['week_index'] + 1
    for target in ['casetime_min', 'turnover_min']:
        latest[f'mu_{target}'] = np.maximum(0.0, models[target].predict(latest[feature_cols]))
    cols = ['provider_id', 'service_line', 'day_of_week', 'forecast_week_index', 'mu_casetime_min', 'mu_turnover_min', 'allocated_min', 'source_mode']
    out = latest[cols].copy()
    out.to_csv(Path(out_dir) / 'point_forecasts.csv', index=False)
    return out
