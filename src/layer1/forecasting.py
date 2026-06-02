"""
Layer 1 — Point Forecasting
===========================

Two models are trained: CaseTime and Turnover. XGBoost is used when available;
otherwise the code falls back to scikit-learn models so Layer 1 can still run.
Validation uses temporal folds by week_index, never random K-fold leakage.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from feature_engineering import feature_columns


TARGETS = ['casetime_min', 'turnover_min']


def _rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def _default_xgb_params(seed: int) -> dict[str, Any]:
    return {
        'n_estimators': 80,
        'max_depth': 3,
        'learning_rate': 0.06,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'reg_alpha': 0.01,
        'reg_lambda': 1.2,
        'gamma': 0.2,
        'objective': 'reg:squarederror',
        'random_state': seed,
        'verbosity': 0,
        'n_jobs': 1,
        'tree_method': 'hist',
    }


def _make_model(config: dict, params: dict | None = None):
    """Build the point-forecast model with robust fallbacks."""
    layer_cfg = config.get('layer1', {}) if isinstance(config, dict) else {}
    seed = int(layer_cfg.get('random_seed', 42))
    preferred = str(layer_cfg.get('point_model', 'xgboost')).lower()

    if preferred in {'xgboost', 'xgb'} and _xgboost_available():
        from xgboost import XGBRegressor
        p = _default_xgb_params(seed)
        if params:
            p.update(params)
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', XGBRegressor(**p)),
        ])
        return model, 'xgboost', p

    if preferred in {'random_forest', 'rf'}:
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(
                n_estimators=150, min_samples_leaf=3, random_state=seed, n_jobs=1
            )),
        ])
        return model, 'sklearn_random_forest', {}

    if preferred in {'hist_gradient_boosting', 'hgb', 'histgb'}:
        try:
            model = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', HistGradientBoostingRegressor(
                    max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
                    l2_regularization=0.01, random_state=seed,
                )),
            ])
            return model, 'sklearn_hist_gradient_boosting', {}
        except Exception:
            pass

    # Safe fallback: fast and deterministic. This is also used when XGBoost is
    # unavailable so Layer 1 validation can still run on a clean environment.
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', Ridge(alpha=1.0)),
    ])
    return model, 'sklearn_ridge', {}


def _temporal_folds(df: pd.DataFrame, n_splits: int = 3, min_train_weeks: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    weeks = sorted(int(w) for w in df['week_index'].dropna().unique())
    if len(weeks) < 3:
        return []

    n_splits = max(1, min(int(n_splits), len(weeks) - 1))
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    # Expanding-window folds over unique week labels.
    val_weeks_chunks = np.array_split(weeks[max(1, len(weeks) - n_splits):], n_splits)
    for chunk in val_weeks_chunks:
        val_weeks = [int(x) for x in chunk]
        if not val_weeks:
            continue
        train_weeks = [w for w in weeks if w < min(val_weeks)]
        if len(train_weeks) < min_train_weeks:
            train_weeks = [w for w in weeks if w < min(val_weeks)]
        if not train_weeks:
            continue
        tr_idx = df.index[df['week_index'].isin(train_weeks)].to_numpy()
        va_idx = df.index[df['week_index'].isin(val_weeks)].to_numpy()
        if len(tr_idx) and len(va_idx):
            folds.append((tr_idx, va_idx))

    if not folds:
        last_week = weeks[-1]
        tr_idx = df.index[df['week_index'] < last_week].to_numpy()
        va_idx = df.index[df['week_index'] == last_week].to_numpy()
        if len(tr_idx) and len(va_idx):
            folds.append((tr_idx, va_idx))
    return folds


def _try_optuna_params(features: pd.DataFrame, cols: list[str], target: str, folds, config: dict, out: Path) -> dict | None:
    layer_cfg = config.get('layer1', {}) if isinstance(config, dict) else {}
    n_trials = int(layer_cfg.get('optuna_trials', 0))
    timeout = layer_cfg.get('optuna_timeout_sec', None)
    seed = int(layer_cfg.get('random_seed', 42))

    if n_trials <= 0 or not _xgboost_available():
        return None

    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except Exception as exc:
        warnings.warn(f'Optuna skipped: {exc}')
        return None

    study_name = f'layer1_{target}'
    storage = f'sqlite:///{out / "optuna_study.db"}'

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 1e-1, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }
        rmses = []
        for k, (tr_idx, va_idx) in enumerate(folds):
            model, _, _ = _make_model(config, params=params)
            model.fit(features.loc[tr_idx, cols], features.loc[tr_idx, target])
            pred = np.maximum(0.0, model.predict(features.loc[va_idx, cols]))
            rmses.append(_rmse(features.loc[va_idx, target], pred))
            trial.report(float(np.mean(rmses)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(rmses))

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction='minimize',
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=max(3, min(10, n_trials // 4))),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    return dict(study.best_params) if study.best_trial else None


def train_point_forecasts(features: pd.DataFrame, out_dir, config: dict) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Keep target rows first.  In weakly-linked case files some provider/day
    # groups have only one synthetic/history-distributed row, so weeks_observed
    # can be 0 for every row.  That should not crash Layer 1; it should run a
    # cold-start validation model and clearly record the situation in metrics.
    f_all = features.copy()
    for t in TARGETS:
        f_all[t] = pd.to_numeric(f_all[t], errors='coerce')
    f_all = f_all[f_all[TARGETS].notna().all(axis=1)].copy()

    f = f_all[f_all['weeks_observed'].fillna(0) >= 1].copy() if 'weeks_observed' in f_all.columns else f_all.copy()
    cold_start_mode = False
    if f.empty and not f_all.empty:
        warnings.warn(
            'No rows have weeks_observed >= 1. Falling back to cold-start training on all target rows. '
            'This usually means the cases file has provider/day totals but no case timestamp or block_id.'
        )
        f = f_all.copy()
        cold_start_mode = True

    cols = feature_columns(f)
    if not cols:
        raise ValueError('No numeric feature columns were generated for Layer 1.')
    if f.empty:
        raise ValueError(
            'No trainable Layer 1 target rows were generated. Check cases provider_id, day_of_week, '
            'block_case_minutes, and turnover_time columns.'
        )

    min_rows = int(config.get('layer1', {}).get('min_history_rows_for_model', 30))
    if len(f) < min_rows:
        warnings.warn(f'Only {len(f)} feature rows (< {min_rows}); metrics may be unstable.')

    folds = _temporal_folds(
        f,
        n_splits=int(config.get('layer1', {}).get('cv_splits', 3)),
        min_train_weeks=int(config.get('layer1', {}).get('min_train_weeks', 4)),
    )
    if not folds:
        # Sparse fallback: deterministic 80/20 split by sorted row order.
        cut = max(1, int(0.8 * len(f)))
        folds = [(f.index[:cut].to_numpy(), f.index[cut:].to_numpy())]

    metrics: dict[str, Any] = {'feature_columns': cols, 'fold_count': len(folds), 'cold_start_mode': bool(cold_start_mode)}
    predictions = f[[
        'provider_id', 'service_line', 'week_start', 'week_index',
        'day_of_week', 'casetime_min', 'turnover_min', 'allocated_min',
    ]].copy()
    predictions['validation_fold'] = -1
    best_params_all: dict[str, Any] = {}
    models: dict[str, Any] = {}

    for target in TARGETS:
        print(f'  tuning/training target: {target}')
        best_params = _try_optuna_params(f, cols, target, folds, config, out)
        best_params_all[target] = best_params or {}

        cv_pred = pd.Series(index=f.index, dtype=float)
        train_rmse_by_fold = []
        for fold_id, (tr_idx, va_idx) in enumerate(folds):
            model, model_name, used_params = _make_model(config, params=best_params)
            model.fit(f.loc[tr_idx, cols], f.loc[tr_idx, target])
            pred = np.maximum(0.0, model.predict(f.loc[va_idx, cols]))
            cv_pred.loc[va_idx] = pred
            tr_pred = np.maximum(0.0, model.predict(f.loc[tr_idx, cols]))
            train_rmse_by_fold.append(_rmse(f.loc[tr_idx, target], tr_pred))
            predictions.loc[va_idx, 'validation_fold'] = fold_id

        valid_mask = cv_pred.notna()
        predictions[f'pred_{target}'] = cv_pred.values
        predictions[f'residual_{target}'] = predictions[target] - predictions[f'pred_{target}']

        final_model, model_name, used_params = _make_model(config, params=best_params)
        final_model.fit(f[cols], f[target])
        final_train_pred = np.maximum(0.0, final_model.predict(f[cols]))

        if valid_mask.any():
            rmse = _rmse(f.loc[valid_mask, target], cv_pred.loc[valid_mask])
            mae = float(mean_absolute_error(f.loc[valid_mask, target], cv_pred.loc[valid_mask]))
            n_valid = int(valid_mask.sum())
        else:
            rmse = np.nan
            mae = np.nan
            n_valid = 0

        train_rmse = _rmse(f[target], final_train_pred)
        gap = float(rmse - train_rmse) if np.isfinite(rmse) else np.nan
        metrics[target] = {
            'model': model_name,
            'rmse_cv': float(rmse) if np.isfinite(rmse) else None,
            'mae_cv': float(mae) if np.isfinite(mae) else None,
            'rmse_train_final': float(train_rmse),
            'generalization_gap_rmse': gap if np.isfinite(gap) else None,
            'n_train_final': int(len(f)),
            'n_valid_oof': n_valid,
            'best_params': best_params or used_params,
        }
        print(f'  {target}: model={model_name}  CV_RMSE={rmse:.1f}  CV_MAE={mae:.1f}  train_RMSE={train_rmse:.1f}')

        joblib.dump(final_model, out / f'{target}_model.joblib')
        models[target] = final_model

    predictions.to_csv(out / 'validation_predictions.csv', index=False)
    predictions.to_csv(out / 'residuals.csv', index=False)

    with open(out / 'model_metrics.json', 'w', encoding='utf-8') as fp:
        json.dump(metrics, fp, indent=2, default=str)
    with open(out / 'xgboost_best_params.json', 'w', encoding='utf-8') as fp:
        json.dump(best_params_all, fp, indent=2, default=str)

    return {
        'models': models,
        'metrics': metrics,
        'predictions': predictions,
        'feature_columns': cols,
    }


def make_next_week_forecasts(
    features: pd.DataFrame,
    models: dict,
    feature_cols: list[str],
    out_dir,
    forecast_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create next-week point forecasts from explicit forecast feature rows."""
    latest = forecast_features.copy() if forecast_features is not None else None
    if latest is None or latest.empty:
        latest = (
            features.sort_values('week_index')
            .groupby(['provider_id', 'day_of_week'], as_index=False, dropna=False)
            .tail(1)
            .copy()
        )
        latest['forecast_week_index'] = latest['week_index'] + 1

    for c in feature_cols:
        if c not in latest.columns:
            latest[c] = np.nan

    for target in TARGETS:
        latest[f'mu_{target}'] = np.maximum(0.0, models[target].predict(latest[feature_cols]))

    keep = [
        'provider_id', 'service_line', 'day_of_week', 'week_index',
        'forecast_week_index', 'rotation_phase',
        'mu_casetime_min', 'mu_turnover_min',
        'allocated_min', 'source_mode', 'weeks_observed',
        'exception_rate_series', 'provider_exception_rate',
    ]
    out_df = latest[[c for c in keep if c in latest.columns]].copy()
    out_df.to_csv(Path(out_dir) / 'point_forecasts.csv', index=False)
    return out_df.reset_index(drop=True)
