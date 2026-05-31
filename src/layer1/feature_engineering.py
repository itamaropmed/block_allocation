from __future__ import annotations

import numpy as np
import pandas as pd

from src.layer1.attention_pooling import attention_summary


def build_features(obs: pd.DataFrame, prelayer: dict, config: dict) -> tuple[pd.DataFrame, dict]:
    rows = []
    wq_case = 0.046
    wq_turn = 0.046

    sl_context = obs.groupby(['service_line', 'week_index', 'day_of_week']).agg(
        service_line_wk_mean=('casetime_util', 'mean')
    ).reset_index()

    for (pid, dow), g in obs.groupby(['provider_id', 'day_of_week'], dropna=False):
        g = g.sort_values('week_index').reset_index(drop=True)
        histories = {}
        for _, r in g.iterrows():
            phase = int(r['rotation_phase'])
            h = histories.setdefault(phase, {'case': [], 'turn': []})
            case_hist = pd.Series(h['case'], dtype=float)
            turn_hist = pd.Series(h['turn'], dtype=float)
            f = r.to_dict()
            for window in [4, 8, 12]:
                ch = case_hist.tail(window)
                th = turn_hist.tail(window)
                f[f'trailing_{window}w_mean_case'] = float(ch.mean()) if len(ch) else np.nan
                f[f'trailing_{window}w_std_case'] = float(ch.std(ddof=0)) if len(ch) else np.nan
                f[f'trailing_{window}w_min_case'] = float(ch.min()) if len(ch) else np.nan
                f[f'trailing_{window}w_max_case'] = float(ch.max()) if len(ch) else np.nan
                f[f'trailing_{window}w_mean_turn'] = float(th.mean()) if len(th) else np.nan
                f[f'trailing_{window}w_std_turn'] = float(th.std(ddof=0)) if len(th) else np.nan
            ac = attention_summary(case_hist.tail(12).values, wq_case)
            at = attention_summary(turn_hist.tail(12).values, wq_turn)
            f['attn_weighted_util'] = ac['weighted']
            f['attn_entropy'] = ac['entropy']
            f['attn_recency_bias'] = ac['recency_bias']
            f['attn_turn_weighted'] = at['weighted']
            f['attn_turn_entropy'] = at['entropy']
            f['attn_turn_recency'] = at['recency_bias']
            f['weeks_observed'] = int(len(case_hist))
            week_num = int(r['week_index']) + 1
            f['sin_woy'] = float(np.sin(2 * np.pi * week_num / 52.0))
            f['cos_woy'] = float(np.cos(2 * np.pi * week_num / 52.0))
            f['sin_woy_2'] = float(np.sin(4 * np.pi * week_num / 52.0))
            f['cos_woy_2'] = float(np.cos(4 * np.pi * week_num / 52.0))
            for d in range(7):
                f[f'dow_{d}'] = 1 if int(dow) == d else 0
            rows.append(f)
            h['case'].append(float(r['casetime_util']))
            h['turn'].append(float(r['turnover_util']))
    features = pd.DataFrame(rows)
    features = features.merge(sl_context, on=['service_line', 'week_index', 'day_of_week'], how='left')

    exc = prelayer.get('block_template', [])
    if exc:
        e = pd.DataFrame(exc)
        e_provider = e.groupby('dominant_provider_id')['exception_rate'].mean().rename('provider_exception_rate').reset_index()
        features = features.merge(e_provider, left_on='provider_id', right_on='dominant_provider_id', how='left')
        features.drop(columns=['dominant_provider_id'], inplace=True, errors='ignore')
    if 'provider_exception_rate' not in features.columns:
        features['provider_exception_rate'] = 0.0
    features['provider_exception_rate'] = features['provider_exception_rate'].fillna(0.0)
    return features.sort_values(['week_index', 'provider_id', 'day_of_week']).reset_index(drop=True), {'wq_case': wq_case, 'wq_turn': wq_turn}


def feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        'provider_id', 'service_line', 'week_start', 'source_mode',
        'casetime_min', 'turnover_min', 'casetime_util', 'turnover_util',
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
