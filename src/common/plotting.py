from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def save_bic_plot(bic_df: pd.DataFrame, path: str | Path) -> None:
    if bic_df.empty:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(bic_df['T'], bic_df['bic'], marker='o')
    best = bic_df.loc[bic_df['bic'].idxmin()]
    plt.scatter([best['T']], [best['bic']], s=90)
    plt.annotate(f"T*={int(best['T'])}", (best['T'], best['bic']), xytext=(8, 8), textcoords='offset points')
    plt.xlabel('Candidate rotation period T')
    plt.ylabel('BIC score')
    plt.title('Pre-Layer Period Detection via BIC')
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()


def save_residual_plot(pred_df: pd.DataFrame, path: str | Path, target: str) -> None:
    if pred_df.empty or target not in pred_df.columns:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pred_col = f'pred_{target}'
    if pred_col not in pred_df.columns:
        return
    residual = pred_df[target] - pred_df[pred_col]
    plt.figure(figsize=(8, 5))
    plt.hist(residual.dropna(), bins=30)
    plt.xlabel('Residual minutes')
    plt.ylabel('Count')
    plt.title(f'Residual distribution: {target}')
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()


def save_pareto_plot(df: pd.DataFrame, path: str | Path) -> None:
    if df.empty:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    x = df.get('changed_blocks', pd.Series(range(len(df))))
    y = df.get('objective_value', pd.Series(range(len(df))))
    plt.scatter(x, y)
    for _, r in df.iterrows():
        plt.annotate(str(r.get('theme', 'candidate')), (r.get('changed_blocks', 0), r.get('objective_value', 0)), xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Changed blocks')
    plt.ylabel('Objective value')
    plt.title('Layer 2 Pareto-style Candidate Diagnostics')
    plt.tight_layout()
    plt.savefig(p, dpi=180)
    plt.close()
