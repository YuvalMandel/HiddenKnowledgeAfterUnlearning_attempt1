#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from hk_utils import REPO, METHODS, method_fname, load_scores_df

OUT_DIR = REPO / 'plots' / 'overlap_boundary_prec8_prediction'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_dataset():
    traj = pd.read_csv(REPO / 'plots' / 'ckpt_layer_trajectory_metrics' / 'trajectory_metrics_by_method_subset.csv')
    mid = traj[traj['layer_band'] == 'mid'][['method', 'subset', 'question_idx', 'auc']].rename(columns={'auc': 'mid_auc'})

    overlap_rows = []
    for method in METHODS:
        m = mid[(mid['method'] == method) & (mid['subset'].isin(['suppressed', 'forgotten']))]
        if m.empty:
            continue
        s = m[m['subset'] == 'suppressed']['mid_auc'].dropna().to_numpy()
        f = m[m['subset'] == 'forgotten']['mid_auc'].dropna().to_numpy()
        if len(s) < 10 or len(f) < 10:
            continue
        low = max(np.quantile(s, 0.1), np.quantile(f, 0.1))
        high = min(np.quantile(s, 0.9), np.quantile(f, 0.9))
        mm = m[(m['mid_auc'] >= low) & (m['mid_auc'] <= high)].copy()
        mm['overlap_low'] = low
        mm['overlap_high'] = high
        overlap_rows.append(mm)
    ov = pd.concat(overlap_rows, ignore_index=True)

    df = load_scores_df()
    rows = []
    for method in METHODS:
        mf = method_fname(method)
        allowed = {'base'} | {f'{mf}_ck{i}' for i in range(1, 8)}
        sub = df[
            (df['domain'] == 'bio') & (df['clf'] == 'LR') & (df['split_type'] == 'cv') & (df['probe_type'] == 'own') &
            (df['model_id'].isin(allowed)) & (df['layer_config'].astype(str).str.match(r'^layer_\d+$'))
        ][['model_id', 'question_idx', 'layer_config', 'fold', 'k_internal', 'k_external']].copy()
        if sub.empty:
            continue
        sub['layer_idx'] = sub['layer_config'].str.extract(r'layer_(\d+)').astype(int)
        sub['checkpoint'] = sub['model_id'].str.replace(f'{mf}_', '', regex=False)
        sub.loc[sub['model_id'] == 'base', 'checkpoint'] = 'base'

        for (qi, ck), g in sub.groupby(['question_idx', 'checkpoint']):
            mid_k = g[(g['layer_idx'] >= 10) & (g['layer_idx'] <= 20)]['k_internal'].mean()
            late_k = g[(g['layer_idx'] >= 21) & (g['layer_idx'] <= 32)]['k_internal'].mean()
            ext_k = g['k_external'].mean()
            rows.append({'method': method, 'question_idx': int(qi), 'checkpoint': ck, 'mid_k': mid_k, 'late_k': late_k, 'ext_k': ext_k})

    feat = pd.DataFrame(rows)
    wide = feat.pivot_table(index=['method', 'question_idx'], columns='checkpoint', values=['mid_k', 'late_k', 'ext_k'], aggfunc='mean')
    wide.columns = [f'{m}_{c}' for m, c in wide.columns]
    wide = wide.reset_index()

    merged = ov.merge(wide, on=['method', 'question_idx'], how='inner')
    merged['target'] = (merged['subset'] == 'suppressed').astype(int)

    # engineered pre-ck8 features
    for m in ['mid_k', 'late_k', 'ext_k']:
        cols = [f'{m}_ck{i}' for i in range(1, 8) if f'{m}_ck{i}' in merged.columns]
        if len(cols) >= 2:
            merged[f'{m}_slope_ck1_to_ck7'] = merged[cols[-1]] - merged[cols[0]]
            merged[f'{m}_min_ck1_to_ck7'] = merged[cols].min(axis=1)
            merged[f'{m}_var_ck1_to_ck7'] = merged[cols].var(axis=1)
    if 'late_k_ck7' in merged.columns and 'mid_k_ck7' in merged.columns:
        merged['late_minus_mid_ck7'] = merged['late_k_ck7'] - merged['mid_k_ck7']
        merged['mid_minus_late_ck7'] = merged['mid_k_ck7'] - merged['late_k_ck7']

    return merged


def eval_model(X, y, model, cv):
    prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        'roc_auc': roc_auc_score(y, prob),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'f1': f1_score(y, pred, zero_division=0),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'tn_fp_fn_tp': confusion_matrix(y, pred).ravel().tolist(),
    }


def main():
    ds = build_dataset()
    ds.to_csv(OUT_DIR / 'boundary_prec8_dataset.csv', index=False)

    forbidden = {'subset', 'target', 'mid_auc', 'overlap_low', 'overlap_high'}
    feature_cols = [c for c in ds.columns if c not in forbidden and c not in ['method', 'question_idx']]
    X = ds[feature_cols].copy()
    y = ds['target'].to_numpy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if len(ds) >= 120 else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    models = {
        'logreg': Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))]),
        'rf': Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', RandomForestClassifier(n_estimators=300, random_state=42, min_samples_leaf=3))]),
        'hgb': Pipeline([('imputer', SimpleImputer(strategy='median')), ('clf', HistGradientBoostingClassifier(random_state=42))]),
    }

    results = []
    baseline = max(np.mean(y == 0), np.mean(y == 1))
    for name, model in models.items():
        m = eval_model(X, y, model, cv)
        results.append({'scope': 'pooled', 'method': 'ALL', 'model': name, 'majority_baseline': baseline, **m})

    for method in METHODS:
        sub = ds[ds['method'] == method]
        if len(sub) < 40 or sub['target'].nunique() < 2:
            continue
        Xm = sub[feature_cols]
        ym = sub['target'].to_numpy()
        cvm = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        b = max(np.mean(ym == 0), np.mean(ym == 1))
        for name, model in models.items():
            m = eval_model(Xm, ym, model, cvm)
            results.append({'scope': 'per_method', 'method': method, 'model': name, 'majority_baseline': b, **m})

    res = pd.DataFrame(results)
    res.to_csv(OUT_DIR / 'prediction_results.csv', index=False)

    # Feature importance (logistic coefficients, pooled fit)
    logreg = models['logreg']
    logreg.fit(X, y)
    coefs = np.abs(logreg.named_steps['clf'].coef_[0])
    fi = pd.DataFrame({'feature': feature_cols, 'importance': coefs}).sort_values('importance', ascending=False)
    fi.to_csv(OUT_DIR / 'feature_importance.csv', index=False)

    perf = res[res['scope'] == 'per_method'].pivot_table(index='method', columns='model', values='roc_auc')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not perf.empty:
        perf.plot(kind='bar', ax=ax)
        ax.set_ylabel('ROC-AUC')
        ax.set_xlabel('Method')
        ax.set_title('Prediction Performance by Method')
        ax.tick_params(axis='x', rotation=20)
        ax.legend(title='Model')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'figure1_prediction_performance_by_method.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure1_prediction_performance_by_method.pdf', bbox_inches='tight')
    plt.close(fig)

    top = fi.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top['feature'], top['importance'])
    ax.set_xlabel('Absolute coefficient magnitude')
    ax.set_ylabel('Feature')
    ax.set_title('Top Logistic Coefficient Magnitudes (Pooled)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'figure2_feature_importance.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure2_feature_importance.pdf', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, method in enumerate(METHODS):
        ax = axes[i]
        m = ds[ds['method'] == method]
        if m.empty or ('ext_k_ck7' not in m.columns):
            ax.axis('off')
            continue
        ycol = 'late_k_ck7' if 'late_k_ck7' in m.columns else ('mid_k_ck7' if 'mid_k_ck7' in m.columns else None)
        if ycol is None:
            ax.axis('off')
            continue
        for t, color, label in [(1, '#d62728', 'suppressed'), (0, '#7f7f7f', 'forgotten')]:
            g = m[m['target'] == t]
            ax.scatter(g['ext_k_ck7'], g[ycol], s=12, alpha=0.4, color=color, label=label)
        ax.set_title(method, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('Figure 3: ck7 State Plot (External vs Internal)', fontsize=12, y=0.995)
    fig.supxlabel('ck7 external K')
    fig.supylabel('ck7 internal K (late if available, else mid)')
    fig.tight_layout(rect=[0.03, 0.06, 1, 0.93])
    fig.savefig(OUT_DIR / 'figure3_ck7_state_plot.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure3_ck7_state_plot.pdf', bbox_inches='tight')
    plt.close(fig)

    best_auc = res[(res['scope'] == 'pooled')]['roc_auc'].max()
    lines = [
        '# Overlap Boundary Pre-ck8 Prediction Summary',
        '',
        f'- Pooled best ROC-AUC: {best_auc:.3f}',
        f'- Majority baseline (pooled): {baseline:.3f}',
        '- Dataset excludes ck8-definitional features by construction.',
    ]
    (OUT_DIR / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved outputs to', OUT_DIR)


if __name__ == '__main__':
    main()
