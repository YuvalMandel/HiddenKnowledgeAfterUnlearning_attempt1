#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hk_utils import REPO, METHODS, KNOWS_THRESHOLD, method_fname, load_scores_df, load_split_indices, compute_prefilter_mask

OUT_DIR = REPO / 'plots' / 'transition_timing'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKPTS = ['base'] + [f'ck{i}' for i in range(1, 9)]
CKPT_TO_T = {c: i for i, c in enumerate(CKPTS)}


def first_drop(vals):
    for i, v in enumerate(vals):
        if np.isfinite(v) and v <= KNOWS_THRESHOLD:
            return i, False
    return 9, True


def main():
    df = load_scores_df()
    te, orig_to_pos = load_split_indices()
    filt_mask = compute_prefilter_mask(te, orig_to_pos, df)
    filt_te = set(te[filt_mask].astype(int).tolist())

    single_full = df[(df['domain'] == 'bio') & (df['clf'] == 'LR') & (df['split_type'] == 'single') & (df['layer_config'] == 'full')].drop_duplicates('question_idx').copy()

    all_rows = []
    for method in METHODS:
        mf = method_fname(method)
        mk = single_full[single_full['model_id'] == f'{mf}_ck8'][['question_idx', 'k_internal', 'k_external']].copy()
        if mk.empty:
            continue
        mk = mk[mk['question_idx'].isin(filt_te)].copy()
        if mk.empty:
            continue
        mk['subset'] = 'forgotten'
        mk.loc[(mk['k_internal'] > KNOWS_THRESHOLD) & (mk['k_external'] > KNOWS_THRESHOLD), 'subset'] = 'retained'
        mk.loc[(mk['k_internal'] > KNOWS_THRESHOLD) & (mk['k_external'] <= KNOWS_THRESHOLD), 'subset'] = 'suppressed'
        mk.loc[(mk['k_internal'] <= KNOWS_THRESHOLD) & (mk['k_external'] > KNOWS_THRESHOLD), 'subset'] = 'lucky'
        subset_map = dict(zip(mk['question_idx'].astype(int), mk['subset']))

        allowed_ids = {'base'} | {f'{mf}_ck{i}' for i in range(1, 9)}
        cv = df[
            (df['domain'] == 'bio') &
            (df['clf'] == 'LR') &
            (df['split_type'] == 'cv') &
            (df['probe_type'] == 'own') &
            (df['model_id'].isin(allowed_ids)) &
            (df['layer_config'].astype(str).str.match(r'^layer_\d+$'))
        ][['model_id', 'question_idx', 'layer_config', 'fold', 'k_internal', 'k_external']].copy()
        if cv.empty:
            continue
        cv['layer_idx'] = cv['layer_config'].str.extract(r'layer_(\d+)').astype(int)
        cv = cv[(cv['layer_idx'] >= 10) & (cv['layer_idx'] <= 20)]
        cv = cv[cv['question_idx'].isin(subset_map.keys())]
        if cv.empty:
            continue
        cv['checkpoint'] = cv['model_id'].str.replace(f'{mf}_', '', regex=False)
        cv.loc[cv['model_id'] == 'base', 'checkpoint'] = 'base'
        cv = cv[cv['checkpoint'].isin(CKPTS)]

        agg = cv.groupby(['question_idx', 'checkpoint'], as_index=False).agg(
            k_internal=('k_internal', 'mean'),
            k_external=('k_external', 'mean')
        )

        for qi, g in agg.groupby('question_idx'):
            pivot = g.set_index('checkpoint').reindex(CKPTS)
            int_vals = pivot['k_internal'].to_numpy(dtype=float)
            ext_vals = pivot['k_external'].to_numpy(dtype=float)
            t_int, c_int = first_drop(int_vals)
            t_ext, c_ext = first_drop(ext_vals)
            all_rows.append({
                'method': method,
                'question_idx': int(qi),
                'subset': subset_map[int(qi)],
                't_ext_drop': int(t_ext),
                't_int_drop': int(t_int),
                'ext_censored': bool(c_ext),
                'int_censored': bool(c_int),
                'suppression_lag': int(t_int - t_ext),
            })

    tdf = pd.DataFrame(all_rows)
    if tdf.empty:
        raise RuntimeError('No transition timing rows produced.')
    tdf.to_csv(OUT_DIR / 'transition_timing_by_question.csv', index=False)

    summ = tdf.groupby(['method', 'subset'], as_index=False).agg(
        n=('question_idx', 'size'),
        mean_t_ext_drop=('t_ext_drop', 'mean'),
        mean_t_int_drop=('t_int_drop', 'mean'),
        mean_suppression_lag=('suppression_lag', 'mean'),
        ext_censored_rate=('ext_censored', 'mean'),
        int_censored_rate=('int_censored', 'mean'),
    )
    summ.to_csv(OUT_DIR / 'transition_timing_summary.csv', index=False)

    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True, sharey=True)
    axes = axes.ravel()
    cmap = {'retained': '#2ca02c', 'suppressed': '#d62728', 'forgotten': '#7f7f7f', 'lucky': '#1f77b4'}
    for i, method in enumerate(METHODS):
        ax = axes[i]
        m = tdf[tdf['method'] == method]
        if m.empty:
            ax.axis('off')
            continue
        for s, g in m.groupby('subset'):
            ax.scatter(g['t_ext_drop'], g['t_int_drop'], s=10, alpha=0.45, color=cmap.get(s, '#333333'), label=s)
        ax.plot([0, 9], [0, 9], '--', color='black', lw=1)
        ax.set_title(method, fontsize=9)
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('Figure 1: Transition Scatter (t_ext_drop vs t_int_drop)', fontsize=12, y=0.995)
    fig.supxlabel('t_ext_drop (first checkpoint where K_external <= 0.5)')
    fig.supylabel('t_int_drop (first checkpoint where K_internal <= 0.5)')
    fig.tight_layout(rect=[0.03, 0.06, 1, 0.93])
    fig.savefig(OUT_DIR / 'figure1_transition_scatter.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure1_transition_scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True, sharey=True)
    axes = axes.ravel()
    bins = np.arange(-9.5, 10.5, 1)
    for i, method in enumerate(METHODS):
        ax = axes[i]
        m = tdf[tdf['method'] == method]
        if m.empty:
            ax.axis('off')
            continue
        for s in ['suppressed', 'forgotten', 'lucky', 'retained']:
            g = m[m['subset'] == s]
            if g.empty:
                continue
            ax.hist(g['suppression_lag'], bins=bins, alpha=0.4, density=True, label=s, color=cmap.get(s, '#333333'))
        ax.set_title(method, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('Figure 2: Suppression Lag Histograms by Final Subset', fontsize=12, y=0.995)
    fig.supxlabel('suppression_lag = t_int_drop - t_ext_drop')
    fig.supylabel('Density')
    fig.tight_layout(rect=[0.03, 0.06, 1, 0.93])
    fig.savefig(OUT_DIR / 'figure2_suppression_lag_histograms.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure2_suppression_lag_histograms.pdf', bbox_inches='tight')
    plt.close(fig)

    pivot = summ.pivot_table(index='method', columns='subset', values='mean_suppression_lag')
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(pivot.index))
    width = 0.2
    for i, s in enumerate(['retained', 'suppressed', 'forgotten', 'lucky']):
        vals = pivot[s].to_numpy() if s in pivot.columns else np.zeros(len(x))
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20)
    ax.set_xlabel('Method')
    ax.set_ylabel('mean suppression_lag')
    ax.set_title('Figure 3: Mean Suppression Lag by Method and Subset')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'figure3_mean_suppression_lag_by_method_subset.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure3_mean_suppression_lag_by_method_subset.pdf', bbox_inches='tight')
    plt.close(fig)

    heat = summ.pivot_table(index=['method', 'subset'], values=['mean_t_ext_drop', 'mean_t_int_drop', 'mean_suppression_lag'])
    fig, ax = plt.subplots(figsize=(10, max(5, 0.2 * len(heat))))
    im = ax.imshow(heat.to_numpy(), aspect='auto', cmap='coolwarm')
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([f'{m}/{s}' for m, s in heat.index], fontsize=8)
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=20)
    ax.set_xlabel('Transition timing metric')
    ax.set_ylabel('Method / subset')
    ax.set_title('Figure 4: Drop Timing Heatmap')
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'figure4_drop_timing_heatmap.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure4_drop_timing_heatmap.pdf', bbox_inches='tight')
    plt.close(fig)

    s_mean = summ[summ['subset'] == 'suppressed']['mean_suppression_lag'].mean()
    f_mean = summ[summ['subset'] == 'forgotten']['mean_suppression_lag'].mean()
    lines = [
        '# Transition Timing Summary',
        '',
        f'- Mean suppression lag (suppressed): {s_mean:.3f}',
        f'- Mean suppression lag (forgotten): {f_mean:.3f}',
        '- Positive lag indicates internal drop later than external drop.',
    ]
    (OUT_DIR / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')

    print('Saved outputs to', OUT_DIR)


if __name__ == '__main__':
    main()
