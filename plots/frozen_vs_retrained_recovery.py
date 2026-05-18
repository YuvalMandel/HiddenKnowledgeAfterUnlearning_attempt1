#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hk_utils import REPO, METHODS, N_OPTIONS, KNOWS_THRESHOLD, method_fname, load_correct_idx, load_scores_df

OUT_DIR = REPO / 'plots' / 'frozen_vs_retrained_recovery'
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXT_DIR = REPO / 'inside_out_ext'


def pairwise_k(arr, correct_idx, qidx):
    k = []
    for qi in qidx:
        c = int(correct_idx[qi])
        wins = 0
        total = 0
        for w in range(N_OPTIONS):
            if w == c:
                continue
            wins += float(arr[qi, c] > arr[qi, w])
            total += 1
        k.append(wins / total)
    return np.array(k, dtype=float)


def quadrant(frozen, retrained):
    fh = frozen > KNOWS_THRESHOLD
    rh = retrained > KNOWS_THRESHOLD
    if fh and rh:
        return 'frozen_high_retrained_high'
    if (not fh) and rh:
        return 'frozen_low_retrained_high'
    if (not fh) and (not rh):
        return 'frozen_low_retrained_low'
    return 'frozen_high_retrained_low'


def main():
    correct_idx = load_correct_idx()
    df = load_scores_df()
    single_multi = df[(df['domain'] == 'bio') & (df['clf'] == 'LR') & (df['split_type'] == 'single') & (df['layer_config'] == 'multi')]

    rows = []
    for method in METHODS:
        mf = method_fname(method)
        int_path = EXT_DIR / f'{mf}_ck8_bio_int_proba.npy'
        if not int_path.exists():
            continue

        mk = single_multi[single_multi['model_id'] == f'{mf}_ck8'][['question_idx', 'k_internal', 'k_external']].copy()
        if mk.empty:
            continue
        mk['subset'] = 'forgotten'
        mk.loc[(mk['k_internal'] > KNOWS_THRESHOLD) & (mk['k_external'] > KNOWS_THRESHOLD), 'subset'] = 'retained'
        mk.loc[(mk['k_internal'] > KNOWS_THRESHOLD) & (mk['k_external'] <= KNOWS_THRESHOLD), 'subset'] = 'suppressed'
        mk.loc[(mk['k_internal'] <= KNOWS_THRESHOLD) & (mk['k_external'] > KNOWS_THRESHOLD), 'subset'] = 'lucky'

        qidx = mk['question_idx'].astype(int).to_numpy()
        subset_map = dict(zip(mk['question_idx'].astype(int), mk['subset']))
        ext_map = dict(zip(mk['question_idx'].astype(int), mk['k_external']))

        # frozen: base-coordinate probability readout at ck8 (from *_int_proba.npy)
        frozen_arr = np.load(int_path)
        frozen_k = pairwise_k(frozen_arr, correct_idx, qidx)
        frozen_map = {int(q): float(k) for q, k in zip(qidx, frozen_k)}

        # retrained: own-probe cv k_internal at ck8 aggregated by band
        allowed = {f'{mf}_ck8'}
        cv = df[
            (df['domain'] == 'bio') & (df['clf'] == 'LR') & (df['split_type'] == 'cv') & (df['probe_type'] == 'own') &
            (df['model_id'].isin(allowed)) & (df['layer_config'].astype(str).str.match(r'^layer_\d+$')) &
            (df['question_idx'].isin(qidx))
        ][['question_idx', 'layer_config', 'k_internal']].copy()
        if cv.empty:
            continue
        cv['layer_idx'] = cv['layer_config'].str.extract(r'layer_(\d+)').astype(int)

        bands = {
            'mid': (10, 20),
            'late': (21, 32),
            'full': (1, 32),
        }
        for band, (lo, hi) in bands.items():
            b = cv[(cv['layer_idx'] >= lo) & (cv['layer_idx'] <= hi)]
            agg = b.groupby('question_idx')['k_internal'].mean()
            for qi, rk in agg.items():
                fk = frozen_map.get(int(qi), np.nan)
                if not np.isfinite(fk):
                    continue
                rows.append({
                    'method': method,
                    'question_idx': int(qi),
                    'subset': subset_map[int(qi)],
                    'layer_band': band,
                    'K_external_ck8': float(ext_map[int(qi)]),
                    'frozen_K_internal': float(fk),
                    'retrained_K_internal': float(rk),
                    'recovery_gain': float(rk - fk),
                    'quadrant': quadrant(float(fk), float(rk)),
                })

    rdf = pd.DataFrame(rows)
    if rdf.empty:
        raise RuntimeError('No recovery rows produced.')

    rdf.to_csv(OUT_DIR / 'recovery_by_question.csv', index=False)

    summ = rdf.groupby(['method', 'subset', 'layer_band'], as_index=False).agg(
        n=('question_idx', 'size'),
        mean_frozen_K_internal=('frozen_K_internal', 'mean'),
        mean_retrained_K_internal=('retrained_K_internal', 'mean'),
        mean_recovery_gain=('recovery_gain', 'mean'),
    )
    summ.to_csv(OUT_DIR / 'recovery_summary_by_method_subset.csv', index=False)

    quad = rdf.groupby(['method', 'subset', 'layer_band', 'quadrant']).size().reset_index(name='count')
    quad['fraction'] = quad['count'] / quad.groupby(['method', 'subset', 'layer_band'])['count'].transform('sum')
    quad.to_csv(OUT_DIR / 'recovery_quadrants.csv', index=False)

    # Figure 1
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True, sharey=True)
    axes = axes.ravel()
    cmap = {'retained': '#2ca02c', 'suppressed': '#d62728', 'forgotten': '#7f7f7f', 'lucky': '#1f77b4'}
    for i, method in enumerate(METHODS):
        ax = axes[i]
        m = rdf[(rdf['method'] == method) & (rdf['layer_band'] == 'mid')]
        if m.empty:
            ax.axis('off')
            continue
        for s, g in m.groupby('subset'):
            ax.scatter(g['frozen_K_internal'], g['retrained_K_internal'], s=12, alpha=0.45, color=cmap.get(s, '#333333'), label=s)
        ax.axvline(0.5, ls='--', color='black', lw=0.8)
        ax.axhline(0.5, ls='--', color='black', lw=0.8)
        ax.set_title(method, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('Figure 1: Frozen vs Retrained Internal K (mid band)', fontsize=12, y=0.995)
    fig.supxlabel('Frozen K_internal')
    fig.supylabel('Retrained K_internal')
    fig.tight_layout(rect=[0.03, 0.06, 1, 0.93])
    fig.savefig(OUT_DIR / 'figure1_frozen_vs_retrained_scatter.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure1_frozen_vs_retrained_scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 2
    fig, ax = plt.subplots(figsize=(10, 5))
    m = rdf[rdf['layer_band'] == 'mid'].copy()
    order = ['retained', 'suppressed', 'forgotten', 'lucky']
    data = [m[m['subset'] == s]['recovery_gain'].to_numpy() for s in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_xlabel('Final subset')
    ax.set_ylabel('recovery_gain')
    ax.set_title('Recovery Gain by Subset (mid band, pooled methods)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'figure2_recovery_gain_by_subset.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure2_recovery_gain_by_subset.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 3
    qmid = quad[quad['layer_band'] == 'mid']
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharey=True)
    axes = axes.ravel()
    qorder = ['frozen_high_retrained_high', 'frozen_low_retrained_high', 'frozen_low_retrained_low', 'frozen_high_retrained_low']
    for i, method in enumerate(METHODS):
        ax = axes[i]
        mm = qmid[qmid['method'] == method]
        if mm.empty:
            ax.axis('off')
            continue
        x = np.arange(4)
        bottom = np.zeros(4)
        for qn in qorder:
            vals = []
            for s in ['retained', 'suppressed', 'forgotten', 'lucky']:
                r = mm[(mm['subset'] == s) & (mm['quadrant'] == qn)]
                vals.append(float(r['fraction'].iloc[0]) if not r.empty else 0.0)
            ax.bar(x, vals, bottom=bottom, label=qn)
            bottom += np.array(vals)
        ax.set_xticks(x)
        ax.set_xticklabels(['ret', 'sup', 'forg', 'luck'])
        ax.set_title(method, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('Figure 3: Recovery Quadrant Fractions (mid band)', fontsize=12, y=0.995)
    fig.supxlabel('Subset: ret=retained, sup=suppressed, forg=forgotten, luck=lucky')
    fig.supylabel('Fraction')
    fig.tight_layout(rect=[0.03, 0.06, 1, 0.93])
    fig.savefig(OUT_DIR / 'figure3_recovery_quadrant_counts.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure3_recovery_quadrant_counts.pdf', bbox_inches='tight')
    plt.close(fig)

    # Figure 4
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, method in enumerate(METHODS):
        ax = axes[i]
        m = rdf[(rdf['method'] == method) & (rdf['layer_band'] == 'mid')]
        if m.empty:
            ax.axis('off')
            continue
        for s, g in m.groupby('subset'):
            ax.scatter(g['K_external_ck8'], g['retrained_K_internal'], s=12, alpha=0.45, color=cmap.get(s, '#333333'), label=s)
        ax.set_title(method, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('Figure 4: Retrained Internal K vs External K at ck8 (mid band)', fontsize=12, y=0.995)
    fig.supxlabel('K_external_ck8')
    fig.supylabel('Retrained K_internal')
    fig.tight_layout(rect=[0.03, 0.06, 1, 0.93])
    fig.savefig(OUT_DIR / 'figure4_recovery_vs_external_k.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT_DIR / 'figure4_recovery_vs_external_k.pdf', bbox_inches='tight')
    plt.close(fig)

    f_sup = summ[(summ['subset'] == 'suppressed') & (summ['layer_band'] == 'mid')]['mean_recovery_gain'].mean()
    f_forg = summ[(summ['subset'] == 'forgotten') & (summ['layer_band'] == 'mid')]['mean_recovery_gain'].mean()
    lines = [
        '# Frozen vs Retrained Recovery Summary',
        '',
        '- Operationalization note: `frozen_K_internal` uses ck8 probability readout artifacts (`*_ck8_bio_int_proba.npy`), and `retrained_K_internal` uses ck8 own-probe CV layer-band means.',
        f'- Mean recovery gain (suppressed, mid): {f_sup:.3f}',
        f'- Mean recovery gain (forgotten, mid): {f_forg:.3f}',
        '- Positive forgotten recovery gain suggests rerouting/displacement under this proxy setup.',
    ]
    (OUT_DIR / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')

    print('Saved outputs to', OUT_DIR)


if __name__ == '__main__':
    main()
