#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hk_utils import REPO, METHODS, N_OPTIONS, KNOWS_THRESHOLD, method_fname, load_correct_idx, load_scores_df, get_parquet_multi_single, load_split_indices, compute_prefilter_mask

OUT_DIR = REPO / 'plots' / 'pair_level_suppressed_forgotten'
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXT_DIR = REPO / 'inside_out_ext'


def pair_rows_for_checkpoint(method, checkpoint, correct_idx, qidx, int_arr, ext_arr, subset_map):
    rows = []
    for qi in qidx:
        c = int(correct_idx[qi])
        for w in range(N_OPTIONS):
            if w == c:
                continue
            i_c = float(int_arr[qi, c])
            i_w = float(int_arr[qi, w])
            e_c = float(ext_arr[qi, c])
            e_w = float(ext_arr[qi, w])
            im = i_c - i_w
            em = e_c - e_w
            rows.append({
                'method': method,
                'checkpoint': checkpoint,
                'question_idx': int(qi),
                'correct_option_idx': c,
                'wrong_option_idx': int(w),
                'subset_ck8': subset_map.get(int(qi), 'unknown'),
                'internal_score_correct': i_c,
                'internal_score_wrong': i_w,
                'external_score_correct': e_c,
                'external_score_wrong': e_w,
                'internal_margin_pair': im,
                'external_margin_pair': em,
                'internal_win_pair': im > 0,
                'external_win_pair': em > 0,
            })
    return rows


def pair_outcome(int_win, ext_win):
    if int_win and not ext_win:
        return 'hidden_pair'
    if (not int_win) and (not ext_win):
        return 'forgotten_pair'
    if int_win and ext_win:
        return 'retained_pair'
    return 'lucky_pair'


def save_fig(fig, stem):
    fig.savefig(OUT_DIR / f'{stem}.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT_DIR / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    correct_idx = load_correct_idx()
    te, orig_to_pos = load_split_indices()
    df = load_scores_df()
    filt_mask = compute_prefilter_mask(te, orig_to_pos, df)
    filt_te = te[filt_mask]

    base_int = np.load(EXT_DIR / 'base_bio_int_proba.npy')
    base_ext = np.load(EXT_DIR / 'base_bio_ext.npy')

    all_rows = []
    outcome_rows = []
    best_rows = []
    summary_lines = ['# Pair-Level Suppressed vs Forgotten Summary', '']

    parquet_multi = get_parquet_multi_single(df)

    for method in METHODS:
        mf = method_fname(method)
        ext_path = EXT_DIR / f'{mf}_ck8_bio_ext.npy'
        int_path = EXT_DIR / f'{mf}_ck8_bio_int_proba.npy'
        if not ext_path.exists() or not int_path.exists():
            continue

        ck8_ext = np.load(ext_path)
        ck8_int = np.load(int_path)

        mk = parquet_multi[parquet_multi['model_id'] == f'{mf}_ck8'][['question_idx', 'k_internal', 'k_external']].copy()
        if mk.empty:
            continue
        mk['subset_ck8'] = 'forgotten'
        mk.loc[(mk['k_internal'] > KNOWS_THRESHOLD) & (mk['k_external'] > KNOWS_THRESHOLD), 'subset_ck8'] = 'retained'
        mk.loc[(mk['k_internal'] > KNOWS_THRESHOLD) & (mk['k_external'] <= KNOWS_THRESHOLD), 'subset_ck8'] = 'suppressed'
        mk.loc[(mk['k_internal'] <= KNOWS_THRESHOLD) & (mk['k_external'] > KNOWS_THRESHOLD), 'subset_ck8'] = 'lucky'
        subset_map = dict(zip(mk['question_idx'].astype(int), mk['subset_ck8']))
        qidx = np.array(sorted(set(mk['question_idx'].astype(int)).intersection(set(filt_te.astype(int)))))
        if len(qidx) == 0:
            continue

        all_rows.extend(pair_rows_for_checkpoint(method, 'base', correct_idx, qidx, base_int, base_ext, subset_map))
        all_rows.extend(pair_rows_for_checkpoint(method, 'ck8', correct_idx, qidx, ck8_int, ck8_ext, subset_map))

        for qi in qidx:
            c = int(correct_idx[qi])
            bw_ext_base = int(np.argmax([base_ext[qi, j] if j != c else -1e9 for j in range(N_OPTIONS)]))
            bw_ext_ck8 = int(np.argmax([ck8_ext[qi, j] if j != c else -1e9 for j in range(N_OPTIONS)]))
            bw_int_base = int(np.argmax([base_int[qi, j] if j != c else -1e9 for j in range(N_OPTIONS)]))
            bw_int_ck8 = int(np.argmax([ck8_int[qi, j] if j != c else -1e9 for j in range(N_OPTIONS)]))
            best_rows.append({
                'method': method,
                'question_idx': int(qi),
                'subset_ck8': subset_map.get(int(qi), 'unknown'),
                'internal_external_best_wrong_agreement_ck8': int(bw_int_ck8 == bw_ext_ck8),
                'external_best_wrong_stability_base_to_ck8': int(bw_ext_base == bw_ext_ck8),
                'internal_best_wrong_stability_base_to_ck8': int(bw_int_base == bw_int_ck8),
            })

    pair_df = pd.DataFrame(all_rows)
    if pair_df.empty:
        raise RuntimeError('No pair-level rows produced.')

    ck8 = pair_df[pair_df['checkpoint'] == 'ck8'].copy()
    ck8['pair_outcome'] = [pair_outcome(i, e) for i, e in zip(ck8['internal_win_pair'], ck8['external_win_pair'])]
    pair_df = pair_df.merge(
        ck8[['method', 'question_idx', 'wrong_option_idx', 'pair_outcome']],
        on=['method', 'question_idx', 'wrong_option_idx'],
        how='left'
    )

    base = pair_df[pair_df['checkpoint'] == 'base'][['method', 'question_idx', 'wrong_option_idx', 'internal_margin_pair', 'external_margin_pair']].rename(
        columns={'internal_margin_pair': 'base_internal_margin_pair', 'external_margin_pair': 'base_external_margin_pair'}
    )
    c8 = pair_df[pair_df['checkpoint'] == 'ck8'][['method', 'question_idx', 'wrong_option_idx', 'internal_margin_pair', 'external_margin_pair']].rename(
        columns={'internal_margin_pair': 'ck8_internal_margin_pair', 'external_margin_pair': 'ck8_external_margin_pair'}
    )
    merge_margin = base.merge(c8, on=['method', 'question_idx', 'wrong_option_idx'], how='inner')
    merge_margin['internal_margin_survival'] = merge_margin['ck8_internal_margin_pair'] - merge_margin['base_internal_margin_pair']
    merge_margin['external_margin_drop'] = merge_margin['ck8_external_margin_pair'] - merge_margin['base_external_margin_pair']
    merge_margin['access_suppression_index'] = merge_margin['internal_margin_survival'] - merge_margin['external_margin_drop']
    merge_margin = merge_margin.merge(ck8[['method', 'question_idx', 'wrong_option_idx', 'pair_outcome', 'subset_ck8']], on=['method', 'question_idx', 'wrong_option_idx'], how='left')

    out_counts = ck8.groupby(['method', 'pair_outcome']).size().reset_index(name='count')
    out_counts['proportion'] = out_counts['count'] / out_counts.groupby('method')['count'].transform('sum')
    out_counts.to_csv(OUT_DIR / 'pair_level_outcomes.csv', index=False)

    margin_summary = merge_margin.groupby(['method', 'pair_outcome'], as_index=False).agg(
        mean_base_internal_margin_pair=('base_internal_margin_pair', 'mean'),
        mean_base_external_margin_pair=('base_external_margin_pair', 'mean'),
        mean_ck8_internal_margin_pair=('ck8_internal_margin_pair', 'mean'),
        mean_ck8_external_margin_pair=('ck8_external_margin_pair', 'mean'),
        mean_internal_margin_survival=('internal_margin_survival', 'mean'),
        mean_external_margin_drop=('external_margin_drop', 'mean'),
        mean_access_suppression_index=('access_suppression_index', 'mean'),
        n=('pair_outcome', 'size'),
    )
    margin_summary.to_csv(OUT_DIR / 'pair_margin_summary.csv', index=False)

    best_df = pd.DataFrame(best_rows)
    best_summary = best_df.groupby(['method', 'subset_ck8'], as_index=False).mean(numeric_only=True)
    best_summary.to_csv(OUT_DIR / 'best_distractor_summary.csv', index=False)

    cmap = {'hidden_pair': '#d62728', 'forgotten_pair': '#7f7f7f', 'retained_pair': '#2ca02c', 'lucky_pair': '#1f77b4'}
    for fig_name, xcol, ycol in [
        ('figure1_base_pair_margins', 'base_external_margin_pair', 'base_internal_margin_pair'),
        ('figure2_ck8_pair_margins', 'ck8_external_margin_pair', 'ck8_internal_margin_pair'),
        ('figure3_margin_survival_vs_external_drop', 'external_margin_drop', 'internal_margin_survival'),
    ]:
        fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=False, sharey=False)
        axes = axes.ravel()
        for i, method in enumerate(METHODS):
            ax = axes[i]
            m = merge_margin[merge_margin['method'] == method]
            if m.empty:
                ax.axis('off')
                continue
            for po, g in m.groupby('pair_outcome'):
                ax.scatter(g[xcol], g[ycol], s=8, alpha=0.35, color=cmap.get(po, '#333333'), label=po)
            ax.set_title(method, fontsize=9)
            ax.grid(alpha=0.2)
        title_map = {
            'figure1_base_pair_margins': 'Figure 1: Base Pair Margins by ck8 Pair Outcome',
            'figure2_ck8_pair_margins': 'Figure 2: ck8 Pair Margins by ck8 Pair Outcome',
            'figure3_margin_survival_vs_external_drop': 'Figure 3: Internal Margin Survival vs External Margin Drop',
        }
        xlabel_map = {
            'figure1_base_pair_margins': 'Base external margin (correct - wrong)',
            'figure2_ck8_pair_margins': 'ck8 external margin (correct - wrong)',
            'figure3_margin_survival_vs_external_drop': 'External margin drop (ck8 - base)',
        }
        ylabel_map = {
            'figure1_base_pair_margins': 'Base internal margin (correct - wrong)',
            'figure2_ck8_pair_margins': 'ck8 internal margin (correct - wrong)',
            'figure3_margin_survival_vs_external_drop': 'Internal margin survival (ck8 - base)',
        }
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))
        fig.suptitle(title_map[fig_name], fontsize=12, y=0.995)
        fig.supxlabel(xlabel_map[fig_name])
        fig.supylabel(ylabel_map[fig_name])
        fig.tight_layout(rect=[0.03, 0.05, 1, 0.93])
        save_fig(fig, fig_name)

    comp = ck8.groupby(['method', 'subset_ck8', 'pair_outcome']).size().reset_index(name='n')
    comp['frac'] = comp['n'] / comp.groupby(['method', 'subset_ck8'])['n'].transform('sum')
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharey=True)
    axes = axes.ravel()
    order_subset = ['retained', 'suppressed', 'forgotten', 'lucky']
    order_out = ['retained_pair', 'hidden_pair', 'forgotten_pair', 'lucky_pair']
    for i, method in enumerate(METHODS):
        ax = axes[i]
        m = comp[comp['method'] == method]
        if m.empty:
            ax.axis('off')
            continue
        bottom = np.zeros(len(order_subset))
        for po in order_out:
            vals = []
            for s in order_subset:
                r = m[(m['subset_ck8'] == s) & (m['pair_outcome'] == po)]
                vals.append(float(r['frac'].iloc[0]) if not r.empty else 0.0)
            ax.bar(order_subset, vals, bottom=bottom, color=cmap.get(po, '#333333'), label=po)
            bottom += np.array(vals)
        ax.set_title(method, fontsize=9)
        ax.tick_params(axis='x', labelrotation=20)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle('Figure 4: Pair Outcome Composition by Question Subset', fontsize=12, y=0.995)
    fig.supxlabel('Question-level subset at ck8')
    fig.supylabel('Fraction of pair outcomes')
    fig.tight_layout(rect=[0.03, 0.06, 1, 0.93])
    save_fig(fig, 'figure4_pair_outcome_composition_by_question_subset')

    pair_df.to_csv(OUT_DIR / 'pair_level_dataset.csv', index=False)

    sup_hidden = comp[(comp['subset_ck8'] == 'suppressed') & (comp['pair_outcome'] == 'hidden_pair')]
    for_forgotten = comp[(comp['subset_ck8'] == 'forgotten') & (comp['pair_outcome'] == 'forgotten_pair')]
    summary_lines.append(f'- Mean hidden_pair fraction inside suppressed questions: {sup_hidden["frac"].mean():.3f}')
    summary_lines.append(f'- Mean forgotten_pair fraction inside forgotten questions: {for_forgotten["frac"].mean():.3f}')
    summary_lines.append('- Pair-level checkpoints available from current artifacts: base and ck8.')
    (OUT_DIR / 'summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')

    print('Saved outputs to', OUT_DIR)


if __name__ == '__main__':
    main()
