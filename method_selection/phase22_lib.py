
from pathlib import Path
import json
import pandas as pd
import numpy as np

PAIR_FEATURES = [
    'pred_true_on_true_stmt','pred_true_on_false_stmt','true_stmt_correct','false_stmt_correct',
    'tf_correct_count','both_correct','both_wrong','exactly_one_wrong',
    'pattern_TT','pattern_TF','pattern_FT','pattern_FF',
    'pair_logically_consistent','pair_contradiction_flag','bias_to_true','bias_to_false',
]
LOGIT_FEATURES = [
    'true_stmt_logit_margin','false_stmt_logit_margin','pair_mean_margin',
    'pair_abs_mean_margin','pair_margin_gap','pair_margin_std','pair_confidence_gap',
]
LABEL_CANDIDATES = ['best_method_label', 'best_method', 'label']
GROUP_CANDIDATES = ['question_group_id', 'original_id', 'question_id']
SPLIT_CANDIDATES = ['split', 'official_split', 'base_split']
TRUTH_CANDIDATES = ['is_true_statement','statement_truth','is_true','gold_label','answer','target','label','truth','true_false']
PRED_CANDIDATES = ['pred_true','pred_label','prediction','pred','label_pred','predicted_label']
MARGIN_CANDIDATES = ['logit_margin','margin','true_false_margin','margin_true_false','tf_margin']
PROB_TRUE_CANDIDATES = ['prob_true','p_true','probability_true']
LOGIT_TRUE_CANDIDATES = ['logit_true','true_logit']
LOGIT_FALSE_CANDIDATES = ['logit_false','false_logit']

METHOD_ORDER = ['PB_J','RMU','GradDiff','ELM','RepNoise','RR','RMU_LAT','TAR']
METHOD_TO_LABEL = {m: i + 1 for i, m in enumerate(METHOD_ORDER)}

def ensure_dir(path):
    path = Path(path); path.mkdir(parents=True, exist_ok=True); return path

def dump_json(path, obj=None):
    path = Path(path); ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def resolve_default_paths(script_dir: Path):
    """
    Resolve project-relative default paths for this repository layout:

      <project_root>/
        data/wmdp_tf_pairs.csv               — TF pair dataset
        checkpoints/                          — logit CSVs (base_bio_logit_*.csv etc.)
        method_selection_out/
          authoritative_row_measurements.csv  — produced by build_authoritative_measurements.py
    """
    # script_dir is method_selection/, its parent is the project root
    root = script_dir.parent
    data_dir = root / 'data'
    ms_out   = root / 'method_selection_out'

    # Fall back to sibling lookup if run from a different CWD
    if not data_dir.exists():
        alt = (root / '..' / 'data').resolve()
        if alt.exists():
            data_dir = alt

    return {
        'data_dir':         data_dir.resolve(),
        'pairs_csv':        (data_dir / 'wmdp_tf_pairs.csv').resolve(),
        'logits_dir':       (root / 'checkpoints').resolve(),
        'measurements_csv': (ms_out / 'authoritative_row_measurements.csv').resolve(),
        'phase22_out':      (ms_out / 'phase22').resolve(),
    }

def normalize_method_name(x: str) -> str:
    s = str(x).strip()
    aliases = {'PB&J':'PB_J','PB_J':'PB_J','PBJ':'PB_J','RMU':'RMU','GradDiff':'GradDiff',
               'GradientDifference':'GradDiff','ELM':'ELM','RepNoise':'RepNoise','RR':'RR',
               'RMU+LAT':'RMU_LAT','RMU_LAT':'RMU_LAT','RMU-LAT':'RMU_LAT','TAR':'TAR'}
    return aliases.get(s, s)

def _infer_train_regime(df: pd.DataFrame) -> pd.Series:
    cols = set(df.columns)
    if 'train_regime' in cols:
        s = df['train_regime'].astype(str)
        uniq = set(s.dropna().unique().tolist())
        if any(u.startswith('traincv') and u != 'traincv' for u in uniq) or ('full' in uniq and len(uniq) > 2):
            return s

    scheme = None
    if 'train_scheme' in cols:
        scheme = df['train_scheme'].astype(str)
    elif 'train_regime' in cols:
        scheme = df['train_regime'].astype(str)
    else:
        raise KeyError('Measurements CSV must contain train_regime or train_scheme')

    window_col = None
    for c in ['traincv_window', 'train_window_id', 'window_id', 'window', 'train_window', 'cv_window']:
        if c in cols:
            window_col = c
            break

    if window_col is None:
        return scheme

    win = df[window_col].astype(str)

    def combine(sc, w):
        s = str(sc).strip()
        ww = str(w).strip()
        if s.lower() == 'full':
            return 'full'
        if s.lower().startswith('traincv'):
            lw = ww.lower()
            if lw.startswith('traincv'):
                return ww
            digits = ''.join(ch for ch in ww if ch.isdigit())
            if digits:
                return f'traincv{digits}'
            return 'traincv'
        return s

    return pd.Series([combine(sc, w) for sc, w in zip(scheme, win)], index=df.index)

def build_scientific_labels(measurements_csv: Path):
    df = pd.read_csv(measurements_csv, low_memory=False)
    if 'question_group_id' not in df.columns or 'method' not in df.columns:
        raise KeyError('Measurements CSV must contain question_group_id and method')

    df['question_group_id'] = df['question_group_id'].astype(str)
    df['method'] = df['method'].map(normalize_method_name)
    df['train_regime'] = _infer_train_regime(df).astype(str)

    required = ['RE_signal','frozen_recovery_mean','retrained_recovery_mean','recovery_gain_mean','leakage_proxy']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f'Measurements CSV missing required columns: {missing}')

    score_rows = []
    label_rows = []
    for regime, g in df.groupby('train_regime', sort=False):
        work = g[['question_group_id','method','train_regime'] + required].copy()

        def norm01(s):
            s = pd.to_numeric(s, errors='coerce')
            lo, hi = s.min(skipna=True), s.max(skipna=True)
            if pd.isna(lo) or pd.isna(hi) or hi <= lo:
                return pd.Series(np.zeros(len(s)), index=s.index)
            return (s - lo) / (hi - lo)

        work['RE_n'] = norm01(work['RE_signal'])
        work['Frozen_n'] = norm01(work['frozen_recovery_mean'])
        work['Retrained_n'] = norm01(work['retrained_recovery_mean'])
        work['Gain_n'] = norm01(work['recovery_gain_mean'])
        work['Suppression_n'] = norm01(work['leakage_proxy'])
        work['scientific_score'] = (
            0.40 * work['RE_n']
            - 0.20 * work['Frozen_n']
            - 0.25 * work['Retrained_n']
            - 0.10 * work['Gain_n']
            - 0.05 * work['Suppression_n']
        )
        score_rows.append(work[['question_group_id','method','train_regime','scientific_score']])

        tmp = work[['question_group_id','method','train_regime','scientific_score','Retrained_n','Frozen_n']].copy()
        tmp['method_rank'] = tmp['method'].map(lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 999)
        tmp = tmp.sort_values(
            ['question_group_id','scientific_score','Retrained_n','Frozen_n','method_rank'],
            ascending=[True,False,True,True,True]
        )
        best = tmp.groupby('question_group_id', as_index=False).first()
        best['best_method_name'] = best['method']
        best['best_method_label'] = best['best_method_name'].map(METHOD_TO_LABEL).astype(int)
        label_rows.append(best[['question_group_id','train_regime','best_method_name','best_method_label']])

    scores = pd.concat(score_rows, ignore_index=True)
    labels = pd.concat(label_rows, ignore_index=True)
    return scores, labels

def find_first(existing_cols, candidates):
    for c in candidates:
        if c in existing_cols: return c
    return None

def normalize_split_value(x):
    s = str(x).lower()
    if s.startswith('val'): return 'val'
    if s.startswith('test'): return 'test'
    return s

def feature_columns_for_mode(mode: str):
    m = str(mode).lower()
    if m in {'2.2','pair','pair_only'}: return PAIR_FEATURES
    if m in {'2.4','logits','logit','logits_only'}: return LOGIT_FEATURES
    if m in {'joint','both','2.2+2.4'}: return PAIR_FEATURES + LOGIT_FEATURES
    raise ValueError(f'Unsupported feature_mode: {mode}')

def discover_logit_file(logits_dir: Path, split: str) -> Path:
    candidates = []
    for parent in [logits_dir, logits_dir / 'logits']:
        if not parent.exists(): continue
        for p in parent.glob('*.csv'):
            name = p.name.lower()
            if 'logit' in name and split in name:
                candidates.append(p)
    preferred = [p for p in candidates if p.name.lower().startswith('base_bio_logit')]
    if preferred: return sorted(preferred)[0]
    if candidates: return sorted(candidates)[0]
    raise FileNotFoundError(f'Could not find a *logit* file for split={split} under {logits_dir}')

def coerce_binary_pred(series: pd.Series) -> pd.Series:
    def f(x):
        if pd.isna(x): return np.nan
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {'true','t','1','yes'}: return 1
            if s in {'false','f','0','no'}: return 0
        try:
            v = float(x)
            return int(v > 0.5) if v not in (0.0, 1.0) else int(v)
        except Exception:
            return np.nan
    return series.map(f)

def extract_margin(df: pd.DataFrame) -> pd.Series:
    cols = set(df.columns)
    c = find_first(cols, MARGIN_CANDIDATES)
    if c: return pd.to_numeric(df[c], errors='coerce')
    pt = find_first(cols, PROB_TRUE_CANDIDATES)
    if pt:
        p = pd.to_numeric(df[pt], errors='coerce')
        return 2 * p - 1
    lt = find_first(cols, LOGIT_TRUE_CANDIDATES)
    lf = find_first(cols, LOGIT_FALSE_CANDIDATES)
    if lt and lf:
        return pd.to_numeric(df[lt], errors='coerce') - pd.to_numeric(df[lf], errors='coerce')
    return pd.Series(np.nan, index=df.index)

def _to_truth_series(series: pd.Series):
    def f(x):
        if pd.isna(x): return np.nan
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {'true','t','1','yes'}: return 1
            if s in {'false','f','0','no'}: return 0
        try:
            v = float(x)
            if v in (0.0, 1.0): return int(v)
        except Exception:
            pass
        return np.nan
    return series.map(f)

def _detect_truth_column(df: pd.DataFrame):
    cols = set(df.columns)
    c = find_first(cols, TRUTH_CANDIDATES)
    if c:
        vals = _to_truth_series(df[c])
        if vals.notna().sum() > 0: return c
    for c in df.columns:
        vals = _to_truth_series(df[c])
        if vals.notna().sum() >= max(2, int(0.8 * len(df))): return c
    return None

def load_base_predictions(logits_dir: Path) -> pd.DataFrame:
    parts = []
    debug = {'files': []}
    for split in ['val','test']:
        path = discover_logit_file(logits_dir, split)
        df = pd.read_csv(path)
        df['official_split'] = split
        parts.append(df)
        debug['files'].append({'split': split, 'path': str(path), 'columns': list(df.columns), 'n_rows': int(len(df))})
    raw = pd.concat(parts, ignore_index=True)
    qid_col = find_first(set(raw.columns), GROUP_CANDIDATES)
    truth_col = _detect_truth_column(raw)
    pred_col = find_first(set(raw.columns), PRED_CANDIDATES)

    out = pd.DataFrame(index=raw.index)
    out['official_split'] = raw['official_split'].map(normalize_split_value)
    out['question_group_id'] = raw[qid_col].astype(str) if qid_col else np.nan
    out['statement_truth'] = _to_truth_series(raw[truth_col]) if truth_col else np.nan

    if pred_col:
        pred_true = coerce_binary_pred(raw[pred_col]); pred_source = f'pred_col:{pred_col}'
    else:
        pt = find_first(set(raw.columns), PROB_TRUE_CANDIDATES)
        if pt:
            pred_true = (pd.to_numeric(raw[pt], errors='coerce') >= 0.5).astype(float); pred_source = f'prob_true:{pt}'
        else:
            margin = extract_margin(raw)
            pred_true = (margin > 0).astype(float); pred_true[pd.isna(margin)] = np.nan; pred_source = 'derived_from_margin'
    out['pred_true'] = pred_true
    out['logit_margin'] = extract_margin(raw)
    out['eval_row_order'] = raw.groupby('official_split').cumcount()
    debug['detected'] = {'qid_col': qid_col, 'truth_col': truth_col, 'pred_col': pred_col, 'pred_source': pred_source}
    out.attrs['debug'] = debug
    return out

def load_pairs(pairs_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(pairs_csv)
    cols = set(df.columns)
    qid_col = find_first(cols, GROUP_CANDIDATES)
    if not qid_col: raise KeyError(f'Could not find question-group column in {pairs_csv}. Columns: {list(df.columns)}')
    split_col = find_first(cols, SPLIT_CANDIDATES)
    if not split_col: raise KeyError(f'Could not find split column in {pairs_csv}. Columns: {list(df.columns)}')
    truth_col = _detect_truth_column(df)
    if not truth_col: raise KeyError(f'Could not infer truth column in {pairs_csv}. Columns: {list(df.columns)}.')
    out = pd.DataFrame()
    out['question_group_id'] = df[qid_col].astype(str)
    out['official_split'] = df[split_col].map(normalize_split_value)
    out = out[out['official_split'].isin(['val','test'])].copy()
    out['statement_truth'] = _to_truth_series(df.loc[out.index, truth_col]).astype(int)
    out['eval_row_order'] = out.groupby('official_split').cumcount()
    out.attrs['debug'] = {'qid_col': qid_col, 'split_col': split_col, 'truth_col': truth_col, 'columns': list(df.columns), 'n_rows': int(len(df))}
    return out.reset_index(drop=True)

def join_pairs_predictions(pairs_csv: Path, logits_dir: Path):
    pairs = load_pairs(pairs_csv)
    preds = load_base_predictions(logits_dir)
    debug = {'pairs_debug': getattr(pairs,'attrs',{}).get('debug',{}), 'preds_debug': getattr(preds,'attrs',{}).get('debug',{})}
    explicit_ok = preds['question_group_id'].notna().all() and preds['statement_truth'].notna().all()
    if explicit_ok:
        merged_explicit = pairs.merge(
            preds[['official_split','question_group_id','statement_truth','pred_true','logit_margin']],
            on=['official_split','question_group_id','statement_truth'], how='left')
        missing_explicit = int(merged_explicit['pred_true'].isna().sum())
    else:
        merged_explicit = None; missing_explicit = None
    debug['explicit_join_missing'] = missing_explicit
    merged_roworder = pairs.merge(
        preds[['official_split','eval_row_order','pred_true','logit_margin']],
        on=['official_split','eval_row_order'], how='left')
    missing_roworder = int(merged_roworder['pred_true'].isna().sum())
    debug['roworder_join_missing'] = missing_roworder
    if merged_explicit is not None and missing_explicit == 0:
        merged = merged_explicit; debug['join_strategy_used'] = 'explicit'
    elif missing_roworder == 0:
        merged = merged_roworder; debug['join_strategy_used'] = 'roworder'
    elif merged_explicit is not None and missing_explicit < missing_roworder:
        merged = merged_explicit; debug['join_strategy_used'] = 'explicit_partial'
    else:
        merged = merged_roworder; debug['join_strategy_used'] = 'roworder_partial'
    merged.attrs['debug'] = debug
    if merged['pred_true'].isna().any():
        raise ValueError(f'Failed to join some base predictions to val/test pairs. Join diagnostics: {debug}')
    return merged

def build_pair_and_logit_features(pairs_csv: Path, logits_dir: Path):
    merged = join_pairs_predictions(pairs_csv, logits_dir)
    rows = []
    for (split, qid), g in merged.groupby(['official_split','question_group_id'], sort=False):
        gt = g.sort_values('statement_truth', ascending=False).reset_index(drop=True)
        if len(gt) != 2: continue
        row_true = gt.iloc[0] if int(gt.iloc[0]['statement_truth']) == 1 else gt.iloc[1]
        row_false = gt.iloc[1] if int(gt.iloc[1]['statement_truth']) == 0 else gt.iloc[0]
        pT, pF = int(row_true['pred_true']), int(row_false['pred_true'])
        cT, cF = int(pT == 1), int(pF == 0)
        mT = float(row_true['logit_margin']) if pd.notna(row_true['logit_margin']) else np.nan
        mF = float(row_false['logit_margin']) if pd.notna(row_false['logit_margin']) else np.nan
        rows.append({
            'question_group_id': str(qid),'official_split': split,
            'pred_true_on_true_stmt': pT,'pred_true_on_false_stmt': pF,
            'true_stmt_correct': cT,'false_stmt_correct': cF,
            'tf_correct_count': cT + cF,'both_correct': int(cT == 1 and cF == 1),
            'both_wrong': int(cT == 0 and cF == 0),'exactly_one_wrong': int((cT + cF) == 1),
            'pattern_TT': int(pT == 1 and pF == 1),'pattern_TF': int(pT == 1 and pF == 0),
            'pattern_FT': int(pT == 0 and pF == 1),'pattern_FF': int(pT == 0 and pF == 0),
            'pair_logically_consistent': int(pT == 1 and pF == 0),
            'pair_contradiction_flag': int((pT == 1 and pF == 1) or (pT == 0 and pF == 0)),
            'bias_to_true': (pT + pF) / 2.0,'bias_to_false': 1.0 - ((pT + pF) / 2.0),
            'true_stmt_logit_margin': mT,'false_stmt_logit_margin': mF,
            'pair_mean_margin': np.nanmean([mT, mF]),
            'pair_abs_mean_margin': np.nanmean([abs(mT) if pd.notna(mT) else np.nan, abs(mF) if pd.notna(mF) else np.nan]),
            'pair_margin_gap': np.nan if pd.isna(mT) or pd.isna(mF) else (mT - mF),
            'pair_margin_std': np.nanstd([mT, mF]),
            'pair_confidence_gap': np.nan if pd.isna(mT) or pd.isna(mF) else (abs(mT) - abs(mF)),
        })
    out = pd.DataFrame(rows); out.attrs['debug'] = getattr(merged,'attrs',{}).get('debug',{})
    if out.empty: raise ValueError('No pair/logit feature rows were built.')
    return out
