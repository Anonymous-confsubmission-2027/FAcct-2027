"""Additional FAMOUS analyses reported in the paper, run after famous_pipeline.py.

Usage:
  python famous_extra.py --runs DIR_MAIN --compare DIR_FR DIR_EN2000
                         --strata evidence_strata.csv --recognition coding.csv
                         --shifts OUT_MAIN/domain_shifts.csv --out OUTDIR

DIR_MAIN, DIR_FR and DIR_EN2000 contain the run files of the English 800-character,
French and English 2,000-character conditions (one CSV per source set, model and regime).
The recognition file has columns model, domain, identified (and optionally identified_strict),
coded from the probe in the language of the runs passed with --runs and --shifts.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from famous_pipeline import load_runs, rag_paired, bh, FUNCTIONAL

CELLS = 'ABCDS'
V = [f'verdict_{c}' for c in CELLS]


def evidence_strata(W, strata, B, rng):
    S = pd.read_csv(strata)
    groups = {'all': None,
              'revealing': set(S[S.stratum.isin(['factcheck_domain', 'verdict_terms', 'original_copy'])].id),
              'strictly_clean': set(S[S.stratum == 'clean_strict'].id)}
    out = []
    for name, ids in groups.items():
        r = rag_paired(W, B, rng, ids=ids)
        g = W if ids is None else W[W.id.isin(ids)]
        acc = g.groupby(['model', 'rag']).apply(lambda x: 100 * (x.verdict_S == x.vrai_label).mean()).unstack()
        real = g.groupby(['model', 'rag']).apply(lambda x: 100 * (x.verdict_S == 'real').mean()).unstack()
        r = r.merge(acc.rename(columns={0: 'acc_S_norag', 1: 'acc_S_rag'}).reset_index(), on='model')
        r = r.merge(real.rename(columns={0: 'real_S_norag', 1: 'real_S_rag'}).reset_index(), on='model')
        out.append(r.assign(stratum=name))
    return pd.concat(out, ignore_index=True)


def refusals(W):
    out = []
    for (m, r), g in W.groupby(['model', 'rag']):
        ind = g[V] == 'indetermine'
        row = {'model': m, 'rag': r, 'pairs_with_indeterminate': 100 * ind.any(axis=1).mean(),
               'flip_without_indeterminate': 100 * g[~ind.any(axis=1)].flip.mean()}
        row.update({f'indeterminate_{c}': 100 * ind[f'verdict_{c}'].mean() for c in CELLS})
        out.append(row)
    return pd.DataFrame(out)


def ols_hc3(y, X):
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, XtXi, X)
    S = (X * ((e / (1 - h)) ** 2)[:, None]).T @ X
    se = np.sqrt(np.diag(XtXi @ S @ XtXi))
    return b, se, 2 * stats.norm.sf(np.abs(b / se))


def recognition_regression(shifts, coding, column):
    D = pd.read_csv(shifts)
    D = D[(D.rag == 0) & D.model.isin(FUNCTIONAL)]
    R = pd.read_csv(coding)
    M = D.merge(R[['model', 'domain', column]].rename(columns={column: 'identified'}), on=['model', 'domain'])
    M['cred'] = M.cell.isin(['A', 'C']).astype(float)
    M['prom'] = M.cell.isin(['A', 'B']).astype(float)
    X = pd.DataFrame({'const': 1.0, 'cred': M.cred, 'identified': M.identified.astype(float),
                      'cred_x_identified': M.cred * M.identified, 'prom': M.prom})
    for m in sorted(M.model.unique())[1:]:
        X[m] = (M.model == m).astype(float)
    b, se, p = ols_hc3(M['shift'].values, X.values)
    res = pd.DataFrame({'term': X.columns, 'coef': b, 'se': se, 'p': p}).iloc[:5]
    means = M.groupby(['identified', 'cred'])['shift'].mean().reset_index()
    return res.assign(coding=column, n=len(M)), means.assign(coding=column)


def compare_conditions(main, others):
    key = ['id', 'model', 'rag', 'set']
    out = []
    for name, Wo in others.items():
        J = main.merge(Wo, on=key, suffixes=('_m', '_o'))
        for (m, r), g in J.groupby(['model', 'rag']):
            agree = np.mean([(g[f'{v}_m'] == g[f'{v}_o']).mean() for v in V])
            out.append({'condition': name, 'model': m, 'rag': r, 'verdict_agreement': 100 * agree,
                        'real_S_main': 100 * (g.verdict_S_m == 'real').mean(),
                        'real_S_other': 100 * (g.verdict_S_o == 'real').mean()})
    return pd.DataFrame(out)


def shift_correlations(shift_files):
    tabs = {k: pd.read_csv(f) for k, f in shift_files.items()}
    names = list(tabs)
    base = tabs[names[0]][['model', 'rag', 'domain', 'shift']].rename(columns={'shift': names[0]})
    for k in names[1:]:
        base = base.merge(tabs[k][['model', 'rag', 'domain', 'shift']].rename(columns={'shift': k}),
                          on=['model', 'rag', 'domain'])
    base = base[base.model.isin(FUNCTIONAL)]
    out = []
    for r, g in base.groupby('rag'):
        for k in names[1:]:
            out.append({'rag': r, 'pair': f'{names[0]}~{k}', 'pearson_r': g[names[0]].corr(g[k])})
    return pd.DataFrame(out)


def significant_shifts(shift_file):
    D = pd.read_csv(shift_file)
    D = D[D.model.isin(FUNCTIONAL)].copy()
    for r in (0, 1):
        m = D.rag == r
        D.loc[m, 'q112'] = bh(D.loc[m, 'p'].values)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--compare', nargs='*', default=[])
    ap.add_argument('--compare-shifts', nargs='*', default=[])
    ap.add_argument('--strata')
    ap.add_argument('--recognition')
    ap.add_argument('--shifts')
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    W = load_runs(a.runs)
    refusals(W).to_csv(f'{a.out}/refusals.csv', index=False)
    if a.strata:
        evidence_strata(W, a.strata, a.B, rng).to_csv(f'{a.out}/retrieval_by_stratum.csv', index=False)
    if a.shifts:
        significant_shifts(a.shifts).to_csv(f'{a.out}/domain_shifts_bh112.csv', index=False)
        if a.recognition:
            cols = [c for c in ('identified', 'identified_strict') if c in pd.read_csv(a.recognition).columns]
            res, means = zip(*[recognition_regression(a.shifts, a.recognition, c) for c in cols])
            pd.concat(res).to_csv(f'{a.out}/recognition_regression.csv', index=False)
            pd.concat(means).to_csv(f'{a.out}/recognition_means.csv', index=False)
    if a.compare:
        others = {os.path.basename(os.path.normpath(d)): load_runs(d) for d in a.compare}
        compare_conditions(W, others).to_csv(f'{a.out}/condition_agreement.csv', index=False)
    if a.shifts and a.compare_shifts:
        files = {'main': a.shifts}
        files.update({os.path.basename(os.path.dirname(f)) or f: f for f in a.compare_shifts})
        shift_correlations(files).to_csv(f'{a.out}/shift_correlations.csv', index=False)


if __name__ == '__main__':
    main()
