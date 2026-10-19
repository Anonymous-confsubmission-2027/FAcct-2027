"""Dependence-aware analyses for FAMOUS, run on existing verdicts (no new model calls).

Usage:
  python famous_revision.py --runs DIR_EN800 --runs-fr DIR_FR800 --recognition recognition_coding.csv
                            --out OUTDIR [--B 10000] [--seed 2026]

Outputs:
  presence_clustered.csv   Net shift in the real rate per cell (named domain minus no source),
                           with article-cluster bootstrap intervals and Holm-adjusted p-values
                           (conditional on the drawn domains) and crossed article/domain intervals
                           (category level), plus direction counts.
  disagreement.csv         Five-condition flip rate, baseline-relative disagreement and pairwise
                           disagreement among named domains, each split into fake/real reversals
                           and changes involving an indeterminate verdict, with article-cluster intervals.
  recognition_dependence.csv  Domain-level regression of the shift on credibility, identification,
                           their interaction and prominence: HC3, domain-clustered (CR1) and wild
                           cluster bootstrap inference, and a crossed bootstrap that reconstructs
                           the shifts from resampled articles and domains.
  reliability_probe.csv    Agreement between model-stated and external reliability, and the
                           association of stated reliability with the observed shift.
"""
import argparse
import os

import numpy as np
import pandas as pd

from famous_pipeline import load_runs, FUNCTIONAL

CELLS = 'ABCD'
PROM = {'A': 1, 'B': 1, 'C': 0, 'D': 0}
CRED = {'A': 1, 'B': 0, 'C': 1, 'D': 0}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p)
    q = np.maximum.accumulate((n - np.arange(n)) * p[o]); out = np.empty(n); out[o] = np.minimum(q, 1)
    return out


def boot_p(draws):
    draws = np.asarray(draws)
    return min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))


def matrices(g):
    """Article x set matrices of verdicts for one model and regime."""
    ids = np.sort(g.id.unique()); sets = np.sort(g.set.unique())
    V = {c: g.pivot_table(index='id', columns='set', values=f'verdict_{c}', aggfunc='first').reindex(index=ids, columns=sets).values
         for c in 'ABCDS'}
    dom = {c: g.groupby('set')[f'source_{c}'].first().reindex(sets).values for c in CELLS}
    return ids, sets, V, dom


def presence_clustered(W, B, rng):
    out = []
    for (m, r), g in W.groupby(['model', 'rag']):
        ids, sets, V, _ = matrices(g)
        S = V['S']; rS = (S == 'real').astype(float)
        n, k = rS.shape
        rows = []
        for c in CELLS:
            X = V[c]; d = (X == 'real').astype(float) - rS
            est = d.mean()
            art = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
            crossed = np.array([d[np.ix_(rng.integers(0, n, n), rng.integers(0, k, k))].mean() for _ in range(B)])
            ind = (X == 'indetermine') | (S == 'indetermine')
            rows.append(dict(model=m, rag=r, cell=c, net_shift=100 * est,
                             art_lo=100 * np.percentile(art, 2.5), art_hi=100 * np.percentile(art, 97.5), p_art=boot_p(art),
                             crossed_lo=100 * np.percentile(crossed, 2.5), crossed_hi=100 * np.percentile(crossed, 97.5),
                             p_crossed=boot_p(crossed),
                             fake_to_real=int(((S == 'fake') & (X == 'real')).sum()),
                             real_to_fake=int(((S == 'real') & (X == 'fake')).sum()),
                             involving_indeterminate=int((ind & (X != S)).sum())))
        rows = pd.DataFrame(rows)
        rows['p_art_holm'] = holm(rows.p_art.values)
        rows['p_crossed_holm'] = holm(rows.p_crossed.values)
        out.append(rows)
    return pd.concat(out, ignore_index=True)


def disagreement(W, B, rng):
    out = []
    pairs = [(a, b) for i, a in enumerate(CELLS) for b in CELLS[i + 1:]]
    for (m, r), g in W.groupby(['model', 'rag']):
        ids, sets, V, _ = matrices(g)
        n = len(ids)
        allv = np.stack([V[c] for c in 'ABCDS'])
        flip = (allv != allv[0]).any(axis=0)
        has_f = (allv == 'fake').any(axis=0); has_r = (allv == 'real').any(axis=0)
        rev5 = has_f & has_r

        def split(x, y):
            diff = x != y
            rev = ((x == 'fake') & (y == 'real')) | ((x == 'real') & (y == 'fake'))
            return diff, rev
        base = [split(V[c], V['S']) for c in CELLS]
        bd = np.mean([b[0] for b in base], axis=0); br = np.mean([b[1] for b in base], axis=0)
        pw = [split(V[a], V[b]) for a, b in pairs]
        pd_ = np.mean([p[0] for p in pw], axis=0); pr = np.mean([p[1] for p in pw], axis=0)
        per_art = {'flip': flip.mean(1), 'flip_reversal': rev5.mean(1),
                   'baseline': bd.mean(1), 'baseline_reversal': br.mean(1),
                   'pairwise': pd_.mean(1), 'pairwise_reversal': pr.mean(1)}
        row = dict(model=m, rag=r)
        idx = [rng.integers(0, n, n) for _ in range(B)]
        for k, v in per_art.items():
            bs = np.array([v[i].mean() for i in idx])
            row[k] = 100 * v.mean(); row[f'{k}_lo'] = 100 * np.percentile(bs, 2.5); row[f'{k}_hi'] = 100 * np.percentile(bs, 97.5)
        out.append(row)
    return pd.DataFrame(out)


def ols(y, X):
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ X.T @ y
    return b, XtXi, y - X @ b


def hc3_se(X, XtXi, e):
    h = np.einsum('ij,jk,ik->i', X, XtXi, X)
    S = (X * ((e / (1 - h)) ** 2)[:, None]).T @ X
    return np.sqrt(np.diag(XtXi @ S @ XtXi))


def cr1_se(X, XtXi, e, cl):
    G = len(np.unique(cl)); n, k = X.shape
    meat = np.zeros((k, k))
    for c in np.unique(cl):
        s = X[cl == c].T @ e[cl == c]; meat += np.outer(s, s)
    V = XtXi @ meat @ XtXi * G / (G - 1) * (n - 1) / (n - k)
    return np.sqrt(np.diag(V))


def wild_cluster_p(y, X, cl, j, B, rng):
    """Wild cluster bootstrap-t (Rademacher weights, null imposed) for coefficient j."""
    b, XtXi, e = ols(y, X)
    t0 = b[j] / cr1_se(X, XtXi, e, cl)[j]
    Xr = np.delete(X, j, axis=1); br, _, er = ols(y, Xr); fitted = Xr @ br
    groups = np.unique(cl); gi = np.searchsorted(groups, cl)
    ts = np.empty(B)
    for i in range(B):
        w = rng.choice([-1.0, 1.0], len(groups))[gi]
        yb = fitted + er * w
        bb, XtXib, eb = ols(yb, X)
        ts[i] = bb[j] / cr1_se(X, XtXib, eb, cl)[j]
    return (np.abs(ts) >= abs(t0)).mean()


def design(D, ident_col):
    X = pd.DataFrame({'const': 1.0, 'cred': D.cred, 'identified': D[ident_col].astype(float),
                      'cred_x_identified': D.cred * D[ident_col].astype(float), 'prom': D.prom})
    for mm in sorted(D.model.unique())[1:]:
        X[mm] = (D.model == mm).astype(float)
    return X


def shift_frame(W, R, ident_col):
    """One row per functional model and drawn domain (no retrieval)."""
    rows = []
    for m, g in W[(W.rag == 0) & W.model.isin(FUNCTIONAL)].groupby('model'):
        ids, sets, V, dom = matrices(g)
        rS = (V['S'] == 'real').astype(float)
        for c in CELLS:
            d = (V[c] == 'real').astype(float) - rS
            for si in range(len(sets)):
                rows.append(dict(model=m, cell=c, set_idx=si, domain=dom[c][si], shift=100 * d[:, si].mean(),
                                 prom=PROM[c], cred=CRED[c]))
    D = pd.DataFrame(rows).merge(R[['model', 'domain', ident_col]], on=['model', 'domain'], how='left')
    return D


def recognition_dependence(W, R, ident_col, label, B, rng):
    D = shift_frame(W, R, ident_col)
    X = design(D, ident_col); y = D['shift'].values; cl = D.domain.values
    b, XtXi, e = ols(y, X.values)
    j = list(X.columns).index('cred_x_identified')
    se_hc3 = hc3_se(X.values, XtXi, e); se_cr = cr1_se(X.values, XtXi, e, cl)
    from scipy import stats
    G = len(np.unique(cl))
    p_hc3 = 2 * stats.norm.sf(abs(b[j] / se_hc3[j])); p_cr = 2 * stats.t.sf(abs(b[j] / se_cr[j]), G - 1)
    p_wild = wild_cluster_p(y, X.values, cl, j, B, rng)
    jp = list(X.columns).index('prom')
    p_wild_prom = wild_cluster_p(y, X.values, cl, jp, B, rng)
    # crossed bootstrap: resample articles (jointly across models and domains) and domains within cells,
    # reconstruct each domain-model shift from the resampled articles, refit.
    mats = {}
    for m, g in W[(W.rag == 0) & W.model.isin(FUNCTIONAL)].groupby('model'):
        ids, sets, V, dom = matrices(g)
        rS = (V['S'] == 'real').astype(float)
        mats[m] = {c: (V[c] == 'real').astype(float) - rS for c in CELLS}
        n, k = rS.shape
    ident = D.set_index(['model', 'cell', 'set_idx'])[ident_col]
    models = sorted(mats)
    est = []
    for _ in range(B):
        ai = rng.integers(0, n, n)
        si = {c: rng.integers(0, k, k) for c in CELLS}
        rows = []
        for m in models:
            for c in CELLS:
                sh = 100 * mats[m][c][ai].mean(axis=0)
                for s in si[c]:
                    rows.append((m, PROM[c], CRED[c], ident.loc[(m, c, s)], sh[s]))
        Db = pd.DataFrame(rows, columns=['model', 'prom', 'cred', ident_col, 'shift'])
        Xb = design(Db, ident_col)
        if np.linalg.matrix_rank(Xb.values) < Xb.shape[1]:
            continue
        bb, _, _ = ols(Db['shift'].values, Xb.values)
        est.append(bb[j])
    est = np.array(est)
    return dict(coding=label, n=len(D), domains=G, coef=b[j], se_hc3=se_hc3[j], p_hc3=p_hc3,
                se_cluster=se_cr[j], p_cluster=p_cr, p_wild_cluster=p_wild,
                crossed_lo=np.percentile(est, 2.5), crossed_hi=np.percentile(est, 97.5), crossed_p=boot_p(est),
                crossed_valid=len(est), prom_coef=b[jp], prom_p_wild_cluster=p_wild_prom)


def reliability_probe(W, R):
    from scipy import stats
    D = shift_frame(W, R, 'identified')
    rel = R[['model', 'domain', 'stated_reliability']].copy()
    rel['stated'] = rel.stated_reliability.map({'reliable': 1.0, 'generally reliable': 1.0, 'uncertain': 0.5,
                                                'unreliable': 0.0})
    D = D.merge(rel, on=['model', 'domain'], how='left')
    rated = D.dropna(subset=['stated'])
    decisive = rated[rated.stated != 0.5]
    out = [dict(measure='share_rated', value=100 * len(rated) / len(D)),
           dict(measure='share_uncertain_among_rated', value=100 * (rated.stated == 0.5).mean()),
           dict(measure='agreement_with_external_among_decisive', value=100 * (decisive.stated == decisive.cred).mean()),
           dict(measure='n_decisive', value=len(decisive))]
    r, p = stats.spearmanr(rated.stated, rated['shift'])
    out += [dict(measure='spearman_stated_vs_shift', value=r), dict(measure='p_spearman', value=p)]
    for m, g in rated.groupby('model'):
        dg = g[g.stated != 0.5]
        out.append(dict(measure=f'agreement_{m}', value=100 * (dg.stated == dg.cred).mean() if len(dg) else np.nan))
    X0 = design(rated, 'identified')
    b0, XtXi0, e0 = ols(rated['shift'].values, X0.values)
    j0 = list(X0.columns).index('cred_x_identified')
    se0 = cr1_se(X0.values, XtXi0, e0, rated.domain.values)
    out.append(dict(measure='coef_cred_x_identified_rated_subset', value=b0[j0]))
    out.append(dict(measure='p_cred_x_identified_rated_subset',
                    value=2 * stats.t.sf(abs(b0[j0] / se0[j0]), rated.domain.nunique() - 1)))
    X = X0.copy(); X['stated'] = rated.stated.values
    b, XtXi, e = ols(rated['shift'].values, X.values)
    se = cr1_se(X.values, XtXi, e, rated.domain.values)
    G = rated.domain.nunique()
    for name in ('cred_x_identified', 'stated'):
        j = list(X.columns).index(name)
        out.append(dict(measure=f'coef_{name}_with_stated', value=b[j]))
        out.append(dict(measure=f'p_{name}_with_stated', value=2 * stats.t.sf(abs(b[j] / se[j]), G - 1)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--runs-fr')
    ap.add_argument('--recognition', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--B', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    W = load_runs(a.runs)
    W = W[W.model.isin(FUNCTIONAL)]
    R = pd.read_csv(a.recognition)
    presence_clustered(W, a.B, rng).to_csv(f'{a.out}/presence_clustered.csv', index=False)
    disagreement(W, min(a.B, 2000), rng).to_csv(f'{a.out}/disagreement.csv', index=False)
    rec = [recognition_dependence(W, R, 'identified', 'primary', min(a.B, 2000), rng),
           recognition_dependence(W, R, 'identified_strict', 'strict', min(a.B, 2000), rng)]
    if a.runs_fr:
        Wf = load_runs(a.runs_fr)
        rec.append(recognition_dependence(Wf[Wf.model.isin(FUNCTIONAL)], R, 'identified_fr', 'french', min(a.B, 2000), rng))
    pd.DataFrame(rec).to_csv(f'{a.out}/recognition_dependence.csv', index=False)
    reliability_probe(W, R).to_csv(f'{a.out}/reliability_probe.csv', index=False)


if __name__ == '__main__':
    main()
