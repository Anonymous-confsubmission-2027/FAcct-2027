"""FAMOUS analysis pipeline for any number of source sets.

Usage:
  python pipeline.py --runs DIR [--glob PATTERN] --out OUTDIR
                     [--exclude-b ajc.com,aljazeera.com]
                     [--recognition FILE] [--english DIR]
                     [--evidence FILE] [--sample FILE]

Run files: one CSV per (source set, model, regime) with columns
id, vraie_source, vrai_label, source_A..source_D, verdict_A..verdict_D, verdict_S.
File names must contain (seed|tirage)<n>_<model>_(RAG|WRAG).
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from scipy import stats

C = 'ABCD'; ALL = 'ABCDS'
FUNCTIONAL = ['qwen2.5_7b', 'phi4-mini', 'llama3.2_3b', 'gemma3_4b']
NAME_RE = re.compile(r'(?:seed|tirage)_?(\d+)_(.+?)_(RAG|WRAG)(?:_[A-Za-z]+)?\.csv$')


def load_runs(folder, pattern='*.csv'):
    L = []
    for f in sorted(glob.glob(os.path.join(folder, pattern))):
        m = NAME_RE.search(os.path.basename(f))
        if not m:
            continue
        s, mod, mode = m.groups()
        d = pd.read_csv(f)
        need = {'id', 'vrai_label', 'verdict_S'} | {f'verdict_{c}' for c in C} | {f'source_{c}' for c in C}
        missing = need - set(d.columns)
        if missing:
            raise ValueError(f'{f}: missing columns {missing}')
        d['model'] = mod; d['rag'] = int(mode == 'RAG'); d['set'] = int(s)
        L.append(d)
    if not L:
        raise SystemExit(f'No run files matched in {folder}')
    W = pd.concat(L, ignore_index=True)
    W['flip'] = W[[f'verdict_{c}' for c in ALL]].nunique(axis=1) > 1
    return W


def check_design(W):
    """Source sets: one domain per cell, constant within a set, no repeats across sets."""
    rows = []
    for s, g in W.groupby('set'):
        row = {'set': s}
        for c in C:
            u = g[f'source_{c}'].unique()
            if len(u) != 1:
                raise ValueError(f'set {s}, cell {c}: {len(u)} domains')
            row[c] = u[0]
        rows.append(row)
    D = pd.DataFrame(rows).sort_values('set')
    allv = D[list(C)].values.ravel()
    dup = pd.Series(allv).value_counts()
    rep = dup[dup > 1].to_dict()
    cover = W.groupby(['model', 'rag']).set.nunique().unstack()
    return D, rep, cover


def boot_articles(g, f, B, rng):
    ids = g.id.unique(); grp = {u: np.flatnonzero(g.id.values == u) for u in ids}
    out = []
    for _ in range(B):
        s = rng.choice(ids, len(ids)); rows = np.concatenate([grp[u] for u in s])
        out.append(f(g.iloc[rows]))
    return np.percentile(out, [2.5, 97.5])


def flip_table(W, B, rng):
    F = []
    for (m, r), g in W.groupby(['model', 'rag']):
        g = g.reset_index(drop=True)
        lo, hi = boot_articles(g, lambda x: x.flip.mean(), B, rng)
        acc = [(g[f'verdict_{c}'] == g.vrai_label).mean() for c in ALL]
        F.append(dict(model=m, rag=r, n_sets=g.set.nunique(), flip=100 * g.flip.mean(), lo=100 * lo, hi=100 * hi,
                      any_set=100 * g.groupby('id').flip.max().mean(),
                      acc_min=100 * min(acc), acc_max=100 * max(acc), acc_S=100 * acc[4],
                      indet=100 * (g[[f'verdict_{c}' for c in ALL]] == 'indetermine').values.mean()))
    return pd.DataFrame(F)


def transitions(W):
    T = []
    for (m, r), g in W.groupby(['model', 'rag']):
        for lab in ['all', 'real', 'fake']:
            h = g if lab == 'all' else g[g.vrai_label == lab]
            oS = h.verdict_S == h.vrai_label
            ci = sum(((oS) & (h[f'verdict_{c}'] != h.vrai_label)).sum() for c in C)
            ic = sum(((~oS) & (h[f'verdict_{c}'] == h.vrai_label)).sum() for c in C)
            n = 4 * len(h)
            T.append(dict(model=m, rag=r, label=lab, n=n, correct_to_incorrect=int(ci), incorrect_to_correct=int(ic),
                          net_pp=100 * (ic - ci) / n, gross_pp=100 * (ic + ci) / n))
    return pd.DataFrame(T)


def class_conditional(W):
    K = []
    for (m, r), g in W.groupby(['model', 'rag']):
        for cond, cols in [('S', ['verdict_S']), ('named', [f'verdict_{c}' for c in C])]:
            rr = np.mean([(g[g.vrai_label == 'real'][c] == 'real').mean() for c in cols])
            rf = np.mean([(g[g.vrai_label == 'fake'][c] == 'fake').mean() for c in cols])
            K.append(dict(model=m, rag=r, cond=cond, recall_real=100 * rr, recall_fake=100 * rf, bal_acc=50 * (rr + rf),
                          flip_real=100 * g[g.vrai_label == 'real'].flip.mean(),
                          flip_fake=100 * g[g.vrai_label == 'fake'].flip.mean()))
    return pd.DataFrame(K)


def domain_shifts(W, B, rng):
    D = []
    for (m, r, s), g in W.groupby(['model', 'rag', 'set']):
        for c in C:
            v = (g[f'verdict_{c}'] == 'real').astype(float).values - (g.verdict_S == 'real').astype(float).values
            bs = [rng.choice(v, len(v)).mean() for _ in range(B)]
            to_r = int(((g[f'verdict_{c}'] == 'real') & (g.verdict_S != 'real')).sum())
            to_f = int(((g[f'verdict_{c}'] != 'real') & (g.verdict_S == 'real')).sum())
            p = stats.binomtest(to_r, to_r + to_f).pvalue if to_r + to_f else 1.0
            D.append(dict(model=m, rag=r, set=s, cell=c, domain=g[f'source_{c}'].iloc[0], shift=100 * v.mean(),
                          lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5),
                          to_real=to_r, to_fake=to_f, p=p))
    D = pd.DataFrame(D)
    D['q_bh'] = np.nan
    for (m, r), idx in D.groupby(['model', 'rag']).groups.items():
        D.loc[idx, 'q_bh'] = bh(D.loc[idx, 'p'].values)
    return D


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def cell_rates(g):
    return {c: (g[f'verdict_{c}'] == 'real').mean() for c in C}


def did_two_way(W, B, rng):
    """Category contrasts with uncertainty from both articles and domains.
    Domains are resampled within each cell (domain = source set's draw for that cell),
    articles are resampled jointly; rates are averaged over resampled domains."""
    out = []
    for (m, r), g in W.groupby(['model', 'rag']):
        sets = np.sort(g.set.unique()); ids = g.id.unique()
        R = {c: g.pivot_table(index='id', columns='set', values=f'verdict_{c}', aggfunc='first').eq('real').astype(float)
             .reindex(index=ids, columns=sets) for c in C}
        Rs = g.pivot_table(index='id', columns='set', values='verdict_S', aggfunc='first').eq('real').astype(float).reindex(index=ids, columns=sets)
        M = {c: R[c].values for c in C}; MS = Rs.values

        def stat(ai, si):
            rr = {c: np.nanmean(M[c][np.ix_(ai, si[c])]) for c in C}
            rS = np.nanmean(MS[np.ix_(ai, si['A'])])
            return dict(prom_lowcred=rr['B'] - rr['D'], cred_lowprom=rr['C'] - rr['D'],
                        cred_highprom=rr['A'] - rr['B'], did=(rr['A'] - rr['B']) - (rr['C'] - rr['D']),
                        presence=np.mean([rr[c] for c in C]) - rS)
        full = stat(np.arange(len(ids)), {c: np.arange(len(sets)) for c in C})
        art_only, two_way = [], []
        for _ in range(B):
            ai = rng.integers(0, len(ids), len(ids))
            art_only.append(stat(ai, {c: np.arange(len(sets)) for c in C}))
            two_way.append(stat(ai, {c: rng.integers(0, len(sets), len(sets)) for c in C}))
        for k in full:
            a = np.percentile([x[k] for x in art_only], [2.5, 97.5]); b = np.percentile([x[k] for x in two_way], [2.5, 97.5])
            out.append(dict(model=m, rag=r, contrast=k, est=100 * full[k], art_lo=100 * a[0], art_hi=100 * a[1],
                            twoway_lo=100 * b[0], twoway_hi=100 * b[1]))
    return pd.DataFrame(out)


def leave_one_set_out(W):
    out = []
    for (m, r), g in W.groupby(['model', 'rag']):
        for s in sorted(g.set.unique()):
            for label, h in [('only', g[g.set == s]), ('without', g[g.set != s])]:
                rr = cell_rates(h)
                out.append(dict(model=m, rag=r, set=s, mode=label, did=100 * ((rr['A'] - rr['B']) - (rr['C'] - rr['D']))))
    return pd.DataFrame(out)


def logit_cluster(X, y, cl):
    X = np.asarray(X, float); y = np.asarray(y, float); b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b)); H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (y - p)); b += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ b)); Hinv = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    sc = X * (y - p)[:, None]; meat = np.zeros((X.shape[1],) * 2)
    for u in np.unique(cl):
        s = sc[cl == u].sum(0); meat += np.outer(s, s)
    G = len(np.unique(cl))
    return b, np.sqrt(np.diag(Hinv @ meat @ Hinv * G / (G - 1)))


def regression(W):
    Lg = []
    for c in C:
        t = W[['id', 'model', 'rag', 'set', 'vrai_label', f'source_{c}', f'verdict_{c}']].copy()
        t.columns = ['id', 'model', 'rag', 'set', 'label', 'domain', 'verdict']; t['cell'] = c; Lg.append(t)
    Lg = pd.concat(Lg, ignore_index=True)
    Lg = Lg[Lg.verdict != 'indetermine']
    Lg['y'] = (Lg.verdict == 'real').astype(int); Lg['true_real'] = (Lg.label == 'real').astype(int)
    Lg['prom'] = Lg.cell.isin(['A', 'B']).astype(int); Lg['cred'] = Lg.cell.isin(['A', 'C']).astype(int)
    out = []
    for (m, r), d in Lg.groupby(['model', 'rag']):
        if m not in FUNCTIONAL:
            continue
        X = pd.DataFrame({'const': 1, 'true_real': d.true_real, 'prom': d.prom, 'cred': d.cred, 'prom_x_cred': d.prom * d.cred})
        for s in sorted(d.set.unique())[1:]:
            X[f'set{s}'] = (d.set == s).astype(int)
        b, se = logit_cluster(X.values, d.y.values, d.id.values)
        for k, bb, ss in zip(X.columns, b, se):
            if k.startswith('set') or k == 'const':
                continue
            out.append(dict(model=m, rag=r, term=k, OR=np.exp(bb), lo=np.exp(bb - 1.96 * ss), hi=np.exp(bb + 1.96 * ss),
                            p=2 * stats.norm.sf(abs(bb / ss))))
    return pd.DataFrame(out)


def presence(W):
    P = []
    for (m, r), g in W.groupby(['model', 'rag']):
        for c in C:
            a = int(((g[f'verdict_{c}'] == 'real') & (g.verdict_S != 'real')).sum())
            b = int(((g[f'verdict_{c}'] != 'real') & (g.verdict_S == 'real')).sum())
            P.append(dict(model=m, rag=r, cell=c, to_real=a, to_fake=b, p=stats.binomtest(a, a + b).pvalue if a + b else 1.0))
    P = pd.DataFrame(P); P['q_holm'] = np.nan
    for _, idx in P.groupby(['model', 'rag']).groups.items():
        p = P.loc[idx, 'p'].values; o = np.argsort(p); n = len(p)
        q = np.maximum.accumulate((n - np.arange(n)) * p[o]); qq = np.empty(n); qq[o] = np.minimum(q, 1)
        P.loc[idx, 'q_holm'] = qq
    return P


def rag_paired(W, B, rng, ids=None):
    out = []
    h = W if ids is None else W[W.id.isin(ids)]
    for m, g in h.groupby('model'):
        p = g.groupby(['id', 'rag']).flip.mean().unstack().dropna()
        d = (p[1] - p[0]).values; bs = [rng.choice(d, len(d)).mean() for _ in range(B)]
        out.append(dict(model=m, n=len(d), flip_norag=100 * p[0].mean(), flip_rag=100 * p[1].mean(), diff=100 * d.mean(),
                        lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5)))
    return pd.DataFrame(out)


def code_recognition(text):
    t = str(text).strip().lower()
    m = re.search(r'(?:connaissance|knowledge)\s*:\s*(\w+)', t)
    if m:
        t = m.group(1)
    t = re.sub(r'^[\s\*"«»\'\-:]+', '', t)
    if re.match(r'^(oui|yes)\b', t):
        return 1
    if re.match(r'^(non|no)\b', t):
        return 0
    return np.nan


def recognition(path, D):
    Rg = pd.read_csv(path)
    cols = {c.lower(): c for c in Rg.columns}
    mcol = cols.get('modele', cols.get('model', cols.get('modèle'))); dcol = cols.get('domaine', cols.get('domain'))
    rcol = cols.get('reponse', cols.get('response', cols.get('réponse')))
    Rg = Rg.rename(columns={mcol: 'model', dcol: 'domain', rcol: 'response'})
    Rg['model'] = Rg.model.str.replace(':', '_', regex=False)
    Rg['known'] = Rg.response.map(code_recognition)
    Rg['stated_reliability'] = Rg.response.str.extract(r'(?:[Ff]i\w*abilit\w*|[Rr]eliability)\s*:\s*([^\n(.,]+)', expand=False).str.strip().str.lower()
    summ = Rg.groupby('model').known.agg(['mean', 'count', lambda s: s.isna().sum()]).rename(columns={'<lambda_0>': 'uncoded'})
    M = D.merge(Rg[['model', 'domain', 'known']], on=['model', 'domain'], how='left')
    M['abs_shift'] = M['shift'].abs()
    by = M.groupby(['model', 'rag', 'known']).abs_shift.agg(['mean', 'count']).reset_index()
    return Rg, summ, M, by


def language_compare(Wfr, Wen):
    key = ['id', 'model', 'rag', 'set']
    J = Wfr.merge(Wen, on=key, suffixes=('_fr', '_en'))
    out = []
    for (m, r), g in J.groupby(['model', 'rag']):
        agree = np.mean([(g[f'verdict_{c}_fr'] == g[f'verdict_{c}_en']).mean() for c in ALL])
        out.append(dict(model=m, rag=r, n_sets=g.set.nunique(), flip_fr=100 * g.flip_fr.mean(), flip_en=100 * g.flip_en.mean(),
                        acc_S_fr=100 * (g.verdict_S_fr == g.vrai_label_fr).mean(),
                        acc_S_en=100 * (g.verdict_S_en == g.vrai_label_en).mean(), verdict_agreement=100 * agree))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True); ap.add_argument('--glob', default='*.csv'); ap.add_argument('--out', required=True)
    ap.add_argument('--exclude-b', default=''); ap.add_argument('--recognition'); ap.add_argument('--english')
    ap.add_argument('--evidence'); ap.add_argument('--strata'); ap.add_argument('--B', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=2026)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(a.seed)
    W = load_runs(a.runs, a.glob)
    design, repeats, cover = check_design(W)
    design.to_csv(f'{a.out}/design.csv', index=False)
    report = {'n_files_rows': len(W), 'sets': design.set.tolist(), 'repeated_domains': repeats,
              'coverage_sets_per_model_regime': cover.to_dict()}
    print(design.to_string(index=False)); print('repeated domains:', repeats or 'none'); print(cover)
    flip_table(W, a.B, rng).to_csv(f'{a.out}/flip.csv', index=False)
    transitions(W).to_csv(f'{a.out}/transitions.csv', index=False)
    class_conditional(W).to_csv(f'{a.out}/class_conditional.csv', index=False)
    D = domain_shifts(W, a.B, rng); D.to_csv(f'{a.out}/domain_shifts.csv', index=False)
    did_two_way(W[W.model.isin(FUNCTIONAL)], a.B, rng).to_csv(f'{a.out}/contrasts_two_way.csv', index=False)
    leave_one_set_out(W[W.model.isin(FUNCTIONAL)]).to_csv(f'{a.out}/did_leave_one_set_out.csv', index=False)
    regression(W).to_csv(f'{a.out}/regression.csv', index=False)
    presence(W).to_csv(f'{a.out}/presence.csv', index=False)
    rag_paired(W, a.B, rng).to_csv(f'{a.out}/rag_all.csv', index=False)
    if a.exclude_b:
        bad = set(a.exclude_b.split(','))
        keep = design[~design.B.isin(bad)].set
        Wr = W[W.set.isin(keep)]
        flip_table(Wr, a.B, rng).to_csv(f'{a.out}/flip_excl_b.csv', index=False)
        did_two_way(Wr[Wr.model.isin(FUNCTIONAL)], a.B, rng).to_csv(f'{a.out}/contrasts_two_way_excl_b.csv', index=False)
        report['sets_without_excluded_b'] = keep.tolist()
    if a.strata:
        S = pd.read_csv(a.strata)
        for s, g in S.groupby('stratum'):
            rag_paired(W, a.B, rng, ids=set(g.id)).assign(stratum=s).to_csv(f'{a.out}/rag_stratum_{s}.csv', index=False)
    if a.recognition:
        Rg, summ, M, by = recognition(a.recognition, D)
        Rg.to_csv(f'{a.out}/recognition_coded.csv', index=False); summ.to_csv(f'{a.out}/recognition_summary.csv')
        by.to_csv(f'{a.out}/shift_by_recognition.csv', index=False)
        print(summ)
    if a.english:
        We = load_runs(a.english, a.glob)
        language_compare(W, We).to_csv(f'{a.out}/language_compare.csv', index=False)
        flip_table(We, a.B, rng).to_csv(f'{a.out}/flip_english.csv', index=False)
    json.dump(report, open(f'{a.out}/report.json', 'w'), indent=1, default=str)
    print(pd.read_csv(f'{a.out}/flip.csv').round(1).to_string(index=False))


if __name__ == '__main__':
    main()
