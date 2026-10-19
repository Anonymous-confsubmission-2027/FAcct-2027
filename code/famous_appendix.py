"""Additional appendix analyses for the main condition (English prompt, 800 characters).

  python code/famous_appendix.py --runs data/runs/EN800 --raw data/raw --articles data/articles_150.csv --out out/appendix

Writes:
  indeterminate.csv     Classification of the indeterminate verdicts of llama3.2:3b and llama3.2:1b from their
                        raw responses: refusal (the model declines to verify), both words (contains "fake" and
                        "real"), true/false (answers "true" or "false" instead), other.
                        For llama3.2:3b, source sets 1-3 kept no raw responses in the main runs; their responses
                        are taken from the second run (data/raw/llama3.2_3b_second_run, verdicts identical) and,
                        with retrieval, for the 10 rerun articles from data/raw/EN800_RAG_rerun_10_articles.
  error_direction.csv   For each functional model, regime and cell, the named-domain verdicts that change
                        correctness relative to no source, split by true label: real_lost (a real article judged
                        real without a source and not real with the domain), real_gained, fake_accepted (a fake
                        article judged real only with the domain), fake_caught. Each cell has 525 verdicts on real
                        and 525 on fake articles (75 articles x 7 source sets).
  flip_by_subset.csv    Flip rates by FakeNewsNet subset (GossipCop, PolitiFact).
  examples.csv          Three articles from source set 2 whose verdict by phi4-mini without retrieval depends on
                        the prominence of the named domain.
"""
import argparse
import glob
import os
import re

import pandas as pd

from famous_pipeline import load_runs, FUNCTIONAL

CELLS = 'ABCDS'
REFUSAL = re.compile(r"\b(can ?not|can't|cannot|couldn't|could not|unable to|not able to|won't|will not)\b"
                     r"[^.]{0,40}\b(verify|determine|confirm|assess|evaluate|provide|help|make|fulfill)\b", re.I)
EXAMPLES = [('gossipcop-914034', 2), ('gossipcop-5112442905', 2), ('politifact627', 2)]


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def classify(resp):
    t = str(resp).lower()
    if REFUSAL.search(str(resp)):
        return 'refusal'
    if re.search(r'\bfake\b', t) and re.search(r'\breal\b', t):
        return 'both words'
    if re.search(r'\b(true|false)\b', t):
        return 'true/false'
    return 'other'


def responses(runs, raw, model, s, regime):
    a = read(f'{runs}/tirage{s}_{model}_{regime}.csv').set_index('id')
    if 'reponse_S' in a.columns:
        return a, a
    src = read(glob.glob(f'{raw}/llama3.2_3b_second_run/*800*tirage{s}_{model}_{regime}.csv')[0]).set_index('id')
    if regime == 'RAG':
        rr = read(f'{raw}/EN800_RAG_rerun_10_articles/en_10_tirage{s}_{model}_RAG.csv').set_index('id')
        src = src.copy()
        for k in CELLS:
            src.loc[rr.index, [f'verdict_{k}', f'reponse_{k}']] = rr[[f'verdict_{k}', f'reponse_{k}']].values
    return a, src


def indeterminate(runs, raw):
    out = []
    for model in ('llama3.2_3b', 'llama3.2_1b'):
        for regime in ('WRAG', 'RAG'):
            for s in range(1, 8):
                a, src = responses(runs, raw, model, s, regime)
                for i in a.index:
                    for k in CELLS:
                        assert a.at[i, f'verdict_{k}'] == src.at[i, f'verdict_{k}']
                        if a.at[i, f'verdict_{k}'] == 'indetermine':
                            out.append(dict(model=model, rag=int(regime == 'RAG'), category=classify(src.at[i, f'reponse_{k}'])))
    d = pd.DataFrame(out)
    return d.groupby(['model', 'rag', 'category']).size().rename('n').reset_index()


def error_direction(W):
    rows = []
    for (m, r), g in W[W.model.isin(FUNCTIONAL)].groupby(['model', 'rag']):
        S, real, fake = g.verdict_S, g.vrai_label == 'real', g.vrai_label == 'fake'
        for c in 'ABCD':
            N = g[f'verdict_{c}']
            rows.append(dict(model=m, rag=r, cell=c,
                             real_lost=int((real & (S == 'real') & (N != 'real')).sum()),
                             real_gained=int((real & (S != 'real') & (N == 'real')).sum()),
                             fake_accepted=int((fake & (S != 'real') & (N == 'real')).sum()),
                             fake_caught=int((fake & (S == 'real') & (N != 'real')).sum())))
    return pd.DataFrame(rows)


def flip_by_subset(W):
    W = W[W.model.isin(FUNCTIONAL)].assign(subset=lambda x: x.id.map(lambda i: 'GossipCop' if i.startswith('gossipcop') else 'PolitiFact'))
    return (W.groupby(['model', 'rag', 'subset']).flip.mean() * 100).rename('flip').reset_index()


def examples(W, articles):
    A = read(articles).set_index('id')
    g = W[(W.model == 'phi4-mini') & (W.rag == 0)]
    rows = []
    for i, s in EXAMPLES:
        r = g[(g.id == i) & (g.set == s)].iloc[0]
        rows.append(dict(id=i, set=s, subset='GossipCop' if i.startswith('gossipcop') else 'PolitiFact',
                         title=A.loc[i, 'title'], label=r.vrai_label,
                         **{f'verdict_{k}': r[f'verdict_{k}'] for k in CELLS},
                         **{f'source_{k}': r[f'source_{k}'] for k in 'ABCD'}))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='data/runs/EN800')
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--articles', default='data/articles_150.csv')
    ap.add_argument('--out', default='out/appendix')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = load_runs(a.runs)
    W = W[0] if isinstance(W, tuple) else W
    indeterminate(a.runs, a.raw).to_csv(f'{a.out}/indeterminate.csv', index=False)
    error_direction(W).to_csv(f'{a.out}/error_direction.csv', index=False)
    flip_by_subset(W).to_csv(f'{a.out}/flip_by_subset.csv', index=False)
    examples(W, a.articles).to_csv(f'{a.out}/examples.csv', index=False)
    for f in ('indeterminate', 'error_direction', 'flip_by_subset', 'examples'):
        print(f'== {f}')
        print(pd.read_csv(f'{a.out}/{f}.csv').round(1).to_string(index=False))
