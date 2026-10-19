"""Rebuild the main-condition verdict files (data/runs/EN800/) from the raw run outputs.

  python code/build_runs.py --raw data/raw --out out/runs [--check data/runs]

data/raw/EN800_original            Outputs of run_inference.ipynb for the English 800-character
                                   condition (11-12 October 2026), one CSV per source set, model and
                                   regime. The files of data/runs/EN2000 and data/runs/FR800 are the
                                   notebook outputs unchanged and are not duplicated here.
data/raw/EN800_RAG_rerun_10_articles
                                   Rerun of 13 October 2026: the 10 articles for which retrieval
                                   found no evidence, in the English 800-character condition with
                                   retrieval, for gemma3:4b, llama3.2:3b, phi4-mini and qwen2.5:7b,
                                   with the no-evidence statement in English
                                   (data/evidence_10_articles_en_placeholder.csv) instead of French.
                                   Everything else was unchanged.

The analysed files are these outputs, except that in EN800 with retrieval the verdicts of
these 10 articles for these four models are taken from the rerun. llama3.2:1b was not rerun.
With --check, the rebuilt files are compared with the released ones.
"""
import argparse
import glob
import os
import re

import pandas as pd

NAME_RE = re.compile(r'tirage(\d+)_(.+?)_(RAG|WRAG)\.csv$')
VERDICTS = ['verdict_A', 'verdict_B', 'verdict_C', 'verdict_D', 'verdict_S']


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def index_runs(folder):
    runs = {}
    for f in glob.glob(os.path.join(folder, '*.csv')):
        m = NAME_RE.search(os.path.basename(f))
        if m:
            runs[(int(m[1]), m[2], m[3])] = f
    return runs


def build(raw, out):
    rerun = index_runs(os.path.join(raw, 'EN800_RAG_rerun_10_articles'))
    for cond in ('EN800',):
        os.makedirs(os.path.join(out, cond), exist_ok=True)
        for (s, model, regime), f in sorted(index_runs(os.path.join(raw, 'EN800_original')).items()):
            d = read(f).set_index('id')
            if cond == 'EN800' and (s, model, regime) in rerun:
                r = read(rerun[(s, model, regime)]).set_index('id')
                assert (r[['vraie_source', 'vrai_label']] == d.loc[r.index, ['vraie_source', 'vrai_label']]).all().all()
                # verdicts (and raw responses, where the run kept them) come from the rerun
                cols = VERDICTS + [c for c in d.columns if c.startswith('reponse_') and c in r.columns]
                d.loc[r.index, cols] = r[cols]
            d.reset_index().to_csv(os.path.join(out, cond, f'tirage{s}_{model}_{regime}.csv'), index=False)


def check(out, ref):
    n, bad = 0, []
    for cond in ('EN800',):
        for f in sorted(glob.glob(os.path.join(ref, cond, '*.csv'))):
            a = read(f)
            b = read(os.path.join(out, cond, os.path.basename(f)))
            n += 1
            if not a.equals(b):
                bad.append(f)
    print(f'{n} files compared, {len(bad)} differ' + (':' if bad else ''))
    for f in bad:
        print('  ', f)
    return not bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--out', default='out/runs')
    ap.add_argument('--check')
    a = ap.parse_args()
    build(a.raw, a.out)
    if a.check:
        check(a.out, a.check)
