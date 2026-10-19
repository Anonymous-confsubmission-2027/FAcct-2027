FAMOUS: supplementary material

Requirements: Python 3 with numpy, pandas and scipy (analyses); ollama, ddgs and statsmodels
(inference notebook). All analyses run from the files in data/.


code/
  prepare_data.py      Data preparation. Builds the labelled FakeNewsNet table, assigns
                       prominence and credibility to publisher domains, removes duplicates and
                       articles from the sources of data/excluded_sources_articles.csv, draws the
                       150-article sample (seed 30), and writes the curated domain pools (final
                       exclusion list). It then compares the sample and the pools with
                       data/articles_150.csv and data/domain_pools.csv (see "Article sample").
                       Needs the third-party inputs listed at the end.
  run_inference.ipynb  Inference: prompts (English and French), evidence retrieval, the keyword
                       rule that maps responses to verdicts (extraire_verdict), the counterfactual
                       runs, the draw of the seven source sets (seed 42) and the recognition probe.
                       Run from the root of this folder; models are served with Ollama.
  run_history/         The two notebook versions with which the runs were made, kept as a record
                       (Kaggle paths replaced by data/ and out/, outputs removed; file names are
                       those used at the time). notebook_runs_2026-10-11.ipynb: the French runs and
                       source sets 1-3 of both English conditions; it stores verdicts only.
                       notebook_runs_2026-10-12_13.ipynb: identical except that it also stores the
                       raw responses; source sets 4-7 of both English conditions, the second
                       llama3.2:3b run and the rerun of 13 October. run_inference.ipynb is a cleaned
                       version of the latter. The prompts and the verdict rule are identical in all
                       three; the excerpt length was set by editing texte[:800] / texte[:2000] in
                       construire_prompt, and the French runs call construire_prompt_fr.
  build_runs.py        Rebuilds data/runs/EN800 from data/raw (see "Runs") and checks the result:
                         python code/build_runs.py --raw data/raw --out out/runs --check data/runs
  famous_pipeline.py   Main analyses for one condition: flip rates, accuracy, transitions,
                       class-conditional recall, logistic regression, presence tests,
                       two-way bootstrap contrasts, domain-level shifts, retrieval comparison.
  famous_extra.py      Analyses combining conditions or using the recognition coding:
                       retrieval by evidence stratum, refusals, Benjamini-Hochberg over the
                       112 domain-model shifts, recognition regression, agreement between
                       conditions, correlation of domain shifts.
  famous_appendix.py   Appendix analyses of the main condition: classification of indeterminate verdicts
                       from raw responses, verdicts that change correctness by true label and cell,
                       flip rates by FakeNewsNet subset, and the examples table:
                         python code/famous_appendix.py --runs data/runs/EN800 --raw data/raw \
                             --articles data/articles_150.csv --out out/appendix
  famous_revision.py   Dependence-aware analyses: source presence per cell with article-cluster
                       and crossed article/domain bootstraps (Holm-adjusted), baseline-relative
                       and pairwise disagreement, recognition regression with domain-clustered,
                       wild cluster and crossed bootstrap inference, and the comparison of
                       model-stated with external reliability.


data/
  articles_150.csv       The 150 FakeNewsNet articles used in all experiments: id, label
                         (real/fake), dossier_origine (subset_label), original publisher
                         (domaine, domaine_principal) with its prominence and credibility, the
                         cleaned title and text shown to the models (title, text) and the crawled
                         originals (title_original, text_original).
  evidence_150.csv       The retrieved evidence (9 October 2026): search query (the original
                         title), snippets (Preuves), result domains and retrieval date. For the 10
                         articles without results, it holds the French statement
                         "(aucune preuve trouvee apres plusieurs tentatives)".
  evidence_10_articles_en_placeholder.csv
                         The same 10 articles with the English statement "(no evidence found after
                         several attempts)", used in the rerun described under "Runs".
  excluded_sources_articles.csv
                         First exclusion list (80 sources: unidentifiable sources, platforms,
                         reference and official sites), created on 5 October 2026 and applied to
                         the articles before the sample was drawn.
  excluded_domains.csv   Final exclusion list (293 domains, an extension of the first), with the
                         reason (in French) and, where available, the number of FakeNewsNet
                         articles; applied to the domain pools.
  domain_pools.csv       The 1,873 curated domains with prominence (s0_connue = top 10,000 of
                         Tranco, s1_peu_connue = other) and credibility (credible, peu_credible,
                         mixte, non_note) labels. Cells A-D contain 101, 7, 125 and 22 domains.
  source_sets.csv        The seven source sets (one stand-in domain per cell).
  tirages_7.csv          The same seven source sets in the format read by run_inference.ipynb.
  evidence_strata.csv    Evidence stratum of each article (first matching rule).
  recognition_coding.csv Recognition probe descriptions (English) with the primary coding
                         (identified), the stricter second reading (identified_strict) and the
                         coding of the French probe (identified_fr).
  runs/EN800/   Main condition: English prompt, first 800 characters.
  runs/FR800/   French prompt, first 800 characters.
  runs/EN2000/  English prompt, first 2,000 characters.
  runs/EN800_larger/, runs/EN2000_larger/
                Supplementary runs of the English 800- and 2,000-character conditions with
                gemma3:12b and qwen2.5:14b (from 17 October 2026, Ollama 0.40.6, same notebook,
                source sets and evidence, with the English no-evidence statement for all 10
                articles; raw responses for all source sets). All were made in one Kaggle session
                except gemma3:12b at 2,000 characters, which ran over three sessions (session time
                limit) with the same model identifier and Ollama version; each file comes from a
                single session. Analysed with:
                  python code/famous_pipeline.py --runs data/runs/EN800_larger --out out/EN800_larger
                  python code/famous_pipeline.py --runs data/runs/EN2000_larger --out out/EN2000_larger
                These are the files analysed in the paper. One CSV per source set (tirage1..7),
                model and regime (WRAG = without retrieval, RAG = with retrieval). Columns: id
                (FakeNewsNet identifier), vraie_source (original publisher, never shown to the
                model), vrai_label, source_A..source_D (stand-in domains), verdict_A..verdict_D,
                verdict_S (no source); verdicts are real, fake or indetermine. Where the run kept
                them, reponse_A..reponse_S hold the raw model responses.
  raw/EN800_original/    The outputs of the English 800-character runs as produced on 11-12
                         October 2026, before the rerun below.
  raw/EN800_RAG_rerun_10_articles/
                         Rerun of 13 October 2026 (see "Runs").
  raw/llama3.2_3b_second_run/
                         A second run of llama3.2:3b on source sets 1-3, in both English
                         conditions and both regimes, with raw responses, made later in the same
                         Kaggle session (models not reloaded). It reproduces all 9,000 verdicts of
                         the original runs (raw/EN800_original and runs/EN2000).
  raw/recognition_probe/en, raw/recognition_probe/fr
                         Raw answers of the five models to the recognition probe, in English and
                         French.


Article sample
  Articles were first deduplicated (same URL, title and text) and filtered (non-empty title and
  text, title not equal to the publisher name, publisher not in
  data/excluded_sources_articles.csv). The sample was then drawn on 8 October 2026 with seed 30,
  balanced by subset and label (38 fake and 38 real GossipCop, 37 fake and 37 real PolitiFact
  articles), and the publisher's name was removed from titles and texts. The pools of stand-in
  domains were curated afterwards, on 11 October 2026, with the extended list
  data/excluded_domains.csv, which applies to the pools only. As a result, 20 sampled articles
  have a publisher that the final list excludes as a stand-in (for example platforms, network
  domains and PolitiFact itself); prepare_data.py lists them. The original publisher is never
  shown to the models.

  The draw depends on the order in which os.listdir returns the article folders, which can
  differ between file systems. On the machine on which it was drawn (macOS 26.7.1, Python
  3.14.3), the script reproduces the sample and the pools exactly; on Kaggle (Linux) it draws a
  different sample, while the pools do not depend on the order. prepare_data.py reports whether
  the sample matches, and data/articles_150.csv remains the reference sample.


Runs
  All runs use temperature 0 and the notebook in code/. The prompt texts are those of
  construire_prompt (English) and construire_prompt_fr (French); the 2,000-character condition
  differs only in the excerpt length (texte[:2000] instead of texte[:800]).

  For the 10 articles for which retrieval found no evidence, the evidence file holds a French
  statement, which the runs of 11-12 October inserted into English prompts too. On 13 October
  2026, these 10 articles were rerun in the English 800-character condition with retrieval, for
  gemma3:4b, llama3.2:3b, phi4-mini and qwen2.5:7b and all seven source sets, with the English
  statement (data/evidence_10_articles_en_placeholder.csv); nothing else changed. In
  data/runs/EN800 their verdicts replace those of the original runs. llama3.2:1b, the French
  condition and the 2,000-character condition keep the French statement. build_runs.py rebuilds
  data/runs/EN800 from data/raw and confirms that the released files contain exactly these
  changes.

  Raw responses were kept for source sets 4-7 of the English runs (all source sets for
  llama3.2:1b), for the rerun, the second llama3.2:3b run and the recognition probe; the French
  runs and source sets 1-3 of the other English runs kept verdicts only.


Analyses
  python code/famous_pipeline.py --runs data/runs/EN800 --out out/EN800 \
      --exclude-b ajc.com,aljazeera.com --strata data/evidence_strata.csv
  python code/famous_pipeline.py --runs data/runs/FR800 --out out/FR800
  python code/famous_pipeline.py --runs data/runs/EN2000 --out out/EN2000
  python code/famous_extra.py --runs data/runs/EN800 \
      --compare data/runs/FR800 data/runs/EN2000 \
      --compare-shifts out/FR800/domain_shifts.csv out/EN2000/domain_shifts.csv \
      --strata data/evidence_strata.csv --recognition data/recognition_coding.csv \
      --shifts out/EN800/domain_shifts.csv --out out/extra
  python code/famous_revision.py --runs data/runs/EN800 --runs-fr data/runs/FR800 \
      --recognition data/recognition_coding.csv --out out/revision
  python code/famous_pipeline.py --runs data/runs/EN800_larger --out out/EN800_larger
  python code/famous_pipeline.py --runs data/runs/EN2000_larger --out out/EN2000_larger

The prompts are reproduced verbatim in Appendix A of the paper.


Third-party inputs, not redistributed
  FakeNewsNet (the crawled folders gossipcop_fake, gossipcop_real, politifact_fake and
  politifact_real, downloaded on 6 September 2026; only news_article.json is used, and the
  social-context files are neither used nor redistributed), the Tranco list K9QYW (6 September
  2026) and the news-media reliability labels of 28 October 2024. They are needed only to rerun
  prepare_data.py; all analyses run from the files in data/.


Generative AI usage
  During the preparation of this work, the authors used ChatGPT (OpenAI) and Claude (Anthropic) to assist with language editing and phrasing, as well as with code documentation, debugging, and implementation troubleshooting. All methodological choices, audit design, parameterisation, evaluation criteria, and interpretation of results were determined by the authors.
