# Benchmarking Synthetic Tabular Data Generation Methods

*Last updated: September 2026*
*Author: melike-Necim* — [https://github.com/melike-analyst]

## Summary

This study benchmarks three approaches to synthetic tabular data generation —
Gaussian Copula, CTGAN, and LLM-based few-shot generation — across two public
datasets (UCI Adult Income and Credit Card Fraud Detection), measured on
statistical fidelity, machine learning utility (Train-Synthetic-Test-Real),
and privacy risk (Distance to Closest Record). On the Adult dataset, CTGAN
achieved the best overall fidelity-utility trade-off (fidelity 0.868, utility
gap 0.037), while Gaussian Copula matched it on fidelity (0.842) but showed a
substantially larger utility gap (0.173). On the more complex, imbalanced
Credit Card dataset, both classical methods lost considerable fidelity
(0.643 and 0.609) yet still preserved near-real downstream classification
performance (utility gap under 0.01), suggesting fidelity and utility capture
different — and sometimes disconnected — aspects of synthetic data quality.
The privacy analysis surfaced a notable finding: the minimum DCR for Gaussian
Copula on the Adult dataset was exactly 0, indicating at least one synthetic
record was an exact duplicate of a real one — a risk that would have been
invisible if only the *mean* DCR had been reported. The LLM-based method
showed an even lower mean/min DCR, but this result is confounded by its
incomplete schema coverage (see Limitations) rather than being a genuine
privacy advantage.

## Motivation

Businesses increasingly rely on synthetic data to unlock data sharing,
software testing, and analytics without exposing sensitive real records. In
practice, this creates a concrete decision problem: **which generation method
should be used, for which dataset, and for which downstream purpose?**
Vendor claims about synthetic data quality are often difficult to verify
independently. This project builds a small, fully reproducible benchmark to
answer that question empirically — comparing a classical statistical method,
a GAN-based deep learning method, and a newer LLM-based approach — rather
than relying on marketing claims from any single tool provider.

## Methodology

**Datasets**
- UCI Adult Income (15 columns, mixed categorical/numerical, ~30K rows after cleaning)
- Credit Card Fraud Detection (31 columns, highly imbalanced target: ~0.17% fraud)
- Both datasets were split 70% train / 30% holdout, stratified on the target
  column, with a fixed random seed (`random_state=42`) for reproducibility.
  The holdout set was never exposed to any generation method — it was
  reserved exclusively for final evaluation to avoid data leakage.

**Methods compared**

| Method | Category | Library |
|---|---|---|
| Gaussian Copula | Classical statistical | SDV |
| CTGAN | GAN-based | SDV |
| LLM few-shot generation | API-based | Google Generative AI (Gemini) |

**Metrics**
- **Fidelity:** SDMetrics Quality Report score, computed on the columns
  common to both the real and synthetic data.
- **Utility (TSTR — Train on Synthetic, Test on Real):** A Random Forest
  classifier was trained on synthetic data and evaluated on the untouched
  holdout set (weighted F1-score), compared against a baseline model trained
  on real data and evaluated on the same holdout set.
  `utility_gap = real_baseline_F1 − synthetic_F1`; smaller is better.
- **Privacy (DCR — Distance to Closest Record):** For each synthetic row,
  the distance to its nearest real record (numeric columns, min-max scaled)
  was computed; both the mean and the minimum across all synthetic rows are
  reported, since a low mean can mask a small number of high-risk outliers.
- **Schema coverage:** The proportion of the real dataset's columns actually
  reproduced by a method's synthetic output.

## Results



## Discussion

**Fidelity and utility do not always move together.** On the Adult dataset,
CTGAN produced the best all-around result: highest fidelity (0.868) and the
smallest utility gap (0.037) — meaning a classifier trained on its synthetic
data came within 3.7 F1 points of a classifier trained on real data. Gaussian
Copula reached comparable fidelity (0.842) but its utility gap was nearly 5x
larger (0.173), showing that a high fidelity score does not guarantee that
downstream models will learn equally well from the data — likely because
Gaussian Copula, by modeling marginal distributions and a single correlation
structure, can distort the specific decision boundaries a classifier depends
on even while matching aggregate statistics reasonably well.

**High-dimensional, imbalanced data is harder to model faithfully — but the
signal a classifier needs can still survive.** On the Credit Card dataset,
both classical methods saw fidelity drop sharply (Gaussian Copula: 0.643,
CTGAN: 0.609) compared to Adult, consistent with the added difficulty of 31
columns and a target class present in only ~0.17% of rows. Despite this, the
utility gap on Credit Card was remarkably small for both methods (0.002 and
0.010 respectively) — synthetic data that only loosely matched the real
distribution overall still preserved enough of the fraud/non-fraud separating
signal for a downstream classifier to perform almost as well as one trained
on real data. This is a useful practical takeaway: fidelity metrics measured
across an entire feature space can understate how usable synthetic data is
for a specific, narrower modeling task.

**Aggregate privacy metrics can hide worst-case risk.** On the Adult dataset,
Gaussian Copula's *mean* DCR (0.091) looked reasonably safe, but its
*minimum* DCR was exactly 0.000 — meaning at least one synthetic record is an
exact duplicate of a real training record. CTGAN's minimum DCR on the same
dataset (0.000005) was almost as close. This shows why reporting only a mean
privacy score is insufficient in practice: a single memorized or near-copied
record is enough to constitute a genuine re-identification risk, and this can
coexist with a mean score that looks acceptable at a glance.

**The LLM method's apparent privacy advantage is very likely a measurement
artifact, not a genuine strength.** The LLM-generated data had the lowest DCR
of all methods on Adult (mean 0.000082, min 0.000000) — at first glance
suggesting either strong memorization risk or, counter-intuitively, unusually
tight clustering around real records. However, this metric was computed on
only 11 of 15 columns (73.3% schema coverage), because the generation prompt
omitted `fnlwgt`, `education.num`, `capital.gain`, and `capital.loss` — three
of which are high-variance numeric columns that would substantially increase
distances if included. Computing DCR on a smaller, mostly low-cardinality
categorical feature subspace mechanically compresses distances, since there
are fewer dimensions across which two rows can differ. This makes the LLM
method's fidelity and privacy scores directly comparable to Gaussian
Copula/CTGAN only with this caveat in mind, and it should not be read as
"the LLM approach is inherently more private."

**Practical implications.** No single method was best on every axis, which is
consistent with AIMultiple's own stated principle of not making blanket
purchase recommendations. For a use case where downstream classifier
performance matters most and the dataset is large and low-dimensional,
CTGAN's fidelity-utility trade-off on Adult looks the strongest here, at the
cost of considerably longer training time. Gaussian Copula is fast and simple
but this study's own DCR results are a concrete reminder to always check
minimum, not just mean, distance-to-real before treating its output as
privacy-safe. The LLM-based approach is attractive for quick prototyping with
minimal setup, but this study shows that prompt/schema completeness directly
determines both what the model measures and how its privacy profile should be
interpreted — a full-schema, larger-scale replication (and a real
membership-inference-style privacy test) would be needed before drawing
stronger conclusions about it.

## Limitations

- **Incomplete LLM generation schema:** the generation prompt covered only 11
  of the Adult dataset's 15 columns (`fnlwgt`, `education.num`, `capital.gain`,
  `capital.loss` were omitted). Fidelity and privacy metrics for this method
  were computed on the 11 common columns only (73.3% schema coverage), while
  Gaussian Copula and CTGAN were evaluated on the full 15-column schema. As
  discussed above, this likely inflates the LLM method's apparent privacy
  advantage and makes direct cross-method comparisons only partially fair.
- **Schema coverage was only explicitly logged for the LLM method.** Gaussian
  Copula and CTGAN are, by construction, fit directly on the real data's full
  metadata and therefore reproduce all columns; their schema coverage is
  treated as 100% in this report even though it is not recorded as a
  numeric field in `raw_results.csv`.
- **LLM generation was limited to 500 rows** for the Adult dataset (vs. ~22,000
  for Gaussian Copula/CTGAN) due to API rate and time constraints, and was not
  attempted at all for the larger, higher-dimensionality Credit Card dataset.
- **No generation time or API cost tracking** was performed in this iteration
  of the study.
- **Single run per method** — results reflect one execution per method/dataset
  pair, without repeated runs across multiple random seeds. A more rigorous
  version of this study would report mean ± standard deviation across 3+ runs
  to separate genuine differences from run-to-run variance.
- **Privacy evaluation is limited to DCR** on numeric columns; more rigorous
  audits (e.g., membership inference attacks, categorical-aware distance
  metrics) were out of scope for this study.
- **No SQL-based analysis** was performed in the final version of this study;
  all analysis was done in Python/pandas.

## Conflict of Interest Disclosure

The author has no financial or commercial relationship with any of the tools
evaluated in this study (SDV, Google Generative AI) beyond standard API usage
for research purposes. No vendor sponsored or reviewed this work prior to
publication.

## Reproducibility

All code, prompts, and raw results are available at:
https://github.com/melike-analyst/synthetic-data-benchmark

To reproduce: clone the repository, run `pip install -r requirements.txt`, add
your own API key to `.env` (see `.env.example`), and run the notebooks in
order (`01` → `06`).