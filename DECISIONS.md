# Decisions

Design decisions for the modeling cohort and splits, implemented in
[`assignment/splits.py`](assignment/splits.py) and tested in
[`tests/test_splits.py`](tests/test_splits.py). Each entry says what we chose and why,
so it can be revisited.

## Splits design (as specified)

- **Positives:** patients with any of the 4 lumbar surgery CPT codes (22630, 22633, 63030, 63047).
- **Data window:** all data in the `X` quarters before surgery. `X` (`lookback_quarters`) is a parameter, so it can be varied.
- **Negatives:** patients with at least 2 years of data who never received or scheduled back surgery. They are sampled at random in proportion to the positives.
- **Train/val/test:** 70/15/15, assigned deterministically from the patient token.
- **Holdout for real inference:** patients-table patients who have not had surgery, with active coverage and data from at least the past quarter.
- **Sources:** train/val/test may draw on patients from the claims feed and the patients table. The holdout comes from the patients table only.

## Decisions

### 1. Split buckets: 0–69 train, 70–84 val, 85–99 test
The original ranges "00–70 / 71–85 / 86–100" cover 101 values and give 71/15/15. We use 0–69 / 70–84 / 85–99 to get exactly 70/15/15.

Tokens are 32-character hex strings. The "last 2 digits" are the last two decimal digits of the token read as a number: `int(token, 16) % 100`. Taking the last two hex characters mod 100 would be uneven, because 256 values don't divide by 100 and buckets 0–55 would get 3 values each against 2 for the rest. The split depends only on the token, so a patient always lands in the same split whatever the lookback, ratio or seed.

### 2. Positives: any surgery code, any status, either source
A patient is positive if any of the 4 codes appears in encounters (`event_code`) or claims (CPT `code`), with any status: performed, scheduled, cancelled, no-show, missing, or any claim status. The spec defines negatives as never having received *or even scheduled* surgery, so anyone with a surgery code is treated as a surgical case. With the current extract that gives 2,597 positives.

`SURGERY_CODES` is a constant, so restricting positives to performed surgeries later is a one-line change.

### 3. Positive index date = earliest surgery-code date across both sources
EDA showed that claims surgery lines are usually dated about 1 day after the matching encounters surgery. Taking the earliest date keeps the surgery itself out of the window.

### 4. The window ends the day before the index date
Each row has `start_date = index_date − X quarters` and `end_date = index_date − 1 day`, both inclusive. Nothing on or after the index date (the surgery, or same-day pre-op activity) can enter the features.

### 5. Ambiguous surgery evidence is excluded everywhere
A `Spine Surgery` encounter with a missing code is evidence of surgery but can't be confirmed as one of the 4 codes. Patients whose only surgery evidence is such a row (26 patients) are neither positive, negative nor holdout.

### 6. Negatives: no surgery evidence at any time, ≥ 2 years of observed data
A negative has no surgery code and no `Spine Surgery` encounter of any status, in either source, at any point in the extract. That includes after their index date, so no negative is a surgery that hadn't happened yet.

"2 years of data" means the span from the patient's first to last dated event across encounters and claims is at least 2 years (`min_history_years`).

### 7. Negative index dates are matched to the positives' index dates
Each sampled negative gets an index date drawn at random from the positives' index dates. The draw is limited to dates after the negative's first event and no later than the day after their last event, so the window sits inside the negative's own data. Candidates with no such date are not eligible.

**Why:** a negative has no natural index date. Anchoring negatives at their last event would put them all at the end of the extract while positives spread across it. The classes would then differ in calendar time and in how often the lookback window runs off the start of the extract (about 21% of positive windows start before 2023-07-01 at a 4-quarter lookback). A model could learn "short or truncated history means surgery." Matched index dates give both classes the same timing and the same truncation rate, which we verified on the real extract:

| | positives | negatives |
|---|---|---|
| windows starting before the extract (4q lookback) | 20.8% | 21.2% |
| index dates by quarter | matched within about 1–2 percentage points per quarter |

### 8. Negative ratio defaults to 1:1
We read "proportionally" as one negative per positive. `neg_ratio` changes this. Sampling uses a fixed `seed` (default 0), so results are reproducible.

### 9. Holdout: reachable patients-table patients with no surgery evidence
The holdout is every patients-table patient who meets all of these:
- no surgery evidence (as defined for negatives);
- not sampled as a negative or positive, so the holdout never shares a patient with train/val/test, and claims-only patients are never in it;
- **active coverage at the extract end:** `coverage_start` ≤ 2026-06-30 and `coverage_end` blank or ≥ 2026-06-30. In this extract the latest `coverage_end` is 2026-06-29, so active means a blank `coverage_end`;
- **recent data:** at least one encounter or claim in the last quarter, 2026-04-01 to 2026-06-30. This is `holdout_recent_quarters` (default 1).

**Why:** the holdout is the population we would actually score for next quarter's outreach. Patients whose coverage has ended are no longer Meridian's to manage. Patients with no recent activity have stale features and may have left the system without their coverage being updated.

Its index date is the day after the extract end (2026-07-01), so its window is the last `X` quarters of data. Holdout patients with less than 2 years of history are kept.

Candidates that fail the coverage or recency filter are dropped from the output entirely; they are not moved into another split. In the current extract, 6,564 are dropped for ended coverage and 921 for having no data since 2026-04-01. That leaves 17,295 holdout patients, down from 24,780.

### 10. Output file
`write_splits(lookback_quarters, neg_ratio=1.0, seed=0)` writes
`artifacts/splits/patient_windows_lookback{X}q_neg{ratio}x_seed{seed}.csv` with one row per patient:
`patient_token, split, label, index_date, start_date, end_date`. `label` is 1 for positives, 0 for negatives, and blank for the holdout.

## Features

Implemented in [`assignment/features.py`](assignment/features.py) and tested in
[`tests/test_features.py`](tests/test_features.py).

### F1. One-hot code presence inside the splits window
Each patient gets one 0/1 column per code, set to 1 if the code appears on any date in that patient's `[start_date, end_date]` window from the splits file, inclusive. Columns are named `{code_system}_{code}` (`CPT_72148`, `ICD10CM_M54.50`, `NDC_65862003900`, `HCPCS_G0446`), so codes from different systems can't collide. Counts and recency are deliberately left out for v1.

### F2. Both sources, any status
Codes come from claims (ICD10CM, CPT, HCPCS, NDC) and encounters (ICD10CM, CPT). A code present in either source sets the same column. All claim statuses (paid, denied and so on) and all encounter statuses count: a denied claim or a cancelled event still shows clinical intent. `sources` can restrict this to one source. Rows with a missing code are dropped.

### F3. Ignored codes default to the surgery codes
`ignore_codes` removes codes before encoding. The default is the 4 surgery CPT codes, because they define the label. The window already ends the day before the index date, so this mainly guards against any earlier surgery code. The list can be extended.

### F4. One row per patient in the splits file
Every patient in the splits file gets a row, even with no vocabulary codes in their window (all zeros). Patients not in the splits file are ignored. Every split (train, val, test, holdout) has exactly the same columns in the same order.

### F4a. Vocabulary chosen on train only: codes more than 10 train patients have
A code becomes a feature only if **more than 10 train patients** have it in their window (`min_train_patients=11`). Val, test and holdout get exactly those columns, and any other code they have is dropped.

**Why:** a column that is all zero, or nearly so, in train can't be learned and only adds noise. Choosing columns on train alone also keeps val and test truly unseen. Future extracts and new clients will bring new codes, and they can't change the columns of a trained model.

The vocabulary is written next to the features as `<features file>_vocabulary.csv` (`feature, train_patients`). Scoring can reuse it with `build_features(..., vocabulary=read_vocabulary(path))`, which gives the same columns and fills codes the scoring data lacks with 0.

At a 4-quarter lookback this keeps 112 of the 114 codes seen. It drops `ICD10CM_Z68.42` (0 train patients, 2 holdout) and `ICD10CM_Z68.20` (under 11 train patients). The rarest kept feature is `ICD10CM_Z68.39`, with 11 train patients.

### F5. File names follow the splits file
`features_onehot_<splits params>_<ignore tag>_mintrain<n>.csv`, for example `features_onehot_lookback4q_neg1x_seed0_excl-surgery_mintrain11.csv`. The ignore tag is `excl-surgery` (default), `excl-none`, or `excl-<n>codes-<hash>` for a custom list. The script also writes `<features file>_coverage.csv` and `<features file>_vocabulary.csv`.

### F6. Coverage report
`feature_coverage()` gives, per feature, the percentage of patients with it in each of train/val/test/holdout, plus a `missing_in_train` flag for features seen elsewhere but never in train. `coverage_summary()` gives, per split, patients, features present, mean codes per patient, and the share of patients with no codes.

`features_present` in the summary counts the columns with at least one patient = 1 in that split. It is not the number of columns, which is the same for every split.

At a 4-quarter lookback, after the train vocabulary filter, there are 112 features. Coverage is similar across splits: about 17 codes per patient, and under 1% of patients with no codes. No feature is missing in train, and train prevalence ranges from 0.3% to 45%.

## Training

Implemented in [`assignment/train.py`](assignment/train.py) and tested in
[`tests/test_train.py`](tests/test_train.py).

### T1. Train on train, evaluate on val, never read test or holdout
`read_features()` drops test and holdout rows as soon as the file is read, so no later step can touch them. A test checks this by filling test and holdout rows with nonsense values and no labels and confirming the model and metrics come out identical. Test is saved for a single final evaluation, and the holdout is only for scoring.

### T2. Model: L2-regularized logistic regression, `C = 1.0`, not tuned
The assignment asks for a simple, well-calibrated risk model and says tuning isn't rewarded. Logistic regression on 0/1 code features is interpretable (one weight per code, written to `coefficients.csv`) and usually close to calibrated without extra steps. `C` is a parameter but is fixed at the sklearn default; val is used for reporting, not for choosing it.

### T3. Validation report
`artifacts/models/logreg_<features params>/metrics_val.json` has:
- ROC AUC, average precision, Brier score and log loss on val;
- precision, recall and lift in the top 5%, 10% and 20% of val by score, which is how an outreach list would be used;
- train AUC, only as an overfitting check.

`calibration_val.csv` compares mean predicted and observed surgery rate by decile of val score. The run also writes `model.pkl` (model plus its feature list) and `coefficients.csv`.

First run (4-quarter lookback, `mintrain11`): val ROC AUC 0.708, average precision 0.735, Brier 0.217. Precision in the top 10% is 0.875, a lift of 1.7 over val's 51% positive rate. Train AUC is 0.759, so there is mild overfitting.

### T4. Predicted probabilities are for the 1:1 sampled population
Train and val are about 50% positive because negatives are sampled 1:1 with positives. In the real population surgery is far rarer, so the model's probabilities are far too high in absolute terms, even though the ranking is unaffected. Before using scores as real risk, they need a prior correction for the true prevalence, or recalibration on a sample with the true base rate.

## Known limitations / open questions

- **No fixed prediction horizon.** Positives are labelled by surgery at the index date. Negatives have no surgery at any time in the extract, so negatives with early index dates have a longer surgery-free follow-up than those with late ones. A fixed horizon ("surgery within N quarters of the index date") would need positives and negatives redefined around it.
- **History requirement is asymmetric.** Negatives must have at least 2 years of data; positives have no minimum. Positives with very little history may look different for that reason alone.
- **Eligibility nudges the negative date distribution.** Negatives whose data ends early can only take early dates, so the match is close but not exact (see the table above).
- **Surgery codes disagree between sources.** Encounters has more surgery patients than claims (2,449 vs 1,729), and the union is used. If one source is judged authoritative, this should be revisited.
