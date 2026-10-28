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
> **Superseded for the surgery model by decision 11.** Surgery features now use all history up to `X` quarters *before* surgery. Enrollment cohorts still use the window below.

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

### 11. Surgery window inverted: all history up to a gap of `X` quarters before surgery
`X` used to be a lookback: the features were the `X` quarters ending the day before surgery. A model trained like that learns what a patient looks like in the months right before surgery. That flags patients who are already close to the operating table, when PT or WM can no longer help.

`X` is now a **gap**, `gap_quarters`:
- **Window:** features use **all history** (`start_date` = `HISTORY_START`, 1900-01-01) up to the day before `index_date − X quarters`. With X = 1, everything except the last quarter before surgery is used; with X = 2, everything except the last two quarters; and so on.
- **Matching:** negatives get matched index dates as before, but only dates more than `X` quarters after their first event, so their window has data.
- **Positives:** positives with no data before the gap are dropped. Otherwise an empty window would only ever mean "positive".
- **Holdout:** the holdout is scored on all history up to the extract end, with **no gap**. The model learned what patients look like at least `X` quarters before surgery, so applying it to today's data flags surgery about `X` or more quarters ahead.
- **Files:** splits files are named `patient_windows_gap{X}q_neg1x_seed0.csv`. `write_splits`, `build_splits`, `write_features` and the CLIs take `gap_quarters`. The pipeline default is `GAP_QUARTERS = 4` (one year of lead time). Enrollment models keep their 4-quarter lookback.

**Sweep** (XGBoost, surgery and consult codes ignored, val):

| gap X (quarters) | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|
| positives (= negatives) | 2,507 | 2,383 | 2,198 | 1,944 | 1,313 | 673 | 233 | 0 |
| val patients | 769 | 733 | 668 | 598 | 413 | 202 | 80 | – |
| val AUC | **0.657** | 0.601 | 0.579 | 0.581 | 0.561 | 0.457 | 0.473 | – |
| lift in top 10% | 1.45 | 1.31 | 1.35 | 1.32 | 0.94 | 1.06 | 0.83 | – |

- **Signal fades quickly with lead time.** Most of the separable signal sits in the last quarter or two before surgery.
- **Beyond 6 quarters there is no signal** (AUC ≤ 0.5), and the extract (about 3 years) leaves too few positives with that much history. At 12 quarters there are none.
- **For comparison:** the old window (the 4 quarters right before surgery) had val AUC 0.684. That score came largely from patients who were already months from surgery.

With X = 4, `main.py train` gives surgery val AUC 0.581 and test AUC 0.571, and `main.py score` lists 3,492 patients (holdout 17,747).

## Features

Implemented in [`assignment/features.py`](assignment/features.py) and tested in
[`tests/test_features.py`](tests/test_features.py).

### F1. Code counts inside the splits window (was: one-hot presence)
Each patient gets one column per code: the **number of rows** with that code on any date in the patient's `[start_date, end_date]` window from the splits file, inclusive, summed across claims and encounters. Columns are named `{code_system}_{code}` (`CPT_72148`, `ICD10CM_M54.50`, `NDC_65862003900`, `HCPCS_G0446`), so codes from different systems can't collide. `encoding="binary"` still gives the earlier 0/1 presence features. Recency is left out for v1.

Counts are raw rows, so one visit can count more than once: as a claim line and as an encounter, or as several claim lines. A count reflects how much activity a code generated, not how many separate visits it took.

Result: counts score about the same as presence. Val AUC differs by −0.005 to +0.006 across the 8 lookbacks, within noise. Counts are kept as the default because they carry strictly more information.

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
`features_<encoding>[_minmax]_<splits params>_<ignore tag>_mintrain<n>[_demo].csv`, for example `features_count_minmax_lookback4q_neg1x_seed0_excl-surgery_mintrain11_demo.csv`.
- The encoding is `count` or `onehot`.
- `_minmax` marks scaled features.
- The ignore tag is `excl-surgery` (default), `excl-none`, or `excl-<n>codes-<hash>` for a custom list.
- `_demo` marks that age and sex are included.

The script also writes `_vocabulary.csv`, `_scaling.csv`, `_coverage.csv` and `_summary.csv` next to the features. Training run folders take the same name with `features_` replaced by `logreg_`.

### F6. Coverage report
`feature_coverage()` gives, per feature, the percentage of patients with it in each of train/val/test/holdout, plus a `missing_in_train` flag for features seen elsewhere but never in train. `coverage_summary()` gives, per split, patients, features present, mean codes per patient, and the share of patients with no codes.

`features_present` in the summary counts the columns with at least one patient = 1 in that split. It is not the number of columns, which is the same for every split.

At a 4-quarter lookback, after the train vocabulary filter, there are 112 features. Coverage is similar across splits: about 17 codes per patient, and under 1% of patients with no codes. No feature is missing in train, and train prevalence ranges from 0.3% to 45%.

### F7. Min-max scaling fitted on train only
The code-count columns and `age` are scaled to `(x − min) / (max − min)`, using each column's min and max over **train rows only**. The same transform is then applied unchanged to val, test and holdout, so nothing about their distribution leaks into training. The sex columns are already 0/1 and are left alone. A column that is constant in train is only shifted.

Values outside train's range are not clipped, so val, test and holdout can fall below 0 or above 1. Clipping would hide real extremes; for example, the largest out-of-range value is 3.0, a patient with three times the train maximum count. The fitted min and max are saved as `<features file>_scaling.csv`, so scoring can apply them with `apply_minmax(features, read_scaling(path))` without refitting.

Coverage and summary reports are computed before scaling, from presence (value > 0).

### F8. Demographics: age and sex
`age = 2026 − birth_year` and one-hot `sex_F`, `sex_M`, `sex_U` come from the patients table and sit right after the ID columns. They are always kept, whatever the train vocabulary filter. A blank sex (143 patients) leaves all three sex columns 0. A patient missing from the patients table would get the table's median age; none are missing in this extract.

Adding them raised val AUC by 0.003–0.011 at every lookback, mostly through age: older patients have surgery less often. The sex weights are unstable.

**Implausible ages are replaced by the median birth year.** Any age over `MAX_PLAUSIBLE_AGE` (100), plus any missing birth year, gets the median *plausible* birth year of the patients table (1955, age 71). In practice that means the 63 patients with `birth_year = 1900`, a placeholder for unknown that gave age 126. Before this, those 63 set the top of the train-fitted min-max range (49–126), which squeezed every real patient into the bottom 61% of the scale. They were also treated as the oldest patients rather than as unknown. The age range is now 49–95/96.

Effect after retraining: val/test AUC moved by at most 0.008 (surgery test 0.721 → 0.720; WM val 0.668 → 0.660, test 0.666 → 0.668). The holdout list went from 3,407 to 3,378 patients. Test-set enrollment comparisons are unchanged to within 0.5 points. XGBoost only uses the order of values, so the squeeze itself didn't matter; the change only affects how the 63 unknown-age patients are treated.

**Still not handled:** 4,367 patients have `birth_year = 1961`, about 4 times the neighbouring years, with only about 210 a year after 1961. That looks like a default or cap. A `birth_year_1961` flag is a possible fix; ask Meridian.

### F9. Provider-specialty counts from claims
There is one column per `provider_specialty` listed in the data dictionary, all 15, named `specialty_<name>` (e.g. `specialty_orthopedic_surgery`). Each counts the patient's claim lines in the window from a provider of that specialty, whatever the code. Lines with no code still count, and lines with an ignored code (surgery) don't. The columns are min-max scaled on train like the code counts, and always kept regardless of the train vocabulary filter, so the column set is stable across extracts. Pharmacy (NDC) lines have no specialty and don't count. Six documented specialties never appear in this extract (Neurosurgery, Rheumatology, Anesthesiology, Cardiology, Nephrology, Endocrinology), so their columns are all zero. File names gain `_spec`; `--no-specialties` turns the columns off.

Result, XGBoost only (val AUC before → after):
- **Surgery:** +0.001 to +0.009 across the 8 lookbacks. The 12-quarter lookback is still best at 0.727, tied with 10 quarters.
- **PT enrollment:** 0.659 → 0.658.
- **WM enrollment:** 0.643 → 0.648.

Every change is well within val noise (about ±0.02). Specialty mostly restates what the codes already say. The exception is WM enrollment, where `specialty_physical_therapy` becomes the single most important feature. It packs the PT-engagement signal that was spread across several CPT codes into one column (see E8).

### F10. Surgical consult codes are forbidden in every model
99204 and 99244, the `Surgical Consult` codes, are never model inputs. They're listed in `features.FORBIDDEN_FEATURE_CODES` and removed by `build_features` whatever `ignore_codes` says, from code columns and from specialty counts alike. `train.py` refuses any features file that contains them, so older files can't be trained on by mistake. File tags reflect the effective set: `excl-surgery-consult`, `excl-surgery-consult-ptref` (PT), or `excl-consult`.

**Why:** they were the top two surgery features at every lookback. But a patient already seeing a surgeon is past the point where PT or WM outreach is likely to help, and a model leaning on consults flags patients late. Part J of the EDA shows outreach is already late: only 25% of eventual surgery patients were contacted before surgery, at a median of 141 days ahead.

**Effect** (XGBoost, val AUC with consults → without):
- **Surgery:**

  | lookback | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
  |---|---|---|---|---|---|---|---|---|
  | with consults | 0.699 | 0.711 | 0.723 | 0.722 | 0.721 | 0.722 | 0.727 | 0.727 |
  | without | 0.696 | 0.704 | **0.708** | 0.686 | 0.685 | 0.688 | 0.687 | 0.688 |

  Short lookbacks lose little (−0.004 to −0.015). Lookbacks of 4 quarters or more lose about 0.035–0.040, so 3 quarters is now the best lookback. The consult signal was concentrated in longer windows. The top features are now pharmacy NDCs, lumbar X-ray (72100) and PT activity (97110, 97001).
- **PT enrollment:** 0.636 → 0.625.
- **WM enrollment:** 0.648 → 0.668. Both are within noise.

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

### T5. XGBoost as a second model, fixed settings
`train(..., model="xgboost")` (or `--model xgboost`) fits gradient-boosted trees on the same features, with the same train/val handling and report. Settings are fixed, not tuned: 300 trees, depth 3, learning rate 0.05, 80% row and column subsampling, `min_child_weight` 5, seed 0, single thread for reproducibility. The report lists feature gain importance instead of coefficients. Min-max scaling has no effect on trees, so the same feature files are used.

`xgboost` is pinned to 3.0.5. Version 3.4.1 needs a newer OpenMP runtime than the `libomp` 18 installed here, and won't load on this Mac.

**Result** (count + min-max + demographics features, val):

| quarters | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|
| logreg val AUC | 0.679 | 0.700 | 0.712 | 0.717 | 0.695 | 0.694 | 0.690 | 0.692 |
| xgboost val AUC | 0.690 | 0.708 | 0.720 | 0.721 | 0.713 | 0.717 | 0.722 | 0.726 |

XGBoost is ahead at every lookback, and the gap grows with history length, from +0.004 at 4 quarters to +0.034 at 12. With logreg, longer windows got worse; with XGBoost they don't, which suggests the extra history holds non-linear signal that logreg can't use. It also overfits more: train AUC is about 0.84 against val 0.72, compared with 0.75 against 0.70 for logreg. Each single comparison is within the noise of the 796-patient val set (about ±0.02), but the direction is consistent across all 8 lookbacks. Both models rank the surgical consult codes (99204, 99244) highest.

### T6. `uv run main.py train`: the three models from the raw extract, plus a test evaluation
`main.py train` calls `assignment.pipeline.train_all(data/, artifacts/)`. It needs only the five extract files, and claims may be `.csv.gz` or `.csv`. This was verified by a run in an empty directory with just those files, with claims gzipped; it takes about 70 seconds on CPU. For each task it builds splits, features and an XGBoost model with the agreed defaults, then scores val and **test**:

| task | splits | ignored codes (plus consults, always) |
|---|---|---|
| surgery | `splits.py`, 4-quarter lookback | surgery codes |
| enroll-PT | `enrollment_splits.py`, no prior PT care | surgery + PT referral codes |
| enroll-WM | `enrollment_splits.py` | surgery codes |

The test split is read only in `assignment.evaluate`, once per trained model, after training is finished. It writes `metrics_test.json`, `calibration_test.csv` and `predictions_test.csv` into the model's run folder. `artifacts/manifest.json` records each task's splits, features, vocabulary, scaling and model paths, for the scoring step. `artifacts/training_summary.csv` holds the val and test metrics.

First test results, alongside val:

| task | val AUC | **test AUC** | test AP (base rate) | test lift in top 10% |
|---|---|---|---|---|
| surgery | 0.686 | **0.721** | 0.711 (0.49) | 1.73 |
| enroll-PT | 0.625 | **0.621** | 0.897 (0.83) | 1.15 |
| enroll-WM | 0.668 | **0.666** | 0.834 (0.72) | 1.23 |

Test agrees with val within noise. Surgery test is 0.035 higher than val, which is still within the ±0.02–0.03 uncertainty of ~750-patient sets. So there is no sign that choosing settings on val overfit them.

## Inference

Implemented in [`assignment/inference.py`](assignment/inference.py), run by `uv run main.py score`, tested in [`tests/test_inference.py`](tests/test_inference.py).

### I1. Holdout → surgery model → PT/WM models → rules → list ordered by surgery score
1. **Population:** the surgery splits' holdout, 17,295 patients. These are patients-table patients with active coverage, data in the last quarter and no surgery evidence. Index date is 2026-07-01, and the window is the 4 quarters before it.
2. **Surgery model** scores everyone. A patient is positive if the score is **≥ 0.5**, the natural cutoff for a model trained on a 50/50 sample.
3. **PT and WM models** score only the surgery positives. A patient is positive for a program if their score is **at least that model's training positive rate** (PT 0.82, WM 0.73), meaning more likely than the average training patient to enroll rather than refuse. The models were trained on mostly-enroller cohorts, so 0.5 would make almost everyone positive.
4. **Rules**, for anyone positive for PT and/or WM:
   - coverage active at the extract end;
   - no surgery evidence (any surgery code, or a `Spine Surgery` encounter, any status);
   - PT dropped if already enrolled in PT *or* with PT-specialty claims before the index date (prior PT care, which the PT model's cohort excludes, see E9);
   - WM dropped if already enrolled in WM.

   Patients left with no program are removed. Coverage and surgery are already guaranteed by the holdout definition but are checked again explicitly.
5. **Output:** `out/targets.csv` with `patient_token, program_recommended, score`. `program_recommended` is `PT`, `WM` or `PT+WM`; `score` is the surgery score. Rows are ordered by score, highest first. `out/targets_detail.csv` adds every model score, threshold, positive flag and rule outcome for every holdout patient, and `main.py score` prints the funnel.

Features are **rebuilt from the raw extract** for each model, using its saved vocabulary, ignored codes and train scaling, rather than read back from the training files. That way the same code can score a newer extract. A check confirms the rebuilt scores equal scores on the training-time holdout rows to within 3×10⁻⁸.

### I2. Feature files are written at full precision
Feature files used to be written rounded to 6 significant digits. The models were trained on those rounded values, while inference rebuilds unrounded values in memory. Values such as 1/3 then fell on the other side of XGBoost split points, and holdout scores differed by up to **0.26**. Files are now written at full precision, which leaves val/test metrics unchanged to 3 decimals. A test now requires the written features to equal the in-memory ones exactly.

### I3. First list (thresholds as above)
| step | patients |
|---|---|
| holdout scored | 17,295 |
| surgery positive (≥ 0.5) | 6,755 |
| PT positive / WM positive | 2,920 / 3,107 |
| PT or WM positive | 4,384 |
| PT removed: already enrolled in PT / prior PT care | 1,188 / 1,833 |
| WM removed: already enrolled in WM | 393 |
| **on the list** | **3,407**: WM only 2,343, PT only 693, both 371 |

The list leans heavily to WM, because the PT rules remove most PT positives: the people the PT model likes are often already in PT care. How far down the list to go is not yet decided. 6,755 surgery positives is far more than the real surgery rate (8.7%) implies, because 0.5 on a 50/50-trained model is not a calibrated real-world risk.

### I4. Scores prior-corrected to population rates
`run_inference` rescales each model's odds from its training positive rate to the rate observed in the extract (`inference.prior_correct`):
- **Surgery:** 2.9%, the patients-table patients with a first surgery code in the extract, per patient per year (`annual_surgery_rate`).
- **PT / WM:** 21.7% / 15.8%, the share of patients outreached for the program who enrolled within 90 days of the first attempt (`outreach_enrollment_rate`).

`score` in `targets.csv` is the corrected surgery risk. `targets_detail.csv` adds `surgery_risk`, `PT_enroll_prob` and `WM_enroll_prob`, and keeps the raw scores. The rates are recomputed from the data on every run and written to `targets_summary.csv`. The correction is monotonic, so thresholds (still applied to raw scores), flags and the ranking are unchanged.

It assumes the training negatives represent the population, which is only approximately true (history requirement and matched dates for surgery; explicit refusers for enrollment). A population backtest with Platt scaling would replace it.

## Program enrollment (PT, WM)

Implemented in [`assignment/labels.py`](assignment/labels.py), with `labels_file=` in `assignment/train.py`. Tested in [`tests/test_labels.py`](tests/test_labels.py) and `tests/test_train.py`.

### E1. Multi-label, modelled as two binary tasks
1,773 of the 10,042 enrolled patients (18%) enrolled in both PT and WM, and 67% of WM enrollees also did PT. So the outcome is multi-label, not multi-class. Both enrollments rarely happen together: the median gap is 193 days, and only 22 patients started both within 90 days.

We train one binary model per program on the same features rather than a single multi-label model. XGBoost's multi-label mode builds independent trees per label anyway, and separate tasks allow per-program exclusions, metrics and calibration. The program to recommend can then be chosen by comparing the two scores.

### E2. Same splits and features; tasks supply their own labels
Splits and features are unchanged. The same patients, splits, index dates, windows, feature vocabulary and train-fitted scaling are reused, and nothing in them depends on labels. `assignment.labels` writes a `labels_<task>_<splits params>.csv` (`patient_token, label`), and `train(..., labels_file=...)` swaps it in for the surgery label. Train/val patients without a label for that task are dropped and counted in the report under `excluded_unlabelled`. Test and holdout are still never read. Run names get the task prefix, e.g. `xgboost_enroll-PT_h4q_count_minmax_lookback4q_...`. Any future task can plug in the same way.

### E3. Label: new enrollment within `horizon_quarters` from the index date
- **Label 1:** an enrollment in the program dated in `[index_date, index_date + N quarters)`. The feature window ends the day before the index date, so features never overlap the outcome period.
- **Excluded (no label):**
  - **prior enrollees:** patients enrolled in *that* program at any time before the index date. The task is new enrollment, and their PT/WM session codes would give the answer away. A prior PT enrollment does not exclude a patient from the WM task.
  - **censored follow-up:** the horizon runs past the extract end (2026-06-30), so a 0 might be an enrollment we can't see yet.
  - **holdout:** no outcome to learn from.
- `horizon_quarters` defaults to 4.

| task (4-quarter lookback) | train labelled / positives | val labelled / positives | val positive rate |
|---|---|---|---|
| PT, 4 quarters | 2,068 / 352 | 441 / 82 | 18.6% |
| PT, 1 quarter | 2,830 / 141 | 597 / 42 | 7.0% |
| WM, 4 quarters | 2,463 / 133 | 533 / 38 | 7.1% |
| WM, 1 quarter | 3,363 / 53 | 721 / 9 | 1.2% |

**1-quarter horizons are too thin to use.** WM has 9 val positives, and PT and WM 1-quarter val AUCs are 0.53–0.67. The 4-quarter horizon is the working choice, even though the outreach decision is for next quarter.

### E4. First results (val, 4-quarter lookback, 4-quarter horizon)
| | logreg AUC | xgboost AUC | logreg lift in top 10% |
|---|---|---|---|
| PT | 0.715 | 0.690 | 2.7 |
| WM | 0.707 | 0.704 | 1.9 (xgboost 2.4) |

With these small labelled sets, XGBoost overfits badly (train AUC 0.88–0.92) and does no better than logreg, the reverse of the surgery task.

### E5. Known issues with reusing the surgery splits for enrollment
- **Post-surgery rehab.** Half the cohort's index date is their surgery date, so their outcome window is right after surgery. Surgery patients enroll in PT within 4 quarters at 22% against 13% for others, and in WM at 8.5% against 3.4%. Part of what the PT model learns is post-operative rehab, not conservative care before surgery, which is Meridian's goal. The surgical consult codes (99204, 99244) are among the top PT features.
- **Censoring removes recent index dates.** With a 4-quarter horizon, 1,211 PT and 1,460 WM patients are dropped for censoring, which shifts the labelled set toward earlier index dates.
- **Cross-program signal.** WM's top features are PT therapy codes (97530, 97140, 97110), and PT's include nutrition therapy (97802, 97803). That fits the overlap in E1: engagement in one program predicts the other.
- **Enrollment isn't outreach response.** 42% of enrolled patients were never contacted, so these models predict enrollment from any source, not the effect of outreach.

### E6. Reworked: an enrollment cohort per program (replaces E2–E5 for modelling)
The labels-on-surgery-splits approach (E2–E5) mostly predicted post-surgery enrollment, because the surgery-patient label windows started at surgery. It's kept in the code but superseded by a cohort built for enrollment: [`assignment/enrollment_splits.py`](assignment/enrollment_splits.py), tested in [`tests/test_enrollment_splits.py`](tests/test_enrollment_splits.py). It writes the standard splits format, so features and training run on it unchanged.

- **Positives:** patients whose **first** enrollment in the program came before their first surgery-code date, or who never had surgery. Index date = that first enrollment. Patients who first enrolled after surgery (413 PT, 217 WM) are left out.
- **Negatives:** patients contacted for that program whose attempt ended in a refusal (**Declined, Opted Out, Unsubscribed**), who never enrolled in it, and whose first refusal came before any surgery. Index date = first refusal. Other results such as voicemail or no answer don't count: we can't tell whether those patients ever got the message.
- **Window:** the `lookback_quarters` quarters before the index date, ending the day before, as everywhere else. Positives' features therefore stop before their first enrollment, so their own program's session codes can't leak in.
- **Splits and holdout:** the same token buckets and holdout rules as the surgery splits (shared `finish_splits()` in `splits.py`). Each program has its own file: `patient_windows_enroll-<PT|WM>_lookback<X>q.csv`.

### E7. Positives limited to the outreach period (`match_period=True`)
Negatives only exist while outreach ran (index dates 2024-07 to 2026-02), but enrollments span the whole extract. Without a limit, 31% of PT positives and 20% of WM positives had lookback windows cut off by the extract start, against 0% of negatives. The index date alone then separated the classes with AUC 0.67 (PT). Keeping positives to the negatives' index-date range removes the truncation, with 0% in both classes, and brings the date-only AUC down to 0.57 (PT) and 0.51 (WM). Val AUC without matching was higher (PT 0.683 against 0.655); part of that difference was the timing shortcut, so matched is the honest number. `match_period=False` is still available (files tagged `_allperiod`).

Some timing difference remains for PT (date-only AUC 0.57). Matching the date distributions exactly, as done for surgery negatives, would remove it.

### E8. Results (4-quarter lookback, matched period, val)
| | train pos / neg | val pos / neg | logreg val AUC | xgboost val AUC | val AP (base rate) | lift in top 10% |
|---|---|---|---|---|---|---|
| PT | 3,713 / 803 | 820 / 184 | 0.655 | 0.659 | 0.899 (0.817) | 1.2 |
| WM | 1,148 / 422 | 264 / 91 | 0.665 | 0.643 | 0.856 (0.744) | 1.2–1.3 |

- **The cohort is mostly positives** (74–82% in val), because refusals are rare. Average precision looks high only because of that base rate; lift in the top 10% is about 1.2. AUC is the fairer comparison.
- **Models:** logreg and XGBoost are similar. XGBoost overfits more (train AUC 0.80–0.90). Calibration is close (mean decile gap 0.03–0.04 for logreg).
- **The top features describe engagement across programs.** PT enrollment, as opposed to refusing PT, is predicted by WM-type activity: nutrition therapy 97802/97803, obesity counselling G0447, BMI Z68.3x, obesity E66.9. WM enrollment is predicted by PT therapy codes (97140, 97530). Older age raises the odds of enrolling in both.
- **Negatives come from outreach, positives from any source.** Positives include patients who enrolled without being contacted. The model separates "enrolls" from "refuses when asked", which is closer to the outreach decision than E3 was, but it's still not the effect of outreach.

### E9. PT cohort excludes patients already receiving physical therapy
`exclude_prior_specialties` in `enrollment_splits.py` drops anyone, holdout included, with a claim line from a listed provider specialty dated **before their index date**. The default is `("Physical Therapy",)` for PT and nothing for WM. Files are tagged, e.g. `patient_windows_enroll-PT_noprior-physical_therapy_lookback4q.csv`.

**Why:**
- All Physical Therapy-specialty claim lines carry the PT therapy codes (97530, 97140, 97110), so they look like PT program delivery.
- A patient with prior PT claims may already be in PT care. Meridian wouldn't market PT to them, and the model was partly learning "already in PT", which isn't a useful prediction.
- "Before the index date" is the right cut. Excluding PT claims at *any* time removes 87% of the cohort and nearly every positive, because enrollees' own sessions appear as PT claims after they enroll.

Prior PT is **not** excluded from WM, where it's real cross-program engagement, or from surgery, where it's a legitimate risk signal.

**Effect** (4-quarter lookback, XGBoost):

| | before | after |
|---|---|---|
| train pos / neg | 3,713 / 803 | 3,004 / 660 |
| val pos / neg | 820 / 184 | 653 / 161 |
| holdout | 14,595 | 8,813 |
| val AUC | 0.658 | 0.627 |
| val AP (base rate) | 0.899 (0.817) | 0.873 (0.802) |
| lift in top 10% | 1.18 | 1.17 |

- `specialty_physical_therapy` is now all zero in train and unused.
- The top PT features are now weight-management and general-engagement signals: obesity counselling G0447, age, nutrition therapy 97802/97803, psychotherapy 90834/90837, `specialty_other` and `specialty_behavioral_health`.
- Encounter-side PT *referral* codes (97110 at rank 47, 97001 at rank 59) remain. They come from the `PT Referral` event category and are a legitimate step before enrollment.
- The lower AUC is the more honest number: the model no longer gets credit for spotting people already in PT.
- **The PT holdout shrinks by 40%.** 5,782 otherwise-eligible patients have had PT claims and are no longer PT prospects. That's a business-relevant finding in itself.

### E10. PT referral codes also removed from the PT model's features
The PT model's features also ignore the PT referral codes **97001** (PT evaluation) and **97110** (therapeutic exercise). These appear only as `PT Referral` events in encounters; in claims, 97110 is billed only by PT providers. `DEFAULT_IGNORE_CODES["PT"]` in `enrollment_splits.py` is the surgery codes plus these two, and is passed as `ignore_codes` to `write_features`. The files are tagged `excl-surgery-ptref`, a named set in `features.NAMED_IGNORE_SETS`. WM keeps the surgery-only default.

With E9 excluding PT-specialty patients, **no PT code reaches the PT model at all**. 97140 and 97530 are billed only by PT providers, so they were already zero.

Effect (XGBoost, 4-quarter lookback): val AUC 0.627 → 0.636, AP 0.873 → 0.877, lift in top 10% 1.17 → 1.15. That's unchanged within noise: the referral codes carried little once prior PT patients were gone. The top features are age, obesity counselling (G0447), nutrition therapy (97802/97803), `specialty_other`, BMI Z68.32, `specialty_chiropractic`, an office visit (99215), psychotherapy (90837) and HbA1c (83036). It's a general-engagement and metabolic-health profile.

## Known limitations / open questions

- **No fixed prediction horizon.** Positives are labelled by surgery at the index date. Negatives have no surgery at any time in the extract, so negatives with early index dates have a longer surgery-free follow-up than those with late ones. A fixed horizon ("surgery within N quarters of the index date") would need positives and negatives redefined around it.
- **History requirement is asymmetric.** Negatives must have at least 2 years of data; positives have no minimum. Positives with very little history may look different for that reason alone.
- **Eligibility nudges the negative date distribution.** Negatives whose data ends early can only take early dates, so the match is close but not exact (see the table above).
- **Surgery codes disagree between sources.** Encounters has more surgery patients than claims (2,449 vs 1,729), and the union is used. If one source is judged authoritative, this should be revisited.
