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

**Known data issues, not yet handled:** 63 patients have `birth_year = 1900`, which looks like a placeholder for unknown (age 126). 4,367 patients have `birth_year = 1961`, about 4 times the neighbouring years, with only about 210 a year after 1961. That looks like a default or cap. The proposed fix is to treat 1900 as missing with an `age_missing` flag, and add a `birth_year_1961` flag.

### F9. Provider-specialty counts from claims
There is one column per `provider_specialty` listed in the data dictionary, all 15, named `specialty_<name>` (e.g. `specialty_orthopedic_surgery`). Each counts the patient's claim lines in the window from a provider of that specialty, whatever the code. Lines with no code still count, and lines with an ignored code (surgery) don't. The columns are min-max scaled on train like the code counts, and always kept regardless of the train vocabulary filter, so the column set is stable across extracts. Pharmacy (NDC) lines have no specialty and don't count. Six documented specialties never appear in this extract (Neurosurgery, Rheumatology, Anesthesiology, Cardiology, Nephrology, Endocrinology), so their columns are all zero. File names gain `_spec`; `--no-specialties` turns the columns off.

Result, XGBoost only (val AUC before → after):
- **Surgery:** +0.001 to +0.009 across the 8 lookbacks. The 12-quarter lookback is still best at 0.727, tied with 10 quarters.
- **PT enrollment:** 0.659 → 0.658.
- **WM enrollment:** 0.643 → 0.648.

Every change is well within val noise (about ±0.02). Specialty mostly restates what the codes already say. The exception is WM enrollment, where `specialty_physical_therapy` becomes the single most important feature. It packs the PT-engagement signal that was spread across several CPT codes into one column (see E8).

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

## Known limitations / open questions

- **No fixed prediction horizon.** Positives are labelled by surgery at the index date. Negatives have no surgery at any time in the extract, so negatives with early index dates have a longer surgery-free follow-up than those with late ones. A fixed horizon ("surgery within N quarters of the index date") would need positives and negatives redefined around it.
- **History requirement is asymmetric.** Negatives must have at least 2 years of data; positives have no minimum. Positives with very little history may look different for that reason alone.
- **Eligibility nudges the negative date distribution.** Negatives whose data ends early can only take early dates, so the match is close but not exact (see the table above).
- **Surgery codes disagree between sources.** Encounters has more surgery patients than claims (2,449 vs 1,729), and the union is used. If one source is judged authoritative, this should be revisited.
