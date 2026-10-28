"""Figures for the README, written to `figures/`.

Run from the repository root after `uv run main.py train` (the creative PCA uses
the trained PT/WM feature vocabularies and scaling):

    uv run python notebooks/readme_figures.py

Each figure repeats an EDA notebook view (`notebooks/meridian_eda.ipynb`, part in
brackets) with the same definitions:

    outreach_timing.png         first outreach attempt relative to first surgery [J]
    birth_year.png              patients by birth year [I]
    outreach_by_quarter.png     patients outreached / enrolled within 90 days per quarter [K]
    creative_enrollment.png     % enrolled within 90 days by campaign creative [M]
    creative_pca.png            PCA of model features, enrolled vs not, per creative [N]
"""

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from assignment.features import (
    SPECIALTIES,
    apply_minmax,
    build_features,
    load_code_events,
    load_demographics,
    read_scaling,
    read_vocabulary,
)
from assignment.inference import load_manifest
from assignment.splits import HOLDOUT, SURGERY_CODES, _window, claims_path

ROOT = Path(__file__).resolve().parents[1]
DATA, ARTIFACTS, FIGURES = ROOT / "data", ROOT / "artifacts", ROOT / "figures"

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.axisbelow": True, "grid.alpha": .25, "font.size": 9})
BLUE, ORANGE, GREEN, GRAY, INK = "#2a6fbb", "#d9822b", "#4f9d69", "#9aa0a6", "#333333"
PROGRAM_COLOR = {"PT": ORANGE, "WM": GREEN}
CREATIVES = {"PT": ["Back in Motion", "Stay in the Game", "Stand Tall"], "WM": ["Fresh Start", "More Moments"]}


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote figures/{name}")


def load_outreach() -> tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.read_csv(DATA / "meridian_health_outreach.csv", dtype={"patient_token": str}, parse_dates=["timestamp"])
    out["date"] = out.timestamp.dt.normalize()
    enr = pd.read_csv(DATA / "meridian_health_enrollments.csv", dtype={"patient_token": str}, parse_dates=["enrollment_date"])
    return out, enr


def enrolled_within(first: pd.DataFrame, enr: pd.DataFrame, keys: list[str], days: int = 90) -> pd.Series:
    """For rows of `first` (keys + `first_attempt`): enrolled in that program within `days` after the first attempt."""
    g = first.merge(enr[["patient_token", "program", "enrollment_date"]], on=["patient_token", "program"])
    g = g[(g.enrollment_date - g.first_attempt).dt.days.between(0, days)]
    hits = set(map(tuple, g[keys].to_numpy()))
    return pd.Series([tuple(k) in hits for k in first[keys].to_numpy()], index=first.index)


def first_surgery_dates() -> pd.Series:
    enc = pd.read_csv(DATA / "meridian_health_encounters.csv", usecols=["patient_token", "event_date", "event_code"],
                      dtype=str, parse_dates=["event_date"])
    clm = pd.read_csv(claims_path(DATA), usecols=["patient_token", "service_date", "code_system", "code"],
                      dtype=str, parse_dates=["service_date"])
    dates = pd.concat([
        enc.loc[enc.event_code.isin(SURGERY_CODES), ["patient_token", "event_date"]].set_axis(["patient_token", "date"], axis=1),
        clm.loc[(clm.code_system == "CPT") & clm.code.isin(SURGERY_CODES), ["patient_token", "service_date"]]
        .set_axis(["patient_token", "date"], axis=1),
    ])
    return dates.groupby("patient_token").date.min()


def outreach_timing(out: pd.DataFrame) -> None:
    first_surgery = first_surgery_dates()
    att = out.merge(first_surgery.rename("first_surgery"), left_on="patient_token", right_index=True)
    lead = (att.first_surgery - att.date).dt.days.groupby(att.patient_token).max()  # earliest attempt
    after, close, early = (lead <= 0).sum(), lead.between(1, 90).sum(), (lead > 90).sum()
    print(f"surgery patients contacted: {len(lead):,}; first contact on/after surgery {after} ({after / len(lead):.0%}), "
          f"1-90 d before {close} ({close / len(lead):.0%}), >90 d before {early} ({early / len(lead):.0%})")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    bins = np.arange(-719.5, 721, 30)  # edges at 0.5, 30.5, ... so day 0 and day 90 fall in the right group
    for mask, color, label in [(lead <= 0, GRAY, f"on/after surgery: {after:,} ({after / len(lead):.0%})"),
                               (lead.between(1, 90), ORANGE, f"1-90 days before: {close:,} ({close / len(lead):.0%})"),
                               (lead > 90, BLUE, f">90 days before: {early:,} ({early / len(lead):.0%})")]:
        ax.hist(lead[mask], bins=bins, color=color, edgecolor="white", linewidth=.6, label=label)
    ax.axvline(0, color=INK, lw=1, ls=":")
    ax.set(title=f"First outreach attempt relative to first surgery ({len(lead):,} surgery patients ever contacted)",
           xlabel="Days from first outreach attempt to first surgery (negative = contacted after surgery)",
           ylabel="Patients")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "outreach_timing.png")


def birth_year() -> None:
    by = pd.read_csv(DATA / "meridian_health_patients.csv").birth_year
    plausible_median = by[2026 - by <= 100].median()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.hist(by, bins=range(int(by.min()), int(by.max()) + 2), color=BLUE, edgecolor="white", linewidth=.3)
    for year, text in [(1900, f"1900: {(by == 1900).sum()} patients\n(placeholder, age 126)"),
                       (1961, f"1961: {(by == 1961).sum():,} patients\n(~4x 1958-60, then ~200/yr)")]:
        ax.annotate(text, (year + .5, (by == year).sum()), xytext=(8, 30 if year == 1900 else -4),
                    textcoords="offset points", fontsize=8, color=INK, va="bottom" if year == 1900 else "top")
    ax.axvline(plausible_median + .5, color=INK, lw=1, ls="--")
    ax.text(plausible_median, ax.get_ylim()[1] * .55, f"median plausible\nbirth year {plausible_median:.0f}  ",
            ha="right", fontsize=8, color=INK)
    ax.set(title=f"Patients by birth year ({len(by):,} patients, 1-year bins)", xlabel="Birth year", ylabel="Patients")
    ax.grid(axis="x", visible=False)
    save(fig, "birth_year.png")


def outreach_by_quarter(out: pd.DataFrame, enr: pd.DataFrame) -> None:
    o = out.assign(quarter=out.timestamp.dt.to_period("Q"))
    pq = o.groupby(["quarter", "patient_token", "program"]).date.min().rename("first_attempt").reset_index()
    pq["enrolled_90d"] = enrolled_within(pq, enr, ["quarter", "patient_token", "program"])
    quarters = pd.period_range(o.quarter.min(), o.quarter.max(), freq="Q")
    by_q = pd.DataFrame({
        "outreached": pq.groupby("quarter").patient_token.nunique(),
        "enrolled_90d": pq[pq.enrolled_90d].groupby("quarter").patient_token.nunique(),
    }).reindex(quarters, fill_value=0).fillna(0)
    camp = by_q[by_q.outreached > 0]
    rate = camp.enrolled_90d / camp.outreached * 100
    mean_rate = rate.mean()

    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(camp))
    w = .38
    ax.bar(x - w / 2, camp.outreached, width=w, color=GRAY, label="patients outreached")
    ax.bar(x + w / 2, camp.enrolled_90d, width=w, color=ORANGE, label="enrolled within 90 days of first attempt")
    for xi, n, e, r in zip(x, camp.outreached, camp.enrolled_90d, rate):
        ax.text(xi + w / 2, e + 40, f"{r:.1f}%", ha="center", fontsize=8, color=INK)
        ax.text(xi - w / 2, n + 40, f"{n:,.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, camp.index.astype(str))
    ax.set(title=f"Outreach campaigns by quarter (mean {mean_rate:.1f}% of outreached patients enrolled within 90 days)",
           ylabel="Patients")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "outreach_by_quarter.png")


def creative_table(out: pd.DataFrame, enr: pd.DataFrame) -> pd.DataFrame:
    """One row per patient x creative: first attempt with it, enrolled in its program within 90 days after."""
    om = out[out.message.notna()]
    pm = om.groupby(["patient_token", "program", "message"]).date.min().rename("first_attempt").reset_index()
    pm["enrolled_90d"] = enrolled_within(pm, enr, ["patient_token", "program", "message"])
    return pm


def creative_enrollment(pm: pd.DataFrame) -> None:
    stats = pm.groupby(["program", "message"]).enrolled_90d.agg(patients="size", rate="mean").reset_index()
    stats["ci"] = 1.96 * np.sqrt(stats.rate * (1 - stats.rate) / stats.patients) * 100
    stats["rate"] *= 100
    order = [m for p in CREATIVES for m in CREATIVES[p]]
    stats = stats.set_index("message").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(stats))[::-1]
    ax.barh(y, stats.rate, xerr=stats.ci, color=stats.program.map(PROGRAM_COLOR), height=.55,
            error_kw=dict(ecolor=INK, capsize=3, lw=1))
    for yi, r, ci, n in zip(y, stats.rate, stats.ci, stats.patients):
        ax.text(r + ci + .4, yi, f"{r:.1f}%  (n={n:,})", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [f"{m} ({p})" for m, p in zip(stats.message, stats.program)])
    ax.set(title="% of patients enrolled within 90 days of first attempt with each creative (95% interval)",
           xlabel="% enrolled within 90 days", xlim=(0, 30))
    ax.grid(axis="y", visible=False)
    save(fig, "creative_enrollment.png")


def program_features(program: str, rows: pd.DataFrame, events: pd.DataFrame, demographics: pd.DataFrame) -> pd.DataFrame:
    """The program model's scaled features for `rows` (patient_token, index_date), in the model's column order."""
    entry = load_manifest(ARTIFACTS)[f"enroll-{program}"]
    windows = _window(pd.Index(rows.patient_token), rows.set_index("patient_token").index_date, entry["lookback_quarters"])
    features = build_features(windows.assign(split=HOLDOUT, label=pd.NA), events, ignore_codes=entry["ignore_codes"],
                              vocabulary=read_vocabulary(entry["vocabulary_file"]), demographics=demographics,
                              specialties=SPECIALTIES)
    scaling = read_scaling(entry["scaling_file"])
    features = apply_minmax(features, scaling)
    with open(Path(entry["run_dir"]) / "model.pkl", "rb") as f:
        columns = pickle.load(f)["features"]
    # leave out the 0/1 sex columns (they split the cloud into bands) and columns constant in train,
    # which the model cannot use and which stay unscaled (e.g. PT claims for the PT model)
    constant = set(scaling.index[scaling["min"] == scaling["max"]])
    columns = [c for c in columns if not c.startswith("sex_") and c not in constant]
    return features.set_index("patient_token")[columns]


def creative_pca(pm: pd.DataFrame) -> None:
    """Per program: PCA of the program model's features over the 4 quarters before each patient's first
    attempt for that program; one panel per creative, enrolled within 90 days vs not."""
    events, demographics = load_code_events(DATA), load_demographics(DATA)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex="row", sharey="row")
    for r, (program, creatives) in enumerate(CREATIVES.items()):
        p = pm[pm.program == program]
        first = p.groupby("patient_token").first_attempt.min().rename("index_date").reset_index()
        X = program_features(program, first, events, demographics)
        pca = PCA(n_components=2, random_state=0)
        coords = pd.DataFrame(pca.fit_transform(X), index=X.index, columns=["pc1", "pc2"])
        var = pca.explained_variance_ratio_ * 100
        for c, creative in enumerate(creatives):
            ax = axes[r, c]
            d = p[p.message == creative].join(coords, on="patient_token")
            no, yes = d[~d.enrolled_90d], d[d.enrolled_90d]
            ax.scatter(no.pc1, no.pc2, s=4, color=GRAY, alpha=.35, linewidths=0, label=f"not enrolled ({len(no):,})")
            ax.scatter(yes.pc1, yes.pc2, s=4, color=PROGRAM_COLOR[program], alpha=.6, linewidths=0,
                       label=f"enrolled within 90 d ({len(yes):,})")
            for grp, color in [(no, INK), (yes, PROGRAM_COLOR[program])]:
                ax.scatter(grp.pc1.mean(), grp.pc2.mean(), s=70, marker="X", color=color, edgecolors="white", linewidths=1.2)
            ax.set(title=f"{creative} ({program})", xlabel=f"PC1 ({var[0]:.0f}% of variance)",
                   ylabel=f"PC2 ({var[1]:.0f}%)" if c == 0 else None)
            ax.legend(frameon=False, fontsize=7, loc="upper right", markerscale=3)
    axes[1, 2].axis("off")
    axes[1, 2].text(0, .5, "Each dot is a patient who received the creative.\n"
                           "Features: the program model's inputs over the\n4 quarters before the first attempt "
                           "for that program,\nscaled as in training, without sex and\ncolumns constant in train; one PCA per\nprogram. X = group mean.",
                    fontsize=8, color=INK, va="center")
    fig.suptitle("PCA of patient features: enrolled within 90 days vs not, by campaign creative", fontsize=11)
    save(fig, "creative_pca.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    out, enr = load_outreach()
    outreach_timing(out)
    birth_year()
    outreach_by_quarter(out, enr)
    pm = creative_table(out, enr)
    creative_enrollment(pm)
    creative_pca(pm)


if __name__ == "__main__":
    main()
