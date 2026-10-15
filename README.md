# Terrorism, GDP growth and FDI in developed and developing countries

The code behind the seminar's action plan and its three definitional files:

- **File 1** – what counts as a market, and rules **Q1–Q6** for admitting one
- **File 2** – developed vs developing, criteria **C1–C3** and the borderline flags
- **File 3** – which terror events qualify (**K1–K6** + magnitude floor) and the
  **Terrorism Severity Index (TSI)**

It builds the country-year panel for 1998–2020 and runs the analysis of plan
§7: four regressions, each for GDP growth (H1) and the FDI ratio (H2).

---

## Running it

Press the green arrow on each file in PyCharm, in order.

| File | Plan | What it does | When |
|---|---|---|---|
| `0_fetch_data.py` | Part 5 | downloads WDI, UNDP HDI, World Bank income-group history, UCDP conflicts, GeoNames cities | once (re-run to refresh) |
| `1_build_terror_cache.py` | — | converts the 104 MB GTD xlsx into a fast cache | once per GTD file |
| `2_build_sample.py` | Part 5, **Stage 1** | K1–K6 + floor → Q1–Q6 → C1–C3 → market-year aggregates; the Stage 1 exports | after any change to `config.py` |
| `3_run_analysis.py` | Part 5, **Stage 2** | TSI, diagnostics, the 4 × 2 regressions, §8 tables, Figures 13A–24, report | whenever you want results |

Open **`outputs/report.html`** for everything on one page, in the plan's order.

To prove the regressions work before trusting them:

```bash
python 3_run_analysis.py --sample
```

This keeps the real panel (markets, years, groups, TSI, controls) but replaces
the outcomes with synthetic ones carrying a **known** TSI effect, then checks
that the four regressions recover it — including b3, the H1/H2 test — and find
nothing in a placebo. It writes to `outputs/sample_check/`, never over the real
results.

Tests (no framework needed; each prints PASS/FAIL):

```bash
python -m tests.test_definitions
```

```bash
python -m tests.test_io_tabular
```

`test_definitions` checks every rule of Files 1–3 and §7 on hand-built cases:
K1–K6 and the floor, the PD weights, blank-is-not-zero, the TSI's pooled
standardisation, Q1–Q6, C1–C3 and the four borderline triggers, recovery of a
planted interaction, and the diagnostic tests.

---

## Where each definition lives

| Definition | Code |
|---|---|
| Window 1998–2020, base year 1998 (Part 0) | `config.py` |
| Q1–Q6 market qualification (File 1 §4) | `seminar/qualification.py` |
| C1–C3 classification, borderline triggers (File 2) | `seminar/classification.py` |
| K1–K6, magnitude floor, severity traits, §5.8 flags (File 3 Part A) | `seminar/terror.py` |
| TSI steps 1–4 (File 3 Part B, plan §5.6) | `seminar/terror.py` (step 1), `seminar/tsi.py` (steps 2–4) |
| Outcomes and controls (§6.2, §6.4) | `config.py` → `seminar/panel.py` |
| Diagnostics and the standard-error choice (§7.4) | `seminar/diagnostics.py` |
| The four regressions, build-up, decomposition, hypothesis tests (§7, §8) | `seminar/regression.py` |
| Figures of the master list (Part 3) | `seminar/viz.py` |

**What is fixed and what is a setting.** Rules Files 1–3 state exactly — K2 =
all three criteria, K3 = `doubtterr = 0`, the floor, the PD weights 3/2/1/1, the
TSI formula, HDI ≥ 0.800 with its 0.02 band — are coded next to their source
and are *not* switches. `config.py` holds only the choices the Files leave
open, each of which the methodology chapter has to state:

| Setting | Why it is open |
|---|---|
| `CANDIDATE_COUNTRIES` | Figure 6 starts from all GTD countries (the default); a published list can be used instead |
| `NON_SOVEREIGN`, `TERRITORIAL_CHANGES` | how Q1's "sovereign … consistently named … over the whole window" is applied |
| `Q3_MIN_TRADE_RATIO`, `Q3_MIN_FDI_RATIO` | File 1 does not quantify "non-trivial" |
| `Q5_EXCLUSIONS` | war / contested statehood is a judgement; seeded with Saeed et al.'s precedent |
| `Q6_MIN_ACTIVE_YEARS` | File 1 does not say how few events is "degenerate" |
| `C2_SOURCE`, `C2_HIGH_INCOME_THRESHOLD` | which vintage of the World Bank grouping C2 uses |
| `INDEX_PROVIDER_DISAGREEMENTS` | borderline trigger 4 (South Korea, per File 2) |
| `CONTROLS` | the §6.4 vector, with each control's precedent |
| `SE_TYPE`, `COMPONENT_ENTRY`, `ROBUSTNESS_EXCLUDE_BORDERLINE` | §7.4, §8.7 and §4.5 choices |

---

## What comes out

```
outputs/stage1/        what Part 5 says to carry into Stage 2
  gtd_qualifying_events_1998_2020.csv   event level, sample markets only (audit)
  gtd_country_year_1998_2020.csv        NA, NK, NI, PD + flag counts per market-year
  sample_markets.csv                    the classified sample (§4.7)
outputs/tables/        CSVs named by plan section: t3_5_exclusion_log.csv is the
                       §3.5 exclusion table, appD_* / appE_* are the appendices
outputs/figures/       fig01_... to fig24_..., numbered as in the master list
outputs/report.html    everything on one page
```

Stage 1 prints the row count at every filter stage — screenshot them for the
provenance record Part 5 asks for. `appE_spot_checks.csv` lists market-years
small enough to verify by hand against the raw GTD, with their event ids.

**How to read the coefficients** (§5.6): the TSI is standardised on this
sample, so every coefficient is the change in the outcome associated with a
**one-standard-deviation** rise in terrorism severity — not one more attack,
and not comparable with studies using counts or dummies.

**Coefficient labels** follow §7.2–§7.3: b1 = TSI(t), b2 = TSI(t−1),
b3 = TSI(t) × Developed (the formal test of H1 and H2), b4 = TSI(t−1) ×
Developed. (Part 0.4 and the glossary still call the interaction "β₂".)

---

## Things to know before writing them up

- **The course rubric.** Part 5 and risk 8 say the figures must be produced in
  the AI conversation. The figures here are a reference and a cross-check;
  upload the Stage 1 exports and rebuild them there.
- **The sample is larger than the plan's "23 markets".** Applying Q1–Q6 to every
  GTD country gives the number printed by `2_build_sample.py` (77 with the
  current settings). To reproduce a fixed list, set `CANDIDATE_COUNTRIES`.
- **The window cannot start before 1998** (File 3 §3.2), and 1998 itself drops
  from the regressions because TSI(t−1) for 1998 would need 1997 events.
- **Blank is not zero** (File 3 §10): a market-year whose every event has a
  blank `nkill` is missing on NK, and its TSI averages the components it has.
- **Terrorism is not randomly assigned.** Every estimate is an association, not
  a causal effect (§10).

### If downloads fail with a certificate error

`CERTIFICATE_VERIFY_FAILED` means the network presents its own certificate.
The scripts already use `truststore`, which makes Python trust the Windows
certificate store. If it is missing:

```bash
pip install truststore
```

---

## Layout

```
config.py                  the only file you edit day to day
0..3_*.py                  the run targets, in order
seminar/
  specs.py        paths.py        countries.py     io_tabular.py
  sources.py      terror.py       qualification.py classification.py
  tsi.py          panel.py        descriptives.py  diagnostics.py
  regression.py   viz.py          report.py        sample_data.py
tests/                     test_definitions.py, test_io_tabular.py
data/
  worldbank/  undp/  classification/  conflict/  reference/  cache/
outputs/                   stage1/, tables/, figures/, report.html
```

Install once:

```bash
pip install -r requirements.txt
```
