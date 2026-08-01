# The effect of terrorism on long-term country metrics

Seminar project. For every metric you choose, this estimates what terrorism does
to it, and whether the effect differs between developing and developed
countries.

The question it answers, per metric: **what changes when terrorism is added to a
regression that already controls for everything else?**

---

## Running it

Open the project in PyCharm and press the green arrow on each file, in order.

| File | What it does | When |
|---|---|---|
| `0_fetch_worldbank.py` | downloads real metric data from the World Bank into `data/` | optional, once |
| `1_build_terror_cache.py` | converts the 104 MB GTD xlsx into a fast cache | once |
| `2_check_data.py` | checks your files against `config.py` and reports problems | after every data change |
| `3_run_analysis.py` | runs the regressions, writes tables, figures and the report | whenever you want results |

Results land in `outputs/`: open **`outputs/report.html`** for everything on one
page, or take the CSVs in `outputs/tables/` straight into the paper.

To prove the machinery works before trusting it on real data:

```bash
python 3_run_analysis.py --sample
```

That builds a synthetic dataset with a **known** terrorism effect planted in it,
runs the whole pipeline, and reports whether the regression recovered the number
it was supposed to. It also includes a placebo metric with no effect planted,
which should come out insignificant.

---

## Adding data: always two steps

**1. Put the file in the right folder**

```
data/metrics/     DEPENDENT variables - each one gets its own regression
data/controls/    INDEPENDENT controls - held fixed in every model
data/filters/     classification data - decides developed vs developing
```

**2. Declare it in `config.py`**

```python
METRICS = {
    "trade_openness": MetricSpec(
        file      = "trade_openness.csv",   # the file you just added
        column    = "value",                # which column holds the number
        transform = None,                   # None|"log"|"pct_change"|"diff"
        min_years = 10,                     # skip countries with less history
        label     = "Trade openness",       # caption for tables and figures
        unit      = "% of GDP",
    ),
}
```

Adding a filtration metric works the same way — drop the file in `data/filters/`
and say what you filter on:

```python
FILTERS = {
    "hdi": FilterSpec(
        file       = "hdi.csv",
        column     = "value",
        rule       = "threshold",
        threshold  = 0.800,        # >= 0.800 -> developed
        high_label = "developed",
        low_label  = "developing",
        mode       = "static",     # one fixed tag per country
    ),
}
ACTIVE_FILTER = "hdi"              # which one this run uses
```

Then run `2_check_data.py`. It validates every declaration against the real
files and tells you exactly what is wrong — missing file, misspelled column,
a threshold no country can reach — before any analysis starts.

---

## What your data files should look like

Either layout works, `.csv` or `.xlsx`, and the program detects which:

**Long** — one row per country-year:

```csv
country,year,value
United States,1990,20.55
United States,1991,20.31
Germany,1990,52.10
```

**Wide** — one column per year:

```csv
country,1990,1991,1992
United States,20.55,20.31,20.94
Germany,52.10,50.02,47.88
```

Rules:

- The country column may hold **names** ("United States") or **ISO3 codes**
  ("USA"). Both work.
- Extra metadata lines above the header are skipped automatically.
- `..`, empty cells and `n/a` are read as missing.
- `1,234.5` with a thousands separator is read as a number.
- A file with **no year column at all** is treated as constant over the whole
  period — useful for a one-off classification snapshot.
- Any country name the program does not recognise is listed in
  `outputs/unmatched_countries.csv`. Add real ones to
  `data/reference/country_aliases.csv` (columns `name,iso3`).

Everything becomes one flat DataFrame with columns `iso3`, `year`, and the
value, so combining data is ordinary `pd.merge(a, b, on=["iso3","year"])`.
**No MultiIndex is used anywhere in this project.**

---

## What the analysis actually does

For each metric independently, four nested models on identical rows:

| | Model | What it is |
|---|---|---|
| **A** | before terrorism | `metric ~ controls + country FE + year FE` |
| **B** | after terrorism | `A + terror` |
| **C** | by group | `B + terror x developed` |
| **D** | long run | `metric ~ controls + terror(t, t-1, t-2, t-3) + FE` |

The **A to B difference is the headline**: how much extra variation terrorism
explains, whether it is significant, how the other coefficients moved once it
entered, and the effect translated into the metric's own units.

Fixed effects are ordinary dummy variables (`C(iso3)`, `C(year)`). Country
dummies absorb everything permanent about a country — institutions, geography,
its normal level of the metric. Year dummies absorb whatever hit the whole world
that year, like 2008. What remains for the terrorism coefficient is
within-country variation relative to the world: that is what *isolating the
effect of terrorism* means here. Standard errors are clustered by country.

**Model C note.** Under `mode="static"` each country stays in one group, so the
`developed` main effect is perfectly absorbed by the country dummies and is not
estimated. The interaction still identifies the difference in slope between the
two groups, which is the comparison the paper needs.

### Figures

Written to `outputs/figures/` at 300 dpi:

1. **coverage heatmap** — which country-years you actually have
2. **before/after chart** — every coefficient in A next to B, plus the R² gain
3. **effect forest plot** — the effect for developed vs developing
4. **lag profile** — the cumulative effect over four years
5. **partial regression plot** — the isolated effect as a scatter, where the
   fitted slope *is* the coefficient
6. **group trends** — the metric over time by group with terrorism overlaid
7. **cross-metric summary** — every metric's effect on one comparable axis

---

## Things worth knowing before you write them up

- **The GTD has no data for 1993 at all.** Those records were lost. The code
  keeps 1993 as missing rather than counting it as a year without terrorism,
  which is why the sample starts at 1997 when four lags are used.
- **A country-year with no attacks in the GTD is a real zero**, not missing
  data, and is treated that way.
- **Terrorism is not randomly assigned.** Weak economies may attract terrorism
  as much as terrorism weakens economies. Fixed effects and lagged terrorism
  reduce this concern but do not remove it. These are associations under strong
  controls, not proven causation — say so in the paper; an examiner will look
  for it.
- **GTD collection methods changed** around 1997 and 2012, so raw counts are not
  perfectly comparable across the whole span. The year fixed effects absorb the
  common part of this.
- Quality filters (`GTD_REQUIRE_ALL_CRITERIA`, `GTD_DROP_DOUBTED` in
  `config.py`) follow the usual practice: keep only events meeting all three GTD
  criteria and drop those flagged as possibly not terrorism. Of 209,706 events,
  174,396 (83%) survive them.
- **The May 2022 GTD release ends in 2020**, which is why `PERIOD` stops there.
  If you point the code at a later release, raise it — any year the release does
  not cover is excluded rather than counted as a year of zero attacks.

### If downloads fail with a certificate error

`0_fetch_worldbank.py` may report `CERTIFICATE_VERIFY_FAILED`. That happens when
the network (a corporate proxy, or antivirus that inspects HTTPS) presents its
own certificate: Windows trusts it, but Python ships its own separate list and
does not. The script already handles this with `truststore`, which tells Python
to use the Windows certificate store. If it is missing:

```bash
pip install truststore
```

This does not disable any security check — it uses the same certificates your
browser already uses.

---

## Layout

```
config.py                  the only file you edit day to day
0..3_*.py                  the run targets, in order
seminar/
  specs.py                 what you declare in config.py
  paths.py                 where every folder is
  countries.py             country name -> ISO3
  io_tabular.py            reads any csv/xlsx of country-year data
  terror.py                the GTD side
  filters.py               developed vs developing tagging
  panel.py                 assembles the table for one metric
  regression.py            the nested models A/B/C/D
  viz.py                   figures
  report.py                CSV tables and the HTML report
  pipeline.py              wiring shared by the run scripts
  sample_data.py           synthetic data with a known effect
tests/test_io_tabular.py   tests for the loader; run it directly
data/                      your data (see above)
outputs/                   figures, tables, report.html
```

Install once:

```bash
pip install -r requirements.txt
```
