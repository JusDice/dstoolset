# dstoolset

[![PyPI](https://img.shields.io/pypi/v/dstoolset)](https://pypi.org/project/dstoolset/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Reusable exploratory data analysis (EDA) helpers for data-science and analytics
projects. `dstoolset` bundles the small pandas/seaborn/scipy utilities you end
up rewriting in every notebook — feature-vs-target plots, full-coverage pair
plots, grouped distribution comparisons, IQR outlier detection, numeric
binning, and quick "which features matter" statistics — behind one clean
import.

```python
import dstoolset as ds
```

## Installation

From PyPI:

```bash
pip install dstoolset
```

The latest version straight from GitHub:

```bash
pip install git+https://github.com/JusDice/dstoolset.git
```

`dstoolset` requires Python 3.9+ and pulls in `pandas`, `numpy`, `matplotlib`,
`seaborn` and `scipy` automatically.

For development, clone the repo and install it in editable mode with the dev
extras (`build`, `twine`, `pytest`):

```bash
git clone https://github.com/JusDice/dstoolset.git
cd dstoolset
pip install -e ".[dev]"
```

## Quick start

```python
import seaborn as sns
import dstoolset as ds

df = sns.load_dataset("tips")

# Which features explain the most variance in `tip`?
ds.variance_ranking(df, target="tip")
# [('total_bill', 0.457), ('size', 0.239), ('day', 0.020), ('time', 0.015), ...]

# Flag rows holding an IQR outlier in any numeric column
mask = ds.get_outliers_rows(df)
df[mask]

# Plot every feature against the target; the plot type is picked automatically
fig, axes = ds.plot_target(df, target="tip")
```

## What's inside

The public API is re-exported at the top level, so everything is reachable as
`ds.<function>` regardless of the internal module it lives in.

| Area | Function | What it does |
| --- | --- | --- |
| Plots | [`plot_target`](#plot_target) | Every feature vs. the target, plot type chosen per feature |
| | [`pair_plots`](#pair_plots) | Every numeric pair, one figure per feature, nothing dropped |
| | [`hist_plots`](#hist_plots) | Count plot for every categorical feature |
| | [`distr_comp`](#distr_comp) | Share of each target category within each feature group |
| | [`box_plot_numeric`](#box_plot_numeric) | Box plot for every numeric feature |
| Statistics | [`variance_ranking`](#variance_ranking) | Rank features by variance of the target they explain ($R^2$ / $\eta^2$) |
| | [`r_squared`, `eta_squared`](#r_squared-and-eta_squared) | The two scores behind that ranking |
| | [`ordinal_association`](#ordinal_association) | Spearman, Kruskal-Wallis and $\eta^2$ for ordinal features |
| | [`multiple_ANOVA`](#multiple_anova) | One-way ANOVA of the target across each feature's groups |
| | [`get_groups`](#get_groups) | Split columns into per-group arrays for your own tests |
| Data prep | [`categorize_numeric`](#categorize_numeric) | Bin a numeric column into ordered categories |
| | [`get_outliers_rows`](#get_outliers_rows) | Boolean mask of rows holding an IQR outlier |

### Shared options for grid plots

`plot_target`, `hist_plots`, `distr_comp` and `box_plot_numeric` draw one
subplot per feature on a single figure and share the same conventions:

- **Layout:** the figure is always 20 inches wide. By default up to 6 panels
  are laid out 3 per row and more are laid out 4 per row; pass `ncols=` to
  choose yourself. Unused trailing axes are hidden.
- **Your own axes:** pass `ax=` (one Axes, or one per panel) to draw into an
  existing figure instead of creating one. Your figure keeps its size and
  layout, and `ncols` is ignored.
- **Single-feature mode:** `plot_target`, `hist_plots` and `distr_comp` accept
  `features="col"` as a bare string to plot just that column; they then
  return `(fig, ax)` with a single Axes instead of an array.
- **Counts under tick labels:** `display_count=True` (on `plot_target`,
  `hist_plots` and `distr_comp`) writes the row count under each category,
  e.g. `n=34.6K`.
- **Return value:** always `(fig, axes)`, so you can tweak panels or call
  `fig.savefig(...)`.

```python
import matplotlib.pyplot as plt

fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
left.set_title("my own plot")
_, ax = ds.plot_target(df, target="tip", features="total_bill", ax=right)
```

`pair_plots` uses the same layout rules and `ncols`, but makes one figure per
feature, so it has no `ax` or single-feature mode.

### Plots

#### `plot_target`

`plot_target(df, target, max_cardinality=10, show_mean=False, features="all", display_count=False, ncols=None, ax=None, **kwargs)`

Plots every feature against the target, choosing the plot type per feature:

- a **scatter** when both the target and the feature are continuous numeric,
- a **box plot** when the target has more than three distinct values and is
  paired with a numeric variable,
- a **histogram** colored by the target otherwise (low-cardinality target, or
  two categorical columns).

A numeric column with fewer than `max_cardinality` distinct values is treated
as categorical. `features` accepts `"all"`, `"numeric"`, `"categorical"`, a
single column name or a list. `show_mean=True` overlays each category's mean
on the box-plot panels as a red dashed line; the legend sits just above the
figure, so save with `fig.savefig(..., bbox_inches="tight")` to keep it.
`**kwargs` go to whichever seaborn function draws a given panel.

```python
fig, axes = ds.plot_target(df, target="tip", show_mean=True)
fig, ax = ds.plot_target(df, target="tip", features="day", display_count=True)
```

#### `pair_plots`

`pair_plots(df, target=None, splits=4, binning="quantile", labels="intervals", features="all", ncols=None, **kwargs)`

Scatters every numeric feature against every other one, producing **one figure
per feature** (that feature on the y-axis, each other feature on the x-axis).
Unlike a chunked `seaborn.pairplot`, no pair is ever dropped. `target` colors
every panel as the hue; a numeric target is binned into `splits` categories
first (`binning="quantile"` or `"width"`, or your own edges such as
`binning=[0, 5, 10, 20]` for $[0, 5)$, $[5, 10)$, $[10, 20]$). Returns a list of `(fig, axes)`
tuples, one per feature.

`labels` controls how those bins appear in the legend: `"intervals"` shows
each bin's value range, `None` numbers them `0, 1, ...`, and a list gives your
own names (lowest bin first). `features` limits which numeric columns are
paired. `**kwargs` go to `seaborn.scatterplot`; a numeric target defaults to
the `"viridis"` palette.

```python
plots = ds.pair_plots(df, target="sex")
fig, axes = plots[0]

ds.pair_plots(df, target="tip", binning=[0, 2, 4, 11], labels=["low", "mid", "high"])
```

#### `hist_plots`

`hist_plots(df, features="all", max_cardinality=10, hue=None, display_count=False, ncols=None, ax=None, **kwargs)`

The categorical counterpart of `box_plot_numeric`: a count plot for every
categorical feature, showing how often each category occurs. A column counts
as categorical if it is non-numeric, or numeric with fewer than
`max_cardinality` distinct values (drawn with one bar per value rather than
binned). `hue` splits each bar by another column. `**kwargs` go to
`seaborn.histplot`, which defaults here to `multiple="dodge"` and `shrink=0.8`;
add e.g. `stat="percent"` to show shares instead of counts.

```python
fig, axes = ds.hist_plots(df, hue="sex")
fig, ax = ds.hist_plots(df, features="day", display_count=True)
```

#### `distr_comp`

`distr_comp(df, features, target, max_cardinality=30, display_count=False, ncols=None, ax=None, **kwargs)`

Compares how a **categorical** target is distributed across the groups of each
feature. Within each feature value the share of every target category is
computed (a row-normalised crosstab, in percent) and drawn as a grouped bar
chart, so each cluster sums to $`100\%`$ and groups of very different sizes are
directly comparable. Target categories keep the same order and colors in every
panel. A numeric feature may have at most `max_cardinality` distinct values;
bin a continuous one first with `categorize_numeric`.

```python
fig, axes = ds.distr_comp(df, features=["day", "size"], target="time")

df["bill_band"] = ds.categorize_numeric(df["total_bill"], labels=["low", "mid", "high", "top"])
fig, ax = ds.distr_comp(df, features="bill_band", target="smoker", display_count=True)
```

#### `box_plot_numeric`

`box_plot_numeric(df, features="all", ncols=None, ax=None)`

Draws a box plot for every numeric feature, each on its own (deliberately
flat) subplot — a fast scan of spread and outliers across the frame.
`features` restricts it to a list of columns.

```python
fig, axes = ds.box_plot_numeric(df)
fig.savefig("boxes.png")
```

### Statistics

#### `variance_ranking`

`variance_ranking(df, target, features="all", min_cardinality=5, ascending=False)`

Ranks features by how much of a **numeric** target's variance each explains,
scoring numeric features with $R^2$ and categorical ones with $\eta^2$, so the two
are directly comparable on a $[0, 1]$ scale. A numeric column with at most
`min_cardinality` distinct values is treated as categorical. `features`
accepts `"all"`, `"numeric"`, `"categorical"` or a list. Returns a list of
`(feature_name, variance_explained)` tuples, most informative first by
default.

```python
ds.variance_ranking(df, target="tip", features="numeric")
# [('total_bill', 0.457), ('size', 0.239)]
```

#### `r_squared` and `eta_squared`

`r_squared(feature, target)` · `eta_squared(feature, target)`

The two scores behind `variance_ranking`, usable on their own with two Series.
`r_squared` is the squared Pearson correlation of a numeric feature with a
numeric target, $R^2 = r^2$ (linear association only). `eta_squared` is the
between-group share of the target's total sum of squares for a categorical
feature, $`\eta^2 = SS_{\text{between}} / SS_{\text{total}}`$. Both drop rows
where either value is missing and return a float in $[0, 1]$.

For a binary feature, $\eta^2 = r_{pb}^2$, the squared point-biserial
correlation. Note that both are squared: $r = 0.3$ becomes $r^2 = 0.09$, so
compare them to each other rather than to plain correlations.

```python
ds.r_squared(df["total_bill"], df["tip"])   # 0.457
ds.eta_squared(df["day"], df["tip"])        # 0.020
```

#### `ordinal_association`

`ordinal_association(df, target, features, max_cardinality=15)`

Tests ordinal features against a numeric target with three complementary
measures: **Spearman's $\rho$** (strength and direction of a monotonic trend), the
**Kruskal-Wallis $H$** test (whether the target's distribution differs across
levels, with no normality assumption) and **$\eta^2$** (variance explained). Each
feature must be numeric (e.g. integer-coded levels) or an ordered categorical
such as the output of `categorize_numeric`. Returns a DataFrame with one row
per feature, strongest $\lvert\rho\rvert$ first.

```python
df["bill_band"] = ds.categorize_numeric(df["total_bill"], labels=["low", "mid", "high", "top"])
ds.ordinal_association(df, target="tip", features=["size", "bill_band"])
#      feature  spearman_rho    p_spearman  kruskal_H     p_kruskal   eta2
# 0  bill_band         0.652  6.100789e-31     104.78  1.454948e-22  0.380
# 1       size         0.468  1.059850e-14      56.02  8.041675e-11  0.249
```

#### `multiple_ANOVA`

`multiple_ANOVA(df, by, target, max_cardinality=15)`

Runs a one-way ANOVA (`scipy.stats.f_oneway`) of a numeric target across the
groups of each feature in `by`, testing whether the group means are equal.
Returns `(groups, summary)`: the per-group target arrays that were tested, and
a printable text summary with the $F$ statistic and $p$-value per feature.

```python
groups, summary = ds.multiple_ANOVA(df, by=["day", "time"], target="tip")
print(summary)
# ANOVA results for day:
# Statistic = 1.6724; p-value = 0.1736
#
# ANOVA results for time:
# Statistic = 3.6338; p-value = 0.0578
```

#### `get_groups`

`get_groups(df, by, columns, max_cardinality=15)`

Splits one or more columns into per-group NumPy arrays, grouped by `by` —
ready to feed into any scipy test that takes one sample per group. Returns
`(groups, labels)`: plain lists for a single column, or dicts keyed by column
name for several. A numeric `by` column is grouped only if it has at most
`max_cardinality` distinct values.

```python
from scipy import stats

groups, labels = ds.get_groups(df, by="day", columns="tip")
labels                  # ['Thur', 'Fri', 'Sat', 'Sun']
stats.kruskal(*groups)
```

### Data prep

#### `categorize_numeric`

`categorize_numeric(values, binning="quantile", splits=4, labels=None)`

Bins a numeric Series into ordered categories, numbered from the lowest values
up:

- `binning="quantile"`: equal-frequency bands (`pandas.qcut`); heavily
  repeated values can give fewer than `splits` bins,
- `binning="width"`: equal-width bands (`pandas.cut`),
- `binning=[0, 10, 20, 60]`: your own edges, giving $[0, 10)$, $[10, 20)$,
  $[20, 60]$ (`splits` is ignored; values outside the edges become missing).

`labels=None` returns integer codes, `"intervals"` labels each bin with its
range, and a list gives your own names. Except for integer codes, the result
is an ordered categorical, whose order `ordinal_association` uses for its
ranks.

```python
df["bill_band"] = ds.categorize_numeric(df["total_bill"], labels=["low", "mid", "high", "top"])
df["bill_bracket"] = ds.categorize_numeric(df["total_bill"], binning=[0, 10, 20, 60], labels="intervals")
df["bill_bracket"].value_counts(sort=False)
# [0, 10)      17
# [10, 20)    130
# [20, 60]     97
```

#### `get_outliers_rows`

`get_outliers_rows(df, cols="all", threshold=1.5, side="both")`

Returns a boolean `Series` aligned to `df.index`, `True` where a row holds an
IQR outlier in any of the given columns (outside
$`[Q_1 - t \cdot \mathrm{IQR},\ Q_3 + t \cdot \mathrm{IQR}]`$, where $t$ is `threshold`). `cols` accepts `"all"`
(every numeric column), a single column name, or a list; `side` restricts to
the `"low"` or `"high"` tail.

```python
mask = ds.get_outliers_rows(df, cols=["total_bill", "tip"], side="high")
df[mask]
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
