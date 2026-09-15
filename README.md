# dstoolset

[![PyPI](https://img.shields.io/pypi/v/dstoolset.svg)](https://pypi.org/project/dstoolset/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Reusable exploratory data analysis (EDA) helpers for data-science and analytics
projects. `dstoolset` bundles the small pandas/seaborn utilities you end up
rewriting in every notebook — feature-vs-target plots, full-coverage pair
plots, IQR outlier detection, and a quick "which features matter" ranking —
behind one clean import.

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
and `seaborn` automatically.

## Quick start

```python
import seaborn as sns
import dstoolset as ds

df = sns.load_dataset("tips")

# Which features explain the most variance in `tip`?
ds.variance_ranking(df, target="tip")
# [('total_bill', 0.46...), ('size', 0.24...), ('day', 0.02...), ...]

# Flag rows holding an IQR outlier in any numeric column
mask = ds.get_outliers_rows(df)
df[mask]

# Plot every feature against the target; the plot type is picked automatically
fig, axes = ds.plot_target(df, target="tip")
```

## What's inside

The public API is re-exported at the top level, so everything is reachable as
`ds.<function>` regardless of the internal module it lives in.

### Visualization

**`pair_plots(df, target=None, splits=4, binning="quantile", **kwargs)`**
Scatters every numeric feature against every other one, producing **one figure
per feature** (that feature on the y-axis, each other feature on the x-axis).
Unlike a chunked `seaborn.pairplot`, no pair is ever dropped. `target` colors
every panel as the hue; a numeric target is binned into `splits` categories
first (`binning="quantile"` or `"width"`). Returns a list of `(fig, axes)`
tuples, one per feature.

```python
plots = ds.pair_plots(df, target="sex")
fig, axes = plots[0]
```

**`plot_target(df, target, max_cardinality=10, **kwargs)`**
Plots every feature against the target on its own subplot, choosing the plot
type per feature: a scatter for continuous-numeric pairs, a box plot when the
target has more than three distinct values against a numeric variable, and a
histogram otherwise. A numeric column with fewer than `max_cardinality`
distinct values is treated as categorical. Returns `(fig, axes)`.

```python
fig, axes = ds.plot_target(df, target="tip")
```

**`box_plot_numeric(df)`**
Draws a box plot for every numeric feature, each on its own subplot — a fast
scan of spread and outliers across the frame. Returns `(fig, axes)`.

```python
fig, axes = ds.box_plot_numeric(df)
fig.savefig("boxes.png")
```

### Diagnostics

**`get_outliers_rows(df, cols="all", threshold=1.5, side="both")`**
Returns a boolean `Series` aligned to `df.index`, `True` where a row holds an
IQR outlier in any of the given columns (outside
`[Q1 - threshold * IQR, Q3 + threshold * IQR]`). `cols` accepts `"all"`, a
single column name, or a list; `side` restricts to the `"low"` or `"high"`
tail.

```python
mask = ds.get_outliers_rows(df, cols=["total_bill", "tip"], side="high")
df[mask]
```

**`variance_ranking(df, target, features="all", min_cardinality=5, ascending=False)`**
Ranks features by how much of a **numeric** target's variance each explains,
scoring numeric features with R² and categorical ones with η² so the two are
directly comparable on a 0–1 scale. Returns a list of
`(feature_name, variance_explained)` tuples, most informative first by default.

```python
ds.variance_ranking(df, target="tip", features="numeric")
# [('total_bill', 0.46...), ('size', 0.24...)]
```

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
