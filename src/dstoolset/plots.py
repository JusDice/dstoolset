"""Plotting helpers for exploratory data analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import seaborn as sns

from .data_exploration import categorize_numeric


def _select_features(df, features, exclude=()):
	"""Resolve a ``features`` argument to a concrete list of column names.

	``features`` is ``"all"`` (every column), ``"numeric"`` (numeric-dtype
	columns), ``"categorical"`` (non-numeric columns), or an explicit sequence
	of column names. Names in ``exclude`` (e.g. the target) are dropped.
	"""
	exclude = set(exclude)
	if isinstance(features, str):
		if features == "all":
			cols = list(df.columns)
		elif features == "numeric":
			cols = list(df.select_dtypes(include="number").columns)
		elif features == "categorical":
			cols = list(df.select_dtypes(exclude="number").columns)
		else:
			raise ValueError(
				"features string must be 'all', 'numeric', or 'categorical'; "
				f"got {features!r} (pass a list to name explicit columns)"
			)
	else:
		cols = list(features)
		missing = [c for c in cols if c not in df.columns]
		if missing:
			raise KeyError(f"features not in df: {missing}")
	return [c for c in cols if c not in exclude]


# Total width (inches) of every multi-panel figure; must stay in [18, 20].
_FIG_WIDTH = 20
# Panel height / width for box plots: single horizontal boxes need little
# height, so they get flatter panels than the default 4:3.
_BOX_ASPECT = 0.3


def _format_count(n):
	"""Shorten a row count for a tick label: 1234 -> "1234", 34561 -> "34.6K".

	Counts below 10,000 are shown in full; larger ones get one decimal and a
	K (thousands), M (millions) or B (billions) suffix.
	"""
	n = int(n)
	if n < 10_000:
		return str(n)
	for divisor, suffix in ((1e3, "K"), (1e6, "M"), (1e9, "B")):
		value = n / divisor
		# Move up a unit when rounding would print e.g. "1000.0K".
		if round(value, 1) < 1000 or suffix == "B":
			return f"{value:.1f}{suffix}"


def _add_counts_to_xticks(ax, data, x, used, numeric_axis=False):
	"""Write the number of rows in each x category under its tick label.

	Counts are taken over the rows where every column in ``used`` (the
	columns the panel actually plots) is present, so they match what is
	drawn. ``numeric_axis=False`` is for categorical axes (seaborn bar/box
	plots, histograms of text columns): existing tick labels are matched to
	categories by their text. ``numeric_axis=True`` is for histograms of a
	low-cardinality numeric column drawn on a real number line: ticks are
	placed at each value present.
	"""
	counts = data.dropna(subset=list(used))[x].value_counts()

	if numeric_axis:
		values = sorted(counts.index)
		ax.set_xticks(values)
		ax.set_xticklabels(
			[f"{v:g}\nn={_format_count(counts[v])}" for v in values]
		)
		return

	by_text = {str(k): c for k, c in counts.items()}
	ticks = ax.get_xticks()
	labels = [t.get_text() for t in ax.get_xticklabels()]
	ax.set_xticks(ticks)
	ax.set_xticklabels(
		[
			f"{label}\nn={_format_count(by_text[label])}"
			if label in by_text
			else label
			for label in labels
		]
	)


def _subplot_grid(n, ncols=None, aspect=3 / 4, ax=None):
	"""Create a figure with a grid of ``n`` subplots, sized for readability.

	Shared layout rule for every multi-panel plot in this module.

	Columns: an explicit ``ncols`` always wins (capped at ``n`` so no column
	is left empty). With ``ncols=None``: up to 6 panels -> at most 3 per row;
	more than 6 -> 4 per row.

	Size: the figure is always ``_FIG_WIDTH`` inches wide, split evenly
	between the columns (``col_width = _FIG_WIDTH / ncols``); each row is
	``col_width * aspect`` inches tall. The default ``aspect=3 / 4`` gives
	4:3 panels; functions whose panels are naturally flat (e.g. box plots)
	pass their own named aspect constant.

	Existing axes: when ``ax`` is given (one Axes or a sequence/array of
	them, exactly one per panel), no figure is created and ``ncols`` /
	``aspect`` are ignored; the caller's figure keeps its own size and
	layout. Callers should then skip ``fig.tight_layout()``.

	Returns ``(fig, axes)`` where ``axes`` is a flat array with exactly
	``nrows * ncols`` entries; the trailing ones beyond ``n`` are hidden.
	With ``ax``, ``axes`` is those axes (flattened) and ``fig`` is the
	figure they belong to.
	"""
	if ax is not None:
		axes = np.array(ax if np.iterable(ax) else [ax], dtype=object).reshape(-1)
		if not all(isinstance(a, Axes) for a in axes):
			raise TypeError("ax must be a matplotlib Axes or a sequence of Axes")
		if len(axes) != n:
			raise ValueError(
				f"ax has {len(axes)} axes but there are {n} panels to draw; "
				"pass exactly one Axes per panel"
			)
		return axes[0].figure, axes

	if ncols is None:
		ncols = 4 if n > 6 else min(n, 3)
	else:
		if int(ncols) != ncols or ncols < 1:
			raise ValueError(f"ncols must be a positive integer, got {ncols!r}")
		ncols = min(int(ncols), n)
	nrows = int(np.ceil(n / ncols))

	col_width = _FIG_WIDTH / ncols
	row_height = col_width * aspect

	fig, axes = plt.subplots(
		nrows, ncols, figsize=(_FIG_WIDTH, row_height * nrows)
	)
	axes = np.atleast_1d(axes).reshape(-1)

	# Hide any leftover empty subplots.
	for ax in axes[n:]:
		ax.set_visible(False)

	return fig, axes


def pair_plots(
	df: pd.DataFrame,
	target=None,
	splits: int = 4,
	binning="quantile",
	labels="intervals",
	features="all",
	ncols=None,
	**kwargs,
) -> list:
	"""Scatter every numeric feature against every other, one figure per feature.

	Unlike a chunked :func:`seaborn.pairplot`, this covers *all* pairs: each
	numeric feature gets its own figure holding one scatter panel per other
	feature (that feature on the y-axis, the other on the x-axis). 
	Univariate distribution panels are not drawn. When a ``target`` is given 
	it colors every panel as the ``hue``; a numeric target
	is first binned into categories.

	Note that every pair therefore appears in two figures (``a`` vs ``b`` lives
	in both ``a``'s figure and ``b``'s), which is the cost of giving each
	feature a dedicated view while guaranteeing full coverage.

	Parameters
	----------
	df : pandas.DataFrame
		The data to plot.
	target : str, optional
		Column used to color observations (``hue``). A numeric target is
		binned into ``splits`` categories first (with
		:func:`categorize_numeric`, see ``binning`` and ``labels``); a
		categorical target is used directly. Defaults to ``None`` (no
		coloring).
	splits : int, default 4
		Number of bins to cut a numeric target into. Ignored when
		``binning`` is a list of edges.
	binning : {"quantile", "width"} or sequence of numbers, default "quantile"
		How a numeric target is binned:  
		``"quantile"`` uses :func:`pandas.qcut` (equal-frequency bands),  
		``"width"`` uses :func:`pandas.cut` (equal-width bands),  
		a sequence of at least two strictly increasing edges, e.g.
		``[0, 5, 10, 20]``, gives bins ``[0, 5)``, ``[5, 10)``, ``[10, 20]``;
		target values outside the edges are left uncolored.
	labels : "intervals", None or list of str, default "intervals"
		How the bins of a numeric target are named in the legend:  
		``"intervals"`` shows each bin's value range, e.g. ``(10.5, 17.2]``
		or ``[0, 5)`` for custom edges,  
		``None`` numbers them ``0, 1, ..., n_bins - 1``,  
		a list of names labels them lowest first and must have one entry
		per bin.  
		Ignored for a categorical target or when ``target`` is ``None``.
	features : "all" or sequence of str, default "all"
		Which columns to consider:  
		``"all"`` every numeric column except the target,  
		a sequence of names selects exactly those columns.
	ncols : int, optional
		Number of subplot columns. Defaults to ``None``: up to 6 panels are
		laid out 3 per row, more than 6 are laid out 4 per row. The figure is
		always 20 inches wide, split evenly between the columns, and each
		panel keeps a 4:3 aspect ratio.
	**kwargs
		Passed through to :func:`seaborn.scatterplot`. For a numeric target,
		``palette`` defaults to ``"viridis"``.

	Returns
	-------
	list of (matplotlib.figure.Figure, numpy.ndarray)
		One ``(fig, axes)`` tuple per numeric feature. ``axes`` is the flat
		array of panels in that figure (unused trailing axes are hidden), so
		you can tweak panels or save the whole figure
		(e.g. ``fig.savefig(...)``).

	Raises
	------
	KeyError
		If ``target`` is not a column of ``df``.
	ValueError
		If ``binning`` is not ``"quantile"``, ``"width"`` or a valid list of
		edges, ``labels`` is invalid or does not have one entry per bin, or
		there are fewer than two numeric features to pair.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import pair_plots
	>>> plots = pair_plots(sns.load_dataset("iris"), target="species")
	>>> fig, axes = plots[0]
	"""
	data = df.copy()
	hue = None
	numeric_data = data.select_dtypes(include="number").copy()

	if target is not None:
		if target not in data.columns:
			raise KeyError(f"target {target!r} is not a column in df")

		hue = target
		if target in numeric_data.columns:
			binned = categorize_numeric(
				numeric_data[target], binning, splits, labels=labels
			)
			if labels is None:
				# Integer codes would get a continuous hue legend; make them
				# categories so every bin gets its own legend entry.
				binned = binned.astype("category")
			numeric_data[target] = binned
			kwargs.setdefault("palette", "viridis")  # sequential colors
		else:
			# Categorical target: attach it so it can be used as hue.
			numeric_data[target] = data[target]

	selected = _select_features(
		df, features, exclude=[target] if target is not None else []
	)
	features = [c for c in selected if c in numeric_data.columns]
	if len(features) < 2:
		raise ValueError(
			"need at least two numeric features to pair, "
			f"got {len(features)}"
		)

	plots = []
	for feature in features:
		others = [c for c in features if c != feature]
		fig, axes = _subplot_grid(len(others), ncols)

		for ax, other in zip(axes, others):
			sns.scatterplot(
				data=numeric_data, x=other, y=feature, hue=hue, ax=ax, **kwargs
			)

		fig.suptitle(f"{feature} vs all other features")
		fig.tight_layout()
		plots.append((fig, axes))

	return plots


def box_plot_numeric(df: pd.DataFrame, features="all", ncols=None, ax=None):
	"""Draw a box plot for every numeric feature, each on its own subplot.

	Numeric columns are laid out on a grid (up to four per row). Handy for
	scanning spread and outliers across all numeric features at once.

	Parameters
	----------
	df : pandas.DataFrame
		The data to plot. Only numeric columns are box-plotted.
	features : {"all"} or sequence of str, default "all"
		Which columns to consider:
		``"all"`` every numeric column,  
		a sequence of names selects exactly those columns.
	ncols : int, optional
		Number of subplot columns. Defaults to ``None``: up to 6 panels are
		laid out 3 per row, more than 6 are laid out 4 per row. The figure is
		always 20 inches wide, split evenly between the columns; each panel
		is 0.3 times as tall as it is wide, which suits a single horizontal
		box.
	ax : matplotlib.axes.Axes or sequence of Axes, optional
		Existing axes to draw into, exactly one per panel, instead of creating
		a new figure (e.g. ``ax=ax1`` to sit next to your own subplots). Your
		figure keeps its own size and layout, and ``ncols`` is ignored; the
		returned ``fig`` is the figure those axes belong to. Defaults to
		``None``: a new figure is created.

	Returns
	-------
	fig : matplotlib.figure.Figure
		The figure holding all the box plots, so you can save the whole grid
		(e.g. ``fig.savefig("boxes.png")``).
	axes : numpy.ndarray
		Flat array of the :class:`matplotlib.axes.Axes` used, so you can
		further tweak them. Any unused trailing axes are hidden.

	Raises
	------
	ValueError
		If there are no numeric columns to plot.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import box_plot_numeric
	>>> fig, axes = box_plot_numeric(sns.load_dataset("iris"))
	"""
	selected = _select_features(df, features)
	df_num = df[selected].select_dtypes(include="number")

	n = df_num.shape[1]
	if n == 0:
		raise ValueError("No numeric columns to plot.")

	own_fig = ax is None  # the loop below reuses the name ``ax``
	fig, axes = _subplot_grid(n, ncols, aspect=_BOX_ASPECT, ax=ax)

	for ax, col in zip(axes, df_num.columns):
		sns.boxplot(x=df_num[col], ax=ax)
		ax.set_title(col)
		ax.set_xlabel("")

	if own_fig:
		fig.tight_layout()
	return fig, axes


def hist_plots(
	df: pd.DataFrame,
	features="all",
	max_cardinality: int = 10,
	hue=None,
	display_count: bool = False,
	ncols=None,
	ax=None,
	**kwargs,
):
	"""Draw a histogram (count plot) for every categorical feature.

	The categorical counterpart of :func:`box_plot_numeric`: each categorical
	feature gets its own subplot showing how often each category occurs, laid
	out on a grid (up to four per row).

	A column counts as categorical if it is non-numeric, or numeric with fewer
	than ``max_cardinality`` distinct values. Low-cardinality numeric columns
	are drawn with one bar per distinct value (``discrete=True``) rather than
	being binned.

	Parameters
	----------
	df : pandas.DataFrame
		The data to plot.
	features : "all", str or sequence of str, default "all"
		Which columns to plot:  
		``"all"`` every categorical column,  
		a single column name plots just that column and returns a single
		Axes (see Returns),  
		a sequence of names selects exactly those columns (any that are
		not categorical are skipped).
	max_cardinality : int, default 10
		A numeric column with fewer than this many distinct values is treated
		as categorical; one with more is skipped as continuous.
	hue : str, optional
		Column used to split each bar by color, passed to
		:func:`seaborn.histplot`. It is never plotted as a feature itself.
		Defaults to ``None`` (no coloring).
	display_count : bool, default False
		If True, the number of rows in each x-axis category is written under
		its tick label (e.g. ``yes`` with ``n=34.6K`` below it). Counts of
		10,000 and above are shortened with K / M / B.
	ncols : int, optional
		Number of subplot columns. Defaults to ``None``: up to 6 panels are
		laid out 3 per row, more than 6 are laid out 4 per row. The figure is
		always 20 inches wide, split evenly between the columns, and each
		panel keeps a 4:3 aspect ratio.
	ax : matplotlib.axes.Axes or sequence of Axes, optional
		Existing axes to draw into, exactly one per panel, instead of creating
		a new figure (e.g. ``ax=ax1`` to sit next to your own subplots). Your
		figure keeps its own size and layout, and ``ncols`` is ignored; the
		returned ``fig`` is the figure those axes belong to. Defaults to
		``None``: a new figure is created.
	**kwargs
		Passed through to :func:`seaborn.histplot`. ``multiple="dodge"`` and
		``shrink=0.8`` are the defaults (bars for each ``hue`` level side by
		side, with a small gap between categories); override them or add
		others such as ``stat="percent"`` or ``palette=...``.

	Returns
	-------
	fig : matplotlib.figure.Figure
		The figure holding all the histograms, so you can save the whole grid
		(e.g. ``fig.savefig("hists.png")``).
	axes : numpy.ndarray or matplotlib.axes.Axes
		Flat array of the :class:`matplotlib.axes.Axes` used, so you can
		further tweak them. Any unused trailing axes are hidden. When
		``features`` is a single column name, this is that one Axes instead
		of an array.

	Raises
	------
	KeyError
		If ``hue`` or a name in ``features`` is not a column of ``df``.
	ValueError
		If there are no categorical columns to plot, or a single ``features``
		column is continuous or is also the ``hue``.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import hist_plots
	>>> fig, axes = hist_plots(sns.load_dataset("tips"), hue="sex")
	>>> fig, ax = hist_plots(sns.load_dataset("tips"), features="day")
	"""
	if hue is not None and hue not in df.columns:
		raise KeyError(f"hue {hue!r} is not a column in df")

	# A plain string that is not one of _select_features' keywords names a
	# single column: plot it alone and return its Axes, not an array.
	single = isinstance(features, str) and features not in (
		"all", "numeric", "categorical"
	)
	if single:
		if features not in df.columns:
			raise KeyError(f"feature {features!r} is not a column in df")
		if features == hue:
			raise ValueError(f"{features!r} cannot be both the feature and the hue")
		if (
			pd.api.types.is_numeric_dtype(df[features])
			and df[features].nunique() >= max_cardinality
		):
			raise ValueError(
				f"{features!r} is continuous ({df[features].nunique()} distinct "
				f"values, max_cardinality={max_cardinality}); bin it first, e.g. "
				"with categorize_numeric(), or raise max_cardinality."
			)
		selected = [features]
	else:
		selected = _select_features(
			df, features, exclude=[hue] if hue is not None else []
		)
	cat_cols = [
		col
		for col in selected
		if not pd.api.types.is_numeric_dtype(df[col])
		or df[col].nunique() < max_cardinality
	]

	n = len(cat_cols)
	if n == 0:
		raise ValueError("No categorical columns to plot.")

	own_fig = ax is None  # the loop below reuses the name ``ax``
	fig, axes = _subplot_grid(n, ncols, ax=ax)

	for ax, col in zip(axes, cat_cols):
		is_numeric = pd.api.types.is_numeric_dtype(df[col])
		col_kwargs = {"multiple": "dodge", "shrink": 0.8, **kwargs}
		if is_numeric:
			# One bar per distinct value instead of binning.
			col_kwargs.setdefault("discrete", True)

		sns.histplot(data=df, x=col, hue=hue, ax=ax, **col_kwargs)
		ax.set_title(col)
		ax.set_xlabel("")

		if display_count:
			used = [col] if hue is None else [col, hue]
			_add_counts_to_xticks(ax, df, col, used, numeric_axis=is_numeric)

		# Long category lists overlap; tilt their labels.
		if not is_numeric and df[col].nunique() > 5:
			ax.tick_params(axis="x", labelrotation=45)

	if own_fig:
		fig.tight_layout()
	if single:
		return fig, axes[0]
	return fig, axes


def distr_comp(
	df: pd.DataFrame,
	features,
	target: str,
	max_cardinality: int = 30,
	display_count: bool = False,
	ncols=None,
	ax=None,
	**kwargs,
):
	"""Compare the distribution of ``target`` across the groups of each feature.

	For every feature, the rows of ``df`` are grouped by that feature's values
	and, within each group, the share of every ``target`` category is
	computed (a row-normalised :func:`pandas.crosstab`, in percent). The
	result is drawn as a grouped bar chart: one cluster of bars per feature
	value, one bar per ``target`` category, and each cluster sums to 100 %.
	This makes groups of very different sizes directly comparable.

	Each feature gets its own subplot. ``target`` categories keep the same
	order and colors in every subplot. Rows where the feature or ``target``
	is missing are left out.

	Parameters
	----------
	df : pandas.DataFrame
		The data to plot.
	features : str or sequence of str
		Feature(s) to group rows by; one subplot per feature. A single
		column name returns a single Axes (see Returns).
	target : str
		Categorical column whose distribution is compared between groups.
	max_cardinality : int, default 30
		Largest number of distinct values a numeric feature may have and
		still be plotted; each distinct value becomes its own cluster of
		bars. Bin a continuous feature first (e.g. with
		:func:`categorize_numeric`) to compare it.
	display_count : bool, default False
		If True, the number of rows in each x-axis category is written under
		its tick label (e.g. ``yes`` with ``n=34.6K`` below it). Counts of
		10,000 and above are shortened with K / M / B.
	ncols : int, optional
		Number of subplot columns. Defaults to ``None``: up to 6 panels are
		laid out 3 per row, more than 6 are laid out 4 per row. The figure is
		always 20 inches wide, split evenly between the columns, and each
		panel keeps a 4:3 aspect ratio.
	ax : matplotlib.axes.Axes or sequence of Axes, optional
		Existing axes to draw into, exactly one per panel, instead of creating
		a new figure (e.g. ``ax=ax1`` to sit next to your own subplots). Your
		figure keeps its own size and layout, and ``ncols`` is ignored; the
		returned ``fig`` is the figure those axes belong to. Defaults to
		``None``: a new figure is created.
	**kwargs
		Passed through to :func:`seaborn.barplot`. ``edgecolor="black"`` and
		``linewidth=1`` are the defaults; override them or add others such as
		``palette=...``.

	Returns
	-------
	fig : matplotlib.figure.Figure
		The figure holding all the bar charts, so you can save the whole grid
		(e.g. ``fig.savefig("distr_comp.png")``).
	axes : numpy.ndarray or matplotlib.axes.Axes
		Flat array of the :class:`matplotlib.axes.Axes` used, so you can
		further tweak them. Any unused trailing axes are hidden. When
		``features`` is a single column name, this is that one Axes instead
		of an array.

	Raises
	------
	KeyError
		If ``target`` or a name in ``features`` is not a column of ``df``.
	TypeError
		If a numeric feature has more than ``max_cardinality`` distinct
		values.
	ValueError
		If ``features`` is empty or contains ``target``.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import distr_comp
	>>> tips = sns.load_dataset("tips")
	>>> fig, axes = distr_comp(tips, features=["day", "size"], target="time")
	>>> fig, ax = distr_comp(tips, features="day", target="time")
	"""
	single = isinstance(features, str)
	features = [features] if single else list(features)
	missing = [c for c in [*features, target] if c not in df.columns]
	if missing:
		raise KeyError(f"columns not in df: {missing}")
	if not features:
		raise ValueError("features must name at least one column")
	if target in features:
		raise ValueError(f"target {target!r} cannot also be a feature")

	# Checked up front so nothing is drawn if any feature is unusable.
	too_many = [
		f"{c!r} ({df[c].nunique()} values)"
		for c in features
		if pd.api.types.is_numeric_dtype(df[c])
		and df[c].nunique() > max_cardinality
	]
	if too_many:
		raise TypeError(
			"numeric features with more than "
			f"max_cardinality={max_cardinality} distinct values: "
			f"{', '.join(too_many)}. Bin them first, e.g. with "
			"categorize_numeric(), or raise max_cardinality."
		)

	# Same category order (and so the same colors) in every subplot.
	if isinstance(df[target].dtype, pd.CategoricalDtype):
		hue_order = list(df[target].cat.categories)
	else:
		hue_order = sorted(df[target].dropna().unique())

	bar_kwargs = {
		"edgecolor": "black",
		"linewidth": 1,
		"hue_order": hue_order,
		**kwargs,
	}

	n = len(features)
	own_fig = ax is None  # the loop below reuses the name ``ax``
	fig, axes = _subplot_grid(n, ncols, ax=ax)

	for ax, feature in zip(axes, features):
		pct = (
			pd.crosstab(df[feature], df[target], normalize="index")
			.mul(100)
			.reset_index()
			.melt(id_vars=feature, var_name=target, value_name="Percent")
		)
		sns.barplot(
			data=pct, x=feature, y="Percent", hue=target, ax=ax, **bar_kwargs
		)
		ax.set_title(f"{target} by {feature}")
		ax.set_ylabel("Percent (%)")

		if display_count:
			_add_counts_to_xticks(ax, df, feature, [feature, target])

	if own_fig:
		fig.tight_layout()
	if single:
		return fig, axes[0]
	return fig, axes


def _overlay_category_means(ax, data, x_var, y_var, half_width=0.5):
	"""Overlay a red dashed line at each x-category's mean of ``y_var``.

	Lines are aligned to seaborn's categorical tick positions (which sit at
	0, 1, 2, ... with the category names as tick labels), so they stay correct
	regardless of group order. Each line spans ``2 * half_width`` centered on
	its tick; ``half_width`` is sized to the box width plus a small margin so
	the line steps slightly past the box edges. Returns True if at least one
	line was drawn.
	"""
	means = data.groupby(x_var, observed=True)[y_var].mean()
	means_by_label = {str(idx): val for idx, val in means.items()}
	drew = False
	for pos, tick in zip(ax.get_xticks(), ax.get_xticklabels()):
		mean = means_by_label.get(tick.get_text())
		if mean is None or pd.isna(mean):
			continue
		ax.hlines(
			y=mean, xmin=pos - half_width, xmax=pos + half_width,
			color="red", linestyle="--", linewidth=2,
		)
		drew = True
	return drew


def plot_target(df: pd.DataFrame, target: str, max_cardinality: int = 10, show_mean: bool = False, features="all", display_count: bool = False, ncols=None, ax=None, **kwargs):
	"""Plot every feature against the target, one subplot per feature.

	The plot type is chosen per feature:

	* both target and feature continuous numeric -> :func:`seaborn.scatterplot`;
	* target with more than three distinct values, paired with a numeric
	  variable -> :func:`seaborn.boxplot` (the numeric variable on the value
	  axis, the other grouped along the category axis), which stays readable
	  where many overlaid histograms would not;
	* everything else (a low-cardinality target, or two categorical columns)
	  -> :func:`seaborn.histplot` colored by the target.

	"Continuous numeric" means a numeric column with at least
	``max_cardinality`` distinct values; a numeric column with fewer distinct
	values is treated as categorical.

	Parameters
	----------
	df : pandas.DataFrame
		The data to plot.
	target : str
		Column that every other column is plotted against.
	max_cardinality : int, default 10
		A numeric column with fewer than this many distinct values is treated
		as categorical rather than continuous.
	show_mean : bool, default False
		If True, overlay a red dashed line at each category's mean of the
		numeric variable on the boxplot panels (aligned to seaborn's category
		ticks), with a single legend entry for it. Scatter and histogram panels
		are left unchanged. The legend sits just above the figure, so save
		with ``fig.savefig(..., bbox_inches="tight")`` to keep it in the image
		(Jupyter's inline display already does this).
	features : {"all", "numeric", "categorical"}, str or sequence of str, default "all"
		Which columns to plot against the target:  
		``"all"`` every column except the target,  
		``"numeric"`` only numeric columns,  
		``"categorical"`` only categorical columns,  
		a single column name plots just that column and returns a single
		Axes (see Returns),  
		a sequence of names selects exactly those columns.
	display_count : bool, default False
		If True, the number of rows in each x-axis category is written under
		its tick label (e.g. ``yes`` with ``n=34.6K`` below it). Counts of
		10,000 and above are shortened with K / M / B.
		Panels with a continuous x-axis (scatter plots, histograms of a
		continuous variable) are left unchanged.
	ncols : int, optional
		Number of subplot columns. Defaults to ``None``: up to 6 panels are
		laid out 3 per row, more than 6 are laid out 4 per row. The figure is
		always 20 inches wide, split evenly between the columns, and each
		panel keeps a 4:3 aspect ratio.
	ax : matplotlib.axes.Axes or sequence of Axes, optional
		Existing axes to draw into, exactly one per panel, instead of creating
		a new figure (e.g. ``ax=ax1`` to sit next to your own subplots). Your
		figure keeps its own size and layout, and ``ncols`` is ignored; the
		returned ``fig`` is the figure those axes belong to. Defaults to
		``None``: a new figure is created.
	**kwargs
		Passed through to whichever seaborn function draws a given panel. The
		histogram panels additionally default to ``multiple="dodge"`` and
		``shrink=1.5``; pass those keys explicitly to override them.

	Returns
	-------
	fig : matplotlib.figure.Figure
		The figure holding all the panels, so you can save the whole grid
		(e.g. ``fig.savefig("target.png")``).
	axes : numpy.ndarray or matplotlib.axes.Axes
		Flat array of the :class:`matplotlib.axes.Axes` used. Any unused
		trailing axes are hidden. When ``features`` is a single column name,
		this is that one Axes instead of an array.

	Raises
	------
	KeyError
		If ``target`` or a name in ``features`` is not a column of ``df``.
	ValueError
		If ``df`` has no other columns to plot, or a single ``features``
		column is the ``target`` itself.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import plot_target
	>>> fig, axes = plot_target(sns.load_dataset("iris"), target="species")
	>>> fig, ax = plot_target(
	...     sns.load_dataset("iris"), target="species", features="sepal_length"
	... )
	"""
	if target not in df.columns:
		raise KeyError(f"target {target!r} is not a column in df")

	# A plain string that is not one of _select_features' keywords names a
	# single column: plot it alone and return its Axes, not an array.
	single = isinstance(features, str) and features not in (
		"all", "numeric", "categorical"
	)
	if single:
		if features not in df.columns:
			raise KeyError(f"feature {features!r} is not a column in df")
		if features == target:
			raise ValueError(f"{features!r} cannot be both the feature and the target")
		features = [features]
	else:
		features = _select_features(df, features, exclude=[target])
	if not features:
		raise ValueError("df has no other columns to plot against the target.")

	def is_continuous(col: str) -> bool:
		return (
			pd.api.types.is_numeric_dtype(df[col])
			and df[col].nunique() >= max_cardinality
		)

	target_continuous = is_continuous(target)
	target_many_classes = df[target].nunique() > 3
	hist_kwargs = {"multiple": "dodge", "shrink": 1.5, **kwargs}
	# Half the box width (seaborn default 0.8) plus an eighth, so the
	# mean line steps proportionally past the box edges.
	_box_width = kwargs.get("width", 0.8)
	mean_half_width = _box_width / 2 + _box_width / 8

	n = len(features)
	own_fig = ax is None  # the loop below reuses the name ``ax``
	fig, axes = _subplot_grid(n, ncols, ax=ax)

	drew_means = False
	for ax, col in zip(axes, features):
		feature_numeric = pd.api.types.is_numeric_dtype(df[col])

		if target_continuous and is_continuous(col):
			sns.scatterplot(data=df, x=col, y=target, ax=ax, **kwargs)
		elif target_continuous:
			# Numeric target, categorical feature: target distribution per group.
			if target_many_classes:  # always true for a continuous target
				sns.boxplot(data=df, x=col, y=target, ax=ax, **kwargs)
				if show_mean:
					drew_means |= _overlay_category_means(ax, df, col, target, mean_half_width)
				if display_count:
					_add_counts_to_xticks(ax, df, col, [col, target])
			else:
				sns.histplot(data=df, x=target, hue=col, ax=ax, **hist_kwargs)
		elif target_many_classes and feature_numeric:
			# Categorical target with many classes, numeric feature.
			sns.boxplot(data=df, x=target, y=col, ax=ax, **kwargs)
			if show_mean:
				drew_means |= _overlay_category_means(ax, df, target, col, mean_half_width)
			if display_count:
				_add_counts_to_xticks(ax, df, target, [col, target])
		else:
			# Few target classes, or two categorical columns.
			col_hist_kwargs = hist_kwargs
			if feature_numeric and not is_continuous(col):
				# Low-cardinality numeric feature: one bar group centred on
				# each value (like hist_plots) instead of automatic bins.
				# shrink 1.5 would overlap neighbouring values, so it drops
				# to 0.8 unless you passed your own.
				col_hist_kwargs = {"discrete": True, **hist_kwargs}
				if "shrink" not in kwargs:
					col_hist_kwargs["shrink"] = 0.8
			sns.histplot(data=df, x=col, hue=target, ax=ax, **col_hist_kwargs)
			if display_count and not is_continuous(col):
				_add_counts_to_xticks(
					ax, df, col, [col, target], numeric_axis=feature_numeric
				)

		ax.set_title(f"{col} vs {target}")

	if own_fig:
		fig.tight_layout()

	if show_mean and drew_means:
		fig.legend(
			handles=[
				Line2D([0], [0], color="red", linestyle="--", linewidth=2,
					   label="Mean value within a category")
			],
			# Pin the legend's bottom edge to the figure's top edge so it
			# sits above the grid instead of covering the middle title.
			loc="lower center",
			bbox_to_anchor=(0.5, 1.0),
			frameon=False,
		)
	if single:
		return fig, axes[0]
	return fig, axes
