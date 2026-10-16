"""Data-exploration diagnostics for exploratory data analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def get_outliers_rows(
	df: pd.DataFrame,
	cols="all",
	threshold: float = 1.5,
	side: str = "both",
) -> pd.Series:
	"""Flag rows that contain an IQR outlier in any of the given columns.

	A value is considered an outlier if it falls outside  
	``[Q1 - threshold * IQR, Q3 + threshold * IQR]``.

	Parameters
	----------
	df : pandas.DataFrame
		The data to inspect.
	cols : str or sequence of str, default "all"
		Columns to check:  
		``"all"`` selects all numeric columns,  
		a single column name selects just that column,  
		a list of names selects those columns.
	threshold : float, default 1.5
		IQR multiplier defining the outlier fences. Larger is more lenient.
	side : {"both", "low", "high"}, default "both"
		Which tail counts as an outlier:  
		``"both"`` - values outside the fences,  
		``"low"`` - values below ``Q1 - threshold * IQR``,  
		``"high"`` - values above ``Q3 + threshold * IQR``.

	Returns
	-------
	pandas.Series
		Boolean Series aligned to ``df.index``, ``True`` where a row has an
		outlier in any of the given columns.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import get_outliers_rows
	>>> iris = sns.load_dataset("iris")
	>>> mask = get_outliers_rows(iris)
	>>> iris[mask]
	"""
	if isinstance(cols, str):
		cols = (
			df.select_dtypes(include="number").columns
			if cols == "all"
			else [cols]
		)

	if side not in ("both", "low", "high"):
		raise ValueError(f"side must be 'both', 'low', or 'high', got {side!r}")

	sub = df[cols]
	q1 = sub.quantile(0.25)
	q3 = sub.quantile(0.75)
	iqr = q3 - q1
	lower_bound = q1 - threshold * iqr
	upper_bound = q3 + threshold * iqr

	if side == "both":
		is_outlier = (sub < lower_bound) | (sub > upper_bound)
	elif side == "low":
		is_outlier = sub < lower_bound
	else:  # "high"
		is_outlier = sub > upper_bound

	return is_outlier.any(axis="columns")


def _is_categorical(series: pd.Series, min_cardinality: int) -> bool:
	"""A column is categorical if it is non-numeric, or numeric with few values."""
	return not (
		pd.api.types.is_numeric_dtype(series)
		and series.nunique() > min_cardinality
	)


def r_squared(feature: pd.Series, target: pd.Series) -> float:
	"""Fraction of a numeric target's variance explained by a numeric feature.

	Computed as the squared Pearson correlation between ``feature`` and
	``target`` -- equivalently the R-squared of a simple linear regression of
	the target on the feature. Rows where either value is missing are dropped.
	Ranges from 0 (no linear association) to 1 (a perfect linear fit); note it
	only captures *linear* association.

	Parameters
	----------
	feature : pandas.Series
		Numeric feature.
	target : pandas.Series
		Numeric target whose variance is being explained.

	Returns
	-------
	float
		R-squared in ``[0, 1]``; ``0.0`` when either column is constant or has
		fewer than two non-missing paired values.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import r_squared
	>>> tips = sns.load_dataset("tips")
	>>> r_squared(tips["total_bill"], tips["tip"])
	"""
	d = pd.DataFrame({"f": feature, "y": target}).dropna()
	if d["f"].nunique() < 2 or d["y"].nunique() < 2:
		return 0.0
	r = d["f"].corr(d["y"])
	return float(r ** 2) if pd.notna(r) else 0.0


def eta_squared(feature: pd.Series, target: pd.Series) -> float:
	"""Fraction of a numeric target's variance explained by a categorical feature.

	Computed as the between-group sum of squares over the total sum of squares
	(eta-squared), grouping the target by the feature's categories. Rows where
	either value is missing are dropped. Ranges from 0 (all group means equal)
	to 1 (all variance lies between groups).

	Parameters
	----------
	feature : pandas.Series
		Categorical (grouping) feature.
	target : pandas.Series
		Numeric target whose variance is being explained.

	Returns
	-------
	float
		Eta-squared in ``[0, 1]``; ``0.0`` when the target has no variance.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import eta_squared
	>>> tips = sns.load_dataset("tips")
	>>> eta_squared(tips["day"], tips["tip"])
	"""
	d = pd.DataFrame({"f": feature, "y": target}).dropna()
	grand_mean = d["y"].mean()
	ss_total = ((d["y"] - grand_mean) ** 2).sum()
	if ss_total == 0:
		return 0.0
	group_means = d.groupby("f", observed=True)["y"]
	ss_between = (group_means.count() * (group_means.mean() - grand_mean) ** 2).sum()
	return float(ss_between / ss_total)


def variance_ranking(
	df: pd.DataFrame,
	target: str,
	features="all",
	min_cardinality: int = 5,
	ascending: bool = False,
) -> "list[tuple[str, float]]":
	"""Rank features by how much of the target's variance each one explains.

	Each feature is scored on a comparable 0-1 scale: a numeric feature gets
	its R-squared (the squared Pearson correlation with the target), 
	and a categorical feature gets its eta-squared.  
	Because both are a fraction of target variance, numeric and categorical features 
	can be ranked together.

	The target must be numeric.

	Parameters
	----------
	df : pandas.DataFrame
		The data to inspect.
	target : str
		Numeric target column whose variance is decomposed.
	features : {"all", "categorical", "numeric"} or sequence of str, default "all"
		Which features to score:  
		``"all"`` uses every column except the target,	
		``"categorical"`` only categorical features,  
		``"numeric"`` only numeric features (except the target),  
		a sequence of names scores exactly those columns.  
		
		A column counts as categorical when it is non-numeric or
		numeric with at most ``min_cardinality`` distinct values (so
		``"numeric"`` means numeric with more than ``min_cardinality``
		distinct values).
	min_cardinality : int, default 5
		A numeric column with at most this many distinct values is treated as
		categorical -- both for selection and for which metric it is scored
		with (eta-squared rather than R-squared).
	ascending : bool, default False
		Sort order:  
		``False`` puts the most informative features first,  
		``True`` puts the least informative first.

	Returns
	-------
	list of (str, float)
		``(feature_name, variance_explained)`` tuples ordered by
		``variance_explained``.

	Raises
	------
	KeyError
		If ``target`` or any named feature is not a column of ``df``.
	TypeError
		If ``target`` is not numeric.
	ValueError
		If ``features`` is a string other than ``"all"``, ``"categorical"``
		or ``"numeric"``.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import variance_ranking
	>>> tips = sns.load_dataset("tips")
	>>> variance_ranking(tips, target="tip")
	"""
	if target not in df.columns:
		raise KeyError(f"target {target!r} is not a column in df")
	if not pd.api.types.is_numeric_dtype(df[target]):
		raise TypeError(
			f"target {target!r} must be numeric to decompose its variance"
		)

	if isinstance(features, str):
		if features == "all":
			selected = [c for c in df.columns if c != target]
		elif features == "categorical":
			selected = [
				c for c in df.columns
				if c != target and _is_categorical(df[c], min_cardinality)
			]
		elif features == "numeric":
			selected = [
				c for c in df.columns
				if c != target and not _is_categorical(df[c], min_cardinality)
			]
		else:
			raise ValueError(
				"features string must be 'all', 'categorical', or 'numeric'; "
				f"got {features!r} (pass a list to name explicit columns)"
			)
	else:
		selected = list(features)
		missing = [c for c in selected if c not in df.columns]
		if missing:
			raise KeyError(f"features not in df: {missing}")
		selected = [c for c in selected if c != target]

	scores = []
	for col in selected:
		if _is_categorical(df[col], min_cardinality):
			score = eta_squared(df[col], df[target])
		else:
			score = r_squared(df[col], df[target])
		scores.append((col, score))

	scores.sort(key=lambda pair: pair[1], reverse=not ascending)
	return scores


def get_groups(df: pd.DataFrame, by, columns, max_cardinality: int = 15):
	"""Split one or more columns into per-group arrays, grouped by ``by``.

	Groups the rows of ``df`` by the ``by`` column and, for each requested
	column, collects that column's values within each group -- handy for
	feeding grouped samples into per-group statistics or tests.

	A numeric ``by`` column is treated as categorical only when its cardinality
	is small (at most ``max_cardinality`` distinct values), in which case it is
	cast to string so it groups cleanly; above that a :class:`TypeError` is
	raised, since grouping a high-cardinality numeric column is rarely intended.

	Parameters
	----------
	df : pandas.DataFrame
		The data to group.
	by : str
		Column to group by.
	columns : str or sequence of str
		Column(s) whose values are collected per group; a single name or a
		list of names.
	max_cardinality : int, default 15
		Largest number of distinct values a numeric ``by`` column may have and
		still be grouped; above it a TypeError is raised.

	Returns
	-------
	groups : list of numpy.ndarray, or dict of {str: list of numpy.ndarray}
		The per-group value arrays (one 1-D array per group of ``by``). When
		``columns`` is a single column name the result is just that list; when
		it is a list of names the result is a dict keyed by column name, each
		value being that column's list of per-group arrays.
	labels : list, or dict of {str: list}
		The group labels, parallel to ``groups``. A plain list for a single
		column, or a dict keyed by column name (same label order for every
		column) for several.

	Raises
	------
	TypeError
		If ``by`` is numeric with more than ``max_cardinality`` distinct values.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import get_groups
	>>> tips = sns.load_dataset("tips")
	>>> groups, labels = get_groups(tips, by="day", columns="tip")
	>>> labels
	['Thur', 'Fri', 'Sat', 'Sun']
	"""
	data = df.copy()
	if pd.api.types.is_numeric_dtype(data[by]):
		if data[by].nunique() > max_cardinality:
			raise TypeError(f"{by!r} column is numeric with high cardinality")
		data[by] = data[by].astype("string")

	single = isinstance(columns, str)
	cols = [columns] if single else list(columns)

	groups = {c: [] for c in cols}
	labels = {c: [] for c in cols}
	for group_name, sub_df in data.groupby(by, observed=True):
		for c in cols:
			groups[c].append(sub_df[c].values)
			labels[c].append(group_name)

	if single:
		return groups[columns], labels[columns]
	return groups, labels


def multiple_ANOVA(df: pd.DataFrame, by, target: str, max_cardinality: int = 15):
	"""Run a one-way ANOVA of ``target`` across the groups of each ``by`` feature.

	For every feature in ``by``, the rows of ``df`` are split into groups by
	that feature's values (with :func:`get_groups`) and
	:func:`scipy.stats.f_oneway` tests whether the mean of ``target`` is the
	same in all groups. A small p-value suggests at least one group mean
	differs. The ``target`` is the same for every feature tested.

	Rows where the feature or ``target`` is missing are left out of that
	feature's test.

	Parameters
	----------
	df : pandas.DataFrame
		The data to test.
	by : str or sequence of str
		Grouping feature(s): a single column name, or a list of names to run
		one ANOVA per feature. A numeric feature is grouped only if it has at
		most ``max_cardinality`` distinct values.
	target : str
		Numeric column whose group means are compared.
	max_cardinality : int, default 15
		Largest number of distinct values a numeric ``by`` feature may have
		and still be grouped (passed to :func:`get_groups`).

	Returns
	-------
	groups : list of numpy.ndarray, or dict of {str: list of numpy.ndarray}
		The ``target`` values of each group, as passed to the test. A plain
		list when ``by`` is a single name; a dict keyed by feature name when
		it is a list.
	summary : str
		One block per feature, in the order given::

			ANOVA results for <feature>:
			Statistic = <F>; p-value = <p>

		with the blocks separated by a blank line (print it to read it).

	Raises
	------
	KeyError
		If ``target`` or a name in ``by`` is not a column of ``df``.
	TypeError
		If ``target`` is not numeric, or a numeric ``by`` feature has more
		than ``max_cardinality`` distinct values.
	ValueError
		If a ``by`` feature has fewer than two groups to compare.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import multiple_ANOVA
	>>> tips = sns.load_dataset("tips")
	>>> groups, summary = multiple_ANOVA(tips, by=["day", "time"], target="tip")
	>>> print(summary)
	"""
	single = isinstance(by, str)
	features = [by] if single else list(by)

	missing = [c for c in [*features, target] if c not in df.columns]
	if missing:
		raise KeyError(f"columns not in df: {missing}")
	if not pd.api.types.is_numeric_dtype(df[target]):
		raise TypeError(f"target {target!r} must be numeric, got dtype {df[target].dtype}")

	groups = {}
	summary = ""
	for feature in features:
		data = df[[feature, target]].dropna()
		feature_groups, _ = get_groups(
			data, by=feature, columns=target, max_cardinality=max_cardinality
		)
		if len(feature_groups) < 2:
			raise ValueError(
				f"{feature!r} has {len(feature_groups)} group(s); "
				"ANOVA needs at least two"
			)

		F, p = stats.f_oneway(*feature_groups)
		groups[feature] = feature_groups
		summary += (
			f"ANOVA results for {feature}:\n"
			f"Statistic = {F:.4f}; p-value = {p:.4g}\n\n"
		)

	if single:
		return groups[by], summary
	return groups, summary


def ordinal_association(
	df: pd.DataFrame,
	target: str,
	features,
	max_cardinality: int = 15,
) -> pd.DataFrame:
	"""Test the association between ordinal features and a numeric target.

	For every feature, three complementary measures are computed against
	``target``:

	* Spearman's rank correlation (:func:`scipy.stats.spearmanr`) -- the
	  strength and direction of a *monotonic* trend;
	* the Kruskal-Wallis H test (:func:`scipy.stats.kruskal`) -- whether the
	  target's distribution differs across the feature's levels, without
	  assuming a trend or normality;
	* eta-squared (:func:`eta_squared`) -- the fraction of the target's
	  variance explained by the feature's levels.

	Rows where the feature or ``target`` is missing are left out of that
	feature's tests.

	Parameters
	----------
	df : pandas.DataFrame
		The data to test.
	target : str
		Numeric column tested against every feature.
	features : str or sequence of str
		Ordinal feature(s): a single column name or a list of names. Each must
		be numeric (e.g. integer-coded levels) or an ordered categorical
		(e.g. the output of :func:`categorize_numeric`), whose category order
		is used for the ranks. A numeric feature is grouped only if it has at
		most ``max_cardinality`` distinct values.
	max_cardinality : int, default 15
		Largest number of distinct values a numeric feature may have and
		still be grouped for the Kruskal-Wallis test (passed to
		:func:`get_groups`).

	Returns
	-------
	pandas.DataFrame
		One row per feature, sorted by ``abs(spearman_rho)`` (strongest
		first), with columns:  
		``feature`` - the feature name,  
		``spearman_rho`` - Spearman's rho (3 decimals),  
		``p_spearman`` - its p-value,  
		``kruskal_H`` - the Kruskal-Wallis H statistic (2 decimals),  
		``p_kruskal`` - its p-value,  
		``eta2`` - eta-squared (3 decimals).

	Raises
	------
	KeyError
		If ``target`` or a feature is not a column of ``df``.
	TypeError
		If ``target`` is not numeric, a feature is neither numeric nor an
		ordered categorical, or a numeric feature has more than
		``max_cardinality`` distinct values.
	ValueError
		If ``target`` is also listed as a feature, or a feature has fewer
		than two levels to compare.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import ordinal_association
	>>> tips = sns.load_dataset("tips")
	>>> ordinal_association(tips, target="tip", features=["size"])
	"""
	features = [features] if isinstance(features, str) else list(features)

	missing = [c for c in [*features, target] if c not in df.columns]
	if missing:
		raise KeyError(f"columns not in df: {missing}")
	if not pd.api.types.is_numeric_dtype(df[target]):
		raise TypeError(f"target {target!r} must be numeric, got dtype {df[target].dtype}")
	if target in features:
		raise ValueError(f"target {target!r} cannot also be a feature")

	rows = []
	for col in features:
		data = df[[col, target]].dropna()
		levels = data[col]
		if isinstance(levels.dtype, pd.CategoricalDtype) and levels.cat.ordered:
			ranks = levels.cat.codes
		elif pd.api.types.is_numeric_dtype(levels):
			ranks = levels
		else:
			raise TypeError(
				f"feature {col!r} must be numeric or an ordered categorical, "
				f"got dtype {levels.dtype}"
			)

		groups, _ = get_groups(
			data, by=col, columns=target, max_cardinality=max_cardinality
		)
		if len(groups) < 2:
			raise ValueError(
				f"{col!r} has {len(groups)} level(s); at least two are needed"
			)

		rho, p_rho = stats.spearmanr(ranks, data[target])
		H, p_kw = stats.kruskal(*groups)
		rows.append({
			"feature": col,
			"spearman_rho": round(rho, 3),
			"p_spearman": p_rho,
			"kruskal_H": round(H, 2),
			"p_kruskal": p_kw,
			"eta2": round(eta_squared(levels, data[target]), 3),
		})

	return (
		pd.DataFrame(rows)
		.sort_values("spearman_rho", key=abs, ascending=False)
		.reset_index(drop=True)
	)


def categorize_numeric(
	values,
	binning="quantile",
	splits: int = 4,
	labels=None,
) -> pd.Series:
	"""Bin a numeric feature into ordered categories.

	The range of ``values`` is split into consecutive bins, and each value is
	replaced by the label of the bin it falls into. Bins are numbered from
	the lowest values up. How the bin edges are placed depends on
	``binning``:

	* ``"quantile"`` -- edges sit at the quantiles of the data, so each bin
	  holds roughly the same number of values (equal-frequency bands, via
	  :func:`pandas.qcut`). With ``splits=4`` the edges are the minimum, the
	  25th, 50th and 75th percentiles and the maximum. If many values repeat,
	  some edges coincide; duplicates are dropped, giving fewer bins.
	* ``"width"`` -- the range from minimum to maximum is divided into
	  ``splits`` bins of equal width (equal-width bands, via
	  :func:`pandas.cut`), so the number of values per bin follows the
	  shape of the distribution and bins can be sparse or empty.
	* a list of edges, e.g. ``[0, 5, 10, 20]`` -- you choose the edges
	  yourself (``splits`` is ignored). Bins are left-closed, and the last
	  one also includes its upper edge: ``[0, 5)``, ``[5, 10)``, ``[10, 20]``.
	  Values outside ``[first edge, last edge]`` are not assigned to any bin
	  and become missing.

	With ``"quantile"`` and ``"width"``, bins are right-closed intervals
	``(lower, upper]`` and the lowest bin also includes the minimum.

	Missing values are not assigned to any bin and stay missing.

	Parameters
	----------
	values : pandas.Series or array-like
		Numeric values to bin. A Series keeps its index and name.
	binning : {"quantile", "width"} or sequence of numbers, default "quantile"
		How the values are binned:  
		``"quantile"`` uses :func:`pandas.qcut` (equal-frequency bands),  
		``"width"`` uses :func:`pandas.cut` (equal-width bands),  
		a sequence of at least two strictly increasing edges, e.g.
		``[0, 5, 10, 20]``, gives bins ``[0, 5)``, ``[5, 10)``, ``[10, 20]``.
	splits : int, default 4
		Number of bins. With ``"quantile"`` binning, duplicate bin edges are
		dropped, so heavily repeated values can produce fewer bins. Ignored
		when ``binning`` is a list of edges (there are ``len(binning) - 1``
		bins).
	labels : None, "intervals" or list of str, default None
		How the bins are labelled:  
		``None`` numbers them ``0, 1, ..., n_bins - 1``,  
		``"intervals"`` labels each bin with its value range, e.g.
		``(10.5, 17.2]``, or ``[0, 5)`` for custom edges (handy for plot
		legends and quick reading),  
		a list of names labels them lowest first and must have one entry
		per bin.

	Returns
	-------
	pandas.Series
		The bin label of each value, aligned with ``values``:  
		``labels=None`` -> integer codes (nullable ``Int64``),  
		``labels="intervals"`` -> an ordered categorical of
		:class:`pandas.Interval` objects, or of strings such as ``"[0, 5)"``
		for custom edges (the last bin ``"[10, 20]"`` is closed on both
		sides, which a single set of pandas Intervals cannot express),  
		a list -> an ordered categorical of those names.  
		Missing values stay missing.

	Raises
	------
	TypeError
		If ``values`` is not numeric.
	ValueError
		If ``binning`` is not ``"quantile"``, ``"width"`` or a sequence of at
		least two strictly increasing edges, ``splits`` is less than 1,
		``labels`` is a string other than ``"intervals"``, or a ``labels``
		list does not have one entry per bin.

	Examples
	--------
	>>> import seaborn as sns
	>>> from dstoolset import categorize_numeric
	>>> tips = sns.load_dataset("tips")
	>>> tips["bill_band"] = categorize_numeric(
	...     tips["total_bill"], labels=["low", "mid", "high", "top"]
	... )
	>>> tips["bill_range"] = categorize_numeric(
	...     tips["total_bill"], binning="width", splits=3, labels="intervals"
	... )
	>>> tips["bill_bracket"] = categorize_numeric(
	...     tips["total_bill"], binning=[0, 10, 20, 60], labels="intervals"
	... )
	"""
	series = values if isinstance(values, pd.Series) else pd.Series(values)
	if not pd.api.types.is_numeric_dtype(series):
		raise TypeError(f"values must be numeric, got dtype {series.dtype}")
	if splits < 1:
		raise ValueError(f"splits must be at least 1, got {splits}")

	if isinstance(labels, str) and labels != "intervals":
		raise ValueError(
			f"labels must be None, 'intervals' or a list, got {labels!r}"
		)
	# pandas labels bins with their intervals for None; False gives codes.
	intervals = isinstance(labels, str)
	pandas_labels = None if intervals else False

	custom = not isinstance(binning, str)
	if custom:
		try:
			edges = np.asarray(list(binning), dtype=float)
		except (TypeError, ValueError):
			edges = None
		if (
			edges is None
			or edges.ndim != 1
			or len(edges) < 2
			or np.any(np.diff(edges) <= 0)
		):
			raise ValueError(
				"binning as a list must hold at least two strictly increasing "
				f"edges, got {binning!r}"
			)
		# Left-closed bins [a, b); pandas would leave the top edge out, so
		# values equal to it are put in the last bin: [.., last].
		codes = pd.cut(series, bins=edges, right=False, labels=False)
		codes = codes.mask(series == edges[-1], len(edges) - 2)
		if intervals:
			labels = [f"[{lo:g}, {hi:g})" for lo, hi in zip(edges[:-2], edges[1:-1])]
			labels.append(f"[{edges[-2]:g}, {edges[-1]:g}]")
			intervals = False
	elif binning == "quantile":
		codes, edges = pd.qcut(
			series, q=splits, labels=pandas_labels, retbins=True, duplicates="drop"
		)
	elif binning == "width":
		codes, edges = pd.cut(
			series, bins=splits, labels=pandas_labels, retbins=True
		)
	else:
		raise ValueError(
			"binning must be 'quantile', 'width' or a list of edges, "
			f"got {binning!r}"
		)

	if intervals:
		return codes

	if labels is None:
		return codes.astype("Int64")

	n_bins = len(edges) - 1
	if len(labels) != n_bins:
		hint = (
			f" ({splits - n_bins} duplicate quantile edge(s) were dropped)"
			if not custom and n_bins < splits
			else ""
		)
		raise ValueError(
			f"labels has {len(labels)} entries but there are {n_bins} bins{hint}"
		)

	categories = pd.Categorical.from_codes(
		codes.fillna(-1).astype(int), categories=labels, ordered=True
	)
	return pd.Series(categories, index=series.index, name=series.name)
