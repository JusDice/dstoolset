"""Plotting helpers for exploratory data analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pair_plots(
    df: pd.DataFrame,
    target=None,
    splits: int = 4,
    binning: str = "quantile",
    **kwargs,
) -> list:
    """Scatter every numeric feature against every other, one figure per feature.

    Unlike a chunked :func:`seaborn.pairplot`, this covers *all* pairs: each
    numeric feature gets its own figure holding one scatter panel per other
    feature (that feature on the y-axis, the other on the x-axis), so no pair
    is ever dropped. Univariate distribution panels are not drawn. When a
    ``target`` is given it colors every panel as the ``hue``; a numeric target
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
        binned into ``splits`` categories first; a categorical target is used
        directly. Defaults to ``None`` (no coloring).
    splits : int, default 4
        Number of bins to cut a numeric target into.
    binning : {"quantile", "width"}, default "quantile"
        How a numeric target is binned:  
        ``"quantile"`` uses :func:`pandas.qcut` (equal-frequency bands),  
        ``"width"`` uses :func:`pandas.cut` (equal-width bands).
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
        If ``binning`` is not ``"quantile"`` or ``"width"``, or there are
        fewer than two numeric features to pair.

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
            if binning == "quantile":
                numeric_data[target] = pd.qcut(
                    numeric_data[target], q=splits, duplicates="drop"
                )
            elif binning == "width":
                numeric_data[target] = pd.cut(numeric_data[target], bins=splits)
            else:
                raise ValueError(
                    f"binning must be 'quantile' or 'width', got {binning!r}"
                )
            kwargs.setdefault("palette", "viridis")  # sequential colors
        else:
            # Categorical target: attach it so it can be used as hue.
            numeric_data[target] = data[target]

    features = [c for c in numeric_data.columns if c != target]
    if len(features) < 2:
        raise ValueError(
            "need at least two numeric features to pair, "
            f"got {len(features)}"
        )

    ncols = min(len(features), 6)

    plots = []
    for feature in features:
        others = [c for c in features if c != feature]
        nrows = int(np.ceil(len(others) / ncols))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 3 * nrows)
        )
        axes = np.atleast_1d(axes).reshape(-1)

        for ax, other in zip(axes, others):
            sns.scatterplot(
                data=numeric_data, x=other, y=feature, hue=hue, ax=ax, **kwargs
            )

        # Hide any leftover empty subplots.
        for ax in axes[len(others):]:
            ax.set_visible(False)

        fig.suptitle(f"{feature} vs all other features")
        fig.tight_layout()
        plots.append((fig, axes))

    return plots


def box_plot_numeric(df: pd.DataFrame):
    """Draw a box plot for every numeric feature, each on its own subplot.

    Numeric columns are laid out on a grid (up to four per row). Handy for
    scanning spread and outliers across all numeric features at once.

    Parameters
    ----------
    df : pandas.DataFrame
        The data to plot. Only numeric columns are used.

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
        If ``df`` has no numeric columns.

    Examples
    --------
    >>> import seaborn as sns
    >>> from dstoolset import box_plot_numeric
    >>> fig, axes = box_plot_numeric(sns.load_dataset("iris"))
    """
    df_num = df.select_dtypes(include="number")

    n = df_num.shape[1]
    if n == 0:
        raise ValueError("No numeric columns to plot.")

    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, col in zip(axes, df_num.columns):
        sns.boxplot(x=df_num[col], ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")

    # Hide any leftover empty subplots.
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes


def plot_target(df: pd.DataFrame, target: str, max_cardinality: int = 10, **kwargs):
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
    **kwargs
        Passed through to whichever seaborn function draws a given panel. The
        histogram panels additionally default to ``multiple="dodge"`` and
        ``shrink=1.5``; pass those keys explicitly to override them.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure holding all the panels, so you can save the whole grid
        (e.g. ``fig.savefig("target.png")``).
    axes : numpy.ndarray
        Flat array of the :class:`matplotlib.axes.Axes` used. Any unused
        trailing axes are hidden.

    Raises
    ------
    KeyError
        If ``target`` is not a column of ``df``.
    ValueError
        If ``df`` has no other columns to plot.

    Examples
    --------
    >>> import seaborn as sns
    >>> from dstoolset import plot_target
    >>> fig, axes = plot_target(sns.load_dataset("iris"), target="species")
    """
    if target not in df.columns:
        raise KeyError(f"target {target!r} is not a column in df")

    features = [c for c in df.columns if c != target]
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

    n = len(features)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, col in zip(axes, features):
        feature_numeric = pd.api.types.is_numeric_dtype(df[col])

        if target_continuous and is_continuous(col):
            sns.scatterplot(data=df, x=col, y=target, ax=ax, **kwargs)
        elif target_continuous:
            # Numeric target, categorical feature: target distribution per group.
            if target_many_classes:  # always true for a continuous target
                sns.boxplot(data=df, x=col, y=target, ax=ax, **kwargs)
            else:
                sns.histplot(data=df, x=target, hue=col, ax=ax, **hist_kwargs)
        elif target_many_classes and feature_numeric:
            # Categorical target with many classes, numeric feature.
            sns.boxplot(data=df, x=target, y=col, ax=ax, **kwargs)
        else:
            # Few target classes, or two categorical columns.
            sns.histplot(data=df, x=col, hue=target, ax=ax, **hist_kwargs)

        ax.set_title(f"{col} vs {target}")

    # Hide any leftover empty subplots.
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    return fig, axes
