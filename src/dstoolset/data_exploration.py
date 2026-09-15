"""Data-exploration diagnostics for exploratory data analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def _r_squared(feature: pd.Series, target: pd.Series) -> float:
    """Fraction of target variance explained by a numeric feature (Pearson r**2)."""
    d = pd.DataFrame({"f": feature, "y": target}).dropna()
    if d["f"].nunique() < 2 or d["y"].nunique() < 2:
        return 0.0
    r = d["f"].corr(d["y"])
    return float(r ** 2) if pd.notna(r) else 0.0


def _eta_squared(feature: pd.Series, target: pd.Series) -> float:
    """Fraction of target variance explained by a categorical feature (eta**2)."""
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
            score = _eta_squared(df[col], df[target])
        else:
            score = _r_squared(df[col], df[target])
        scores.append((col, score))

    scores.sort(key=lambda pair: pair[1], reverse=not ascending)
    return scores
