"""dstoolset — reusable exploratory data analysis (EDA) helpers.

The public API is re-exported here so the internal module layout stays free to
change without breaking your imports. Import from the top level, e.g.::

	from dstoolset import pair_plots

or grab the whole namespace::

	import dstoolset as ds
	ds.pair_plots(df)
"""

from .plots import (
	pair_plots,
	box_plot_numeric,
	hist_plots,
	plot_target,
	distr_comp,
)
from .data_exploration import (
	get_outliers_rows,
	variance_ranking,
	r_squared,
	eta_squared,
	get_groups,
	multiple_ANOVA,
	ordinal_association,
	categorize_numeric,
)

__version__ = "0.2.0"

# Public surface. Keep this list in sync as you add helpers.
__all__ = [
	"pair_plots",
	"get_outliers_rows",
	"box_plot_numeric",
	"hist_plots",
	"plot_target",
	"distr_comp",
	"variance_ranking",
	"r_squared",
	"eta_squared",
	"get_groups",
	"multiple_ANOVA",
	"ordinal_association",
	"categorize_numeric",
]
