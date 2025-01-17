"""
Labeling techniques used in financial machine learning.
"""

from famlafl.labeling.labeling import (add_vertical_barrier, apply_pt_sl_on_t1, barrier_touched, drop_labels,
                                        get_bins, get_events)
from famlafl.labeling.trend_scanning import trend_scanning_labels
from famlafl.labeling.tail_sets import TailSetLabels
from famlafl.labeling.fixed_time_horizon import fixed_time_horizon
from famlafl.labeling.matrix_flags import MatrixFlagLabels
from famlafl.labeling.excess_over_median import excess_over_median
from famlafl.labeling.raw_return import raw_return
from famlafl.labeling.return_vs_benchmark import return_over_benchmark
from famlafl.labeling.excess_over_mean import excess_over_mean
