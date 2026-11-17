"""Progress-bin summaries of subject-level optimal choices for PPC."""
import numpy as np


def optimal_by_bin(actions, best, eligible, bins=8):
    """Split all trials in order, then exclude tied rewards within each bin."""
    cuts = np.linspace(0, len(actions), bins + 1, dtype=int)
    counts = np.diff(np.r_[0, np.cumsum(eligible)][cuts])
    hits = np.diff(np.r_[0, np.cumsum((actions == best) & eligible)][cuts])
    return np.divide(hits, counts, out=np.full(bins, np.nan), where=counts > 0)


def timecourse_rows(load, observed, replicated, summarize):
    # Average participants within each replicate before taking predictive quantiles.
    means = np.nanmean(replicated, axis=0)
    return [
        {"load": load, "metric": "optimal_rate", "bin": i + 1,
         **summarize(means[:, i], obs)}
        for i, obs in enumerate(np.nanmean(observed, axis=0))
    ]
