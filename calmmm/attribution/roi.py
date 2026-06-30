from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from calmmm.attribution.contributions import marginal_contributions

if TYPE_CHECKING:
    from calmmm.model.fit import MMMFit


def compute_roi(fit: "MMMFit") -> pd.DataFrame:
    """
    iROI per (KPI, channel) from a fitted MMMFit.

    Uses marginal (counterfactual removal) contributions: the outcome lost if
    a channel's spend dropped to zero, all else equal.  This is the correct
    denominator for iROI / budget optimisation — use channel_contributions()
    if you need additive decomposition for a pie chart instead.

    Returns a DataFrame with columns:
        kpi, channel, total_contribution, total_spend, roi

    total_spend is summed over training time steps only.
    total_contribution is the sum of marginal removals over training time steps.
    """
    if fit._mmm is None:
        raise ValueError(
            "compute_roi() requires a MMMFit produced by HierarchicalMMM.fit(); "
            "fit._mmm is None."
        )

    contrib_df = marginal_contributions(fit)

    total_contrib = (
        contrib_df.groupby(["kpi", "channel"])["contribution"]
        .sum()
        .reset_index()
        .rename(columns={"contribution": "total_contribution"})
    )

    # Filter to training times before summing spend
    train_times = set(
        t for t, m in zip(fit.data.times, fit._mmm._train_mask) if m
    )
    media_train = fit.data.media[fit.data.media["time"].isin(train_times)]
    spend_by_channel = (
        media_train.groupby("channel")["spend"]
        .sum()
        .reset_index()
        .rename(columns={"spend": "total_spend"})
    )

    merged = total_contrib.merge(spend_by_channel, on="channel", how="left")
    merged["roi"] = merged["total_contribution"] / merged["total_spend"]

    return merged[["kpi", "channel", "total_contribution", "total_spend", "roi"]].reset_index(drop=True)
