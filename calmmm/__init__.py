# calmmm/__init__.py
__all__ = [
    "MMMData", "IncrementalityTests",
    "HierarchicalMMM", "MMMFit",
    "CalibrationTarget",
    "ChannelInteraction", "InteractionGraph",
    "channel_contributions", "marginal_contributions", "compute_roi", "saturation_curve",
    "spend_response_report",
]


def __getattr__(name):
    if name in ("MMMData", "IncrementalityTests"):
        from calmmm.data.containers import MMMData, IncrementalityTests
        globals()["MMMData"] = MMMData
        globals()["IncrementalityTests"] = IncrementalityTests
        return globals()[name]
    if name in ("HierarchicalMMM", "MMMFit"):
        from calmmm.model.mmm import HierarchicalMMM
        from calmmm.model.fit import MMMFit
        globals()["HierarchicalMMM"] = HierarchicalMMM
        globals()["MMMFit"] = MMMFit
        return globals()[name]
    if name == "CalibrationTarget":
        from calmmm.calibration.targets import CalibrationTarget
        globals()["CalibrationTarget"] = CalibrationTarget
        return CalibrationTarget
    if name in ("ChannelInteraction", "InteractionGraph"):
        from calmmm.model.interactions import ChannelInteraction, InteractionGraph
        globals()["ChannelInteraction"] = ChannelInteraction
        globals()["InteractionGraph"] = InteractionGraph
        return globals()[name]
    if name in ("channel_contributions", "marginal_contributions", "compute_roi", "saturation_curve", "spend_response_report"):
        from calmmm.attribution.contributions import channel_contributions, marginal_contributions
        from calmmm.attribution.roi import compute_roi
        from calmmm.attribution.curves import saturation_curve, spend_response_report
        globals()["channel_contributions"] = channel_contributions
        globals()["marginal_contributions"] = marginal_contributions
        globals()["compute_roi"] = compute_roi
        globals()["saturation_curve"] = saturation_curve
        globals()["spend_response_report"] = spend_response_report
        return globals()[name]
    raise AttributeError(f"module 'calmmm' has no attribute {name!r}")
