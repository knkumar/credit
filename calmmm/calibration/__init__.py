__all__ = [
    "CalibrationTarget",
    "build_calibration_targets",
    "add_calibration_likelihood",
    "compute_model_lift",
]


def __getattr__(name):
    if name in ("CalibrationTarget", "build_calibration_targets"):
        from calmmm.calibration.targets import CalibrationTarget, build_calibration_targets
        globals()["CalibrationTarget"] = CalibrationTarget
        globals()["build_calibration_targets"] = build_calibration_targets
        return globals()[name]
    if name == "add_calibration_likelihood":
        from calmmm.calibration.likelihood import add_calibration_likelihood
        globals()["add_calibration_likelihood"] = add_calibration_likelihood
        return add_calibration_likelihood
    if name == "compute_model_lift":
        from calmmm.calibration.lift import compute_model_lift
        globals()["compute_model_lift"] = compute_model_lift
        return compute_model_lift
    raise AttributeError(f"module 'calmmm.calibration' has no attribute {name!r}")
