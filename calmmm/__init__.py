# calmmm/__init__.py
__all__ = ["MMMData", "IncrementalityTests", "HierarchicalMMM", "MMMFit", "CalibrationTarget"]


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
    raise AttributeError(f"module 'calmmm' has no attribute {name!r}")
