# calmmm/__init__.py
__all__ = ["MMMData", "IncrementalityTests"]

def __getattr__(name):
    if name in ("MMMData", "IncrementalityTests"):
        from calmmm.data.containers import MMMData, IncrementalityTests
        globals()["MMMData"] = MMMData
        globals()["IncrementalityTests"] = IncrementalityTests
        return globals()[name]
    raise AttributeError(f"module 'calmmm' has no attribute {name!r}")
