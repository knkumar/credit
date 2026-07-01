__all__ = ["build_summary_table", "render_reporting_outputs"]


def __getattr__(name):
    if name in __all__:
        from calmmm.reporting import visualization

        return getattr(visualization, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
