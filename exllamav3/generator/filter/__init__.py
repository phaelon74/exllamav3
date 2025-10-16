from .filter import Filter

# Lazy import to avoid pydantic compatibility issues with formatron
def __getattr__(name):
    if name == "FormatronFilter":
        from .formatron import FormatronFilter
        return FormatronFilter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")