def __getattr__(name):
    if name == "LongTermMemory":
        from memory.long_term import LongTermMemory
        return LongTermMemory
    raise AttributeError(f"module 'memory' has no attribute {name!r}")

__all__ = ["LongTermMemory"]
