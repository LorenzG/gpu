from collections.abc import Iterable

def coerce_iterable(obj):
    if not isinstance(obj, Iterable):
        return [obj]
    return obj