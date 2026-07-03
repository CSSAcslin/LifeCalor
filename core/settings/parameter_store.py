from typing import Any, Callable, Dict


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def coerce_param_value(value: Any, default_value: Any) -> Any:
    if isinstance(default_value, bool):
        return _to_bool(value)
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        try:
            return int(value)
        except (ValueError, TypeError):
            return default_value
    if isinstance(default_value, float):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default_value
    if isinstance(default_value, str):
        if value is None or value == "":
            return default_value
        return str(value)
    return value if value is not None else default_value


def load_param_group(reader: Callable[[str, Any], Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    params = {}
    for key, default_value in defaults.items():
        params[key] = coerce_param_value(reader(key, default_value), default_value)
    return params
