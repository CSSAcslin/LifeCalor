import math


def format_hover_value(value):
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    if number == 0:
        return "0"

    abs_value = abs(number)
    if abs_value < 1e-3:
        return f"{number:.2e}"
    if abs_value < 1:
        return f"{number:.6f}".rstrip("0").rstrip(".")
    if abs_value < 10000:
        return f"{number:.3f}".rstrip("0").rstrip(".")
    return f"{number:.6g}"
