def normalize_progress(current, total=None, maximum_limit=1000):
    """Return QProgressBar-safe current/max values plus original percent."""
    if current == -1:
        return -1, None, 0.0
    if total is None:
        return int(current), None, 0.0

    total = int(total)
    current = int(current)
    if total <= 0:
        return 0, 0, 0.0

    percent = max(0.0, min(100.0, current / total * 100.0))
    if total <= maximum_limit:
        return max(0, min(current, total)), total, percent

    scaled_current = int(round(percent / 100.0 * maximum_limit))
    return max(0, min(scaled_current, maximum_limit)), maximum_limit, percent
