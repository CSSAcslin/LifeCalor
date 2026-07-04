def render_status_update(status: str, message: str):
    if status == "failed":
        return message, "failed"
    return None
