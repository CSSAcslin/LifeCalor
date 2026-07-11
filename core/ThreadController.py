from typing import Any, Callable, Optional, Type


def is_thread_active(thread: Any, expected_type: Optional[Type] = None, is_deleted: Callable[[Any], bool] = lambda _: False) -> bool:
    if thread is None:
        return False
    if expected_type is not None and not isinstance(thread, expected_type):
        return False
    if is_deleted(thread):
        return False
    is_running = getattr(thread, "isRunning", None)
    return bool(callable(is_running) and is_running())


def stop_thread(thread: Any, expected_type: Optional[Type] = None, is_deleted: Callable[[Any], bool] = lambda _: False,
                wait_ms: int = 1000) -> bool:
    """Stop an event-loop thread with a bounded wait; never block the GUI indefinitely."""
    if not is_thread_active(thread, expected_type=expected_type, is_deleted=is_deleted):
        return False
    thread.quit()
    stopped = thread.wait(max(0, int(wait_ms)))
    if stopped is False:
        return False
    thread.deleteLater()
    return True
