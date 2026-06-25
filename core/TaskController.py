from typing import Any, Callable, Optional, Type

from ThreadController import is_thread_active


def ensure_thread_running(
    thread: Any,
    task_state,
    expected_type: Optional[Type] = None,
    is_deleted: Callable[[Any], bool] = lambda _: False,
) -> bool:
    task_state.start()
    if is_thread_active(thread, expected_type=expected_type, is_deleted=is_deleted):
        return False
    thread.start()
    return True
