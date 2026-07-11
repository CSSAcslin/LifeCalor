# LifeCalor 1.0.6 Task, I/O, Rendering, and Error Architecture

This release introduces a multi-task-ready registry while keeping the current compact foreground progress UI. Each task owns an ID, category, state, progress, cancellation token, and optional executor cancellation callback.

Cache and NPY I/O use chunked reads and atomic writes. Cache restore, persistence, cleanup, and full deletion run outside the GUI thread. Escape requests cancellation through the coordinator and never waits in the GUI event handler.

Canvas export snapshots its render parameters and reuses `FrameRenderer` frame by frame. The legacy full-stack RGBA conversion path is removed.

Diagnostics log every severity. Warnings update logs and status only; errors and uncaught exceptions produce a deduplicated popup. A logging bridge covers legacy modules until all call sites emit structured issues directly.

The task registry already permits multiple active records. A future task panel can list and cancel tasks independently without changing worker contracts.
