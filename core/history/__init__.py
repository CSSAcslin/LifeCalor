__all__ = ["HistoryController"]


def __getattr__(name):
    if name == "HistoryController":
        from .controller import HistoryController
        return HistoryController
    raise AttributeError(name)
