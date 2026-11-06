"""Qt fixtures must destroy windows before later tests replace the app style."""

import unittest

from PyQt5 import sip
from PyQt5.QtCore import QCoreApplication, QEvent, Qt
from PyQt5.QtWidgets import QApplication


def dispose_new_windows(app, existing_windows):
    windows = [
        window for window in app.topLevelWidgets()
        if window not in existing_windows and window.windowType() != Qt.Desktop
    ]
    for window in windows:
        if not sip.isdeleted(window):
            window.close()
            window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


class QtTestCase(unittest.TestCase):
    def run(self, result=None):
        # Register before subclass setUp, including fixtures with their own setUp.
        app = QApplication.instance()
        if app is not None:
            self.addCleanup(dispose_new_windows, app, set(app.topLevelWidgets()))
        return super().run(result)
