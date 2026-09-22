from __future__ import annotations

import logging

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from appearance import get_theme_manager

from .pages import (
    AppearancePage,
    CachePage,
    ComputePageAdapter,
    GeneralPage,
    LifetimePage,
    PlotPage,
)


class PreferencesDialog(QDialog):
    settingsApplied = pyqtSignal()

    PAGE_TYPES = (
        GeneralPage,
        AppearancePage,
        PlotPage,
        ComputePageAdapter,
        LifetimePage,
        CachePage,
    )
    PAGE_PRESENTATION = {
        "常规与更新": (
            ":/icons/icon_reset.svg",
            "启动、更新与日志",
            "管理启动行为、更新检查和诊断信息位置。",
        ),
        "外观与画布": (
            ":/icons/icon_color.svg",
            "主题与画布工具",
            "设置界面主题，以及新绘制操作采用的工具默认值。",
        ),
        "绘图默认值": (
            ":/icons/icon_line.svg",
            "曲线与热图",
            "配置结果图的线条、标记、网格和颜色映射。",
        ),
        "计算与加速": (
            ":/icons/icon_v-line.svg",
            "硬件、后端与精度",
            "查看硬件状态并设置新任务的计算策略。",
        ),
        "寿命拟合默认值": (
            ":/icons/icon_anchor.svg",
            "拟合范围与质量",
            "设置之后提交的单指数和双指数拟合默认参数。",
        ),
        "缓存与存储": (
            ":/icons/icon_export.svg",
            "目录、阈值与内存",
            "控制大数组落盘位置、阈值和交互内存预算。",
        ),
    }

    def __init__(self, window, parent=None):
        super().__init__(parent or window)
        self.window = window
        self.setWindowTitle("选项")
        self.setObjectName("PreferencesDialog")
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setMinimumSize(980, 650)
        self.resize(1120, 760)
        self.pages = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("preferencesSidebar")
        sidebar.setFixedWidth(258)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(18, 20, 14, 16)
        sidebar_layout.setSpacing(10)
        sidebar_title = QLabel("选项")
        sidebar_title.setObjectName("preferencesSidebarTitle")
        sidebar_subtitle = QLabel("应用偏好与默认行为")
        sidebar_subtitle.setObjectName("preferencesSidebarSubtitle")
        sidebar_layout.addWidget(sidebar_title)
        sidebar_layout.addWidget(sidebar_subtitle)
        self.navigation = QListWidget()
        self.navigation.setObjectName("preferencesNavigation")
        self.navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.navigation.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.navigation.setSpacing(6)
        self.navigation.setFocusPolicy(Qt.NoFocus)
        sidebar_layout.addWidget(self.navigation, 1)

        content = QWidget()
        content.setObjectName("preferencesContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 20, 24, 0)
        content_layout.setSpacing(12)
        self.page_title = QLabel()
        self.page_title.setObjectName("preferencesPageTitle")
        self.page_description = QLabel()
        self.page_description.setObjectName("preferencesPageDescription")
        self.page_description.setWordWrap(True)
        content_layout.addWidget(self.page_title)
        content_layout.addWidget(self.page_description)
        divider = QFrame()
        divider.setObjectName("preferencesHeaderDivider")
        divider.setFrameShape(QFrame.HLine)
        content_layout.addWidget(divider)
        self.stack = QStackedWidget()
        content_layout.addWidget(self.stack, 1)
        body.addWidget(sidebar)
        body.addWidget(content, 1)
        root.addLayout(body, 1)

        for page_type in self.PAGE_TYPES:
            page = page_type(self.window, self)
            self.pages[page.title] = page
            icon, subtitle, description = self.PAGE_PRESENTATION[page.title]
            item = QListWidgetItem()
            item.setData(Qt.UserRole, page.title)
            item.setData(Qt.AccessibleTextRole, page.title)
            item.setToolTip(description)
            item.setSizeHint(QSize(224, 66))
            self.navigation.addItem(item)
            self.navigation.setItemWidget(
                item,
                self._navigation_widget(icon, page.title, subtitle),
            )
            if isinstance(page, ComputePageAdapter):
                self.stack.addWidget(page)
            else:
                container = QWidget()
                layout = QVBoxLayout(container)
                layout.setContentsMargins(0, 0, 0, 0)
                scroll = QScrollArea()
                scroll.setWidgetResizable(True)
                scroll.setFrameShape(QScrollArea.NoFrame)
                scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                scroll.setWidget(page)
                layout.addWidget(scroll)
                self.stack.addWidget(container)

        self.navigation.currentRowChanged.connect(self._page_changed)
        self.navigation.setCurrentRow(0)

        footer_frame = QFrame()
        footer_frame.setObjectName("preferencesFooter")
        footer = QHBoxLayout()
        footer.setContentsMargins(20, 12, 20, 14)
        footer_frame.setLayout(footer)
        self.page_hint = QLabel("")
        self.page_hint.setObjectName("preferencesStatus")
        footer.addWidget(self.page_hint, 1)
        self.defaults_button = QPushButton("恢复本页默认")
        self.apply_button = QPushButton("应用")
        self.ok_button = QPushButton("确定")
        self.cancel_button = QPushButton("取消")
        self.defaults_button.clicked.connect(self.reset_current_page)
        self.apply_button.clicked.connect(self.apply_changes)
        self.ok_button.clicked.connect(self.accept_changes)
        self.cancel_button.clicked.connect(self.reject)
        footer.addWidget(self.defaults_button)
        footer.addWidget(self.apply_button)
        footer.addWidget(self.ok_button)
        footer.addWidget(self.cancel_button)
        root.addWidget(footer_frame)

        manager = get_theme_manager()
        if manager is not None:
            manager.themeChanged.connect(self._apply_theme)
            self._apply_theme(manager.tokens)

    @staticmethod
    def _navigation_widget(icon_path, title, subtitle):
        widget = QWidget()
        widget.setObjectName("preferencesNavigationItem")
        widget.setAttribute(Qt.WA_TransparentForMouseEvents)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 8, 8, 8)
        layout.setSpacing(11)
        icon = QLabel()
        icon.setObjectName("preferencesNavigationIcon")
        icon.setFixedSize(28, 28)
        icon.setPixmap(QIcon(icon_path).pixmap(24, 24))
        icon.setAlignment(Qt.AlignCenter)
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("preferencesNavigationTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("preferencesNavigationSubtitle")
        text_layout.addWidget(title_label)
        text_layout.addWidget(subtitle_label)
        layout.addWidget(icon)
        layout.addLayout(text_layout, 1)
        return widget

    def _page_changed(self, index):
        self.stack.setCurrentIndex(index)
        if index < 0:
            return
        item = self.navigation.item(index)
        title = item.data(Qt.UserRole)
        _icon, _subtitle, description = self.PAGE_PRESENTATION[title]
        self.page_title.setText(title)
        self.page_description.setText(description)

    def _apply_theme(self, tokens):
        colors = tokens.colors
        self.setStyleSheet(
            f"""
            QFrame#preferencesSidebar {{
                background: {colors['raised_bg']};
                border-right: 1px solid {colors['border']};
            }}
            QLabel#preferencesSidebarTitle {{
                color: {colors['text']};
                font-size: 16pt;
                font-weight: 600;
            }}
            QLabel#preferencesSidebarSubtitle,
            QLabel#preferencesNavigationSubtitle,
            QLabel#preferencesPageDescription,
            QLabel#preferencesStatus {{
                color: {colors['secondary']};
            }}
            QListWidget#preferencesNavigation {{
                background: transparent;
                border: 0;
                padding: 4px 0;
                outline: 0;
            }}
            QListWidget#preferencesNavigation::item {{
                background: transparent;
                border: 0;
                border-left: 3px solid transparent;
                border-radius: 4px;
            }}
            QListWidget#preferencesNavigation::item:hover {{
                background: {colors['hover_bg']};
            }}
            QListWidget#preferencesNavigation::item:selected {{
                background: {colors['selection']};
                border-left: 3px solid {colors['accent']};
            }}
            QWidget#preferencesNavigationItem {{
                background: transparent;
            }}
            QLabel#preferencesNavigationTitle {{
                color: {colors['text']};
                font-weight: 600;
            }}
            QLabel#preferencesNavigationSubtitle {{
                font-size: 9pt;
            }}
            QWidget#preferencesContent {{
                background: {colors['panel_bg']};
            }}
            QLabel#preferencesPageTitle {{
                color: {colors['text']};
                font-size: 17pt;
                font-weight: 600;
            }}
            QLabel#preferencesPageDescription {{
                font-size: 9.5pt;
            }}
            QFrame#preferencesHeaderDivider,
            QFrame#preferencesToolDivider {{
                color: {colors['border']};
                background: {colors['border']};
                max-height: 1px;
            }}
            QFrame#preferencesFooter {{
                background: {colors['raised_bg']};
                border-top: 1px solid {colors['border']};
            }}
            QWidget#preferencesToolSection {{
                background: transparent;
            }}
            QLabel#preferencesToolTitle {{
                color: {colors['text']};
                font-weight: 600;
                font-size: 11pt;
            }}
            QLabel#preferencesToolDescription,
            QLabel#preferencesToolScopeTitle,
            QLabel#preferencesToolScopeText {{
                color: {colors['secondary']};
            }}
            QLabel#preferencesToolScopeText {{
                background: {colors['raised_bg']};
                border: 1px solid {colors['border']};
                border-radius: 3px;
                padding: 2px 5px;
            }}
            """
        )

    def show_page(self, title):
        titles = list(self.pages)
        if title in self.pages:
            self.navigation.setCurrentRow(titles.index(title))
        self.show()
        self.raise_()
        self.activateWindow()

    def current_page(self):
        title = self.navigation.currentItem().data(Qt.UserRole)
        return self.pages[title]

    def reset_current_page(self):
        page = self.current_page()
        page.reset_defaults()
        self.page_hint.setText("已恢复本页默认草稿，点击“应用”或“确定”后保存。")

    def _dirty_pages(self):
        return [page for page in self.pages.values() if page.has_changes()]

    def apply_changes(self):
        dirty = self._dirty_pages()
        if not dirty:
            self.page_hint.setText("没有需要保存的更改。")
            return True
        for page in dirty:
            message = page.validate_page()
            if message:
                self.show_page(page.title)
                QMessageBox.warning(self, f"{page.title}设置无效", message)
                return False
            conflicts = page.conflicts()
            if conflicts:
                self.show_page(page.title)
                QMessageBox.warning(
                    self,
                    "设置已在其他位置更改",
                    "以下设置在窗口打开后又被其他入口修改，请重新打开选项后再设置：\n"
                    + "、".join(conflicts),
                )
                return False

        order = (
            "常规与更新",
            "绘图默认值",
            "计算与加速",
            "寿命拟合默认值",
            "缓存与存储",
            "外观与画布",
        )
        try:
            for title in order:
                page = self.pages[title]
                if page in dirty:
                    page.apply_page()
        except Exception as exc:
            logging.exception("应用选项失败")
            QMessageBox.critical(
                self,
                "选项保存失败",
                f"设置未能完整应用：{type(exc).__name__}: {exc}\n"
                "请检查日志，并重新打开选项确认当前值。",
            )
            return False

        self.window.settings.setValue(
            "preferences/last_page", self.navigation.currentRow()
        )
        self.window.settings.sync()
        self.page_hint.setText("设置已保存。正在运行和已排队的任务保持原提交参数。")
        self.settingsApplied.emit()
        logging.info(
            "统一选项已应用: pages=%s",
            ", ".join(page.title for page in dirty),
        )
        return True

    def accept_changes(self):
        if self.apply_changes():
            self.accept()
