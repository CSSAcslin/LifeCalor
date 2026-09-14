from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


THEME_LIGHT = "light"
THEME_DARK = "dark"


@dataclass(frozen=True)
class ThemeTokens:
    theme_id: str
    label: str
    colors: Mapping[str, str]

    def __getitem__(self, key):
        return self.colors[key]


_LIGHT = {
    "app_bg": "#F5F7F6",
    "panel_bg": "#FFFFFF",
    "raised_bg": "#F7FAF8",
    "input_bg": "#FFFFFF",
    "hover_bg": "#E8F5E9",
    "border": "#C8E6C9",
    "outline": "#9AD19A",
    "text": "#202A24",
    "secondary": "#607066",
    "disabled": "#9AA49D",
    "accent": "#2E7D32",
    "accent_dark": "#1B5E20",
    "selection": "#D8EBE3",
    "on_accent": "#FFFFFF",
    "info": "#265E92",
    "success": "#17613C",
    "warning": "#805B0B",
    "error": "#A83232",
}

_DARK = {
    "app_bg": "#181A1B",
    "panel_bg": "#212426",
    "raised_bg": "#292D30",
    "input_bg": "#191C1E",
    "hover_bg": "#30373A",
    "border": "#414A50",
    "outline": "#718087",
    "text": "#E8EDF0",
    "secondary": "#B0BAC1",
    "disabled": "#7D878E",
    "accent": "#63D5B0",
    "accent_dark": "#42BB98",
    "selection": "#354C45",
    "on_accent": "#10271F",
    "info": "#93C6FC",
    "success": "#83DDA8",
    "warning": "#F0C070",
    "error": "#FF9690",
}

_THEMES = {
    THEME_LIGHT: ThemeTokens(THEME_LIGHT, "清爽浅色", MappingProxyType(_LIGHT)),
    THEME_DARK: ThemeTokens(THEME_DARK, "石墨深色", MappingProxyType(_DARK)),
}


def normalize_theme_id(value) -> str:
    value = str(value or "").strip().lower()
    return value if value in _THEMES else THEME_LIGHT


def theme_tokens(theme_id) -> ThemeTokens:
    return _THEMES[normalize_theme_id(theme_id)]
