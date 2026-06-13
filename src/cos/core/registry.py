# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Darri Eythorsson <dae5@hi.is>
"""Connector registry — discovers and manages observation-connector plugins.

Mirrors CSFS's registry: a slug → connector-class map populated by importing the
connector modules. COS connectors each serve exactly one :class:`ObservationKind`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cos.connectors.base import BaseObservationConnector

_REGISTRY: dict[str, type[BaseObservationConnector]] = {}


def register(slug: str):
    """Decorator to register a connector class under a provider slug."""

    def wrapper(cls: type[BaseObservationConnector]) -> type[BaseObservationConnector]:
        _REGISTRY[slug] = cls
        return cls

    return wrapper


def get_connector(slug: str) -> type[BaseObservationConnector]:
    if slug not in _REGISTRY:
        raise KeyError(f"No connector registered for provider '{slug}'")
    return _REGISTRY[slug]


def list_providers() -> list[str]:
    return sorted(_REGISTRY.keys())


def discover() -> None:
    """Import all connector modules to trigger registration."""
    import importlib
    import pkgutil

    import cos.connectors as pkg

    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name != "base":
            importlib.import_module(f"cos.connectors.{info.name}")
