"""Utilities to parse OpenSim .osim XML files into plain Python dictionaries.

Public API:
- parse_osim(path: str) -> dict
"""

from .parser import parse_osim, OSIMModel  # noqa: F401

__all__ = ["parse_osim", "OSIMModel"]


