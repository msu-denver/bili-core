"""
AETHER Config Module

Loaders for declarative MAS configuration from YAML files and Python dicts.
"""

from .loader import load_mas_from_dict, load_mas_from_yaml

__all__ = [
    "load_mas_from_yaml",
    "load_mas_from_dict",
]
