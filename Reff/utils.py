from functools import wraps
from inspect import signature
from typing import get_args, get_origin, get_type_hints, Union

import numpy as np

from schemas import BaseSharpModel


class CSharpConversionError(TypeError):
    """Raised when conversion from C# object to Python model is not possible."""


def convert_csharp_models(func):
    """
    Decorator that converts incoming C# objects into Python models
    based on function argument type hints.
    """
    sig = signature(func)
    hints = get_type_hints(func)
    builtin_passthrough_types = {bool, int, float, str, bytes, dict, list, tuple, set}

    def _resolve_hint_options(type_hint):
        origin = get_origin(type_hint)
        if origin is Union:
            return get_args(type_hint)
        return (type_hint,)

    def _is_none_type(tp):
        return tp is type(None)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        converted = {}

        for arg_name, value in bound.arguments.items():
            if arg_name not in hints:
                raise CSharpConversionError(
                    f"Type for argument '{arg_name}' is not defined in '{func.__name__}'."
                )

            target_type = hints[arg_name]
            options = _resolve_hint_options(target_type)

            if value is None and any(_is_none_type(tp) for tp in options):
                converted[arg_name] = None
                continue

            matched = False
            for option in options:
                if _is_none_type(option):
                    continue

                if option in builtin_passthrough_types:
                    converted[arg_name] = value
                    matched = True
                    break

                if isinstance(option, type) and issubclass(option, BaseSharpModel):
                    if isinstance(value, option):
                        converted[arg_name] = value
                        matched = True
                        break
                    try:
                        converted[arg_name] = option.from_csharp(value)
                        matched = True
                        break
                    except Exception:
                        # Try next union option before failing.
                        continue

            if not matched:
                raise CSharpConversionError(
                    f"Could not match argument '{arg_name}' in '{func.__name__}' "
                    f"to supported type hint '{target_type}'."
                )

        return func(**converted)

    return wrapper
