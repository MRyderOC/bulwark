# -*- coding: utf-8 -*-
"""
Module for useful generic functions.
"""
from itertools import chain, cycle

import numpy as np
import pandas as pd


def bad_locations(df):
    """Indicates bad cells in `df`."""
    columns = df.columns
    all_locs = chain.from_iterable(zip(df.index, cycle([col])) for col in columns)
    bad = pd.Series(list(all_locs))[np.asarray(df).ravel(order='F')]
    msg = bad.values

    return msg


def snake_to_camel(snake_str):
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)


def series_dtype_check(ser: pd.Series, dtype: str) -> bool:
    """For a given Series specifies if elements have dtypes.

    Args:
        ser (pd.Series): Any Series
        dtype (str): Accepted values: [
                'object', 'bool', 'string',
            # number related
                'numeric', 'float', 'complex', 'int', 'int64',
            # signed or unsigned int
                ('signed_int', 'signed-int', 'signed int', 'signedint', 'sint'),
                ('unsigned_int', 'unsigned-int', 'unsigned int', 'unsignedint', 'uint'),
            # time related
                'datetime', 'datetime64', 'datetime64_ns', 'datetime64_tz',
                'timedelta64', 'timedelta64_ns'
        ]

    Returns:
        A boolean value
    """
    if dtype == "object":
        result = pd.api.types.is_object_dtype(ser)
    elif dtype == "bool":
        result = pd.api.types.is_bool_dtype(ser)
    elif dtype == "string":
        result = pd.api.types.is_string_dtype(ser)
    elif dtype == "numeric":
        result = pd.api.types.is_numeric_dtype(ser)
    elif dtype == "float":
        result = pd.api.types.is_float_dtype(ser)
    elif dtype == "complex":
        result = pd.api.types.is_complex_dtype(ser)
    elif dtype == "int":
        result = pd.api.types.is_integer_dtype(ser)
    elif dtype == "int64":
        result = pd.api.types.is_int64_dtype(ser)
    elif dtype in ["signed_int", "signed-int", "signed int", "signedint", "sint"]:
        result = pd.api.types.is_signed_integer_dtype(ser)
    elif dtype in ["unsigned_int", "unsigned-int", "unsigned int", "unsignedint", "uint"]:
        result = pd.api.types.is_unsigned_integer_dtype(ser)
    elif dtype == "datetime":
        result = pd.api.types.is_datetime64_any_dtype(ser)
    elif dtype == "datetime64":
        result = pd.api.types.is_datetime64_dtype(ser)
    elif dtype == "datetime64_ns":
        result = pd.api.types.is_datetime64_ns_dtype(ser)
    elif dtype == "datetime64tz":
        result = pd.api.types.is_datetime64tz_dtype(ser)
    elif dtype == "timedelta64":
        result = pd.api.types.is_timedelta64_dtype(ser)
    elif dtype == "timedelta64_ns":
        result = pd.api.types.is_timedelta64_ns_dtype(ser)
    else:
        raise ValueError(
            f"This function doesn't support for `{dtype}` checking."
        )

    return result
