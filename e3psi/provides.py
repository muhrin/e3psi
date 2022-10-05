# -*- coding: utf-8 -*-
from . import graphs


def get_historian_types():
    """The central entry point to provide historian type helpers"""
    types = list()
    types.extend(graphs.HISTORIAN_TYPES)

    return types
