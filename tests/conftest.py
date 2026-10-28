# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for pretrainer.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""
# import pytest
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../feature-extractor"))
)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs a real CLI on a data fixture (tens of seconds); select with -m slow")
