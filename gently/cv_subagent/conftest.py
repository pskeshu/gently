"""Pytest configuration for CV Subagent tests"""

import pytest


# Only test with asyncio backend (not trio)
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "anyio: mark test as async with anyio"
    )


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio backend for anyio tests"""
    return "asyncio"
