# tests/conftest.py
import os
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires running services)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m not integration')",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--integration"):
        os.environ["INTEGRATION_TESTS"] = "1"
