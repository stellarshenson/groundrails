"""Shared pytest fixtures."""

import pytest

from groundrails import settings


@pytest.fixture(autouse=True)
def _grounder_ready():
    """Run each test as if ``groundrails.init`` has been called - the grounder is
    ready. The readiness-gate test resets this explicitly to assert the refusal."""
    settings.mark_ready()
    yield
