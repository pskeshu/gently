"""
Tests for gently/harness/roles.py — EmbryoRole registry.

Covers:
- role_class field presence and correct values for all built-in roles
- lineaging role entry: existence, role_class, ui_color, ui_icon
- accessor functions: get_role, is_valid_role, list_roles
- existing roles' other fields unchanged
"""

import pytest

from gently.harness.roles import (
    DEFAULT_ROLE,
    REGISTRY,
    EmbryoRole,
    get_role,
    is_valid_role,
    list_roles,
)

# ---------------------------------------------------------------------------
# role_class field — all existing roles
# ---------------------------------------------------------------------------


def test_test_role_class_is_subject():
    assert REGISTRY["test"].role_class == "subject"


def test_unassigned_role_class_is_subject():
    assert REGISTRY["unassigned"].role_class == "subject"


def test_calibration_role_class_is_reference():
    assert REGISTRY["calibration"].role_class == "reference"


# ---------------------------------------------------------------------------
# lineaging role — existence and fields
# ---------------------------------------------------------------------------


def test_lineaging_role_exists_in_registry():
    assert "lineaging" in REGISTRY


def test_lineaging_role_class_is_reference():
    assert REGISTRY["lineaging"].role_class == "reference"


def test_lineaging_ui_color():
    # Must be distinct from calibration (#00cccc) and test (#ff66cc)
    color = REGISTRY["lineaging"].ui_color
    assert color == "#33cc88"
    assert color != REGISTRY["calibration"].ui_color
    assert color != REGISTRY["test"].ui_color


def test_lineaging_ui_icon():
    assert REGISTRY["lineaging"].ui_icon == "triangle"


def test_lineaging_description_contains_lineage():
    desc = REGISTRY["lineaging"].description.lower()
    assert "lineage" in desc or "nuclei" in desc


def test_lineaging_has_cadence():
    assert REGISTRY["lineaging"].default_cadence_seconds > 0


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------


def test_get_role_lineaging_returns_correct_entry():
    role = get_role("lineaging")
    assert isinstance(role, EmbryoRole)
    assert role.name == "lineaging"
    assert role.role_class == "reference"


def test_is_valid_role_lineaging_true():
    assert is_valid_role("lineaging") is True


def test_is_valid_role_unknown_false():
    assert is_valid_role("nonexistent_role") is False


def test_list_roles_includes_lineaging():
    assert "lineaging" in list_roles()


def test_list_roles_sorted():
    roles = list_roles()
    assert roles == sorted(roles)


def test_get_role_unknown_raises_key_error():
    with pytest.raises(KeyError, match="Unknown embryo role"):
        get_role("not_a_role")


# ---------------------------------------------------------------------------
# Existing roles — other fields unchanged
# ---------------------------------------------------------------------------


def test_test_role_fields_unchanged():
    role = REGISTRY["test"]
    assert role.name == "test"
    assert role.ui_color == "#ff66cc"
    assert role.ui_icon == "star"
    assert role.photodose_budget_multiplier == 1.0
    assert role.detector_name == "dopaminergic_signal"
    assert role.no_object_consecutive_terminal == 5


def test_calibration_role_fields_unchanged():
    role = REGISTRY["calibration"]
    assert role.name == "calibration"
    assert role.ui_color == "#00cccc"
    assert role.ui_icon == "diamond"
    assert role.photodose_budget_multiplier == 10.0
    assert role.detector_name == "perception"
    assert role.no_object_consecutive_terminal == 2


def test_unassigned_role_fields_unchanged():
    role = REGISTRY["unassigned"]
    assert role.name == "unassigned"
    assert role.ui_color == "#888888"
    assert role.ui_icon == "circle"
    assert role.detector_name is None
    assert role.no_object_consecutive_terminal is None


# ---------------------------------------------------------------------------
# DEFAULT_ROLE sanity
# ---------------------------------------------------------------------------


def test_default_role_is_in_registry():
    assert DEFAULT_ROLE in REGISTRY


def test_default_role_is_test():
    assert DEFAULT_ROLE == "test"


# ---------------------------------------------------------------------------
# EmbryoRole is frozen (immutable)
# ---------------------------------------------------------------------------


def test_embryo_role_is_frozen():
    role = REGISTRY["test"]
    with pytest.raises(AttributeError):
        role.name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# role_class field has a safe default for keyword construction
# ---------------------------------------------------------------------------


def test_embryo_role_default_role_class():
    """New EmbryoRole without role_class defaults to 'subject'."""
    role = EmbryoRole(name="dummy", description="test only")
    assert role.role_class == "subject"
