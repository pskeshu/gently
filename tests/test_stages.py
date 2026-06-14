"""
Tests for C. elegans developmental stage definitions.

Tests cover:
- Ordered stage values
- get_order ordinal mapping
- compare() stage comparison
- is_terminal detection
- is_valid for normal and special states
- is_special_state identification
- get_adjacent_stages for middle, first, and last stages
- STAGE_CRITERIA structure validation
"""

import pytest

from gently.organisms.celegans.stages import (
    STAGE_CRITERIA,
    STAGES,
    DevelopmentalStage,
    get_adjacent_stages,
)


class TestOrderedValues:
    def test_ordered_values(self):
        vals = DevelopmentalStage.ordered_values()
        assert vals == [
            "early",
            "bean",
            "comma",
            "1.5fold",
            "2fold",
            "pretzel",
            "hatching",
            "hatched",
        ]

    def test_stages_global_matches(self):
        assert STAGES == DevelopmentalStage.ordered_values()


class TestGetOrder:
    def test_get_order(self):
        assert DevelopmentalStage.get_order("early") == 0
        assert DevelopmentalStage.get_order("hatched") == 7

    def test_get_order_invalid_raises(self):
        with pytest.raises(ValueError):
            DevelopmentalStage.get_order("nonexistent")


class TestCompare:
    def test_compare_earlier(self):
        assert DevelopmentalStage.compare("early", "comma") == -1

    def test_compare_same(self):
        assert DevelopmentalStage.compare("comma", "comma") == 0

    def test_compare_later(self):
        assert DevelopmentalStage.compare("pretzel", "bean") == 1


class TestTerminalAndValid:
    def test_is_terminal_hatched(self):
        assert DevelopmentalStage.is_terminal("hatched") is True

    def test_is_terminal_other(self):
        assert DevelopmentalStage.is_terminal("comma") is False

    def test_is_valid_normal(self):
        assert DevelopmentalStage.is_valid("early") is True
        assert DevelopmentalStage.is_valid("pretzel") is True

    def test_is_valid_special(self):
        assert DevelopmentalStage.is_valid("arrested") is True
        assert DevelopmentalStage.is_valid("no_object") is True

    def test_is_valid_invalid(self):
        assert DevelopmentalStage.is_valid("larva_L2") is False


class TestSpecialState:
    def test_is_special_state(self):
        assert DevelopmentalStage.is_special_state("arrested") is True
        assert DevelopmentalStage.is_special_state("no_object") is True

    def test_is_not_special_state(self):
        assert DevelopmentalStage.is_special_state("early") is False
        assert DevelopmentalStage.is_special_state("hatched") is False


class TestAdjacentStages:
    def test_adjacent_stages_middle(self):
        prev, next_ = get_adjacent_stages("comma")
        assert prev == "bean"
        assert next_ == "1.5fold"

    def test_adjacent_stages_first(self):
        prev, next_ = get_adjacent_stages("early")
        assert prev is None
        assert next_ == "bean"

    def test_adjacent_stages_last(self):
        prev, next_ = get_adjacent_stages("hatched")
        assert prev == "hatching"
        assert next_ is None

    def test_adjacent_stages_invalid(self):
        prev, next_ = get_adjacent_stages("nonexistent")
        assert prev is None
        assert next_ is None


class TestStageCriteria:
    def test_stage_criteria_has_required_keys(self):
        for stage_name, criteria in STAGE_CRITERIA.items():
            assert "features" in criteria, f"{stage_name} missing 'features'"
            assert "NOT_if" in criteria, f"{stage_name} missing 'NOT_if'"
            assert isinstance(criteria["features"], list)
            assert isinstance(criteria["NOT_if"], list)
            assert len(criteria["features"]) > 0, f"{stage_name} has empty features"

    def test_all_ordered_stages_have_criteria(self):
        for stage in STAGES:
            assert stage in STAGE_CRITERIA, f"Missing criteria for stage '{stage}'"
