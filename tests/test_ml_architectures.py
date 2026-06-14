"""
Tests for ML architecture registry.
"""

from gently.ml.architectures import ARCHITECTURE_REGISTRY, get_suitable_architectures


class TestArchitectureRegistry:
    def test_all_entries_have_required_fields(self):
        required = {
            "name",
            "family",
            "param_count_m",
            "min_vram_gb",
            "recommended_vram_gb",
            "min_dataset_size",
            "recommended_dataset_size",
            "training_speed",
            "suitability",
        }
        for arch_id, meta in ARCHITECTURE_REGISTRY.items():
            for field in required:
                assert field in meta, f"{arch_id} missing '{field}'"

    def test_registry_not_empty(self):
        assert len(ARCHITECTURE_REGISTRY) >= 6

    def test_get_suitable_filters_by_vram(self):
        results = get_suitable_architectures(dataset_size=500, vram_gb=2.0)
        # Only small models should be returned
        for r in results:
            assert r["min_vram_gb"] <= 2.0

    def test_get_suitable_filters_by_dataset_size(self):
        results = get_suitable_architectures(dataset_size=50, vram_gb=24.0)
        # Only models that work with 50 samples
        for r in results:
            assert r["min_dataset_size"] <= 50

    def test_zero_vram_returns_empty(self):
        results = get_suitable_architectures(dataset_size=500, vram_gb=0.0)
        assert len(results) == 0

    def test_large_dataset_recommends_more_models(self):
        small = get_suitable_architectures(dataset_size=100, vram_gb=24.0)
        large = get_suitable_architectures(dataset_size=2000, vram_gb=24.0)
        assert len(large) >= len(small)

    def test_results_have_reason(self):
        results = get_suitable_architectures(dataset_size=500, vram_gb=24.0)
        for r in results:
            assert "reason" in r
            assert isinstance(r["reason"], str)

    def test_results_sorted_by_fit_score(self):
        results = get_suitable_architectures(dataset_size=500, vram_gb=24.0)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["fit_score"] >= results[i + 1]["fit_score"]
