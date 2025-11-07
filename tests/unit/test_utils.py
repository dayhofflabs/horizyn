"""
Unit tests for utility functions.
"""

import pytest
import torch

from horizyn.utils import InMemoryCache, cached_method, default_collate, dict_collate_fn


class TestInMemoryCache:
    """Tests for the InMemoryCache class."""

    def test_initialization(self):
        """Test cache initialization."""
        cache = InMemoryCache()
        assert len(cache) == 0
        assert not cache.has("any_key")

    def test_set_and_get(self):
        """Test setting and getting values."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", [1, 2, 3])
        cache.set("key3", {"nested": "dict"})

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == [1, 2, 3]
        assert cache.get("key3") == {"nested": "dict"}
        assert len(cache) == 3

    def test_get_with_default(self):
        """Test getting with default value."""
        cache = InMemoryCache()
        assert cache.get("missing_key", "default") == "default"
        assert cache.get("missing_key") is None

    def test_has(self):
        """Test checking if key exists."""
        cache = InMemoryCache()
        cache.set("key1", "value1")

        assert cache.has("key1")
        assert not cache.has("key2")

    def test_contains_operator(self):
        """Test 'in' operator."""
        cache = InMemoryCache()
        cache.set("key1", "value1")

        assert "key1" in cache
        assert "key2" not in cache

    def test_clear(self):
        """Test clearing the cache."""
        cache = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0
        assert not cache.has("key1")
        assert not cache.has("key2")

    def test_overwrite(self):
        """Test overwriting existing values."""
        cache = InMemoryCache()
        cache.set("key1", "old_value")
        assert cache.get("key1") == "old_value"

        cache.set("key1", "new_value")
        assert cache.get("key1") == "new_value"
        assert len(cache) == 1  # Still only one key

    def test_different_key_types(self):
        """Test using different hashable types as keys."""
        cache = InMemoryCache()
        cache.set("string_key", "value1")
        cache.set(42, "value2")
        cache.set((1, 2, 3), "value3")

        assert cache.get("string_key") == "value1"
        assert cache.get(42) == "value2"
        assert cache.get((1, 2, 3)) == "value3"

    def test_none_value(self):
        """Test storing None as a value."""
        cache = InMemoryCache()
        cache.set("key1", None)

        assert cache.has("key1")
        assert cache.get("key1") is None


class TestCachedMethod:
    """Tests for the cached_method decorator."""

    def test_basic_caching(self):
        """Test that results are cached."""

        class Calculator:
            def __init__(self):
                self._cache = InMemoryCache()
                self.call_count = 0

            @cached_method()
            def expensive_operation(self, x):
                self.call_count += 1
                return x * 2

        calc = Calculator()

        # First call should compute
        result1 = calc.expensive_operation(5)
        assert result1 == 10
        assert calc.call_count == 1

        # Second call should use cache
        result2 = calc.expensive_operation(5)
        assert result2 == 10
        assert calc.call_count == 1  # No additional calls

        # Different key should compute
        result3 = calc.expensive_operation(10)
        assert result3 == 20
        assert calc.call_count == 2

    def test_caching_with_complex_values(self):
        """Test caching complex objects."""

        class Generator:
            def __init__(self):
                self._cache = InMemoryCache()

            @cached_method()
            def generate(self, key):
                return {"key": key, "data": [1, 2, 3]}

        gen = Generator()

        result1 = gen.generate("key1")
        result2 = gen.generate("key1")

        # Should return same object from cache
        assert result1 is result2

    def test_custom_cache_attribute(self):
        """Test using a custom cache attribute name."""

        class CustomClass:
            def __init__(self):
                self.my_custom_cache = InMemoryCache()

            @cached_method(cache_attr="my_custom_cache")
            def method(self, key):
                return key.upper()

        obj = CustomClass()
        result = obj.method("test")
        assert result == "TEST"
        assert "test" in obj.my_custom_cache

    def test_missing_cache_attribute(self):
        """Test error when cache attribute doesn't exist."""

        class BrokenClass:
            @cached_method()
            def method(self, key):
                return key

        obj = BrokenClass()
        with pytest.raises(AttributeError, match="has no attribute '_cache'"):
            obj.method("key")

    def test_with_additional_args(self):
        """Test cached method with additional arguments."""

        class Calculator:
            def __init__(self):
                self._cache = InMemoryCache()

            @cached_method()
            def compute(self, key, multiplier=1):
                return key * multiplier

        calc = Calculator()

        # Cache is based only on first argument (key)
        result1 = calc.compute(5, multiplier=2)
        assert result1 == 10

        # Same key returns cached result, even with different multiplier
        result2 = calc.compute(5, multiplier=3)
        assert result2 == 10  # Cached value, not 15


class TestCollation:
    """Tests for collation functions."""

    def test_default_collate_tensors(self):
        """Test default_collate with simple tensors."""
        batch = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5, 6]),
            torch.tensor([7, 8, 9]),
        ]

        result = default_collate(batch)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 3)
        assert torch.equal(result[0], torch.tensor([1, 2, 3]))
        assert torch.equal(result[1], torch.tensor([4, 5, 6]))
        assert torch.equal(result[2], torch.tensor([7, 8, 9]))

    def test_dict_collate_fn_simple(self):
        """Test dict_collate_fn with simple dictionaries."""
        batch = [
            {"query_vec": torch.tensor([1, 2, 3]), "target_vec": torch.tensor([4, 5, 6])},
            {"query_vec": torch.tensor([7, 8, 9]), "target_vec": torch.tensor([10, 11, 12])},
        ]

        result = dict_collate_fn(batch)

        assert isinstance(result, dict)
        assert "query_vec" in result
        assert "target_vec" in result

        assert result["query_vec"].shape == (2, 3)
        assert result["target_vec"].shape == (2, 3)

        assert torch.equal(result["query_vec"][0], torch.tensor([1, 2, 3]))
        assert torch.equal(result["target_vec"][1], torch.tensor([10, 11, 12]))

    def test_dict_collate_fn_mixed_types(self):
        """Test dict_collate_fn with mixed data types."""
        batch = [
            {"vec": torch.tensor([1.0, 2.0]), "label": 0, "id": "sample1"},
            {"vec": torch.tensor([3.0, 4.0]), "label": 1, "id": "sample2"},
        ]

        result = dict_collate_fn(batch)

        assert result["vec"].shape == (2, 2)
        assert torch.equal(result["label"], torch.tensor([0, 1]))
        assert result["id"] == ["sample1", "sample2"]

    def test_dict_collate_fn_empty_batch(self):
        """Test dict_collate_fn with empty batch."""
        result = dict_collate_fn([])
        assert result == {}

    def test_dict_collate_fn_multi_dimensional(self):
        """Test dict_collate_fn with multi-dimensional tensors."""
        batch = [
            {"features": torch.randn(10, 5)},
            {"features": torch.randn(10, 5)},
            {"features": torch.randn(10, 5)},
        ]

        result = dict_collate_fn(batch)

        assert result["features"].shape == (3, 10, 5)

    def test_dict_collate_fn_consistency_with_default(self):
        """Test that dict_collate_fn produces same results as default_collate."""
        batch = [
            {"a": torch.tensor([1, 2]), "b": torch.tensor([3, 4])},
            {"a": torch.tensor([5, 6]), "b": torch.tensor([7, 8])},
        ]

        dict_result = dict_collate_fn(batch)
        default_result = default_collate(batch)

        assert torch.equal(dict_result["a"], default_result["a"])
        assert torch.equal(dict_result["b"], default_result["b"])

    def test_collate_with_nested_dict(self):
        """Test collation with nested dictionaries."""
        batch = [
            {"outer": {"inner": torch.tensor([1, 2])}},
            {"outer": {"inner": torch.tensor([3, 4])}},
        ]

        result = dict_collate_fn(batch)

        assert "outer" in result
        assert "inner" in result["outer"]
        assert result["outer"]["inner"].shape == (2, 2)

    def test_collate_with_variable_length(self):
        """Test that collation fails gracefully with variable-length data."""
        batch = [
            {"vec": torch.tensor([1, 2, 3])},
            {"vec": torch.tensor([4, 5])},  # Different length
        ]

        # This should raise an error because tensors have different sizes
        with pytest.raises(RuntimeError):
            dict_collate_fn(batch)


class TestIntegration:
    """Integration tests for utilities working together."""

    def test_cache_with_collation(self):
        """Test using cache and collation together."""

        class FingerprintGenerator:
            def __init__(self):
                self._cache = InMemoryCache()

            @cached_method()
            def generate(self, smiles):
                # Simulate expensive fingerprint generation
                return torch.randn(1024)

        gen = FingerprintGenerator()

        # Generate fingerprints
        fp1 = gen.generate("CCO")
        fp2 = gen.generate("CC=O")
        fp3 = gen.generate("CCO")  # Should be cached

        # Verify caching worked
        assert torch.equal(fp1, fp3)
        assert not torch.equal(fp1, fp2)

        # Create a batch of cached fingerprints
        batch = [
            {"fingerprint": gen.generate("CCO"), "id": "rxn1"},
            {"fingerprint": gen.generate("CC=O"), "id": "rxn2"},
        ]

        # Collate the batch
        result = dict_collate_fn(batch)

        assert result["fingerprint"].shape == (2, 1024)
        assert result["id"] == ["rxn1", "rxn2"]
