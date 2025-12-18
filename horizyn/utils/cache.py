"""
Simple in-memory caching utilities for fingerprint generation.
"""

from functools import wraps
from typing import Any, Callable, Dict, Hashable


class InMemoryCache:
    """
    Simple in-memory cache for storing computed values.

    This cache stores key-value pairs in memory using a Python dictionary.
    It's designed for caching fingerprints or embeddings that are expensive
    to compute but can fit in RAM.

    Attributes:
        _cache (Dict): Internal dictionary storing cached values.

    Example:
        >>> cache = InMemoryCache()
        >>> cache.set("key1", [1, 2, 3])
        >>> cache.get("key1")
        [1, 2, 3]
        >>> cache.has("key2")
        False
        >>> len(cache)
        1
    """

    def __init__(self):
        """Initialize an empty cache."""
        self._cache: Dict[Hashable, Any] = {}

    def get(self, key: Hashable, default: Any = None) -> Any:
        """
        Get a value from the cache.

        Args:
            key: Key to look up.
            default: Default value to return if key not found. Defaults to None.

        Returns:
            Cached value if key exists, otherwise default.
        """
        return self._cache.get(key, default)

    def set(self, key: Hashable, value: Any) -> None:
        """
        Store a value in the cache.

        Args:
            key: Key to store value under.
            value: Value to cache.
        """
        self._cache[key] = value

    def has(self, key: Hashable) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Key to check.

        Returns:
            True if key exists, False otherwise.
        """
        return key in self._cache

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return the number of cached items."""
        return len(self._cache)

    def __contains__(self, key: Hashable) -> bool:
        """Check if key is in cache (enables 'in' operator)."""
        return key in self._cache


def cached_method(cache_attr: str = "_cache"):
    """
    Decorator for caching method results in an instance attribute.

    This decorator caches the results of a method call in an InMemoryCache
    stored as an instance attribute. The cache key is the first argument
    to the method (typically an ID or key).

    Args:
        cache_attr: Name of the instance attribute containing the cache.
            Defaults to "_cache".

    Example:
        >>> class FingerprintGenerator:
        ...     def __init__(self):
        ...         self._cache = InMemoryCache()
        ...
        ...     @cached_method()
        ...     def generate(self, smiles: str):
        ...         # Expensive computation
        ...         return compute_fingerprint(smiles)
        >>>
        >>> gen = FingerprintGenerator()
        >>> fp1 = gen.generate("CCO")  # Computed
        >>> fp2 = gen.generate("CCO")  # Cached
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, key: Hashable, *args, **kwargs):
            # Get cache from instance
            cache = getattr(self, cache_attr, None)
            if cache is None:
                raise AttributeError(
                    f"Instance has no attribute '{cache_attr}'. "
                    f"Ensure the instance has an InMemoryCache at '{cache_attr}'."
                )

            # Check cache
            if cache.has(key):
                return cache.get(key)

            # Compute and cache
            result = func(self, key, *args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper

    return decorator
