"""
Unit tests for dataset classes.
"""

import pytest
import torch
from horizyn.datasets import BaseDataset, WrapperDataset, MergeDataset, TupleDataset


class TestBaseDataset:
    """Tests for the BaseDataset class."""

    def test_initialization_with_keys_and_data(self):
        """Test basic initialization with keys and array data."""
        keys = ["a", "b", "c"]
        data = torch.randn(3, 10)

        dataset = BaseDataset(keys=keys, array_data=data)

        assert len(dataset) == 3
        assert dataset.keys == keys
        assert torch.equal(dataset.array_data, data)

    def test_initialization_keys_only(self):
        """Test initialization with only keys."""
        keys = ["a", "b", "c"]
        dataset = BaseDataset(keys=keys)

        assert len(dataset) == 3
        assert dataset.keys == keys

    def test_initialization_data_only(self):
        """Test initialization with only array data."""
        data = torch.randn(5, 10)
        dataset = BaseDataset(array_data=data)

        assert len(dataset) == 5
        assert torch.equal(dataset.array_data, data)

    def test_duplicate_keys_error(self):
        """Test that duplicate keys raise an error."""
        keys = ["a", "b", "a", "c"]  # "a" is duplicated
        data = torch.randn(4, 10)

        with pytest.raises(ValueError, match="Keys must be unique"):
            BaseDataset(keys=keys, array_data=data)

    def test_length_mismatch_error(self):
        """Test that mismatched keys and data lengths raise an error."""
        keys = ["a", "b", "c"]
        data = torch.randn(5, 10)  # 5 != 3

        with pytest.raises(ValueError, match="must have same length"):
            BaseDataset(keys=keys, array_data=data)

    def test_key_to_idx_mapping(self):
        """Test automatic key-to-idx mapping creation."""
        keys = ["x", "y", "z"]
        data = torch.randn(3, 5)

        dataset = BaseDataset(keys=keys, array_data=data)

        assert dataset.key_to_idx == {"x": 0, "y": 1, "z": 2}

    def test_key_to_idx_without_data(self):
        """Test explicit key-to-idx mapping without data."""
        keys = ["x", "y", "z"]

        dataset = BaseDataset(keys=keys, use_key_to_idx=True)

        assert dataset.key_to_idx == {"x": 0, "y": 1, "z": 2}

    def test_key_to_idx_error_without_keys(self):
        """Test that use_key_to_idx without keys raises an error."""
        with pytest.raises(ValueError, match="Cannot create key-to-idx mapping"):
            BaseDataset(use_key_to_idx=True)

    def test_getitem_by_key(self):
        """Test accessing data by key."""
        keys = ["a", "b", "c"]
        data = torch.tensor([[1.0], [2.0], [3.0]])

        dataset = BaseDataset(keys=keys, array_data=data)

        assert torch.equal(dataset["a"], torch.tensor([1.0]))
        assert torch.equal(dataset["b"], torch.tensor([2.0]))
        assert torch.equal(dataset["c"], torch.tensor([3.0]))

    def test_getitem_by_index(self):
        """Test accessing data by integer index."""
        keys = ["a", "b", "c"]
        data = torch.tensor([[1.0], [2.0], [3.0]])

        dataset = BaseDataset(keys=keys, array_data=data)

        assert torch.equal(dataset[0], torch.tensor([1.0]))
        assert torch.equal(dataset[1], torch.tensor([2.0]))
        assert torch.equal(dataset[2], torch.tensor([3.0]))

    def test_getitem_key_not_found(self):
        """Test that accessing non-existent key raises an error."""
        keys = ["a", "b"]
        data = torch.randn(2, 5)

        dataset = BaseDataset(keys=keys, array_data=data)

        with pytest.raises(KeyError, match="not found"):
            dataset["z"]

    def test_getitem_index_out_of_bounds(self):
        """Test that out-of-bounds index raises an error."""
        keys = ["a", "b"]
        data = torch.randn(2, 5)

        dataset = BaseDataset(keys=keys, array_data=data)

        with pytest.raises(IndexError, match="out of bounds"):
            dataset[10]

    def test_transforms(self):
        """Test that transforms are applied."""
        keys = ["a", "b"]
        data = torch.tensor([[1.0], [2.0]])

        def double_transform(key, data):
            return data * 2

        dataset = BaseDataset(keys=keys, array_data=data, transforms=double_transform)

        assert torch.equal(dataset["a"], torch.tensor([2.0]))
        assert torch.equal(dataset["b"], torch.tensor([4.0]))

    def test_iteration(self):
        """Test that dataset is iterable."""
        keys = ["a", "b", "c"]
        data = torch.tensor([[1.0], [2.0], [3.0]])

        dataset = BaseDataset(keys=keys, array_data=data)

        items = [dataset[i] for i in range(len(dataset))]
        assert len(items) == 3

    def test_properties_not_set_errors(self):
        """Test that accessing unset properties raises errors."""
        dataset = BaseDataset()

        with pytest.raises(AttributeError, match="Keys are not set"):
            _ = dataset.keys

        with pytest.raises(AttributeError, match="Array data is not set"):
            _ = dataset.array_data

        with pytest.raises(AttributeError, match="Key-to-idx mapping is not set"):
            _ = dataset.key_to_idx

    def test_len_without_keys_or_data(self):
        """Test that len() without keys or data raises an error."""
        dataset = BaseDataset()

        with pytest.raises(ValueError, match="Cannot determine length"):
            len(dataset)


class TestWrapperDataset:
    """Tests for the WrapperDataset class."""

    def test_basic_wrapping(self):
        """Test basic dataset wrapping."""
        base_dataset = BaseDataset(keys=["a", "b"], array_data=torch.tensor([[1.0], [2.0]]))

        wrapper = WrapperDataset(base_dataset)

        assert len(wrapper) == 2
        assert wrapper.keys == ["a", "b"]
        assert torch.equal(wrapper["a"], torch.tensor([1.0]))

    def test_wrapper_with_transform(self):
        """Test wrapper with transform."""
        base_dataset = BaseDataset(keys=["a", "b"], array_data=torch.tensor([[1.0], [2.0]]))

        def add_ten(key, data):
            return data + 10

        wrapper = WrapperDataset(base_dataset, transforms=add_ten)

        assert torch.equal(wrapper["a"], torch.tensor([11.0]))
        assert torch.equal(wrapper["b"], torch.tensor([12.0]))

    def test_wrapper_preserves_base_dataset(self):
        """Test that wrapper doesn't modify base dataset."""
        base_dataset = BaseDataset(keys=["a"], array_data=torch.tensor([[1.0]]))

        def multiply_by_100(key, data):
            return data * 100

        wrapper = WrapperDataset(base_dataset, transforms=multiply_by_100)

        # Wrapper applies transform
        assert torch.equal(wrapper["a"], torch.tensor([100.0]))
        # Base dataset unchanged
        assert torch.equal(base_dataset["a"], torch.tensor([1.0]))

    def test_wrapper_properties(self):
        """Test that wrapper exposes base dataset properties."""
        base_dataset = BaseDataset(keys=["x", "y"], array_data=torch.randn(2, 5))

        wrapper = WrapperDataset(base_dataset)

        assert wrapper.keys == base_dataset.keys
        assert torch.equal(wrapper.array_data, base_dataset.array_data)
        assert wrapper.key_to_idx == base_dataset.key_to_idx


class TestMergeDataset:
    """Tests for the MergeDataset class."""

    def test_basic_merge(self):
        """Test basic merging of two datasets."""
        ds1 = BaseDataset(keys=["a", "b", "c"], array_data=torch.tensor([[1.0], [2.0], [3.0]]))
        ds2 = BaseDataset(keys=["a", "b", "c"], array_data=torch.tensor([[10.0], [20.0], [30.0]]))

        merged = MergeDataset({"ds1": ds1, "ds2": ds2})

        assert len(merged) == 3
        assert set(merged.keys) == {"a", "b", "c"}

        sample = merged["a"]
        assert torch.equal(sample["ds1"], torch.tensor([1.0]))
        assert torch.equal(sample["ds2"], torch.tensor([10.0]))

    def test_merge_with_intersection(self):
        """Test merging with partial key overlap."""
        ds1 = BaseDataset(keys=["a", "b", "c"], array_data=torch.randn(3, 5))
        ds2 = BaseDataset(keys=["b", "c", "d"], array_data=torch.randn(3, 5))

        merged = MergeDataset({"ds1": ds1, "ds2": ds2})

        # Only common keys (b, c) should be in merged dataset
        assert len(merged) == 2
        assert set(merged.keys) == {"b", "c"}

    def test_merge_no_common_keys_error(self):
        """Test that no common keys raises an error."""
        ds1 = BaseDataset(keys=["a", "b"], array_data=torch.randn(2, 5))
        ds2 = BaseDataset(keys=["c", "d"], array_data=torch.randn(2, 5))

        with pytest.raises(ValueError, match="No common keys"):
            MergeDataset({"ds1": ds1, "ds2": ds2})

    def test_merge_with_dict_results(self):
        """Test merging when datasets return dicts."""
        # Create datasets that return dicts
        ds1 = BaseDataset(
            keys=["a", "b"],
            array_data=[{"feature1": 1, "feature2": 2}, {"feature1": 3, "feature2": 4}],
        )
        ds2 = BaseDataset(keys=["a", "b"], array_data=[{"feature3": 5}, {"feature3": 6}])

        merged = MergeDataset({"ds1": ds1, "ds2": ds2})

        sample = merged["a"]
        assert sample == {"feature1": 1, "feature2": 2, "feature3": 5}

    def test_merge_with_prefix(self):
        """Test merging with add_prefix=True."""
        ds1 = BaseDataset(keys=["a"], array_data=[{"value": 1}])
        ds2 = BaseDataset(keys=["a"], array_data=[{"value": 2}])

        merged = MergeDataset({"ds1": ds1, "ds2": ds2}, add_prefix=True)

        sample = merged["a"]
        assert sample == {"ds1_value": 1, "ds2_value": 2}

    def test_merge_duplicate_key_error(self):
        """Test that duplicate keys in dict results raise an error."""
        ds1 = BaseDataset(keys=["a"], array_data=[{"value": 1}])
        ds2 = BaseDataset(keys=["a"], array_data=[{"value": 2}])

        merged = MergeDataset({"ds1": ds1, "ds2": ds2}, add_prefix=False)

        with pytest.raises(ValueError, match="Duplicate key"):
            merged["a"]

    def test_merge_empty_datasets_error(self):
        """Test that empty datasets dict raises an error."""
        with pytest.raises(ValueError, match="Must provide at least one dataset"):
            MergeDataset({})

    def test_merge_multiple_datasets(self):
        """Test merging more than two datasets."""
        ds1 = BaseDataset(keys=["a", "b"], array_data=torch.tensor([[1.0], [2.0]]))
        ds2 = BaseDataset(keys=["a", "b"], array_data=torch.tensor([[3.0], [4.0]]))
        ds3 = BaseDataset(keys=["a", "b"], array_data=torch.tensor([[5.0], [6.0]]))

        merged = MergeDataset({"ds1": ds1, "ds2": ds2, "ds3": ds3})

        assert len(merged) == 2
        sample = merged["a"]
        assert len(sample) == 3  # Data from 3 datasets


class TestTupleDataset:
    """Tests for the TupleDataset class."""

    def test_basic_tuple_dataset(self):
        """Test basic tuple dataset functionality."""
        # Source datasets
        queries = BaseDataset(
            keys=["q1", "q2", "q3"], array_data=torch.tensor([[1.0], [2.0], [3.0]])
        )
        targets = BaseDataset(
            keys=["t1", "t2", "t3"], array_data=torch.tensor([[10.0], [20.0], [30.0]])
        )

        # Tuple dataset defining pairs
        pairs = BaseDataset(
            keys=["pair1", "pair2"],
            array_data=[{"query": "q1", "target": "t2"}, {"query": "q3", "target": "t1"}],
        )

        tuple_ds = TupleDataset(
            tuple_dataset=pairs, key_name_to_dataset={"query": queries, "target": targets}
        )

        assert len(tuple_ds) == 2

        sample1 = tuple_ds["pair1"]
        assert torch.equal(sample1["query"], torch.tensor([1.0]))
        assert torch.equal(sample1["target"], torch.tensor([20.0]))

        sample2 = tuple_ds["pair2"]
        assert torch.equal(sample2["query"], torch.tensor([3.0]))
        assert torch.equal(sample2["target"], torch.tensor([10.0]))

    def test_tuple_dataset_with_rename(self):
        """Test tuple dataset with key renaming."""
        queries = BaseDataset(keys=["q1"], array_data=torch.tensor([[1.0]]))
        targets = BaseDataset(keys=["t1"], array_data=torch.tensor([[10.0]]))

        pairs = BaseDataset(keys=["pair1"], array_data=[{"query_id": "q1", "target_id": "t1"}])

        tuple_ds = TupleDataset(
            tuple_dataset=pairs,
            key_name_to_dataset={"query_id": queries, "target_id": targets},
            rename_map={"query_id": "reaction", "target_id": "protein"},
        )

        sample = tuple_ds["pair1"]
        assert "reaction" in sample
        assert "protein" in sample
        assert torch.equal(sample["reaction"], torch.tensor([1.0]))
        assert torch.equal(sample["protein"], torch.tensor([10.0]))

    def test_tuple_dataset_non_dict_error(self):
        """Test that non-dict tuple data raises an error."""
        queries = BaseDataset(keys=["q1"], array_data=torch.randn(1, 5))

        # Tuple dataset that doesn't return dicts
        pairs = BaseDataset(keys=["pair1"], array_data=torch.tensor([[1, 2]]))

        # With skip_missing=False, the error happens during __getitem__
        tuple_ds = TupleDataset(
            tuple_dataset=pairs, key_name_to_dataset={"query": queries}, skip_missing=False
        )

        with pytest.raises(TypeError, match="must return a dict"):
            tuple_ds["pair1"]

    def test_tuple_dataset_missing_key_error(self):
        """Test that missing key in tuple dict raises an error."""
        queries = BaseDataset(keys=["q1"], array_data=torch.randn(1, 5))
        targets = BaseDataset(keys=["t1"], array_data=torch.randn(1, 5))

        # Tuple dataset missing "target" key
        pairs = BaseDataset(keys=["pair1"], array_data=[{"query": "q1"}])  # Missing "target"

        # With skip_missing=False, dataset is created but error happens during access
        tuple_ds = TupleDataset(
            tuple_dataset=pairs,
            key_name_to_dataset={"query": queries, "target": targets},
            skip_missing=False,
        )

        # Error occurs when trying to access the pair
        with pytest.raises(KeyError, match="not found in tuple_dict"):
            tuple_ds["pair1"]

    def test_tuple_dataset_with_dict_results(self):
        """Test tuple dataset when source datasets return dicts."""
        queries = BaseDataset(
            keys=["q1"], array_data=[{"smiles": "CCO", "fingerprint": torch.randn(10)}]
        )
        targets = BaseDataset(keys=["t1"], array_data=[{"embedding": torch.randn(20)}])

        pairs = BaseDataset(keys=["pair1"], array_data=[{"query": "q1", "target": "t1"}])

        tuple_ds = TupleDataset(
            tuple_dataset=pairs, key_name_to_dataset={"query": queries, "target": targets}
        )

        sample = tuple_ds["pair1"]
        assert "smiles" in sample
        assert "fingerprint" in sample
        assert "embedding" in sample

    def test_tuple_dataset_with_prefix(self):
        """Test tuple dataset with add_prefix=True."""
        queries = BaseDataset(keys=["q1"], array_data=[{"data": 1}])
        targets = BaseDataset(keys=["t1"], array_data=[{"data": 2}])

        pairs = BaseDataset(keys=["pair1"], array_data=[{"query": "q1", "target": "t1"}])

        tuple_ds = TupleDataset(
            tuple_dataset=pairs,
            key_name_to_dataset={"query": queries, "target": targets},
            add_prefix=True,
        )

        sample = tuple_ds["pair1"]
        assert "query_data" in sample
        assert "target_data" in sample

    def test_tuple_dataset_properties(self):
        """Test that tuple dataset exposes relevant properties."""
        queries = BaseDataset(keys=["q1"], array_data=torch.randn(1, 5))
        targets = BaseDataset(keys=["t1"], array_data=torch.randn(1, 5))

        pairs = BaseDataset(
            keys=["pair1", "pair2"],
            array_data=[{"query": "q1", "target": "t1"}, {"query": "q1", "target": "t1"}],
        )

        tuple_ds = TupleDataset(
            tuple_dataset=pairs, key_name_to_dataset={"query": queries, "target": targets}
        )

        # Keys should match since all pairs are valid
        assert tuple_ds.keys == pairs.keys

        # array_data delegates to tuple_dataset
        assert tuple_ds.array_data == pairs.array_data

        # key_to_idx is based on filtered keys (though in this case all are valid)
        assert len(tuple_ds.key_to_idx) == len(pairs.keys)
        assert all(key in tuple_ds.key_to_idx for key in pairs.keys)

    def test_skip_missing_filters_invalid_pairs(self):
        """Test that skip_missing=True filters out pairs with missing keys."""
        # Source datasets
        queries = BaseDataset(
            keys=["q1", "q2", "q3"], array_data=torch.tensor([[1.0], [2.0], [3.0]])
        )
        targets = BaseDataset(
            keys=["t1", "t2", "t3"], array_data=torch.tensor([[10.0], [20.0], [30.0]])
        )

        # Pairs with some invalid references
        pairs = BaseDataset(
            keys=["pair1", "pair2", "pair3", "pair4"],
            array_data=[
                {"query": "q1", "target": "t2"},  # valid
                {"query": "q99", "target": "t1"},  # invalid: q99 doesn't exist
                {"query": "q2", "target": "t99"},  # invalid: t99 doesn't exist
                {"query": "q3", "target": "t3"},  # valid
            ],
        )

        # With skip_missing=True (default), should filter out invalid pairs
        tuple_ds = TupleDataset(
            tuple_dataset=pairs, key_name_to_dataset={"query": queries, "target": targets}
        )

        # Should only have 2 valid pairs
        assert len(tuple_ds) == 2
        assert "pair1" in tuple_ds.keys
        assert "pair4" in tuple_ds.keys
        assert "pair2" not in tuple_ds.keys  # Filtered out
        assert "pair3" not in tuple_ds.keys  # Filtered out

        # Can access valid pairs
        sample1 = tuple_ds["pair1"]
        assert torch.equal(sample1["query"], torch.tensor([1.0]))
        assert torch.equal(sample1["target"], torch.tensor([20.0]))

    def test_skip_missing_false_strict_mode(self):
        """Test that skip_missing=False keeps all pairs and raises errors on access."""
        queries = BaseDataset(keys=["q1"], array_data=torch.tensor([[1.0]]))
        targets = BaseDataset(keys=["t1"], array_data=torch.tensor([[10.0]]))

        pairs = BaseDataset(
            keys=["pair1", "pair2"],
            array_data=[
                {"query": "q1", "target": "t1"},  # valid
                {"query": "q99", "target": "t1"},  # invalid
            ],
        )

        # With skip_missing=False, all pairs are kept
        tuple_ds = TupleDataset(
            tuple_dataset=pairs,
            key_name_to_dataset={"query": queries, "target": targets},
            skip_missing=False,
        )

        # Should have all pairs
        assert len(tuple_ds) == 2

        # Valid pair works
        sample1 = tuple_ds["pair1"]
        assert torch.equal(sample1["query"], torch.tensor([1.0]))

        # Invalid pair raises KeyError when accessed
        with pytest.raises(KeyError, match="not found"):
            tuple_ds["pair2"]

    def test_skip_missing_with_all_valid_pairs(self):
        """Test that skip_missing=True doesn't filter when all pairs are valid."""
        queries = BaseDataset(keys=["q1", "q2"], array_data=torch.tensor([[1.0], [2.0]]))
        targets = BaseDataset(keys=["t1", "t2"], array_data=torch.tensor([[10.0], [20.0]]))

        pairs = BaseDataset(
            keys=["pair1", "pair2"],
            array_data=[{"query": "q1", "target": "t1"}, {"query": "q2", "target": "t2"}],
        )

        tuple_ds = TupleDataset(
            tuple_dataset=pairs, key_name_to_dataset={"query": queries, "target": targets}
        )

        # All pairs should be present
        assert len(tuple_ds) == 2
        assert tuple_ds.keys == pairs.keys

    def test_skip_missing_preserves_original_keys(self):
        """Test that skip_missing preserves query_id/target_id in output."""
        queries = BaseDataset(keys=["q1"], array_data=torch.tensor([[1.0]]))
        targets = BaseDataset(keys=["t1"], array_data=torch.tensor([[10.0]]))

        pairs = BaseDataset(keys=["pair1"], array_data=[{"query_id": "q1", "target_id": "t1"}])

        tuple_ds = TupleDataset(
            tuple_dataset=pairs,
            key_name_to_dataset={"query_id": queries, "target_id": targets},
            rename_map={"query_id": "query_vec", "target_id": "target_vec"},
        )

        sample = tuple_ds["pair1"]

        # Should have both original IDs and renamed vectors
        assert "query_id" in sample
        assert "target_id" in sample
        assert "query_vec" in sample
        assert "target_vec" in sample
        assert sample["query_id"] == "q1"
        assert sample["target_id"] == "t1"


class TestDatasetEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_element_dataset(self):
        """Test dataset with a single element."""
        dataset = BaseDataset(keys=["only"], array_data=torch.tensor([[1.0]]))

        assert len(dataset) == 1
        assert torch.equal(dataset["only"], torch.tensor([1.0]))

    def test_large_dataset(self):
        """Test dataset with many elements."""
        n = 10000
        keys = [f"key_{i}" for i in range(n)]
        data = torch.randn(n, 100)

        dataset = BaseDataset(keys=keys, array_data=data)

        assert len(dataset) == n
        assert torch.equal(dataset["key_0"], data[0])
        assert torch.equal(dataset["key_9999"], data[9999])

    def test_different_key_types(self):
        """Test dataset with different key types."""
        # String keys
        ds_str = BaseDataset(keys=["a", "b"], array_data=torch.randn(2, 5))
        assert "a" in ds_str.keys

        # Integer keys (use BaseDataset directly with ints)
        ds_int = BaseDataset(keys=[1, 2, 3], array_data=torch.randn(3, 5))
        assert 1 in ds_int.keys

    def test_transforms_with_key_access(self):
        """Test that transforms receive the correct key."""
        keys_received = []

        def track_key_transform(key, data):
            keys_received.append(key)
            return data

        dataset = BaseDataset(
            keys=["a", "b"], array_data=torch.randn(2, 5), transforms=track_key_transform
        )

        dataset["a"]
        dataset["b"]

        assert keys_received == ["a", "b"]
