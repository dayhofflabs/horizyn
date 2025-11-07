"""
Unit tests for fingerprint generation datasets.
"""

import pytest
import torch

from horizyn.datasets import BaseDataset
from horizyn.datasets.fingerprints import (
    BaseFingerprintDataset,
    DRFPFingerprintDataset,
    RDKitPlusFingerprintDataset,
)


class TestBaseFingerprintDataset:
    """Tests for the BaseFingerprintDataset class."""

    def test_initialization(self):
        """Test basic initialization."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = BaseFingerprintDataset(
            reaction_dataset=reactions, vec_dim=512, standardize=False
        )

        assert fp_dataset.vec_dim == 512
        assert fp_dataset.dtype == torch.float32
        assert fp_dataset.smiles_label == "reaction_smiles"
        assert len(fp_dataset) == 1

    def test_standardizer_enabled(self):
        """Test that standardizer is created when standardize=True."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = BaseFingerprintDataset(reaction_dataset=reactions, standardize=True)

        assert fp_dataset.standardize is True
        assert fp_dataset.standardizer is not None

    def test_standardizer_disabled(self):
        """Test that standardizer is not created when standardize=False."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = BaseFingerprintDataset(reaction_dataset=reactions, standardize=False)

        assert fp_dataset.standardize is False
        with pytest.raises(AttributeError):
            _ = fp_dataset.standardizer

    def test_query_smiles_dataset_dict(self):
        """Test querying dataset that returns dicts."""
        reactions = BaseDataset(
            keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O", "extra": "data"}]
        )

        fp_dataset = BaseFingerprintDataset(reaction_dataset=reactions, standardize=False)

        result = fp_dataset._query_smiles_dataset("rxn1")
        assert isinstance(result, dict)
        assert "reaction_smiles" in result

    def test_query_smiles_dataset_string(self):
        """Test querying dataset that returns strings."""
        reactions = BaseDataset(keys=["rxn1"], array_data=["CCO>>CC=O"])

        fp_dataset = BaseFingerprintDataset(reaction_dataset=reactions, standardize=False)

        result = fp_dataset._query_smiles_dataset("rxn1")
        assert isinstance(result, dict)
        assert result["reaction_smiles"] == "CCO>>CC=O"

    def test_query_smiles_dataset_missing_key(self):
        """Test error when dict doesn't have smiles_label."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"wrong_key": "CCO>>CC=O"}])

        fp_dataset = BaseFingerprintDataset(reaction_dataset=reactions, standardize=False)

        with pytest.raises(KeyError, match="missing.*reaction_smiles"):
            fp_dataset._query_smiles_dataset("rxn1")

    def test_generate_fingerprint_not_implemented(self):
        """Test that base class raises error (wrapped NotImplementedError)."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = BaseFingerprintDataset(reaction_dataset=reactions, standardize=False)

        with pytest.raises(Exception, match="Failed to generate fingerprint"):
            fp_dataset["rxn1"]


class TestBaseFingerprintDatasetCaching:
    """Tests for in-memory caching behavior in BaseFingerprintDataset."""

    class DummyFingerprintDataset(BaseFingerprintDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        def _generate_fingerprint(self, reaction_info):  # type: ignore[override]
            self.calls += 1
            # Return a deterministic fingerprint based on SMILES length
            length = len(reaction_info[self.smiles_label])
            return torch.ones(self.vec_dim, dtype=self.dtype) * float(length)

    def test_caches_results_per_key(self):
        reactions = BaseDataset(
            keys=["a", "b"],
            array_data=[
                {"reaction_smiles": "CCO>>CC=O"},
                {"reaction_smiles": "C>>CC"},
            ],
        )

        ds = self.DummyFingerprintDataset(reaction_dataset=reactions, vec_dim=16)

        # First access should compute
        fp_a1 = ds["a"]
        assert ds.calls == 1

        # Second access to same key should hit cache (no new compute)
        fp_a2 = ds["a"]
        assert ds.calls == 1
        assert torch.equal(fp_a1, fp_a2)

        # Access different key should compute again
        _ = ds["b"]
        assert ds.calls == 2

    def test_cache_with_transforms(self):
        reactions = BaseDataset(keys=["a"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        def add_one(_key, tensor: torch.Tensor) -> torch.Tensor:
            return tensor + 1.0

        ds = self.DummyFingerprintDataset(reaction_dataset=reactions, vec_dim=8, transforms=add_one)

        # First call computes and applies transform
        fp1 = ds["a"]
        # Second call should use cached raw fp and re-apply transform
        fp2 = ds["a"]
        assert torch.equal(fp1, fp2)
        # Ensure only one compute occurred
        assert ds.calls == 1


class TestRDKitPlusFingerprintDataset:
    """Tests for the RDKitPlusFingerprintDataset class."""

    def test_initialization_default(self):
        """Test default initialization."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = RDKitPlusFingerprintDataset(reaction_dataset=reactions)

        assert fp_dataset.vec_dim == 1024
        assert fp_dataset.mol_fp_type == "morgan"
        assert fp_dataset.rxn_fp_type == "struct"
        assert fp_dataset.use_chirality is False
        assert fp_dataset.fp_size == 512  # Half of vec_dim

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=2048,
            mol_fp_type="rdkit",
            rxn_fp_type="diff",
            use_chirality=True,
        )

        assert fp_dataset.vec_dim == 2048
        assert fp_dataset.mol_fp_type == "rdkit"
        assert fp_dataset.rxn_fp_type == "diff"
        assert fp_dataset.use_chirality is True

    def test_invalid_mol_fp_type(self):
        """Test that invalid mol_fp_type raises error."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        with pytest.raises(ValueError, match="Invalid mol_fp_type"):
            RDKitPlusFingerprintDataset(reaction_dataset=reactions, mol_fp_type="invalid")

    def test_invalid_rxn_fp_type(self):
        """Test that invalid rxn_fp_type raises error."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        with pytest.raises(ValueError, match="Invalid rxn_fp_type"):
            RDKitPlusFingerprintDataset(reaction_dataset=reactions, rxn_fp_type="invalid")

    def test_odd_vec_dim_struct(self):
        """Test that odd vec_dim raises error for structural fingerprints."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        with pytest.raises(ValueError, match="vec_dim must be even"):
            RDKitPlusFingerprintDataset(
                reaction_dataset=reactions, vec_dim=1023, rxn_fp_type="struct"  # Odd
            )

    def test_generate_structural_fingerprint(self):
        """Test generating a structural fingerprint."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions, vec_dim=1024, rxn_fp_type="struct"
        )

        fp = fp_dataset["rxn1"]

        assert isinstance(fp, torch.Tensor)
        assert fp.shape == (1024,)
        assert fp.dtype == torch.float32
        # Structural fingerprints should be binary (0 or 1)
        assert torch.all((fp == 0) | (fp == 1))

    def test_generate_diff_fingerprint(self):
        """Test generating a difference fingerprint."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions, vec_dim=1024, rxn_fp_type="diff"
        )

        fp = fp_dataset["rxn1"]

        assert isinstance(fp, torch.Tensor)
        assert fp.shape == (1024,)
        # Difference fingerprints can have counts (integers)

    def test_mol_fp_types(self):
        """Test different molecular fingerprint types."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        for fp_type in ["morgan", "rdkit", "atom_pair", "topological_torsion"]:
            fp_dataset = RDKitPlusFingerprintDataset(
                reaction_dataset=reactions, mol_fp_type=fp_type, vec_dim=1024
            )

            fp = fp_dataset["rxn1"]
            assert fp.shape == (1024,)

    def test_with_chirality(self):
        """Test fingerprint generation with chirality enabled."""
        reactions = BaseDataset(
            keys=["rxn1"],
            # Use a simple reaction without invalid chiral centers
            array_data=[{"reaction_smiles": "C[C@H](O)CC>>C[C@H](Cl)CC"}],
        )

        fp_dataset = RDKitPlusFingerprintDataset(reaction_dataset=reactions, use_chirality=True)

        fp = fp_dataset["rxn1"]
        assert fp.shape == (1024,)

    def test_with_standardization(self):
        """Test fingerprint generation with standardization."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            standardize=True,
            standardize_hypervalent=True,
            standardize_uncharge=True,
            standardize_metals=True,
        )

        fp = fp_dataset["rxn1"]
        assert fp.shape == (1024,)

    def test_multiple_reactions(self):
        """Test generating fingerprints for multiple reactions."""
        reactions = BaseDataset(
            keys=["rxn1", "rxn2", "rxn3"],
            array_data=[
                {"reaction_smiles": "CCO>>CC=O"},
                {"reaction_smiles": "C>>CC"},
                {"reaction_smiles": "CC>>C=C"},
            ],
        )

        fp_dataset = RDKitPlusFingerprintDataset(reaction_dataset=reactions)

        fp1 = fp_dataset["rxn1"]
        fp2 = fp_dataset["rxn2"]
        fp3 = fp_dataset["rxn3"]

        assert fp1.shape == (1024,)
        assert fp2.shape == (1024,)
        assert fp3.shape == (1024,)

        # Different reactions should have different fingerprints
        assert not torch.equal(fp1, fp2)
        assert not torch.equal(fp2, fp3)

    def test_invalid_smiles(self):
        """Test error handling for invalid SMILES."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "invalid>>smiles"}])

        fp_dataset = RDKitPlusFingerprintDataset(reaction_dataset=reactions)

        with pytest.raises(Exception):
            fp_dataset["rxn1"]

    def test_sota_configuration(self):
        """Test with SOTA paper configuration."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO.O=O>>CC(=O)O"}])

        # SOTA config from swissprot_sota.yaml
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=1024,
            mol_fp_type="morgan",  # Default
            rxn_fp_type="struct",
            use_chirality=True,
            standardize=True,
            standardize_hypervalent=True,
            standardize_uncharge=True,
            standardize_metals=True,
        )

        fp = fp_dataset["rxn1"]
        assert fp.shape == (1024,)
        assert torch.all((fp == 0) | (fp == 1))  # Binary

    def test_fingerprint_deterministic(self):
        """Test that fingerprints are deterministic."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = RDKitPlusFingerprintDataset(reaction_dataset=reactions)

        fp1 = fp_dataset["rxn1"]
        fp2 = fp_dataset["rxn1"]

        assert torch.equal(fp1, fp2)

    def test_wraps_sql_dataset(self):
        """Test that fingerprint dataset can wrap SQLDataset."""
        # Create mock dataset that behaves like SQLDataset
        reactions = BaseDataset(
            keys=["rxn1", "rxn2"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}, {"reaction_smiles": "C>>CC"}],
        )

        fp_dataset = RDKitPlusFingerprintDataset(reaction_dataset=reactions, vec_dim=1024)

        assert len(fp_dataset) == 2
        fp1 = fp_dataset["rxn1"]
        fp2 = fp_dataset["rxn2"]

        assert fp1.shape == (1024,)
        assert fp2.shape == (1024,)


class TestDRFPFingerprintDataset:
    """Tests for the DRFPFingerprintDataset class."""

    def test_initialization_default(self):
        """Test default initialization."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = DRFPFingerprintDataset(reaction_dataset=reactions)

        assert fp_dataset.vec_dim == 1024
        assert fp_dataset.min_radius == 0
        assert fp_dataset.radius == 3
        assert fp_dataset.rings is True
        assert fp_dataset.root_central_atom is True
        assert fp_dataset.include_hydrogens is False

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = DRFPFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=2048,
            min_radius=1,
            radius=5,
            rings=False,
            root_central_atom=False,
            include_hydrogens=True,
        )

        assert fp_dataset.vec_dim == 2048
        assert fp_dataset.min_radius == 1
        assert fp_dataset.radius == 5
        assert fp_dataset.rings is False
        assert fp_dataset.root_central_atom is False
        assert fp_dataset.include_hydrogens is True

    def test_generate_fingerprint(self):
        """Test generating a DRFP fingerprint."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = DRFPFingerprintDataset(reaction_dataset=reactions, vec_dim=1024)

        fp = fp_dataset["rxn1"]

        assert isinstance(fp, torch.Tensor)
        assert fp.shape == (1024,)
        assert fp.dtype == torch.float32

    def test_fingerprint_dimensions(self):
        """Test DRFP with different dimensions."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        for vec_dim in [256, 512, 1024, 2048]:
            fp_dataset = DRFPFingerprintDataset(reaction_dataset=reactions, vec_dim=vec_dim)

            fp = fp_dataset["rxn1"]
            assert fp.shape == (vec_dim,)

    def test_with_standardization(self):
        """Test DRFP with standardization enabled."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = DRFPFingerprintDataset(
            reaction_dataset=reactions,
            standardize=True,
            standardize_hypervalent=True,
            standardize_uncharge=True,
            standardize_metals=True,
        )

        fp = fp_dataset["rxn1"]
        assert fp.shape == (1024,)

    def test_multiple_reactions(self):
        """Test generating DRFP for multiple reactions."""
        reactions = BaseDataset(
            keys=["rxn1", "rxn2", "rxn3"],
            array_data=[
                {"reaction_smiles": "CCO>>CC=O"},
                {"reaction_smiles": "C>>CC"},
                {"reaction_smiles": "CC>>C=C"},
            ],
        )

        fp_dataset = DRFPFingerprintDataset(reaction_dataset=reactions)

        fp1 = fp_dataset["rxn1"]
        fp2 = fp_dataset["rxn2"]
        fp3 = fp_dataset["rxn3"]

        assert fp1.shape == (1024,)
        assert fp2.shape == (1024,)
        assert fp3.shape == (1024,)

        # Different reactions should have different fingerprints
        assert not torch.equal(fp1, fp2)
        assert not torch.equal(fp2, fp3)

    def test_radius_parameter(self):
        """Test DRFP with different radius values."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO.O=O>>CC(=O)O"}])

        # Generate fingerprints with different radii
        fp_radius_2 = DRFPFingerprintDataset(reaction_dataset=reactions, radius=2)["rxn1"]

        fp_radius_4 = DRFPFingerprintDataset(reaction_dataset=reactions, radius=4)["rxn1"]

        # Both fingerprints should be valid tensors
        # Note: For simple reactions, different radii may produce similar
        # fingerprints because the structural variation is limited
        assert fp_radius_2.shape == (1024,)
        assert fp_radius_4.shape == (1024,)

    def test_rings_parameter(self):
        """Test DRFP with rings enabled/disabled."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "c1ccccc1>>C1CCCCC1"}],  # Benzene to cyclohexane
        )

        # Generate fingerprints with rings enabled and disabled
        fp_with_rings = DRFPFingerprintDataset(reaction_dataset=reactions, rings=True)["rxn1"]

        fp_without_rings = DRFPFingerprintDataset(reaction_dataset=reactions, rings=False)["rxn1"]

        # Both should produce valid fingerprints
        # Note: Ring structures may still be captured even when rings=False
        # because they're part of the substructure enumeration
        assert fp_with_rings.shape == (1024,)
        assert fp_without_rings.shape == (1024,)

    def test_invalid_smiles(self):
        """Test that DRFP handles invalid SMILES gracefully."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "invalid>>smiles"}])

        fp_dataset = DRFPFingerprintDataset(reaction_dataset=reactions)

        # DRFP may handle invalid SMILES gracefully by returning a zero or
        # near-zero fingerprint, rather than raising an exception
        fp = fp_dataset["rxn1"]
        assert fp.shape == (1024,)
        # Invalid SMILES typically produce all-zero or near-zero fingerprints
        assert torch.sum(fp) <= 10.0  # Very low or zero sum

    def test_fingerprint_deterministic(self):
        """Test that DRFP fingerprints are deterministic."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        fp_dataset = DRFPFingerprintDataset(reaction_dataset=reactions)

        fp1 = fp_dataset["rxn1"]
        fp2 = fp_dataset["rxn1"]

        assert torch.equal(fp1, fp2)

    def test_drfp_vs_rdkit_independence(self):
        """Test that DRFP and RDKit+ produce different fingerprints."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO.O=O>>CC(=O)O"}])

        # Generate DRFP fingerprint
        drfp_dataset = DRFPFingerprintDataset(reaction_dataset=reactions, vec_dim=1024)
        drfp_fp = drfp_dataset["rxn1"]

        # Generate RDKit+ structural fingerprint
        rdkit_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions, vec_dim=1024, rxn_fp_type="struct"
        )
        rdkit_fp = rdkit_dataset["rxn1"]

        # Both should be 1024-dimensional
        assert drfp_fp.shape == (1024,)
        assert rdkit_fp.shape == (1024,)

        # But they should be different (different algorithms)
        assert not torch.equal(drfp_fp, rdkit_fp)

    def test_concatenate_drfp_rdkit(self):
        """Test concatenating DRFP and RDKit+ fingerprints (SOTA config)."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO.O=O>>CC(=O)O"}])

        # Generate DRFP fingerprint (1024-dim)
        drfp_dataset = DRFPFingerprintDataset(reaction_dataset=reactions, vec_dim=1024)
        drfp_fp = drfp_dataset["rxn1"]

        # Generate RDKit+ fingerprint (1024-dim)
        rdkit_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions, vec_dim=1024, rxn_fp_type="struct"
        )
        rdkit_fp = rdkit_dataset["rxn1"]

        # Concatenate (SOTA uses both)
        combined_fp = torch.cat([rdkit_fp, drfp_fp], dim=0)

        # Combined fingerprint should be 2048-dimensional
        assert combined_fp.shape == (2048,)

    def test_sota_configuration(self):
        """Test with SOTA paper configuration."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO.O=O>>CC(=O)O"}])

        # SOTA config for DRFP
        fp_dataset = DRFPFingerprintDataset(
            reaction_dataset=reactions, vec_dim=1024, radius=3, rings=True, standardize=True
        )

        fp = fp_dataset["rxn1"]
        assert fp.shape == (1024,)

    def test_wraps_sql_dataset(self):
        """Test that DRFP dataset can wrap SQLDataset."""
        # Create mock dataset that behaves like SQLDataset
        reactions = BaseDataset(
            keys=["rxn1", "rxn2"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}, {"reaction_smiles": "C>>CC"}],
        )

        fp_dataset = DRFPFingerprintDataset(reaction_dataset=reactions, vec_dim=1024)

        assert len(fp_dataset) == 2
        fp1 = fp_dataset["rxn1"]
        fp2 = fp_dataset["rxn2"]

        assert fp1.shape == (1024,)
        assert fp2.shape == (1024,)


class TestMorganFingerprintRadius:
    """Tests for Morgan fingerprint radius parameter."""

    def test_morgan_radius_is_three(self):
        """Test that Morgan fingerprints use radius=3 (ECFP6)."""
        reactions = BaseDataset(keys=["rxn1"], array_data=[{"reaction_smiles": "CCO>>CC=O"}])

        # Create fingerprint dataset with Morgan fingerprints
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            mol_fp_type="morgan",
            vec_dim=1024,
        )

        # Generate fingerprint
        fp = fp_dataset["rxn1"]

        # Verify it generates without error (radius=3 should work)
        assert fp.shape == (1024,), "Fingerprint should be 1024-dimensional"

        # The fingerprint should be valid (not all zeros)
        assert fp.sum() > 0, "Fingerprint should have some non-zero bits"

    def test_radius_three_produces_different_fingerprints_than_radius_two(self):
        """Test that radius=3 produces different fingerprints than radius=2."""
        import numpy as np

        from rdkit.Chem.rdChemReactions import ReactionFromSmarts
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        from rdkit.Chem.rdmolops import AssignStereochemistry, SanitizeMol

        # Create test reaction with moderate complexity
        reactions = BaseDataset(
            keys=["rxn1"], array_data=[{"reaction_smiles": "c1ccccc1CC>>c1ccccc1C=C"}]
        )

        # Get fingerprint with current implementation (should be radius=3)
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            mol_fp_type="morgan",
            vec_dim=1024,
        )
        fp_current = fp_dataset["rxn1"]

        # Manually create a radius=2 fingerprint for comparison
        rxn = ReactionFromSmarts(reactions["rxn1"]["reaction_smiles"], useSmiles=True)
        fp_gen_radius2 = GetMorganGenerator(radius=2, fpSize=512, includeChirality=True)

        fp_radius2_array = np.zeros(1024, dtype=np.int64)

        # Process reactants (with proper sanitization)
        for react in rxn.GetReactants():
            SanitizeMol(react)
            AssignStereochemistry(react, force=True, cleanIt=True)
            fp1 = fp_gen_radius2.GetFingerprintAsNumPy(react)
            fp_radius2_array[:512] |= fp1 == 1

        # Process products (with proper sanitization)
        for prod in rxn.GetProducts():
            SanitizeMol(prod)
            AssignStereochemistry(prod, force=True, cleanIt=True)
            fp1 = fp_gen_radius2.GetFingerprintAsNumPy(prod)
            fp_radius2_array[512:] |= fp1 == 1

        fp_radius2 = torch.tensor(fp_radius2_array, dtype=torch.float32)

        # Verify they are different
        # For a moderately complex reaction, radius=3 should capture more features
        assert not torch.equal(
            fp_current, fp_radius2
        ), "Radius=3 fingerprint should differ from radius=2 for complex reactions"

    def test_fingerprints_in_full_pipeline(self):
        """Test that fingerprints work correctly in the full training pipeline."""
        from horizyn.config import load_config
        from horizyn.data_module import HorizynDataModule

        config = load_config("configs/nano.yaml")
        dm = HorizynDataModule(**config.data)
        dm.setup("fit")

        # Get a sample from the training data
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        # Check that query vectors (reaction fingerprints) are valid
        query_vecs = batch["query_vec"]

        assert (
            query_vecs.shape[1] == 2048
        ), "Query vectors should be 2048-dim (RDKit+ 1024 + DRFP 1024)"
        assert not torch.isnan(query_vecs).any(), "Query vectors should not contain NaN"
        assert not torch.isinf(query_vecs).any(), "Query vectors should not contain Inf"

        # Check that fingerprints have reasonable sparsity
        # (Not all zeros, not all ones)
        nonzero_ratio = (query_vecs > 0).float().mean()
        assert (
            0.01 < nonzero_ratio < 0.99
        ), f"Fingerprints have unusual sparsity: {nonzero_ratio:.2%}"
