"""
Unit tests for fingerprint generation datasets.
"""

import pytest
import torch
from horizyn.datasets import BaseDataset
from horizyn.datasets.fingerprints import BaseFingerprintDataset, RDKitPlusFingerprintDataset


class TestBaseFingerprintDataset:
    """Tests for the BaseFingerprintDataset class."""

    def test_initialization(self):
        """Test basic initialization."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        fp_dataset = BaseFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=512,
            standardize=False
        )
        
        assert fp_dataset.vec_dim == 512
        assert fp_dataset.dtype == torch.float32
        assert fp_dataset.smiles_label == "reaction_smiles"
        assert len(fp_dataset) == 1

    def test_standardizer_enabled(self):
        """Test that standardizer is created when standardize=True."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        fp_dataset = BaseFingerprintDataset(
            reaction_dataset=reactions,
            standardize=True
        )
        
        assert fp_dataset.standardize is True
        assert fp_dataset.standardizer is not None

    def test_standardizer_disabled(self):
        """Test that standardizer is not created when standardize=False."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        fp_dataset = BaseFingerprintDataset(
            reaction_dataset=reactions,
            standardize=False
        )
        
        assert fp_dataset.standardize is False
        with pytest.raises(AttributeError):
            _ = fp_dataset.standardizer

    def test_query_smiles_dataset_dict(self):
        """Test querying dataset that returns dicts."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O", "extra": "data"}]
        )
        
        fp_dataset = BaseFingerprintDataset(
            reaction_dataset=reactions,
            standardize=False
        )
        
        result = fp_dataset._query_smiles_dataset("rxn1")
        assert isinstance(result, dict)
        assert "reaction_smiles" in result

    def test_query_smiles_dataset_string(self):
        """Test querying dataset that returns strings."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=["CCO>>CC=O"]
        )
        
        fp_dataset = BaseFingerprintDataset(
            reaction_dataset=reactions,
            standardize=False
        )
        
        result = fp_dataset._query_smiles_dataset("rxn1")
        assert isinstance(result, dict)
        assert result["reaction_smiles"] == "CCO>>CC=O"

    def test_query_smiles_dataset_missing_key(self):
        """Test error when dict doesn't have smiles_label."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"wrong_key": "CCO>>CC=O"}]
        )
        
        fp_dataset = BaseFingerprintDataset(
            reaction_dataset=reactions,
            standardize=False
        )
        
        with pytest.raises(KeyError, match="missing.*reaction_smiles"):
            fp_dataset._query_smiles_dataset("rxn1")

    def test_generate_fingerprint_not_implemented(self):
        """Test that base class raises error (wrapped NotImplementedError)."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        fp_dataset = BaseFingerprintDataset(
            reaction_dataset=reactions,
            standardize=False
        )
        
        with pytest.raises(Exception, match="Failed to generate fingerprint"):
            fp_dataset["rxn1"]


class TestRDKitPlusFingerprintDataset:
    """Tests for the RDKitPlusFingerprintDataset class."""

    def test_initialization_default(self):
        """Test default initialization."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions
        )
        
        assert fp_dataset.vec_dim == 1024
        assert fp_dataset.mol_fp_type == "morgan"
        assert fp_dataset.rxn_fp_type == "struct"
        assert fp_dataset.use_chirality is False
        assert fp_dataset.fp_size == 512  # Half of vec_dim

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=2048,
            mol_fp_type="rdkit",
            rxn_fp_type="diff",
            use_chirality=True
        )
        
        assert fp_dataset.vec_dim == 2048
        assert fp_dataset.mol_fp_type == "rdkit"
        assert fp_dataset.rxn_fp_type == "diff"
        assert fp_dataset.use_chirality is True

    def test_invalid_mol_fp_type(self):
        """Test that invalid mol_fp_type raises error."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        with pytest.raises(ValueError, match="Invalid mol_fp_type"):
            RDKitPlusFingerprintDataset(
                reaction_dataset=reactions,
                mol_fp_type="invalid"
            )

    def test_invalid_rxn_fp_type(self):
        """Test that invalid rxn_fp_type raises error."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        with pytest.raises(ValueError, match="Invalid rxn_fp_type"):
            RDKitPlusFingerprintDataset(
                reaction_dataset=reactions,
                rxn_fp_type="invalid"
            )

    def test_odd_vec_dim_struct(self):
        """Test that odd vec_dim raises error for structural fingerprints."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        with pytest.raises(ValueError, match="vec_dim must be even"):
            RDKitPlusFingerprintDataset(
                reaction_dataset=reactions,
                vec_dim=1023,  # Odd
                rxn_fp_type="struct"
            )

    def test_generate_structural_fingerprint(self):
        """Test generating a structural fingerprint."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=1024,
            rxn_fp_type="struct"
        )
        
        fp = fp_dataset["rxn1"]
        
        assert isinstance(fp, torch.Tensor)
        assert fp.shape == (1024,)
        assert fp.dtype == torch.float32
        # Structural fingerprints should be binary (0 or 1)
        assert torch.all((fp == 0) | (fp == 1))

    def test_generate_diff_fingerprint(self):
        """Test generating a difference fingerprint."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=1024,
            rxn_fp_type="diff"
        )
        
        fp = fp_dataset["rxn1"]
        
        assert isinstance(fp, torch.Tensor)
        assert fp.shape == (1024,)
        # Difference fingerprints can have counts (integers)

    def test_mol_fp_types(self):
        """Test different molecular fingerprint types."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        for fp_type in ["morgan", "rdkit", "atom_pair", "topological_torsion"]:
            fp_dataset = RDKitPlusFingerprintDataset(
                reaction_dataset=reactions,
                mol_fp_type=fp_type,
                vec_dim=1024
            )
            
            fp = fp_dataset["rxn1"]
            assert fp.shape == (1024,)

    def test_with_chirality(self):
        """Test fingerprint generation with chirality enabled."""
        reactions = BaseDataset(
            keys=["rxn1"],
            # Use a simple reaction without invalid chiral centers
            array_data=[{"reaction_smiles": "C[C@H](O)CC>>C[C@H](Cl)CC"}]
        )
        
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            use_chirality=True
        )
        
        fp = fp_dataset["rxn1"]
        assert fp.shape == (1024,)

    def test_with_standardization(self):
        """Test fingerprint generation with standardization."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            standardize=True,
            standardize_hypervalent=True,
            standardize_uncharge=True,
            standardize_metals=True
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
                {"reaction_smiles": "CC>>C=C"}
            ]
        )
        
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions
        )
        
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
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "invalid>>smiles"}]
        )
        
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions
        )
        
        with pytest.raises(Exception):
            fp_dataset["rxn1"]

    def test_sota_configuration(self):
        """Test with SOTA paper configuration."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO.O=O>>CC(=O)O"}]
        )
        
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
            standardize_metals=True
        )
        
        fp = fp_dataset["rxn1"]
        assert fp.shape == (1024,)
        assert torch.all((fp == 0) | (fp == 1))  # Binary

    def test_fingerprint_deterministic(self):
        """Test that fingerprints are deterministic."""
        reactions = BaseDataset(
            keys=["rxn1"],
            array_data=[{"reaction_smiles": "CCO>>CC=O"}]
        )
        
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions
        )
        
        fp1 = fp_dataset["rxn1"]
        fp2 = fp_dataset["rxn1"]
        
        assert torch.equal(fp1, fp2)

    def test_wraps_sql_dataset(self):
        """Test that fingerprint dataset can wrap SQLDataset."""
        # Create mock dataset that behaves like SQLDataset
        reactions = BaseDataset(
            keys=["rxn1", "rxn2"],
            array_data=[
                {"reaction_smiles": "CCO>>CC=O"},
                {"reaction_smiles": "C>>CC"}
            ]
        )
        
        fp_dataset = RDKitPlusFingerprintDataset(
            reaction_dataset=reactions,
            vec_dim=1024
        )
        
        assert len(fp_dataset) == 2
        fp1 = fp_dataset["rxn1"]
        fp2 = fp_dataset["rxn2"]
        
        assert fp1.shape == (1024,)
        assert fp2.shape == (1024,)

