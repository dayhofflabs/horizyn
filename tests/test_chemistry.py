"""Tests for chemistry utilities (molecular and reaction standardization)."""

import pytest

from horizyn.chemistry import (
    HypervalentStandardizer,
    KekulizeStandardizer,
    MetalStandardizer,
    RemoveHsStandardizer,
    Standardizer,
    UnchargeStandardizer,
    is_smiles_aromatic,
)


class TestIsAromatic:
    """Tests for is_smiles_aromatic function."""

    def test_aromatic_benzene(self):
        """Test that benzene is detected as aromatic."""
        assert is_smiles_aromatic("c1ccccc1") is True

    def test_nonaromatic_ethanol(self):
        """Test that ethanol is not aromatic."""
        assert is_smiles_aromatic("CCO") is False

    def test_invalid_smiles(self):
        """Test that invalid SMILES raises ValueError."""
        with pytest.raises(ValueError, match="invalid SMILES"):
            is_smiles_aromatic("invalid")


class TestHypervalentStandardizer:
    """Tests for HypervalentStandardizer."""

    def test_standardize_simple_molecule(self):
        """Test standardization of a simple molecule."""
        standardizer = HypervalentStandardizer()
        smiles = "CCO"  # Ethanol
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_standardize_invalid_smiles(self):
        """Test that invalid SMILES raises ValueError."""
        standardizer = HypervalentStandardizer()
        with pytest.raises(ValueError, match="Invalid SMILES"):
            standardizer.standardize_molecule("invalid_smiles")


class TestRemoveHsStandardizer:
    """Tests for RemoveHsStandardizer."""

    def test_remove_explicit_hydrogens(self):
        """Test removal of explicit hydrogens."""
        standardizer = RemoveHsStandardizer()
        smiles = "[H]C([H])([H])C([H])([H])O[H]"  # Ethanol with explicit H
        result = standardizer.standardize_molecule(smiles)
        # Result should be canonical SMILES without explicit H
        assert "[H]" not in result
        assert "C" in result
        assert "O" in result

    def test_standardize_simple_molecule(self):
        """Test standardization of molecule without explicit H."""
        standardizer = RemoveHsStandardizer()
        smiles = "CCO"
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)
        assert len(result) > 0


class TestKekulizeStandardizer:
    """Tests for KekulizeStandardizer."""

    def test_kekulize_benzene(self):
        """Test kekulization of benzene."""
        standardizer = KekulizeStandardizer()
        smiles = "c1ccccc1"  # Aromatic benzene
        result = standardizer.standardize_molecule(smiles)
        # Kekulized form should have explicit double bonds
        assert "C" in result  # Capital C (not lowercase c)
        assert isinstance(result, str)

    def test_kekulize_aliphatic(self):
        """Test that aliphatic compounds work fine."""
        standardizer = KekulizeStandardizer()
        smiles = "CCO"
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)


class TestUnchargeStandardizer:
    """Tests for UnchargeStandardizer."""

    def test_uncharge_ammonium(self):
        """Test uncharging of ammonium ion."""
        standardizer = UnchargeStandardizer()
        smiles = "[NH4+]"
        result = standardizer.standardize_molecule(smiles)
        # Should be uncharged
        assert "+" not in result
        assert isinstance(result, str)

    def test_uncharge_carboxylate(self):
        """Test uncharging of carboxylate."""
        standardizer = UnchargeStandardizer()
        smiles = "CC([O-])=O"  # Acetate
        result = standardizer.standardize_molecule(smiles)
        # Should be protonated to acetic acid
        assert "-" not in result
        assert isinstance(result, str)

    def test_standardize_reaction_with_charges(self):
        """Test reaction standardization balances charges with protons."""
        standardizer = UnchargeStandardizer()
        # Simple reaction with charge imbalance
        reaction = "CC(O)=O.[OH-]>>CC([O-])=O"
        result = standardizer.standardize_reaction(reaction)
        # Should contain >> separator
        assert ">>" in result
        assert isinstance(result, str)


class TestMetalStandardizer:
    """Tests for MetalStandardizer."""

    def test_disconnect_metal_bond(self):
        """Test disconnection of metal-oxygen bond."""
        standardizer = MetalStandardizer()
        # Sodium ethoxide
        smiles = "CC[O-].[Na+]"
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)
        # Metal should be disconnected (in separate fragment)
        assert "." in result or "[Na" in result

    def test_simple_molecule_unchanged(self):
        """Test that non-metal molecules work fine."""
        standardizer = MetalStandardizer()
        smiles = "CCO"
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)


class TestStandardizer:
    """Tests for the main Standardizer class."""

    def test_default_standardization(self):
        """Test standardization with default settings."""
        standardizer = Standardizer()
        smiles = "CCO"
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_standardize_with_all_options(self):
        """Test standardization with all options enabled."""
        standardizer = Standardizer(
            standardize_hypervalent=True,
            standardize_remove_hs=True,
            standardize_kekulize=True,
            standardize_uncharge=True,
            standardize_metals=True,
        )
        smiles = "CCO"
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)

    def test_standardize_with_minimal_options(self):
        """Test standardization with minimal options."""
        standardizer = Standardizer(
            standardize_hypervalent=False,
            standardize_remove_hs=False,
            standardize_kekulize=False,
            standardize_uncharge=False,
            standardize_metals=False,
        )
        smiles = "CCO"
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)

    def test_standardize_reaction(self):
        """Test reaction standardization."""
        standardizer = Standardizer()
        # Simple oxidation reaction
        reaction = "CCO.O=O>>CC(=O)O"
        result = standardizer.standardize_reaction(reaction)
        assert isinstance(result, str)
        assert ">>" in result
        # Check reactants and products are present
        parts = result.split(">>")
        assert len(parts) == 2
        assert len(parts[0]) > 0  # Reactants
        assert len(parts[1]) > 0  # Products

    def test_standardize_complex_molecule(self):
        """Test standardization of a more complex molecule."""
        standardizer = Standardizer()
        # Aspirin
        smiles = "CC(=O)Oc1ccccc1C(=O)O"
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)
        assert "C" in result
        assert "O" in result

    def test_standardize_charged_molecule(self):
        """Test standardization handles charges (if uncharge is enabled)."""
        standardizer = Standardizer(standardize_uncharge=True)
        smiles = "[NH4+]"
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)
        # With uncharge enabled, should be uncharged
        assert "+" not in result

    def test_standardize_aromatic_molecule(self):
        """Test standardization handles aromatic molecules."""
        standardizer = Standardizer()
        smiles = "c1ccccc1"  # Benzene
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_sequential_standardization(self):
        """Test that multiple standardization steps work in sequence."""
        standardizer = Standardizer(
            standardize_hypervalent=True,
            standardize_remove_hs=True,
            standardize_uncharge=True,
        )
        # Molecule with explicit H and charge
        smiles = "[H]C([H])([H])[NH3+]"
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)
        # Should have no explicit H and be uncharged
        assert "[H]" not in result
        assert "+" not in result

    def test_invalid_molecule_raises_error(self):
        """Test that invalid SMILES raises appropriate error."""
        standardizer = Standardizer()
        with pytest.raises(ValueError):
            standardizer.standardize_molecule("definitely_not_valid_smiles_123")

    def test_standardize_reaction_preserves_structure(self):
        """Test that reaction standardization preserves arrow direction."""
        standardizer = Standardizer()
        reaction = "C.C>>CC"  # Simple coupling
        result = standardizer.standardize_reaction(reaction)
        assert ">>" in result
        # Should not be reversed
        parts = result.split(">>")
        assert len(parts) == 2

