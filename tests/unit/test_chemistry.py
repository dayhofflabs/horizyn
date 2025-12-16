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

    def test_empty_string_raises(self):
        """Empty SMILES should raise ValueError."""
        with pytest.raises(ValueError, match="empty SMILES"):
            is_smiles_aromatic("")


class TestHypervalentStandardizer:
    """Tests for HypervalentStandardizer."""

    def test_standardize_simple_molecule(self):
        """Test standardization of a simple molecule."""
        standardizer = HypervalentStandardizer()
        smiles = "CCO"  # Ethanol
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chloric_acid(self):
        """Test standardization of chloric acid."""
        standardizer = HypervalentStandardizer()
        assert standardizer.standardize_molecule("O=Cl(=O)O") == "[O-][Cl+2]([O-])O"
        assert standardizer.standardize_molecule("[O-][Cl+2]([O-])O") == "[O-][Cl+2]([O-])O"

    def test_perchlorate(self):
        """Test standardization of perchlorate."""
        standardizer = HypervalentStandardizer()
        assert standardizer.standardize_molecule("O=Cl(=O)[O-]") == "[O-][Cl+2]([O-])[O-]"

    def test_nitric_acid(self):
        """Test standardization of nitric acid."""
        standardizer = HypervalentStandardizer()
        assert standardizer.standardize_molecule("O=N(=O)O") == "O=[N+]([O-])O"

    def test_phosphoric_acid_unchanged(self):
        """Test that phosphoric acid is unchanged."""
        standardizer = HypervalentStandardizer()
        assert standardizer.standardize_molecule("O=P(O)(O)O") == "O=P(O)(O)O"

    def test_sulfuric_acid_unchanged(self):
        """Test that sulfuric acid is unchanged."""
        standardizer = HypervalentStandardizer()
        assert standardizer.standardize_molecule("O=S(=O)(O)O") == "O=S(=O)(O)O"

    def test_water_with_explicit_h(self):
        """Test that water with explicit H is unchanged."""
        standardizer = HypervalentStandardizer()
        assert standardizer.standardize_molecule("[H]O[H]") == "[H]O[H]"

    def test_pyridine_unchanged(self):
        """Test that aromatic pyridine is unchanged."""
        standardizer = HypervalentStandardizer()
        assert standardizer.standardize_molecule("c1ccncc1") == "c1ccncc1"

    def test_formate_unchanged(self):
        """Test that formate is unchanged."""
        standardizer = HypervalentStandardizer()
        assert standardizer.standardize_molecule("O=C[O-]") == "O=C[O-]"

    def test_standardize_reaction(self):
        """Test standardization of a reaction."""
        standardizer = HypervalentStandardizer()
        result = standardizer.standardize_reaction("O=[Cl:1](=O)O.[H][H]>>O=[Cl:1]O.O")
        assert result == "[H][H].[O-][Cl+2:1]([O-])O>>O.[O-][Cl+:1]O"

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

    def test_water_explicit_h(self):
        """Test removal of explicit H from water."""
        standardizer = RemoveHsStandardizer()
        assert standardizer.standardize_molecule("[H]O[H]") == "O"

    def test_hydrogen_molecule_preserved(self):
        """Test that H2 molecule is preserved."""
        standardizer = RemoveHsStandardizer()
        assert standardizer.standardize_molecule("[H][H]") == "[H][H]"

    def test_aromatic_benzene_explicit_h(self):
        """Test removal of explicit H from aromatic benzene."""
        standardizer = RemoveHsStandardizer()
        assert (
            standardizer.standardize_molecule("c([H])1c([H])c([H])c([H])c([H])c([H])1")
            == "c1ccccc1"
        )

    def test_kekulized_benzene_explicit_h(self):
        """Test removal of explicit H from kekulized benzene."""
        standardizer = RemoveHsStandardizer()
        assert (
            standardizer.standardize_molecule("C([H])1=C([H])C([H])=C([H])C([H])=C([H])1")
            == "C1=CC=CC=C1"
        )

    def test_hypervalent_unchanged(self):
        """Test that hypervalent structure is unchanged."""
        standardizer = RemoveHsStandardizer()
        assert standardizer.standardize_molecule("O=Cl(=O)O") == "O=Cl(=O)O"

    def test_pyridine_unchanged(self):
        """Test that pyridine is unchanged."""
        standardizer = RemoveHsStandardizer()
        assert standardizer.standardize_molecule("c1ccncc1") == "c1ccncc1"

    def test_formate_unchanged(self):
        """Test that formate is unchanged."""
        standardizer = RemoveHsStandardizer()
        assert standardizer.standardize_molecule("O=C[O-]") == "O=C[O-]"

    def test_standardize_reaction(self):
        """Test reaction standardization."""
        standardizer = RemoveHsStandardizer()
        result = standardizer.standardize_reaction("[H]N([H])[H].[H+]>>[H][N+]([H])([H])[H]")
        assert ">>" in result

    def test_peroxide_reaction(self):
        """Test reaction with peroxide formation."""
        standardizer = RemoveHsStandardizer()
        result = standardizer.standardize_reaction("O=O.[H]O[H].[H]O[H]>>[H]OO[H].[H]OO[H]")
        assert ">>" in result

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

    def test_kekulize_pyridine(self):
        """Test kekulization of pyridine."""
        standardizer = KekulizeStandardizer()
        assert standardizer.standardize_molecule("c1ccncc1") == "C1=CC=NC=C1"

    def test_kekulize_already_kekulized(self):
        """Test that already kekulized structures are unchanged."""
        standardizer = KekulizeStandardizer()
        assert standardizer.standardize_molecule("C1=CC=NC=C1") == "C1=CC=NC=C1"

    def test_kekulize_canonical(self):
        """Test that kekulization produces canonical form."""
        standardizer = KekulizeStandardizer()
        # Different kekulized forms should canonicalize to the same result
        assert standardizer.standardize_molecule("C1=CC=CC=N1") == "C1=CC=NC=C1"

    def test_kekulize_complex_nad(self):
        """Test kekulization of complex NAD+ molecule."""
        standardizer = KekulizeStandardizer()
        nad = (
            "NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP([O-])(=O)OP([O-])(=O)"
            "OC[C@H]2O[C@H]([C@H](O)[C@@H]2O)n2cnc3c(N)ncnc23)[C@@H](O)[C@H]1O"
        )
        result = standardizer.standardize_molecule(nad)
        # Should be kekulized - check for capital letters
        assert "C" in result
        assert "N" in result

    def test_hypervalent_unchanged(self):
        """Test that hypervalent structures are unchanged."""
        standardizer = KekulizeStandardizer()
        assert standardizer.standardize_molecule("O=Cl(=O)O") == "O=Cl(=O)O"

    def test_water_unchanged(self):
        """Test that water is unchanged."""
        standardizer = KekulizeStandardizer()
        assert standardizer.standardize_molecule("[H]O[H]") == "[H]O[H]"

    def test_formate_unchanged(self):
        """Test that formate is unchanged."""
        standardizer = KekulizeStandardizer()
        assert standardizer.standardize_molecule("O=C[O-]") == "O=C[O-]"

    def test_kekulize_reaction(self):
        """Test kekulization of reactions."""
        standardizer = KekulizeStandardizer()
        reaction = (
            "[H+].O[C@H]1C=C[C@](CC(=O)C([O-])=O)(C=C1)C([O-])=O>>"
            "[O-]C(=O)C(=O)Cc1ccccc1.O=C=O.O"
        )
        result = standardizer.standardize_reaction(reaction)
        assert ">>" in result
        # Products should have kekulized benzene
        assert "C1=CC=CC=C1" in result or "C=C" in result

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
        assert standardizer.standardize_molecule("[NH4+]") == "N"
        assert standardizer.standardize_molecule("N") == "N"

    def test_uncharge_carboxylate(self):
        """Test uncharging of carboxylate."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("C(=O)[O-]") == "O=CO"

    def test_uncharge_formate(self):
        """Test that formate gets protonated."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("O=C[O-]") == "O=CO"

    def test_uncharge_carbonate(self):
        """Test uncharging of carbonate."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("[O-]C(=O)[O-]") == "O=C(O)O"
        assert standardizer.standardize_molecule("OC(=O)[O-]") == "O=C(O)O"

    def test_uncharge_sulfate(self):
        """Test uncharging of sulfate."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("O=S(=O)(O)[O-]") == "O=S(=O)(O)O"
        assert standardizer.standardize_molecule("O=S(=O)([O-])[O-]") == "O=S(=O)(O)O"

    def test_uncharge_phosphate(self):
        """Test uncharging of phosphate."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("[O-]P(=O)([O-])[O-]") == "O=P(O)(O)O"
        assert standardizer.standardize_molecule("[O-]P(=O)([O-])O") == "O=P(O)(O)O"
        assert standardizer.standardize_molecule("[O-]P(=O)(O)O") == "O=P(O)(O)O"

    def test_proton_preserved(self):
        """Test that isolated protons are preserved."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("[H+]") == "[H+]"

    def test_uncharge_primary_amine(self):
        """Test uncharging of primary ammonium."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("CC[NH3+]") == "CCN"
        assert standardizer.standardize_molecule("CCN") == "CCN"

    def test_uncharge_secondary_amine(self):
        """Test uncharging of secondary ammonium."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("CC[NH2+]CC") == "CCNCC"
        assert standardizer.standardize_molecule("CCNCC") == "CCNCC"

    def test_uncharge_tertiary_amine(self):
        """Test uncharging of tertiary ammonium."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("CC[NH+](CC)CC") == "CCN(CC)CC"

    def test_quaternary_amine_preserved(self):
        """Test that quaternary ammonium is preserved."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("C[N+](C)(C)C") == "C[N+](C)(C)C"

    def test_uncharge_guanidine(self):
        """Test uncharging of guanidinium."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("CNC(=[NH2+])N") == "CNC(=N)N"

    def test_uncharge_imidazole(self):
        """Test uncharging of protonated imidazole."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("C1=[NH+]C=CN1") == "C1=CNC=N1"

    def test_uncharge_pyridine(self):
        """Test uncharging of pyridinium."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("C1=C[NH+]=CC=C1") == "C1=CC=NC=C1"
        assert standardizer.standardize_molecule("c1ccc[nH+]c1") == "c1ccncc1"

    def test_uncharge_amino_acid(self):
        """Test uncharging of zwitterionic amino acid."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("[NH3+]CC(=O)[O-]") == "NCC(=O)O"

    def test_uncharge_atp(self):
        """Test uncharging of ATP."""
        standardizer = UnchargeStandardizer()
        atp = (
            "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP([O-])(=O)OP([O-])(=O)"
            "OP([O-])([O-])=O)[C@@H](O)[C@H]1O"
        )
        result = standardizer.standardize_molecule(atp)
        # Should have no charges
        assert "[O-]" not in result
        assert "OP(=O)(O)" in result

    def test_standardize_reaction_with_charges(self):
        """Test reaction standardization balances charges with protons."""
        standardizer = UnchargeStandardizer()
        result = standardizer.standardize_reaction("C(=O)[O-].[NH4+]>>O=CN.O")
        assert result == "N.O=CO>>NC=O.O"

    def test_hypervalent_unchanged(self):
        """Test that hypervalent structures are unchanged."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("O=Cl(=O)O") == "O=Cl(=O)O"

    def test_water_unchanged(self):
        """Test that water is unchanged."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("[H]O[H]") == "[H]O[H]"

    def test_pyridine_aromatic_unchanged(self):
        """Test that aromatic pyridine is unchanged."""
        standardizer = UnchargeStandardizer()
        assert standardizer.standardize_molecule("c1ccncc1") == "c1ccncc1"


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

    def test_iron_porphyrin(self):
        """Test metal standardization of iron porphyrin."""
        standardizer = MetalStandardizer()
        # Simplified heme structure
        heme = "C12=CC3=NC(C=C3)=CC4=CC=C5N4[Fe]N1C(C=C2)=CC6=NC(C=C6)=C5"
        result = standardizer.standardize_molecule(heme)
        # Should have disconnected iron and anionic nitrogens
        assert "[Fe" in result
        assert "[N-]" in result
        assert "." in result  # Separate fragments

    def test_iron_porphyrin_canonical(self):
        """Test canonical output for iron porphyrin."""
        standardizer = MetalStandardizer()
        heme = "C12=CC3=NC(C=C3)=CC4=CC=C5N4[Fe]N1C(C=C2)=CC6=NC(C=C6)=C5"
        result = standardizer.standardize_molecule(heme)
        # Check that it produces the expected output
        assert "C1=CC2=NC1=CC1=CC=C(C=C3C=CC(=N3)C=C3C=CC(=C2)[N-]3)[N-]1.[Fe+2]" == result

    def test_iron_porphyrin_alternate(self):
        """Test metal standardization with alternate porphyrin notation."""
        standardizer = MetalStandardizer()
        porphyrin = "c1(=CC2=NC(C=C2)=Cc3n4c(C=C5C=CC(C=6)=N5)cc3)n([Fe]4)c6cc1"
        result = standardizer.standardize_molecule(porphyrin)
        # Should disconnect metal
        assert "[Fe" in result
        assert "." in result

    def test_simple_molecule_unchanged(self):
        """Test that non-metal molecules work fine."""
        standardizer = MetalStandardizer()
        smiles = "CCO"
        result = standardizer.standardize_molecule(smiles)
        assert isinstance(result, str)

    def test_organic_nitrogen_unchanged(self):
        """Test that organic nitrogen compounds are not affected."""
        standardizer = MetalStandardizer()
        # Pyridine should be unchanged
        assert "c1ccncc1" in standardizer.standardize_molecule("c1ccncc1")


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

    def test_invalid_reaction_raises_error(self):
        """Invalid reaction strings should raise a ValueError with context."""
        standardizer = Standardizer()
        with pytest.raises(ValueError):
            standardizer.standardize_reaction("not_a_reaction")

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

    def test_empty_molecule_raises_error(self):
        """Empty SMILES input should raise ValueError."""
        standardizer = Standardizer()
        with pytest.raises(ValueError):
            standardizer.standardize_molecule("")

    def test_canonicalization_ethanol(self):
        """Non-canonical inputs should canonicalize to a consistent SMILES."""
        standardizer = Standardizer()
        # "OCC" should canonicalize to "CCO"
        result = standardizer.standardize_molecule("OCC")
        assert result == "CCO"

    def test_standardize_reaction_preserves_structure(self):
        """Test that reaction standardization preserves arrow direction."""
        standardizer = Standardizer()
        reaction = "C.C>>CC"  # Simple coupling
        result = standardizer.standardize_reaction(reaction)
        assert ">>" in result
        # Should not be reversed
        parts = result.split(">>")
        assert len(parts) == 2

    def test_default_standardizer_peroxide(self):
        """Test default standardizer with peroxide."""
        standardizer = Standardizer()
        assert standardizer.standardize_molecule("[OH][OH]") == "OO"

    def test_default_standardizer_hypervalent(self):
        """Test default standardizer handles hypervalent structures."""
        standardizer = Standardizer()
        assert standardizer.standardize_molecule("O=Cl(=O)O") == "[O-][Cl+2]([O-])O"
        assert standardizer.standardize_molecule("[O-][Cl+2]([O-])O") == "[O-][Cl+2]([O-])O"
        assert standardizer.standardize_molecule("O=Cl(=O)[O-]") == "[O-][Cl+2]([O-])[O-]"

    def test_default_standardizer_water(self):
        """Test default standardizer removes explicit H from water."""
        standardizer = Standardizer()
        assert standardizer.standardize_molecule("[H]O[H]") == "O"

    def test_default_standardizer_benzene(self):
        """Test default standardizer with benzene."""
        standardizer = Standardizer()
        assert (
            standardizer.standardize_molecule("C([H])1=C([H])C([H])=C([H])C([H])=C([H])1")
            == "C1=CC=CC=C1"
        )

    def test_default_standardizer_pyridine(self):
        """Test default standardizer leaves aromatic pyridine aromatic."""
        standardizer = Standardizer()
        assert standardizer.standardize_molecule("c1ccncc1") == "c1ccncc1"

    def test_default_standardizer_kekulized_benzene(self):
        """Test that kekulized benzene stays kekulized with default settings."""
        standardizer = Standardizer()
        assert standardizer.standardize_molecule("C1=CC=CC=C1") == "C1=CC=CC=C1"

    def test_default_standardizer_charged(self):
        """Test default standardizer with charged species."""
        standardizer = Standardizer()
        # Note: default standardizer typically does NOT uncharge by default
        assert standardizer.standardize_molecule("O=C[O-]") == "O=C[O-]"

    def test_default_standardizer_metal(self):
        """Test default standardizer with metal complex."""
        standardizer = Standardizer()
        heme = "C12=CC3=NC(C=C3)=CC4=CC=C5N4[Fe]N1C(C=C2)=CC6=NC(C=C6)=C5"
        result = standardizer.standardize_molecule(heme)
        # Should disconnect metal
        assert "[Fe+2]" in result
        assert "[N-]" in result

    def test_all_options_enabled(self):
        """Test standardizer with all options enabled."""
        standardizer = Standardizer(
            standardize_hypervalent=True,
            standardize_remove_hs=True,
            standardize_kekulize=True,
            standardize_uncharge=True,
            standardize_metals=True,
        )
        # Peroxide with explicit H
        assert standardizer.standardize_molecule("[OH][OH]") == "OO"
        # Hypervalent
        assert standardizer.standardize_molecule("O=Cl(=O)O") == "[O-][Cl+2]([O-])O"
        # Aromatic → Kekulized
        assert standardizer.standardize_molecule("c1ccncc1") == "C1=CC=NC=C1"
        # Charged → Uncharged
        assert standardizer.standardize_molecule("O=C[O-]") == "O=CO"

    def test_all_options_disabled(self):
        """Test standardizer with all options disabled (null standardizer)."""
        standardizer = Standardizer(
            standardize_hypervalent=False,
            standardize_remove_hs=False,
            standardize_kekulize=False,
            standardize_uncharge=False,
            standardize_metals=False,
        )
        # Everything should be canonicalized but not transformed
        assert standardizer.standardize_molecule("[OH][OH]") == "[OH][OH]"
        assert standardizer.standardize_molecule("O=Cl(=O)O") == "O=Cl(=O)O"
        assert standardizer.standardize_molecule("[H]O[H]") == "[H]O[H]"
        assert standardizer.standardize_molecule("c1ccncc1") == "c1ccncc1"
        assert standardizer.standardize_molecule("C(=O)[O-]") == "C(=O)[O-]"

    def test_combined_standardization_complex(self):
        """Test combined standardization on complex molecule."""
        standardizer = Standardizer(
            standardize_hypervalent=True,
            standardize_remove_hs=True,
            standardize_kekulize=True,
            standardize_uncharge=True,
            standardize_metals=True,
        )
        # Benzoate with explicit H
        result = standardizer.standardize_molecule("c1ccccc1C(=O)[O-]")
        # Should be kekulized, uncharged
        assert "C1=CC=CC=C1" in result or "C=C" in result
        assert "[O-]" not in result
        assert "O=C(O)" in result or "C(=O)O" in result
