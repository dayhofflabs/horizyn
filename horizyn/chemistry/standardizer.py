"""
Molecular and reaction standardization classes.

Extracted from dayhoff-tools for standalone use in Horizyn.
"""

from abc import ABC, abstractmethod

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")


def is_smiles_aromatic(smiles: str) -> bool:
    """Check if SMILES string contains aromatic atoms.

    Args:
        smiles: Input SMILES string

    Returns:
        True if aromatic atoms are found

    Raises:
        ValueError: If SMILES string is invalid
    """
    rdmol = Chem.MolFromSmiles(smiles, sanitize=False)  # type: ignore
    if rdmol is None:
        raise ValueError("invalid SMILES string")
    return any(at.GetIsAromatic() for at in rdmol.GetAtoms())


class BaseStandardizer(ABC):
    """Abstract base class for normalizing molecules and reactions."""

    @abstractmethod
    def standardize_molecule(self, smiles: str) -> str:
        """Standardize molecules as SMILES strings.

        Args:
            smiles: Input SMILES string

        Returns:
            Standardized SMILES string

        Raises:
            ValueError: If SMILES string is invalid
        """
        pass

    def standardize_reaction(self, smiles: str) -> str:
        """Standardize reactions as SMILES/SMARTS strings.

        Args:
            smiles: Input reaction SMILES/SMARTS string

        Returns:
            Standardized reaction SMILES string

        Raises:
            ValueError: If reaction string is invalid
        """
        rdrxn = AllChem.ReactionFromSmarts(smiles, useSmiles=True)  # type: ignore
        rdrxn1 = AllChem.ChemicalReaction()  # type: ignore
        for rdmol in rdrxn.GetReactants():
            smiles1 = Chem.MolToSmiles(rdmol, canonical=False)  # type: ignore
            smiles2 = self.standardize_molecule(smiles1)
            rdmol1 = Chem.MolFromSmiles(smiles2, sanitize=False)  # type: ignore
            rdfrags = Chem.GetMolFrags(rdmol1, asMols=True, sanitizeFrags=False)  # type: ignore
            if len(rdfrags) == 1:
                rdrxn1.AddReactantTemplate(rdmol1)
            else:
                for rdfrag in rdfrags:
                    rdrxn1.AddReactantTemplate(rdfrag)
        for rdmol in rdrxn.GetProducts():
            smiles1 = Chem.MolToSmiles(rdmol, canonical=False)  # type: ignore
            smiles2 = self.standardize_molecule(smiles1)
            rdmol1 = Chem.MolFromSmiles(smiles2, sanitize=False)  # type: ignore
            rdfrags = Chem.GetMolFrags(rdmol1, asMols=True, sanitizeFrags=False)  # type: ignore
            if len(rdfrags) == 1:
                rdrxn1.AddProductTemplate(rdmol1)
            else:
                for rdfrag in rdfrags:
                    rdrxn1.AddProductTemplate(rdfrag)
        return AllChem.ReactionToSmiles(rdrxn1)  # type: ignore


class HypervalentStandardizer(BaseStandardizer):
    """Standardizer for converting double to single bonds in hypervalent compounds."""

    def standardize_molecule(self, smiles: str) -> str:
        """Standardize molecules as SMILES strings.

        Args:
            smiles: Input SMILES string

        Returns:
            Standardized SMILES string

        Raises:
            ValueError: If SMILES string is invalid or sanitization fails
        """
        rdmol = Chem.MolFromSmiles(smiles, sanitize=False)  # type: ignore
        if rdmol is None:
            raise ValueError(f"Invalid SMILES input '{smiles}'")
        ret = Chem.SanitizeMol(rdmol, sanitizeOps=Chem.SANITIZE_CLEANUP)  # type: ignore
        if ret > 0:
            raise ValueError(f"Sanitization failed for SMILES input '{smiles}'")
        return Chem.MolToSmiles(rdmol)  # type: ignore


class RemoveHsStandardizer(BaseStandardizer):
    """Standardizer for removing explicit hydrogens from molecules."""

    def standardize_molecule(self, smiles: str) -> str:
        """Standardize molecules as SMILES strings.

        Args:
            smiles: Input SMILES string

        Returns:
            Standardized SMILES string with explicit hydrogens removed

        Raises:
            ValueError: If SMILES string is invalid or sanitization fails
        """
        rdmol = Chem.MolFromSmiles(smiles, sanitize=False)  # type: ignore
        if rdmol is None:
            raise ValueError(f"Invalid SMILES input '{smiles}'")
        rdmol1 = Chem.RemoveHs(rdmol, sanitize=False)  # type: ignore
        ret = Chem.SanitizeMol(rdmol1, sanitizeOps=Chem.SANITIZE_FINDRADICALS)  # type: ignore
        if ret > 0:
            raise ValueError(f"Sanitization failed for SMILES input '{smiles}'")
        return Chem.MolToSmiles(rdmol1, canonical=True)  # type: ignore


class KekulizeStandardizer(BaseStandardizer):
    """Standardizer for kekulizing aromatic compounds."""

    def standardize_molecule(self, smiles: str) -> str:
        """Standardize molecules as SMILES strings by kekulizing aromatics.

        Args:
            smiles: Input SMILES string

        Returns:
            Kekulized SMILES string

        Raises:
            ValueError: If SMILES string is invalid
        """
        rdmol = Chem.MolFromSmiles(smiles, sanitize=False)  # type: ignore
        if rdmol is None:
            raise ValueError(f"Invalid SMILES input '{smiles}'")
        rdmol.UpdatePropertyCache(strict=False)
        Chem.Kekulize(rdmol, clearAromaticFlags=True)  # type: ignore
        return Chem.MolToSmiles(rdmol, canonical=True)  # type: ignore


class UnchargeStandardizer(BaseStandardizer):
    """Standardizer for removing charges from molecules by protonation/deprotonation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._uncharger = rdMolStandardize.Uncharger()

    def standardize_molecule(self, smiles: str) -> str:
        """Standardize molecules as SMILES strings by removing charges.

        Args:
            smiles: Input SMILES string

        Returns:
            Uncharged SMILES string

        Raises:
            ValueError: If SMILES string is invalid
        """
        rdmol = Chem.MolFromSmiles(smiles, sanitize=False)  # type: ignore
        if rdmol is None:
            raise ValueError(f"Invalid SMILES input '{smiles}'")
        rdmol1 = self._uncharger.uncharge(rdmol)
        return Chem.MolToSmiles(rdmol1)  # type: ignore

    def standardize_reaction(self, smiles: str) -> str:
        """Standardize reactions by removing charges and balancing with protons.

        Args:
            smiles: Input reaction SMILES/SMARTS string

        Returns:
            Uncharged and balanced reaction SMILES string

        Raises:
            ValueError: If reaction string is invalid
        """
        rdrxn = AllChem.ReactionFromSmarts(smiles, useSmiles=True)  # type: ignore
        rdrxn1 = AllChem.ChemicalReaction()  # type: ignore

        # Remove all explicit protons from the reaction
        reactant_total_charge = 0
        product_total_charge = 0
        for rdmol in rdrxn.GetReactants():
            smiles1 = Chem.MolToSmiles(rdmol, canonical=False)  # type: ignore
            if smiles1 != "[H+]":
                smiles2 = self.standardize_molecule(smiles1)
                rdmol1 = Chem.MolFromSmiles(smiles2, sanitize=False)  # type: ignore
                reactant_total_charge += Chem.GetFormalCharge(rdmol1)  # type: ignore
                rdrxn1.AddReactantTemplate(rdmol1)
        for rdmol in rdrxn.GetProducts():
            smiles1 = Chem.MolToSmiles(rdmol, canonical=False)  # type: ignore
            if smiles1 != "[H+]":
                smiles2 = self.standardize_molecule(smiles1)
                rdmol1 = Chem.MolFromSmiles(smiles2, sanitize=False)  # type: ignore
                product_total_charge += Chem.GetFormalCharge(rdmol1)  # type: ignore
                rdrxn1.AddProductTemplate(rdmol1)

        # Rebalance reaction with protons
        if reactant_total_charge > product_total_charge:
            rdmol1 = Chem.MolFromSmiles("[H+]", sanitize=False)  # type: ignore
            for _ in range(reactant_total_charge - product_total_charge):
                rdrxn1.AddProductTemplate(rdmol1)
        elif product_total_charge > reactant_total_charge:
            rdmol1 = Chem.MolFromSmiles("[H+]", sanitize=False)  # type: ignore
            for _ in range(product_total_charge - reactant_total_charge):
                rdrxn1.AddReactantTemplate(rdmol1)
        return AllChem.ReactionToSmiles(rdrxn1)  # type: ignore


class MetalStandardizer(BaseStandardizer):
    """Standardizer for disconnecting bonds between metals and N, O, F atoms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._disconnector = rdMolStandardize.MetalDisconnector()

    def standardize_molecule(self, smiles: str) -> str:
        """Standardize molecules by disconnecting metal-heteroatom bonds.

        Args:
            smiles: Input SMILES string

        Returns:
            Standardized SMILES string with metal bonds disconnected

        Raises:
            ValueError: If SMILES string is invalid
        """
        rdmol = Chem.MolFromSmiles(smiles, sanitize=False)  # type: ignore
        if rdmol is None:
            raise ValueError(f"Invalid SMILES input '{smiles}'")

        flags = Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES  # type: ignore
        if not is_smiles_aromatic(smiles):
            flags ^= Chem.SANITIZE_SETAROMATICITY  # type: ignore
        Chem.SanitizeMol(rdmol, sanitizeOps=flags)  # type: ignore
        rdmol1 = self._disconnector.Disconnect(rdmol)
        return Chem.MolToSmiles(rdmol1)  # type: ignore


class Standardizer(BaseStandardizer):
    """Aggregate standardizer for molecules and reactions.

    This is the main standardizer class used in Horizyn for consistent
    fingerprint generation. It combines multiple standardization steps
    in sequence.
    """

    def __init__(
        self,
        *,
        standardize_hypervalent: bool = True,
        standardize_remove_hs: bool = True,
        standardize_kekulize: bool = False,
        standardize_uncharge: bool = False,
        standardize_metals: bool = True,
    ):
        """Initialize the standardizer with specified operations.

        Args:
            standardize_hypervalent: Convert double to single bonds in
                hypervalent compounds
            standardize_remove_hs: Remove explicit hydrogen atoms
            standardize_kekulize: Kekulize aromatic compounds
            standardize_uncharge: Remove charges from molecules by
                protonation/deprotonation
            standardize_metals: Disconnect bonds between metals and
                N, O, F atoms
        """
        self._standardizers = []
        if standardize_hypervalent:
            self._standardizers.append(HypervalentStandardizer())
        if standardize_remove_hs:
            self._standardizers.append(RemoveHsStandardizer())
        if standardize_kekulize:
            self._standardizers.append(KekulizeStandardizer())
        if standardize_uncharge:
            self._standardizers.append(UnchargeStandardizer())
        if standardize_metals:
            self._standardizers.append(MetalStandardizer())

    def standardize_molecule(self, smiles: str) -> str:
        """Standardize molecules as SMILES strings.

        Applies all configured standardization steps in sequence.

        Args:
            smiles: Input SMILES string

        Returns:
            Standardized SMILES string

        Raises:
            ValueError: If SMILES string is invalid or standardization fails
        """
        smiles1 = smiles
        for standardizer in self._standardizers:
            smiles1 = standardizer.standardize_molecule(smiles1)
        return smiles1

    def standardize_reaction(self, smiles: str) -> str:
        """Standardize reactions as SMILES/SMARTS strings.

        Applies all configured standardization steps in sequence.

        Args:
            smiles: Input reaction SMILES/SMARTS string

        Returns:
            Standardized reaction SMILES string

        Raises:
            ValueError: If reaction string is invalid or standardization fails
        """
        smiles1 = smiles
        for standardizer in self._standardizers:
            smiles1 = standardizer.standardize_reaction(smiles1)
        return smiles1
