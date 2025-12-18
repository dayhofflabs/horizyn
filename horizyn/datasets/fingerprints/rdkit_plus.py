"""
RDKit+ structural fingerprint generation for reactions.
"""

from typing import Any, Dict

import numpy as np
import torch
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetRDKitFPGenerator,
    GetTopologicalTorsionGenerator,
)
from rdkit.Chem.rdmolfiles import MolToSmiles
from rdkit.Chem.rdmolops import AssignStereochemistry, SanitizeMol

from horizyn.datasets.base import BaseDataset, K
from horizyn.datasets.fingerprints.base import BaseFingerprintDataset


class RDKitPlusFingerprintDataset(BaseFingerprintDataset):
    """
    RDKit+ reaction fingerprint dataset.

    Generates structural fingerprints for chemical reactions using RDKit's
    molecular fingerprint generators. The reaction fingerprint concatenates
    binary fingerprints of reactants and products.

    This is the "RDKit+" variant that uses improved fingerprint generators
    and supports various molecular fingerprint types.

    Supported fingerprint types:
        - "morgan": Morgan (ECFP-like) fingerprints
        - "rdkit": RDKit topological fingerprints
        - "atom_pair": Atom pair fingerprints
        - "topological_torsion": Topological torsion fingerprints

    Fingerprint modes:
        - "struct": Structural fingerprint (concatenate reactants | products)
        - "diff": Difference fingerprint (count differences)

    Citation:
        Schneider, N.; Lowe, D. M.; Sayle, R. A.; Landrum, G. A. Development
        of a Novel Fingerprint for Chemical Reactions and Its Application to
        Large-Scale Reaction Classification and Similarity. J. Chem. Inf.
        Model. 2015, 55 (1), 39–53. DOI: https://doi.org/10.1021/ci5006614

    Attributes:
        mol_fp_type (str): Type of molecular fingerprint.
        rxn_fp_type (str): Type of reaction fingerprint ("struct" or "diff").
        use_chirality (bool): Whether to include chirality in fingerprints.
        fp_size (int): Size of molecular fingerprint (half of vec_dim).

    Example:
        >>> # Create RDKit+ fingerprint dataset
        >>> reactions = CSVDataset(
        ...     file_path="data/train_rxns.csv",
        ...     key_column="reaction_id",
        ...     columns=["reaction_smiles"]
        ... )
        >>> fp_dataset = RDKitPlusFingerprintDataset(
        ...     reaction_dataset=reactions,
        ...     vec_dim=1024,  # 512 for reactants + 512 for products
        ...     mol_fp_type="morgan",
        ...     rxn_fp_type="struct",
        ...     use_chirality=True,
        ...     standardize=True
        ... )
        >>> fp = fp_dataset["rxn1"]  # Returns tensor of shape [1024]
    """

    # Mapping from fingerprint type names to generator functions
    _FP_GENERATORS = {
        "morgan": GetMorganGenerator,
        "rdkit": GetRDKitFPGenerator,
        "atom_pair": GetAtomPairGenerator,
        "topological_torsion": GetTopologicalTorsionGenerator,
    }

    def __init__(
        self,
        reaction_dataset: BaseDataset[K],
        vec_dim: int = 1024,
        mol_fp_type: str = "morgan",
        rxn_fp_type: str = "struct",
        use_chirality: bool = False,
        **kwargs,
    ):
        """
        Initialize RDKit+ fingerprint dataset.

        Args:
            reaction_dataset: Dataset containing reaction SMILES strings.
            vec_dim: Total fingerprint dimension. For structural fingerprints,
                this is split evenly between reactants and products. Must be even.
                Defaults to 1024.
            mol_fp_type: Type of molecular fingerprint to use. Options:
                "morgan", "rdkit", "atom_pair", "topological_torsion".
                Defaults to "morgan".
            rxn_fp_type: Type of reaction fingerprint. Options:
                "struct" (structural - concatenate binary fingerprints)
                "diff" (difference - count differences).
                Defaults to "struct".
            use_chirality: Whether to include chirality in fingerprints.
                Defaults to False.
            **kwargs: Additional arguments passed to BaseFingerprintDataset.

        Raises:
            ValueError: If mol_fp_type or rxn_fp_type is invalid.
            ValueError: If vec_dim is odd for structural fingerprints.
        """
        if mol_fp_type not in self._FP_GENERATORS:
            raise ValueError(
                f"Invalid mol_fp_type '{mol_fp_type}'. "
                f"Options: {list(self._FP_GENERATORS.keys())}"
            )

        if rxn_fp_type not in ["struct", "diff"]:
            raise ValueError(f"Invalid rxn_fp_type '{rxn_fp_type}'. Options: ['struct', 'diff']")

        if rxn_fp_type == "struct" and vec_dim % 2 != 0:
            raise ValueError(f"vec_dim must be even for structural fingerprints, got {vec_dim}")

        self.mol_fp_type = mol_fp_type
        self.rxn_fp_type = rxn_fp_type
        self.use_chirality = use_chirality

        # Calculate molecular fingerprint size
        if rxn_fp_type == "struct":
            self.fp_size = vec_dim // 2  # Half for reactants, half for products
        else:
            self.fp_size = vec_dim

        # Initialize base class
        super().__init__(reaction_dataset, vec_dim=vec_dim, **kwargs)

        # Create fingerprint generator
        self._create_fingerprint_generator()

    def _create_fingerprint_generator(self):
        """Create the appropriate RDKit fingerprint generator."""
        generator_func = self._FP_GENERATORS[self.mol_fp_type]

        if self.mol_fp_type == "morgan":
            # Morgan fingerprints (ECFP-like)
            self._fp_gen = generator_func(
                radius=3,  # ECFP6 (radius 3 = diameter 6) - matches API/hatchery
                fpSize=self.fp_size,
                includeChirality=self.use_chirality,
            )
        elif self.mol_fp_type == "rdkit":
            # RDKit topological fingerprints
            self._fp_gen = generator_func(
                fpSize=self.fp_size,
            )
        elif self.mol_fp_type == "atom_pair":
            # Atom pair fingerprints
            self._fp_gen = generator_func(
                fpSize=self.fp_size,
                includeChirality=self.use_chirality,
            )
        elif self.mol_fp_type == "topological_torsion":
            # Topological torsion fingerprints
            self._fp_gen = generator_func(
                fpSize=self.fp_size,
                includeChirality=self.use_chirality,
            )

    def _generate_struct_fingerprint(self, rxn) -> torch.Tensor:
        """
        Generate structural fingerprint (concatenate reactants | products).

        Args:
            rxn: RDKit ChemicalReaction object.

        Returns:
            Fingerprint tensor of shape [vec_dim].
        """
        fp = np.zeros(self.vec_dim, dtype=np.int64)

        # Process reactants (first half of fingerprint)
        for react in rxn.GetReactants():
            ret = SanitizeMol(react, catchErrors=True)
            if ret > 0:
                raise Exception(f"Error sanitizing reactant: {MolToSmiles(react)}")

            if self.use_chirality:
                AssignStereochemistry(react, force=True, cleanIt=True)

            # Get fingerprint as numpy array
            fp1 = self._fp_gen.GetFingerprintAsNumPy(react)
            # OR operation for binary fingerprint
            fp[: self.fp_size] |= fp1 == 1

        # Process products (second half of fingerprint)
        for prod in rxn.GetProducts():
            ret = SanitizeMol(prod, catchErrors=True)
            if ret > 0:
                raise Exception(f"Error sanitizing product: {MolToSmiles(prod)}")

            if self.use_chirality:
                AssignStereochemistry(prod, force=True, cleanIt=True)

            fp1 = self._fp_gen.GetFingerprintAsNumPy(prod)
            fp[self.fp_size :] |= fp1 == 1

        return torch.tensor(fp, dtype=self.dtype)

    def _generate_diff_fingerprint(self, rxn) -> torch.Tensor:
        """
        Generate difference fingerprint (count differences).

        Args:
            rxn: RDKit ChemicalReaction object.

        Returns:
            Fingerprint tensor of shape [vec_dim].
        """
        fp_reactants = np.zeros(self.fp_size, dtype=np.int64)
        fp_products = np.zeros(self.fp_size, dtype=np.int64)

        # Process reactants
        for react in rxn.GetReactants():
            ret = SanitizeMol(react, catchErrors=True)
            if ret > 0:
                raise Exception(f"Error sanitizing reactant: {MolToSmiles(react)}")

            if self.use_chirality:
                AssignStereochemistry(react, force=True, cleanIt=True)

            fp1 = self._fp_gen.GetCountFingerprintAsNumPy(react)
            fp_reactants += fp1

        # Process products
        for prod in rxn.GetProducts():
            ret = SanitizeMol(prod, catchErrors=True)
            if ret > 0:
                raise Exception(f"Error sanitizing product: {MolToSmiles(prod)}")

            if self.use_chirality:
                AssignStereochemistry(prod, force=True, cleanIt=True)

            fp1 = self._fp_gen.GetCountFingerprintAsNumPy(prod)
            fp_products += fp1

        # Compute difference
        fp = fp_products - fp_reactants

        return torch.tensor(fp, dtype=self.dtype)

    def _generate_fingerprint(self, reaction_info: Dict[str, Any]) -> torch.Tensor:
        """
        Generate RDKit+ fingerprint for a reaction.

        Args:
            reaction_info: Dict containing reaction SMILES.

        Returns:
            Fingerprint tensor of shape [vec_dim].

        Raises:
            Exception: If reaction parsing or fingerprint generation fails.
        """
        smiles = reaction_info[self.smiles_label]

        # Parse reaction SMILES
        try:
            rxn = ReactionFromSmarts(smiles, useSmiles=True)
        except Exception as e:
            raise Exception(f"Failed to parse reaction SMILES: {smiles}") from e

        # Generate fingerprint based on type
        if self.rxn_fp_type == "struct":
            return self._generate_struct_fingerprint(rxn)
        elif self.rxn_fp_type == "diff":
            return self._generate_diff_fingerprint(rxn)
        else:
            raise ValueError(f"Invalid rxn_fp_type: {self.rxn_fp_type}")
