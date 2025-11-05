"""
DRFP (Differential Reaction Fingerprint) generation for reactions.
"""

from typing import Any, Dict

import torch
from drfp import DrfpEncoder

from horizyn.datasets.base import BaseDataset, K
from horizyn.datasets.fingerprints.base import BaseFingerprintDataset


class DRFPFingerprintDataset(BaseFingerprintDataset):
    """
    Differential Reaction Fingerprint (DRFP) dataset.

    Generates DRFP fingerprints for chemical reactions using canonical SMILES
    strings of molecular substructures. The fingerprint is computed from the
    symmetric difference of the hashed SMILES strings of reactants and products
    and folded to desired length.

    DRFP fingerprints capture the structural changes that occur during a reaction
    by comparing molecular fragments in reactants versus products. This makes them
    particularly effective for reaction classification and similarity tasks.

    Citation:
        Probst, D.; Schwaller, P.; Reymond, J.-L. Reaction Classification and
        Yield Prediction Using the Differential Reaction Fingerprint DRFP.
        Digit. Discov. 2022, 1 (2), 91–97.
        DOI: https://doi.org/10.1039/d1dd00006c

    Implementation:
        https://github.com/reymond-group/drfp

    Attributes:
        min_radius (int): Minimum fingerprint radius (0 = only query atoms).
        radius (int): Maximum fingerprint radius for substructures.
        rings (bool): Whether to include full rings as substructures.
        root_central_atom (bool): Whether to root substructure SMILES at query atoms.
        include_hydrogens (bool): Whether to include hydrogens in SMILES.

    Example:
        >>> # Create DRFP fingerprint dataset
        >>> reactions = SQLDataset(
        ...     file_path="data/reactions.db",
        ...     table_name="reactions",
        ...     search_key="reaction_id",
        ...     columns=["reaction_smiles"]
        ... )
        >>> fp_dataset = DRFPFingerprintDataset(
        ...     reaction_dataset=reactions,
        ...     vec_dim=1024,
        ...     radius=3,
        ...     standardize=True
        ... )
        >>> fp = fp_dataset["rxn1"]  # Returns tensor of shape [1024]
    """

    def __init__(
        self,
        reaction_dataset: BaseDataset[K],
        vec_dim: int = 1024,
        min_radius: int = 0,
        radius: int = 3,
        rings: bool = True,
        root_central_atom: bool = True,
        include_hydrogens: bool = False,
        **kwargs,
    ):
        """
        Initialize DRFP fingerprint dataset.

        Args:
            reaction_dataset: Dataset containing reaction SMILES strings.
            vec_dim: Fingerprint dimension (folded length). Defaults to 1024.
            min_radius: Minimum fingerprint radius. 0 means only query atoms.
                Defaults to 0.
            radius: Maximum fingerprint radius for substructures. Defaults to 3.
            rings: Whether to include full rings as substructures. Defaults to True.
            root_central_atom: Whether to root substructure SMILES at the query
                atoms. Defaults to True.
            include_hydrogens: Whether to include hydrogens in substructure SMILES.
                Defaults to False.
            **kwargs: Additional arguments passed to BaseFingerprintDataset.
        """
        self.min_radius = min_radius
        self.radius = radius
        self.rings = rings
        self.root_central_atom = root_central_atom
        self.include_hydrogens = include_hydrogens

        # Initialize DRFP encoder
        self._drfp_encoder = DrfpEncoder()

        # Initialize base class
        super().__init__(reaction_dataset, vec_dim=vec_dim, **kwargs)

    def _generate_fingerprint(self, reaction_info: Dict[str, Any]) -> torch.Tensor:
        """
        Generate DRFP fingerprint for a reaction.

        Args:
            reaction_info: Dict containing reaction SMILES.

        Returns:
            Fingerprint tensor of shape [vec_dim].

        Raises:
            Exception: If DRFP fingerprint generation fails.
        """
        smiles = reaction_info[self.smiles_label]

        # Generate DRFP fingerprint
        try:
            fps = self._drfp_encoder.encode(
                X=smiles,
                n_folded_length=self.vec_dim,
                min_radius=self.min_radius,
                radius=self.radius,
                rings=self.rings,
                mapping=False,
                atom_index_mapping=False,
                root_central_atom=self.root_central_atom,
                include_hydrogens=self.include_hydrogens,
                show_progress_bar=False,
            )
        except Exception as e:
            raise Exception(f"Failed to generate DRFP for reaction SMILES: {smiles}") from e

        # Convert to tensor (fps is list with single fingerprint)
        return torch.tensor(fps[0], dtype=self.dtype)
