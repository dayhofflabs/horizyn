"""
Base class for fingerprint generation datasets.
"""

from typing import Any, Callable, Dict, Optional
import torch
from horizyn.datasets.base import BaseDataset, WrapperDataset, K
from horizyn.chemistry.standardizer import Standardizer
from horizyn.utils.cache import InMemoryCache


class BaseFingerprintDataset(WrapperDataset[K]):
    """
    Base class for reaction fingerprint datasets.

    This class wraps a dataset containing reaction SMILES strings and generates
    fingerprint vectors on-the-fly. It supports optional standardization of
    reactions before fingerprint generation.

    Subclasses must implement the `_generate_fingerprint` method to define
    the specific fingerprint algorithm.

    Attributes:
        vec_dim (int): Dimensionality of fingerprint vectors.
        dtype (torch.dtype): Data type for fingerprint tensors.
        smiles_label (str): Key for SMILES string in dataset dict.
        standardize (bool): Whether to standardize reactions.
        _standardizer (Standardizer | None): Standardizer instance if enabled.

    Example:
        >>> # Wrap a CSV dataset with fingerprint generation
        >>> reactions = CSVDataset(
        ...     file_path="data/train_rxns.csv",
        ...     key_column="reaction_id",
        ...     columns=["reaction_smiles"]
        ... )
        >>> fp_dataset = RDKitPlusFingerprintDataset(
        ...     reaction_dataset=reactions,
        ...     vec_dim=1024,
        ...     standardize=True
        ... )
        >>> fingerprint = fp_dataset["rxn1"]  # Returns tensor of shape [1024]
    """

    def __init__(
        self,
        reaction_dataset: BaseDataset[K],
        vec_dim: int = 1024,
        dtype: torch.dtype = torch.float32,
        smiles_label: str = "reaction_smiles",
        standardize: bool = False,
        standardize_hypervalent: bool = True,
        standardize_remove_hs: bool = True,
        standardize_kekulize: bool = False,
        standardize_uncharge: bool = False,
        standardize_metals: bool = True,
        transforms: Optional[Callable[[K, Any], Any]] = None,
        **kwargs,
    ):
        """
        Initialize the fingerprint dataset.

        Args:
            reaction_dataset: Dataset containing reaction SMILES strings.
                Should return either a string or dict with smiles_label key.
            vec_dim: Dimensionality of fingerprint vectors. Defaults to 1024.
            dtype: PyTorch data type for fingerprints. Defaults to torch.float32.
            smiles_label: Key for SMILES string if dataset returns dicts.
                Defaults to "reaction_smiles".
            standardize: Whether to standardize reactions before fingerprinting.
                Defaults to False.
            standardize_hypervalent: Standardize double bonds in hypervalent
                compounds. Only used if standardize=True. Defaults to True.
            standardize_remove_hs: Remove explicit hydrogen atoms. Only used if
                standardize=True. Defaults to True.
            standardize_kekulize: Kekulize aromatic compounds. Only used if
                standardize=True. Defaults to False.
            standardize_uncharge: Convert to neutral species by protonation/
                deprotonation. Only used if standardize=True. Defaults to False.
            standardize_metals: Disconnect bonds between metals and N, O, F atoms.
                Only used if standardize=True. Defaults to True.
            transforms: Optional transform function. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.vec_dim = vec_dim
        self.dtype = dtype
        self.smiles_label = smiles_label
        self.standardize = standardize

        # Initialize wrapper
        super().__init__(reaction_dataset, transforms=transforms, **kwargs)

        # Initialize in-memory cache for fingerprints
        self._cache: InMemoryCache = InMemoryCache()

        # Initialize standardizer if enabled
        self._standardizer = None
        if self.standardize:
            self._standardizer = Standardizer(
                standardize_hypervalent=standardize_hypervalent,
                standardize_remove_hs=standardize_remove_hs,
                standardize_kekulize=standardize_kekulize,
                standardize_uncharge=standardize_uncharge,
                standardize_metals=standardize_metals,
            )

    @property
    def standardizer(self) -> Standardizer:
        """Get the standardizer instance."""
        if self._standardizer is None:
            raise AttributeError("Standardizer is not set (standardize=False)")
        return self._standardizer

    def __getitem__(self, key: K) -> torch.Tensor:
        """
        Get fingerprint for a reaction.

        Args:
            key: The key of the reaction.

        Returns:
            Fingerprint tensor of shape [vec_dim].

        Raises:
            Exception: If fingerprint generation fails.
        """
        # Return cached fingerprint if available (apply transforms on top)
        if self._cache.has(key):
            return self._apply_transforms(key, self._cache.get(key))

        # Get reaction info from wrapped dataset
        reaction_info = self._query_smiles_dataset(key)

        # Standardize if enabled
        reaction_info = self._preprocess_reaction(reaction_info)

        # Generate and cache fingerprint
        try:
            fingerprint = self._generate_fingerprint(reaction_info)
        except Exception as e:
            raise Exception(f"Failed to generate fingerprint for reaction '{key}': {e}") from e

        self._cache.set(key, fingerprint)

        # Apply transforms if any
        return self._apply_transforms(key, fingerprint)

    def _query_smiles_dataset(self, key: K) -> Dict[str, Any]:
        """
        Query the wrapped dataset and ensure correct format.

        Args:
            key: The key to query.

        Returns:
            Dictionary with smiles_label key containing SMILES string.

        Raises:
            KeyError: If dict result doesn't have smiles_label key.
            ValueError: If result is neither string nor dict.
        """
        # Get from wrapped dataset (bypass transform)
        result = self.dataset[key]

        if isinstance(result, dict):
            if self.smiles_label not in result:
                raise KeyError(
                    f"Sample dict missing '{self.smiles_label}' key. "
                    f"Available keys: {list(result.keys())}"
                )
            return result
        elif isinstance(result, str):
            # Convert string to dict format
            return {self.smiles_label: result}
        else:
            raise ValueError(f"Sample must be string or dict, got {type(result)}: {result}")

    def _preprocess_reaction(self, reaction_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optionally standardize the reaction.

        Args:
            reaction_info: Dict containing reaction SMILES.

        Returns:
            Dict with potentially standardized SMILES.
        """
        if self.standardize:
            smiles = reaction_info[self.smiles_label]
            standardized = self.standardizer.standardize_reaction(smiles)
            reaction_info[self.smiles_label] = standardized

        return reaction_info

    def _generate_fingerprint(self, reaction_info: Dict[str, Any]) -> torch.Tensor:
        """
        Generate fingerprint for a reaction.

        This method must be implemented by subclasses.

        Args:
            reaction_info: Dict containing reaction SMILES with smiles_label key.

        Returns:
            Fingerprint tensor of shape [vec_dim].

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement _generate_fingerprint")
