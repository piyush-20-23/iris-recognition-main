from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from config import DATASET_DIR
from src.data.inventory import DatasetInventory
from src.data.labels import IrisSampleMetadata
from src.preprocessing.image_io import load_image, load_mask


class UbiPrDatasetLoader:
    """
    Clean dataset loader providing subject-grouped and session-filtered
    access to the UBIPR both-eyes biometric images and ground truth masks.
    """

    def __init__(self, dataset_dir: Path = DATASET_DIR):
        self.dataset_dir = Path(dataset_dir)
        self.inventory = DatasetInventory(self.dataset_dir)
        self._is_indexed = False

    def index(self) -> "UbiPrDatasetLoader":
        if not self._is_indexed:
            self.inventory.run_inventory(sample_decode_limit=None)
            self._is_indexed = True
        return self

    def get_subject_samples(self) -> Dict[str, List[IrisSampleMetadata]]:
        self.index()
        return self.inventory.subject_samples

    def get_session_samples(self, session_id: int) -> List[IrisSampleMetadata]:
        self.index()
        return [s for s in self.inventory.valid_samples if s.session_id == session_id]

    @staticmethod
    def load_sample_arrays(sample: IrisSampleMetadata) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Loads the decoded BGR image and optional ground-truth segmentation mask.
        """
        img = load_image(sample.image_path)
        mask = load_mask(sample.mask_path) if sample.mask_path else None
        return img, mask
