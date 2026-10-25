import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

from config import (
    IGNORE_PATTERNS,
    IGNORE_PREFIXES,
    VALID_IMAGE_EXTS,
    MASK_CLASS_BG_SKIN,
    MASK_CLASS_IRIS,
    MASK_CLASS_SCLERA,
    MASK_CLASS_EYEBROWS,
)
from src.data.labels import IrisSampleMetadata, parse_sample_metadata


def is_ignored_file(path: Path) -> bool:
    """Returns True if file is OS artifact or hidden."""
    if path.name in IGNORE_PATTERNS:
        return True
    if any(path.name.startswith(p) for p in IGNORE_PREFIXES):
        return True
    return False


class DatasetInventory:
    """
    Audits and inventories the UBIPR iris dataset directory.
    Maintains complete data integrity without modifying source files.
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.all_files: List[Path] = []
        self.ignored_files: List[Path] = []
        self.image_files: List[Path] = []
        self.mask_files: List[Path] = []
        self.unpaired_images: List[Path] = []
        self.unpaired_masks: List[Path] = []
        self.unparseable_files: List[Path] = []
        self.unreadable_images: List[Path] = []
        self.valid_samples: List[IrisSampleMetadata] = []
        self.subject_samples: Dict[str, List[IrisSampleMetadata]] = defaultdict(list)
        self.session_counts: Counter = Counter()

    def run_inventory(self, sample_decode_limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Scan directory and compute audit statistics.
        If sample_decode_limit is None, checks all image files.
        """
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        for root, dirs, files in os.walk(self.dataset_dir):
            root_path = Path(root)
            for f in files:
                p = root_path / f
                self.all_files.append(p)
                if is_ignored_file(p):
                    self.ignored_files.append(p)
                elif p.suffix.lower() == ".jpg":
                    self.image_files.append(p)
                elif p.suffix.lower() == ".png":
                    self.mask_files.append(p)

        # Index masks by stem
        mask_map: Dict[str, Path] = {m.stem: m for m in self.mask_files}
        jpg_stems = set(f.stem for f in self.image_files)
        png_stems = set(mask_map.keys())

        # Check pairing
        for img in self.image_files:
            if img.stem not in mask_map:
                self.unpaired_images.append(img)
        for mask_stem in png_stems - jpg_stems:
            self.unpaired_masks.append(mask_map[mask_stem])

        # Parse labels and check pairing
        for img_path in sorted(self.image_files):
            mask_path = mask_map.get(img_path.stem)
            meta = parse_sample_metadata(img_path, mask_path)
            if meta is None:
                self.unparseable_files.append(img_path)
            else:
                self.valid_samples.append(meta)
                self.subject_samples[meta.subject_id].append(meta)
                self.session_counts[meta.session_id] += 1

        # Check decodability and resolutions
        resolutions = []
        aspect_ratios = []
        channels_set = set()
        dtypes_set = set()

        files_to_check = self.valid_samples
        if sample_decode_limit is not None:
            # Check a representative subset for quick audits
            files_to_check = self.valid_samples[:sample_decode_limit]

        for s in files_to_check:
            img = cv2.imread(str(s.image_path))
            if img is None:
                self.unreadable_images.append(s.image_path)
            else:
                h, w = img.shape[:2]
                ch = img.shape[2] if img.ndim == 3 else 1
                resolutions.append((w, h))
                aspect_ratios.append(round(w / h, 2))
                channels_set.add(ch)
                dtypes_set.add(str(img.dtype))

        # Check mask classes on a sample
        mask_classes_found = set()
        for s in files_to_check[:50]:
            if s.mask_path:
                m = cv2.imread(str(s.mask_path), cv2.IMREAD_UNCHANGED)
                if m is not None:
                    mask_classes_found.update(int(v) for v in np.unique(m))

        counts_per_sub = [len(v) for v in self.subject_samples.values()]

        stats = {
            "dataset_dir": str(self.dataset_dir),
            "total_files": len(self.all_files),
            "ignored_files_count": len(self.ignored_files),
            "ignored_files_samples": [p.name for p in self.ignored_files[:10]],
            "jpg_images_count": len(self.image_files),
            "png_masks_count": len(self.mask_files),
            "unpaired_images_count": len(self.unpaired_images),
            "unpaired_masks_count": len(self.unpaired_masks),
            "unparseable_files_count": len(self.unparseable_files),
            "valid_samples_count": len(self.valid_samples),
            "unique_subjects_count": len(self.subject_samples),
            "session_distribution": dict(self.session_counts),
            "images_per_subject": {
                "min": int(min(counts_per_sub)) if counts_per_sub else 0,
                "max": int(max(counts_per_sub)) if counts_per_sub else 0,
                "mean": round(float(np.mean(counts_per_sub)), 2) if counts_per_sub else 0.0,
                "median": int(np.median(counts_per_sub)) if counts_per_sub else 0,
            },
            "unreadable_images_count": len(self.unreadable_images),
            "channels_found": sorted(list(channels_set)),
            "dtypes_found": sorted(list(dtypes_set)),
            "mask_classes_found": sorted(list(mask_classes_found)),
            "mask_class_semantics": {
                "0": "Background / Skin",
                "85": "Iris (Both Eyes)",
                "170": "Sclera",
                "255": "Eyebrows"
            },
            "resolutions_sample_count": len(resolutions),
            "min_resolution": list(min(resolutions, key=lambda r: r[0]*r[1])) if resolutions else [],
            "max_resolution": list(max(resolutions, key=lambda r: r[0]*r[1])) if resolutions else [],
        }

        return stats

    def generate_contact_sheet(
        self,
        output_path: Path,
        rows: int = 4,
        cols: int = 4,
        thumb_size: Tuple[int, int] = (300, 100)
    ) -> Path:
        """
        Creates a labeled contact sheet grid highlighting iris mask overlay
        and sample metadata (Subject ID, Session, Image ID).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n_samples = rows * cols
        if not self.valid_samples:
            raise ValueError("No valid samples available to create contact sheet.")

        # Pick evenly spaced samples across different subjects
        step = max(1, len(self.valid_samples) // n_samples)
        selected_samples = self.valid_samples[::step][:n_samples]

        sheet_h = rows * thumb_size[1]
        sheet_w = cols * thumb_size[0]
        sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

        for idx, sample in enumerate(selected_samples):
            r = idx // cols
            c = idx % cols
            y1 = r * thumb_size[1]
            y2 = y1 + thumb_size[1]
            x1 = c * thumb_size[0]
            x2 = x1 + thumb_size[0]

            img = cv2.imread(str(sample.image_path))
            if img is None:
                continue

            # Overlay iris mask if available (green overlay on iris region)
            if sample.mask_path and sample.mask_path.exists():
                mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
                if mask is not None and mask.shape[:2] == img.shape[:2]:
                    iris_pixels = (mask == MASK_CLASS_IRIS)
                    overlay = img.copy()
                    overlay[iris_pixels] = [0, 255, 0]  # Green overlay
                    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

            thumb = cv2.resize(img, thumb_size, interpolation=cv2.INTER_AREA)

            # Draw text label banner
            label = f"{sample.subject_id} S{sample.session_id} I{sample.image_id}"
            cv2.rectangle(thumb, (0, 0), (thumb_size[0], 22), (0, 0, 0), -1)
            cv2.putText(thumb, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            sheet[y1:y2, x1:x2] = thumb

        cv2.imwrite(str(output_path), sheet)
        return output_path
