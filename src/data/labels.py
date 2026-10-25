import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Regex for UBIPR naming: C<subject>_S<session>_I<image_id>
# e.g., C101_S1_I1.jpg -> subject="C101", session=1, image_id=1
UBIPR_FILENAME_PATTERN = re.compile(
    r"^C(?P<subject_num>\d+)_S(?P<session>\d+)_I(?P<image_id>\d+)$",
    re.IGNORECASE
)


@dataclass(frozen=True)
class IrisSampleMetadata:
    stem: str
    subject_id: str
    session_id: int
    image_id: int
    image_path: Path
    mask_path: Optional[Path] = None

    def __repr__(self) -> str:
        return (f"IrisSampleMetadata(stem='{self.stem}', subject='{self.subject_id}', "
                f"session={self.session_id}, img={self.image_id})")


def parse_sample_stem(stem: str) -> Optional[dict]:
    """
    Parse a file stem according to the official UBIPR naming standard.
    Returns a dictionary of parsed components or None if unparseable.
    """
    match = UBIPR_FILENAME_PATTERN.match(stem)
    if not match:
        return None
    d = match.groupdict()
    return {
        "subject_id": f"C{d['subject_num']}",
        "session_id": int(d["session"]),
        "image_id": int(d["image_id"]),
    }


def parse_sample_metadata(
    image_path: Path,
    mask_path: Optional[Path] = None
) -> Optional[IrisSampleMetadata]:
    """
    Parse metadata from an image path and its optional mask path.
    Returns IrisSampleMetadata if filename follows UBIPR convention, otherwise None.
    """
    stem = image_path.stem
    parsed = parse_sample_stem(stem)
    if parsed is None:
        return None

    return IrisSampleMetadata(
        stem=stem,
        subject_id=parsed["subject_id"],
        session_id=parsed["session_id"],
        image_id=parsed["image_id"],
        image_path=image_path,
        mask_path=mask_path if mask_path and mask_path.exists() else None,
    )
