import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepface_adapter import load_deepface_backend


def test_load_deepface_backend_provides_represent_and_find() -> None:
    backend = load_deepface_backend()

    assert hasattr(backend, "represent")
    assert hasattr(backend, "find")

    results = backend.represent(img_path=np.zeros((120, 120, 3), dtype=np.uint8), enforce_detection=False)
    assert isinstance(results, list)
