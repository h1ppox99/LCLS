\"\"\"Example adapter contract for ``review.py``; replace with the real bridge.\"\"\"
from __future__ import annotations

import numpy as np


def mask_image(image: np.ndarray) -> np.ndarray:
    \"\"\"Return a placeholder mask marking non-finite pixels only.\"\"\"
    return ~np.isfinite(image)