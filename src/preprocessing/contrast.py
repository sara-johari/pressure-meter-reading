import cv2
import numpy as np


def enhance_contrast(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the
    L channel of the LAB colour space, then apply gamma correction.

    CLAHE boosts local contrast across the dial face without blowing out
    highlights. Gamma lifts midtones for underexposed images.

    Args:
        image: BGR image as numpy array
        gamma: gamma value > 1 brightens midtones (use 1.0 to skip)

    Returns:
        Contrast-enhanced BGR image
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # Gamma correction
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype(np.uint8)
        enhanced = cv2.LUT(enhanced, table)

    return enhanced
