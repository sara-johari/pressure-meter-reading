import cv2
import numpy as np


def enhance_pointer(image: np.ndarray, edge_alpha: float = 0.4) -> np.ndarray:
    """
    Make the needle/pointer stand out from the dial face using two techniques:

    1. HSV channel boost — saturates red/orange hues (common needle colours)
    2. Canny edge overlay — blends detected edges semi-transparently so the
       thin needle appears as a distinct line even if it shares colour with
       tick marks.

    Args:
        image:      BGR image as numpy array
        edge_alpha: blend weight for the Canny edge overlay (0.0–1.0)

    Returns:
        Pointer-enhanced BGR image
    """
    # --- 1. HSV saturation boost for warm hues (red / orange needles) ---
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # Red hues: 0-15 and 160-180 in OpenCV HSV
    red_mask = ((h <= 15) | (h >= 160)).astype(np.float32)
    boost = 1.0 + 0.6 * red_mask          # boost red by 60%, others unchanged
    s = np.clip(s * boost, 0, 255)

    hsv_boosted = cv2.merge([h, s, v]).astype(np.uint8)
    result = cv2.cvtColor(hsv_boosted, cv2.COLOR_HSV2BGR)

    # --- 2. Canny edge overlay ---
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=40, threshold2=120)

    # Convert edges to 3-channel and blend
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(result, 1.0, edges_3ch, edge_alpha, 0)

    return result
