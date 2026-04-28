import cv2
import numpy as np


def crop_to_dial(image: np.ndarray, padding: float = 0.08) -> np.ndarray:
    """
    Detect the circular meter bezel using Hough circle transform
    and crop tightly around it.

    Falls back to returning the original image if no circle is found.

    Args:
        image:   BGR image as numpy array
        padding: fractional padding around the detected circle (default 8%)

    Returns:
        Cropped BGR image centred on the dial
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    h, w = image.shape[:2]
    min_radius = int(min(h, w) * 0.2)
    max_radius = int(min(h, w) * 0.55)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) * 0.3,
        param1=100,
        param2=40,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        # No circle found — return original
        return image

    circles = np.round(circles[0, :]).astype(int)
    # Pick the largest detected circle
    cx, cy, r = sorted(circles, key=lambda c: c[2], reverse=True)[0]

    pad = int(r * padding)
    x1 = max(cx - r - pad, 0)
    y1 = max(cy - r - pad, 0)
    x2 = min(cx + r + pad, w)
    y2 = min(cy + r + pad, h)

    cropped = image[y1:y2, x1:x2]
    return cropped
