import cv2
import numpy as np


def enhance_scale(image: np.ndarray, blend_alpha: float = 0.3) -> np.ndarray:
    """
    Enhance tick marks and scale numbers using adaptive thresholding blended
    back over the colour image.

    Adaptive thresholding handles uneven lighting (e.g. a shadow on one side
    of the dial) that global thresholding would miss.

    Args:
        image:       BGR image as numpy array
        blend_alpha: weight of the threshold layer in the final blend (0.0–1.0)

    Returns:
        Scale-enhanced BGR image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold: block size 15 works well for dial-sized images
    thresh = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=15,
        C=4,
    )

    # Invert so tick marks are white on black (easier to blend)
    thresh_inv = cv2.bitwise_not(thresh)
    thresh_3ch = cv2.cvtColor(thresh_inv, cv2.COLOR_GRAY2BGR)

    # Blend: mostly original colour, small dose of thresholded detail
    result = cv2.addWeighted(image, 1.0, thresh_3ch, blend_alpha, 0)
    return result
