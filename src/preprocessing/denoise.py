import cv2
import numpy as np


def denoise(image: np.ndarray) -> np.ndarray:
    """
    Apply bilateral filter denoising.
    Smooths flat regions (dial face) while preserving hard edges
    (needle, tick marks, bezel).

    Args:
        image: BGR image as numpy array

    Returns:
        Denoised BGR image
    """
    # d=9: diameter of pixel neighbourhood
    # sigmaColor=75: filter sigma in color space
    # sigmaSpace=75: filter sigma in coordinate space
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return denoised
