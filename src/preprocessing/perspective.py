import cv2
import numpy as np


def fix_perspective(image: np.ndarray) -> np.ndarray:
    """
    Detect the meter bezel as an ellipse and warp it back to a circle.
    If no ellipse is confidently detected, returns the original image unchanged.

    Args:
        image: BGR image as numpy array

    Returns:
        Perspective-corrected BGR image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_ellipse = None
    best_score = 0

    for cnt in contours:
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (ma, mi), angle = ellipse

        # Skip if axes are nearly equal (already circular) or too small
        if mi < 20:
            continue

        aspect_ratio = ma / mi if mi > 0 else 1.0
        area = np.pi * (ma / 2) * (mi / 2)
        img_area = image.shape[0] * image.shape[1]

        # Score: prefer large ellipses with noticeable distortion
        if 1.05 < aspect_ratio < 3.0 and area > 0.05 * img_area:
            score = area * (aspect_ratio - 1.0)
            if score > best_score:
                best_score = score
                best_ellipse = ellipse

    if best_ellipse is None:
        return image

    (cx, cy), (ma, mi), angle = best_ellipse
    h, w = image.shape[:2]

    # Build source points from ellipse axes
    src_pts = cv2.ellipse2Poly(
        (int(cx), int(cy)),
        (int(ma / 2), int(mi / 2)),
        int(angle), 0, 360, 10
    ).astype(np.float32)

    # Target: a circle with radius = average of the two axes
    radius = (ma + mi) / 4
    dst_pts = []
    for i, angle_deg in enumerate(range(0, 360, 10)):
        a = np.deg2rad(angle_deg)
        dst_pts.append([cx + radius * np.cos(a), cy + radius * np.sin(a)])
    dst_pts = np.array(dst_pts, dtype=np.float32)

    # Use affine transform on a subset of points (fast approximation)
    transform, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
    if transform is None:
        return image

    corrected = cv2.warpPerspective(image, transform, (w, h))
    return corrected
