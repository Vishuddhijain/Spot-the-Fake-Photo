"""
features.py
Handcrafted, classic-CV features that separate a REAL photo of a real object
from a RECAPTURE (photo of a phone/laptop screen, or of a printout).

No deep learning. Everything here is fast (<15ms on a laptop CPU for a
downscaled image) and interpretable, which matters for both latency/cost
and for the "why this is interesting" honesty the assignment asks for.

Why these particular signals:
  1. Moire / periodicity energy  -> camera sensor grid sampling a screen's
     pixel grid (or a printout's halftone dot grid) creates aliasing that
     shows up as sharp, localized peaks in the FFT magnitude spectrum.
     A real object's spectrum falls off smoothly with no such peaks.
  2. High-frequency energy ratio -> screens/printouts often show a subtly
     different high-frequency profile than natural texture (either extra
     aliasing energy, or a slight softness from refocusing on a screen).
  3. Specular highlight fraction -> glossy screens/glass produce small,
     hard, near-fully-saturated (in V) highlight blobs; matte real objects
     rarely do, at least not in the same size/shape distribution.
  4. Rectangular bezel/frame evidence -> recaptures frequently include a
     sliver of the physical screen edge, bezel, or printout border as a
     long straight line near the image boundary.
  5. Color gamut / banding -> screens have a narrower, more saturated gamut
     and can show quantization banding in smooth gradients; measured via
     smooth-region gradient histogram peakiness.
  6. Laplacian sharpness -> included as a weak secondary signal (not
     reliable alone, but useful in combination).

Each function returns a single float. `extract_features` returns a fixed
order feature vector used by both train.py and predict.py.
"""

import cv2
import numpy as np

FEATURE_NAMES = [
    "moire_peak_ratio",
    "hf_energy_ratio",
    "specular_fraction",
    "border_line_score",
    "banding_score",
    "laplacian_var_norm",
]

TARGET_SIZE = 512  # downscale for speed; recapture artifacts survive downscaling


def _load_gray_and_bgr(path_or_array):
    if isinstance(path_or_array, str):
        bgr = cv2.imread(path_or_array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {path_or_array}")
    else:
        bgr = path_or_array

    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.ndim == 3 and bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)

    h, w = bgr.shape[:2]
    scale = TARGET_SIZE / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def moire_peak_ratio(gray):
    """FFT-based periodicity score. High for screens/printouts, low for real photos."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # exclude DC / very-low frequency (overall brightness, big smooth shapes)
    low_r = 0.04 * min(h, w)
    high_r = 0.45 * min(h, w)
    band = (r > low_r) & (r < high_r)

    band_vals = mag[band]
    if band_vals.size == 0:
        return 0.0

    band_mean = band_vals.mean() + 1e-6
    band_max = np.percentile(band_vals, 99.5)
    peakiness = band_max / band_mean  # natural photos: low; periodic patterns: high
    return float(peakiness)


def hf_energy_ratio(gray):
    """Fraction of spectral energy in the high-frequency band."""
    f = np.fft.fft2(gray.astype(np.float32))
    mag2 = np.abs(np.fft.fftshift(f)) ** 2
    h, w = mag2.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / min(h, w)

    total = mag2.sum() + 1e-9
    hf = mag2[r > 0.25].sum()
    return float(hf / total)


def specular_fraction(bgr):
    """Fraction of pixels that are small, hard, near-saturated bright blobs."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    bright_mask = ((v > 245) & (s < 60)).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

    total_px = bgr.shape[0] * bgr.shape[1]
    small_blob_px = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.01 * total_px:  # "small, hard" highlight, not a huge blown-out sky etc.
            small_blob_px += area

    return float(small_blob_px / total_px)


def border_line_score(gray):
    """Evidence of a long straight line (bezel/printout edge) near the image border."""
    edges = cv2.Canny(gray, 60, 150)
    h, w = edges.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=min(h, w) * 0.35, maxLineGap=10)
    if lines is None:
        return 0.0

    margin = 0.12  # near-border zone
    score = 0.0
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = line
        near_border = (
            min(x1, x2) < margin * w or max(x1, x2) > (1 - margin) * w or
            min(y1, y2) < margin * h or max(y1, y2) > (1 - margin) * h
        )
        length = np.hypot(x2 - x1, y2 - y1)
        is_axis_aligned = (abs(x1 - x2) < 0.03 * w) or (abs(y1 - y2) < 0.03 * h)
        if near_border and is_axis_aligned:
            score += length / max(h, w)

    return float(min(score, 3.0))  # cap to avoid outlier domination


def banding_score(gray):
    """Peakiness of the gradient histogram in smooth regions -> quantization banding."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    smooth_mask = grad_mag < np.percentile(grad_mag, 60)
    smooth_vals = blur[smooth_mask]
    if smooth_vals.size < 100:
        return 0.0

    hist, _ = np.histogram(smooth_vals, bins=64, range=(0, 255))
    hist = hist.astype(np.float32) + 1e-6
    hist /= hist.sum()
    peak = hist.max()
    return float(peak)


def laplacian_var_norm(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var() / 1000.0)  # normalize to a roughly [0, few] range


def extract_features(path_or_array):
    bgr, gray = _load_gray_and_bgr(path_or_array)
    vec = np.array([
        moire_peak_ratio(gray),
        hf_energy_ratio(gray),
        specular_fraction(bgr),
        border_line_score(gray),
        banding_score(gray),
        laplacian_var_norm(gray),
    ], dtype=np.float32)
    return vec
