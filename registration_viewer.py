"""
Registration Quality Viewer v4
===============================
Two-window matplotlib viewer for T1w vs T1c registration + label quality.

WINDOW 1 — Intensity comparison (always open):
  Panel 1: T1w (grayscale)
  Panel 2: T1c (grayscale)
  Panel 3: T1w + difference heatmap overlay
  Panel 4: Click-to-zoom inset

WINDOW 2 — Label disagreement (opens when labels provided):
  Panel 1: T1w + label boundaries (green contours)
  Panel 2: T1c + label boundaries (green contours)
  Panel 3: T1w + disagreement overlay (red = labels disagree)
  Includes per-slice disagreement % metric

Both windows share the same slice slider and axis controls.

Controls:
  - Scroll wheel / arrow keys: change slice
  - Click on Window 1 panels: zoom into region (panel 4)
  - Axis radio buttons: switch axial/coronal/sagittal
  - +/- buttons: adjust zoom level
  - Escape: reset zoom

Usage:
------
  # Intensity only (no label window)
  python registration_viewer_v4.py t1w.nii.gz t1c.nii.gz

  # With label disagreement window
  python registration_viewer_v4.py t1w.nii.gz t1c.nii.gz \
      --labels_t1w labels_t1w.nii.gz --labels_t1c labels_t1c.nii.gz

Requirements:
  pip install nibabel numpy matplotlib scikit-image scipy
"""

import argparse
import sys

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity


# ---------------------------------------------------------------------------
# FreeSurfer label name lookup (106 structures + background)
# ---------------------------------------------------------------------------

FREESURFER_LUT = {
    0: "Background", 2: "L-Cerebral-WM", 3: "L-Cerebral-Cortex",
    4: "L-Lateral-Ventricle", 5: "L-Inf-Lat-Vent", 7: "L-Cerebellum-WM",
    8: "L-Cerebellum-Cortex", 10: "L-Thalamus", 11: "L-Caudate",
    12: "L-Putamen", 13: "L-Pallidum", 14: "3rd-Ventricle",
    15: "4th-Ventricle", 16: "Brain-Stem", 17: "L-Hippocampus",
    18: "L-Amygdala", 24: "CSF", 26: "L-Accumbens",
    28: "L-VentralDC", 30: "L-Vessel", 31: "L-Choroid-Plexus",
    41: "R-Cerebral-WM", 42: "R-Cerebral-Cortex",
    43: "R-Lateral-Ventricle", 44: "R-Inf-Lat-Vent",
    46: "R-Cerebellum-WM", 47: "R-Cerebellum-Cortex",
    49: "R-Thalamus", 50: "R-Caudate", 51: "R-Putamen",
    52: "R-Pallidum", 53: "R-Hippocampus", 54: "R-Amygdala",
    58: "R-Accumbens", 60: "R-VentralDC", 62: "R-Vessel",
    63: "R-Choroid-Plexus", 77: "WM-Hypointensities",
    85: "Optic-Chiasm", 251: "CC_Posterior", 252: "CC_Mid_Posterior",
    253: "CC_Central", 254: "CC_Mid_Anterior", 255: "CC_Anterior",
    # DKT cortical parcellation (1000+ = left, 2000+ = right)
    1002: "L-caudalanteriorcingulate", 1003: "L-caudalmiddlefrontal",
    1005: "L-cuneus", 1006: "L-entorhinal", 1007: "L-fusiform",
    1008: "L-inferiorparietal", 1009: "L-inferiortemporal",
    1010: "L-isthmuscingulate", 1011: "L-lateraloccipital",
    1012: "L-lateralorbitofrontal", 1013: "L-lingual",
    1014: "L-medialorbitofrontal", 1015: "L-middletemporal",
    1016: "L-parahippocampal", 1017: "L-paracentral",
    1018: "L-parsopercularis", 1019: "L-parsorbitalis",
    1020: "L-parstriangularis", 1021: "L-pericalcarine",
    1022: "L-postcentral", 1023: "L-posteriorcingulate",
    1024: "L-precentral", 1025: "L-precuneus",
    1026: "L-rostralanteriorcingulate", 1027: "L-rostralmiddlefrontal",
    1028: "L-superiorfrontal", 1029: "L-superiorparietal",
    1030: "L-superiortemporal", 1031: "L-supramarginal",
    1034: "L-transversetemporal", 1035: "L-insula",
    2002: "R-caudalanteriorcingulate", 2003: "R-caudalmiddlefrontal",
    2005: "R-cuneus", 2006: "R-entorhinal", 2007: "R-fusiform",
    2008: "R-inferiorparietal", 2009: "R-inferiortemporal",
    2010: "R-isthmuscingulate", 2011: "R-lateraloccipital",
    2012: "R-lateralorbitofrontal", 2013: "R-lingual",
    2014: "R-medialorbitofrontal", 2015: "R-middletemporal",
    2016: "R-parahippocampal", 2017: "R-paracentral",
    2018: "R-parsopercularis", 2019: "R-parsorbitalis",
    2020: "R-parstriangularis", 2021: "R-pericalcarine",
    2022: "R-postcentral", 2023: "R-posteriorcingulate",
    2024: "R-precentral", 2025: "R-precuneus",
    2026: "R-rostralanteriorcingulate", 2027: "R-rostralmiddlefrontal",
    2028: "R-superiorfrontal", 2029: "R-superiorparietal",
    2030: "R-superiortemporal", 2031: "R-supramarginal",
    2034: "R-transversetemporal", 2035: "R-insula",
}

def label_name(label_id):
    """Get FreeSurfer structure name from label ID."""
    return FREESURFER_LUT.get(int(label_id), f"ID:{int(label_id)}")


# ---------------------------------------------------------------------------
# Data helpers (unchanged)
# ---------------------------------------------------------------------------

def load_nifti(path):
    """Load NIfTI, reorient to RAS canonical, return float32 3D array."""
    img = nib.load(path)
    # Reorient to RAS+ (Right-Anterior-Superior) regardless of original orientation
    # This ensures consistent axis meaning: dim0=R->L, dim1=P->A, dim2=I->S
    img_canonical = nib.as_closest_canonical(img)
    return np.asarray(img_canonical.dataobj, dtype=np.float32)

def normalise_01(vol):
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol)
    return (vol - vmin) / (vmax - vmin)

def create_brain_mask(vol, threshold_pct=5.0):
    thresh = np.percentile(vol[vol > 0], threshold_pct) if (vol > 0).any() else 0
    return (vol > thresh).astype(np.uint8)

def histogram_match(source, reference, mask=None):
    if mask is None:
        mask = np.ones_like(source, dtype=bool)
    else:
        mask = mask.astype(bool)
    src_vals = source[mask]
    ref_vals = reference[mask]
    if len(src_vals) == 0 or len(ref_vals) == 0:
        return source.copy()
    src_sorted = np.sort(src_vals)
    ref_sorted = np.sort(ref_vals)
    interp_values = np.interp(
        np.linspace(0, 1, len(src_sorted)),
        np.linspace(0, 1, len(ref_sorted)),
        ref_sorted,
    )
    sort_idx = np.argsort(src_vals)
    matched_flat = np.zeros_like(src_vals)
    matched_flat[sort_idx] = interp_values
    result = source.copy()
    result[mask] = matched_flat
    return result

def extract_label_boundaries(label_vol):
    """Binary boundary mask: voxel=1 if any 6-neighbour has different label."""
    boundaries = np.zeros_like(label_vol, dtype=np.uint8)
    for ax in range(3):
        slc_a = [slice(None)] * 3
        slc_b = [slice(None)] * 3
        slc_a[ax] = slice(1, None)
        slc_b[ax] = slice(None, -1)
        diff = label_vol[tuple(slc_a)] != label_vol[tuple(slc_b)]
        boundaries[tuple(slc_a)] |= diff.astype(np.uint8)
        boundaries[tuple(slc_b)] |= diff.astype(np.uint8)
    return boundaries

def cosine_similarity_2d(a, b, mask=None):
    if mask is not None:
        a, b = a[mask.astype(bool)], b[mask.astype(bool)]
    else:
        a, b = a.ravel(), b.ravel()
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0

def compute_ssim_2d(a, b):
    dr = max(a.max() - a.min(), b.max() - b.min(), 1e-10)
    try:
        return structural_similarity(a, b, data_range=dr)
    except Exception:
        return 0.0

def get_slice(vol, axis, idx):
    """
    Extract a 2D slice for display in RADIOLOGICAL convention
    (patient's left on screen right), matching 3D Slicer default.
    
    After as_closest_canonical, data is RAS+:
      dim0 = Right->Left  (i)
      dim1 = Posterior->Anterior  (j)
      dim2 = Inferior->Superior  (k)
    
    Radiological display:
      Axial:    screen-x = R->L (no flip needed, already radiological),
                screen-y = A->P (flip anterior to top)
      Coronal:  screen-x = R->L, screen-y = S->I (flip superior to top)
      Sagittal: screen-x = A->P (anterior left), screen-y = S->I
    """
    if axis == "axial":
        # vol[i, j, k] -> slice at k=idx, display (i=L/R, j=A/P)
        sl = vol[:, :, idx].T  # shape (j, i) = (AP, LR)
        sl = np.flipud(sl)     # flip so anterior is at top
        sl = np.fliplr(sl)     # flip L-R for radiological convention
        return sl
    elif axis == "coronal":
        # slice at j=idx, display (i=L/R, k=S/I)
        sl = vol[:, idx, :].T  # shape (k, i) = (SI, LR)
        sl = np.flipud(sl)     # flip so superior is at top
        sl = np.fliplr(sl)     # flip L-R for radiological convention
        return sl
    elif axis == "sagittal":
        # slice at i=idx, display (j=A/P, k=S/I)
        sl = vol[idx, :, :].T  # shape (k, j) = (SI, AP)
        sl = np.flipud(sl)     # flip so superior is at top
        return sl

def axis_max(vol, axis):
    return {"axial": vol.shape[2], "coronal": vol.shape[1],
            "sagittal": vol.shape[0]}[axis] - 1


# ---------------------------------------------------------------------------
# RGBA compositing helpers
# ---------------------------------------------------------------------------

def compute_edge_magnitude(vol_2d):
    """Sobel edge magnitude for a 2D slice."""
    sx = sobel(vol_2d, axis=0)
    sy = sobel(vol_2d, axis=1)
    return np.sqrt(sx**2 + sy**2)


def compute_structural_diff_slice(sl_t1w, sl_t1c, sl_mask, zscore_threshold=1.5):
    """
    Hybrid structural difference combining two complementary signals:
    
    1. EDGE DIFFERENCE — catches boundary misalignment.
    2. LOCAL RANK DIFFERENCE — catches interior structural changes 
       (cavities filling, regions appearing/disappearing).
    
    The rank component is aggressively thresholded to only fire on 
    large local changes (not normal gadolinium contrast variation).
    
    Returns normalised difference map in [0, 1].
    """
    from scipy.ndimage import uniform_filter
    
    # --- Component 1: Edge difference ---
    edges_t1w = compute_edge_magnitude(sl_t1w)
    edges_t1c = compute_edge_magnitude(sl_t1c)
    e_max_t1w = edges_t1w.max()
    e_max_t1c = edges_t1c.max()
    if e_max_t1w > 1e-8:
        edges_t1w = edges_t1w / e_max_t1w
    if e_max_t1c > 1e-8:
        edges_t1c = edges_t1c / e_max_t1c
    edge_diff = np.abs(edges_t1w - edges_t1c)
    
    # --- Component 2: Local z-score difference (heavily filtered) ---
    patch_size = 25  # larger patch = more stable z-scores
    
    local_mean_t1w = uniform_filter(sl_t1w, size=patch_size)
    local_mean_t1c = uniform_filter(sl_t1c, size=patch_size)
    local_sq_t1w = uniform_filter(sl_t1w**2, size=patch_size)
    local_sq_t1c = uniform_filter(sl_t1c**2, size=patch_size)
    local_std_t1w = np.sqrt(np.maximum(local_sq_t1w - local_mean_t1w**2, 0) + 1e-8)
    local_std_t1c = np.sqrt(np.maximum(local_sq_t1c - local_mean_t1c**2, 0) + 1e-8)
    
    z_t1w = (sl_t1w - local_mean_t1w) / local_std_t1w
    z_t1c = (sl_t1c - local_mean_t1c) / local_std_t1c
    
    rank_diff_raw = np.abs(z_t1w - z_t1c)
    
    # Suppress low-signal regions (background noise)
    # Only keep rank diff where BOTH scans have meaningful signal
    signal_thresh = 0.05
    signal_mask = (sl_t1w > signal_thresh) & (sl_t1c > signal_thresh)
    rank_diff_raw[~signal_mask] = 0
    
    # Hard threshold: only keep large z-score shifts
    # Normal gadolinium variation is typically < 1 std shift locally
    rank_diff_raw[rank_diff_raw < zscore_threshold] = 0
    
    # Normalise the surviving signal
    if sl_mask is not None and sl_mask.any():
        surviving = rank_diff_raw[sl_mask > 0]
        rd_max = np.percentile(surviving[surviving > 0], 95) if (surviving > 0).any() else 1.0
    else:
        rd_max = 1.0
    if rd_max > 1e-8:
        rank_diff = rank_diff_raw / rd_max
    else:
        rank_diff = rank_diff_raw
    rank_diff = np.clip(rank_diff, 0, 1)
    
    # Smooth the rank diff slightly to reduce speckle
    rank_diff = uniform_filter(rank_diff, size=3)
    
    # --- Combine: weighted blend, edges dominant ---
    diff = np.maximum(edge_diff, rank_diff * 0.7)
    
    # Zero outside brain
    if sl_mask is not None:
        diff[sl_mask == 0] = 0
    
    return diff


def make_overlay_rgba(gray_slice, diff_slice, cmap_name="hot",
                       vmin=0, vmax=0.5, alpha_scale=0.7):
    """T1w grayscale base + heatmap overlay scaled by difference magnitude."""
    g = np.clip(gray_slice, 0, 1)
    base = np.stack([g, g, g, np.ones_like(g)], axis=-1)
    cmap = plt.get_cmap(cmap_name)
    normed = np.clip((diff_slice - vmin) / (vmax - vmin + 1e-10), 0, 1)
    heat_rgba = cmap(normed)
    alpha = normed * alpha_scale
    out = base.copy()
    out[..., :3] = base[..., :3] * (1 - alpha[..., np.newaxis]) + \
                   heat_rgba[..., :3] * alpha[..., np.newaxis]
    out[..., 3] = 1.0
    return np.clip(out, 0, 1)


def make_scan_with_boundaries(gray_slice, boundary_slice, 
                                color=(0.0, 1.0, 0.4), alpha=0.8):
    """
    Grayscale scan with coloured label boundary contours overlaid.
    Returns RGBA array.
    """
    g = np.clip(gray_slice, 0, 1)
    out = np.stack([g, g, g, np.ones_like(g)], axis=-1)
    mask = boundary_slice > 0
    for c in range(3):
        out[mask, c] = out[mask, c] * (1 - alpha) + color[c] * alpha
    return np.clip(out, 0, 1)


def make_disagreement_overlay(gray_slice, disagree_slice, brain_mask_slice,
                                color=(1.0, 0.15, 0.15), alpha=0.65):
    """
    T1w grayscale base + red overlay where labels_t1w != labels_t1c.
    Transparent where they agree or outside brain.
    Returns RGBA array.
    """
    g = np.clip(gray_slice, 0, 1)
    out = np.stack([g, g, g, np.ones_like(g)], axis=-1)
    # Only show disagreement inside brain
    mask = (disagree_slice > 0) & (brain_mask_slice > 0)
    for c in range(3):
        out[mask, c] = out[mask, c] * (1 - alpha) + color[c] * alpha
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def run_viewer(t1w, t1c, brain_mask,
               labels_t1w_vol, labels_t1c_vol, bnd_t1w, bnd_t1c,
               pred_t1w_vol, pred_t1c_vol,
               initial_axis, initial_slice):

    current_axis = [initial_axis]
    zoom_center = [None]      # (row, col) in image coords
    zoom_radius = [30]        # half-width of zoom box in pixels
    current_zoom_rw = [0]     # actual ROI pixel width (after clamping)

    has_labels = labels_t1w_vol is not None
    has_pred_t1w = pred_t1w_vol is not None and labels_t1w_vol is not None
    has_pred_t1c = pred_t1c_vol is not None and labels_t1c_vol is not None
    has_pred_cross = pred_t1w_vol is not None and pred_t1c_vol is not None
    has_label_window = has_labels or has_pred_t1w or has_pred_t1c or has_pred_cross

    # Precompute disagreement volumes
    # Panel 1: T1w GT vs Pred
    if has_pred_t1w:
        disagree_t1w_vol = (labels_t1w_vol != pred_t1w_vol).astype(np.uint8)
        bg = (labels_t1w_vol == 0) & (pred_t1w_vol == 0)
        disagree_t1w_vol[bg] = 0

    # Panel 2: T1c GT vs Pred
    if has_pred_t1c:
        disagree_t1c_vol = (labels_t1c_vol != pred_t1c_vol).astype(np.uint8)
        bg = (labels_t1c_vol == 0) & (pred_t1c_vol == 0)
        disagree_t1c_vol[bg] = 0

    # Panel 3: Pred T1w vs Pred T1c (cross-contrast prediction agreement)
    if has_pred_cross:
        disagree_pred_cross_vol = (pred_t1w_vol != pred_t1c_vol).astype(np.uint8)
        bg = (pred_t1w_vol == 0) & (pred_t1c_vol == 0)
        disagree_pred_cross_vol[bg] = 0

    # ===================================================================
    # WINDOW 1 — Intensity comparison (3 panels, resizable)
    # ===================================================================
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6),
                                          facecolor="#111111", num="Intensity")
    fig1.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.18,
                          wspace=0.05)
    fig1.suptitle("Registration Quality — Intensity", color="white",
                  fontsize=14, fontweight="bold")

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])

    ax1.set_title("T1w", color="white", fontsize=11, pad=4)
    ax2.set_title("T1c", color="white", fontsize=11, pad=4)
    ax3.set_title("T1w + structural Δ", color="white", fontsize=11, pad=4)

    # Slider
    ax_slider = fig1.add_axes([0.12, 0.06, 0.45, 0.025], facecolor="#333")
    slider = Slider(ax_slider, 'Slice', 0, axis_max(t1w, initial_axis),
                    valinit=initial_slice, valstep=1, color="#4a9eff")

    # Threshold slider for structural diff sensitivity
    ax_thresh = fig1.add_axes([0.12, 0.02, 0.45, 0.025], facecolor="#333")
    thresh_slider = Slider(ax_thresh, 'Z-thresh', 0.5, 4.0,
                            valinit=1.5, valstep=0.1, color="#ff8844")

    # Axis radio
    ax_radio = fig1.add_axes([0.65, 0.01, 0.10, 0.12], facecolor="#111111")
    radio = RadioButtons(ax_radio, ("axial", "coronal", "sagittal"),
                         active=["axial", "coronal", "sagittal"].index(initial_axis))
    for lb in radio.labels:
        lb.set_color("white"); lb.set_fontsize(9)

    # Metrics text
    ax_text = fig1.add_axes([0.01, 0.01, 0.10, 0.12], facecolor="#111111")
    ax_text.set_xticks([]); ax_text.set_yticks([])
    for sp in ax_text.spines.values(): sp.set_visible(False)
    metrics_text = ax_text.text(0.05, 0.5, "", color="#00ff88", fontsize=10,
                                fontfamily="monospace", transform=ax_text.transAxes,
                                verticalalignment="center")

    # Initial images
    s0 = get_slice(t1w, initial_axis, initial_slice)
    im1 = ax1.imshow(s0, cmap="gray", vmin=0, vmax=1, aspect="equal")
    im2 = ax2.imshow(s0, cmap="gray", vmin=0, vmax=1, aspect="equal")
    im3 = ax3.imshow(np.zeros((*s0.shape, 4)), aspect="equal")

    # Yellow rectangle showing zoom ROI on main panels
    from matplotlib.patches import Rectangle
    zoom_rects = []
    for ax in [ax1, ax2, ax3]:
        rect = Rectangle((0, 0), 1, 1, linewidth=1.5, edgecolor="#ffcc00",
                          facecolor="none", visible=False)
        ax.add_patch(rect)
        zoom_rects.append(rect)

    # ===================================================================
    # WINDOW 2 — Disagreement analysis (3 panels, resizable)
    # Panel 1: T1w GT vs Pred  (orange, if pred_t1w provided)
    # Panel 2: T1c GT vs Pred  (orange, if pred_t1c provided)
    # Panel 3: Pred T1w vs Pred T1c  (red, cross-contrast pred agreement)
    # ===================================================================
    fig2 = None
    lax1 = lax2 = lax3 = None
    lim1 = lim2 = lim3 = None
    label_metrics = None

    if has_label_window:
        fig2, (lax1, lax2, lax3) = plt.subplots(1, 3, figsize=(16, 6),
                                                   facecolor="#111111",
                                                   num="Disagreement")
        fig2.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.18,
                              wspace=0.05)
        fig2.suptitle("Disagreement Analysis",
                      color="white", fontsize=14, fontweight="bold")

        for ax in [lax1, lax2, lax3]:
            ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])

        # Titles based on what's available
        t1w_title = "T1w: GT vs Pred" if has_pred_t1w else "T1w + GT boundaries"
        t1c_title = "T1c: GT vs Pred" if has_pred_t1c else "T1c + GT boundaries"
        cross_title = "Pred T1w vs Pred T1c" if has_pred_cross else "(need both preds)"

        lax1.set_title(t1w_title, color="#ffaa00", fontsize=11, pad=4)
        lax2.set_title(t1c_title, color="#ffaa00", fontsize=11, pad=4)
        lax3.set_title(cross_title, color="#ff4444", fontsize=11, pad=4)

        lim1 = lax1.imshow(np.zeros((*s0.shape, 4)), aspect="equal")
        lim2 = lax2.imshow(np.zeros((*s0.shape, 4)), aspect="equal")
        lim3 = lax3.imshow(np.zeros((*s0.shape, 4)), aspect="equal")

        # Slider on disagreement window (synced to main)
        ax_slider2 = fig2.add_axes([0.12, 0.06, 0.55, 0.025], facecolor="#333")
        slider2 = Slider(ax_slider2, 'Slice', 0, axis_max(t1w, initial_axis),
                          valinit=initial_slice, valstep=1, color="#ff8844")

        # Axis radio on disagreement window
        ax_radio2 = fig2.add_axes([0.75, 0.01, 0.10, 0.12], facecolor="#111111")
        radio2 = RadioButtons(ax_radio2, ("axial", "coronal", "sagittal"),
                               active=["axial", "coronal", "sagittal"].index(initial_axis))
        for lb in radio2.labels:
            lb.set_color("white"); lb.set_fontsize(9)

        # Metrics
        lax_text = fig2.add_axes([0.01, 0.01, 0.65, 0.04], facecolor="#111111")
        lax_text.set_xticks([]); lax_text.set_yticks([])
        for sp in lax_text.spines.values(): sp.set_visible(False)
        label_metrics = lax_text.text(0.02, 0.5, "", color="#ffaa00", fontsize=10,
                                       fontfamily="monospace",
                                       transform=lax_text.transAxes,
                                       verticalalignment="center")

        # Hover annotation for label names
        hover_text = fig2.text(0.5, 0.93, "", color="#00ffcc", fontsize=10,
                                fontfamily="monospace", ha="center", va="bottom")
    else:
        hover_text = None

    # Yellow rectangles on label panels too (for label zoom)
    label_zoom_rects = []
    if has_label_window:
        from matplotlib.patches import Rectangle as Rect2
        for ax in [lax1, lax2, lax3]:
            rect = Rect2((0, 0), 1, 1, linewidth=1.5, edgecolor="#ffcc00",
                          facecolor="none", visible=False)
            ax.add_patch(rect)
            label_zoom_rects.append(rect)

    # ===================================================================
    # WINDOW 3 — Intensity zoom popout
    # ===================================================================
    fig_z = plt.figure(figsize=(8, 5), facecolor="#111111", num="Zoom — Intensity")
    ax_z = fig_z.add_axes([0.02, 0.10, 0.96, 0.82])
    ax_z.set_facecolor("black")
    ax_z.set_xticks([]); ax_z.set_yticks([])
    ax_z.set_title("T1w  |  T1c   (click main window to inspect)",
                    color="#ffcc00", fontsize=11)
    im_z = ax_z.imshow(np.zeros((10, 10, 4)), aspect="equal")

    # +/- buttons on intensity zoom
    ax_zin = fig_z.add_axes([0.35, 0.01, 0.12, 0.06])
    ax_zout = fig_z.add_axes([0.53, 0.01, 0.12, 0.06])
    btn_zin = Button(ax_zin, '+ Zoom In', color="#333", hovercolor="#555")
    btn_zout = Button(ax_zout, '− Zoom Out', color="#333", hovercolor="#555")
    btn_zin.label.set_color("white"); btn_zin.label.set_fontsize(9)
    btn_zout.label.set_color("white"); btn_zout.label.set_fontsize(9)

    # ===================================================================
    # WINDOW 4 — Label zoom popout (only if labels)
    # ===================================================================
    fig_lz = None
    ax_lz = None
    im_lz = None
    label_zoom_center = [None]

    if has_label_window:
        fig_lz = plt.figure(figsize=(8, 5), facecolor="#111111",
                            num="Zoom — Disagreement")
        ax_lz = fig_lz.add_axes([0.02, 0.10, 0.96, 0.82])
        ax_lz.set_facecolor("black")
        ax_lz.set_xticks([]); ax_lz.set_yticks([])
        ax_lz.set_title("T1w+labels  |  T1c+labels  |  Disagreement   "
                        "(click label window)",
                        color="#ffcc00", fontsize=10)
        im_lz = ax_lz.imshow(np.zeros((10, 10, 4)), aspect="equal")

        # +/- buttons on label zoom
        ax_lzin = fig_lz.add_axes([0.35, 0.01, 0.12, 0.06])
        ax_lzout = fig_lz.add_axes([0.53, 0.01, 0.12, 0.06])
        btn_lzin = Button(ax_lzin, '+ Zoom In', color="#333", hovercolor="#555")
        btn_lzout = Button(ax_lzout, '− Zoom Out', color="#333", hovercolor="#555")
        btn_lzin.label.set_color("white"); btn_lzin.label.set_fontsize(9)
        btn_lzout.label.set_color("white"); btn_lzout.label.set_fontsize(9)

    # ===================================================================
    # WINDOW 5 — Error Correlation (2 panels: T1w + T1c with dual overlay)
    # Red = prediction error (GT != Pred)
    # Blue = structural difference (from diff map)
    # Purple = both (error correlates with structural difference)
    # ===================================================================
    has_error_corr = has_pred_t1w or has_pred_t1c
    fig_ec = None
    ax_ec1 = ax_ec2 = None
    im_ec1 = im_ec2 = None
    ec_metrics = None

    if has_error_corr:
        fig_ec, (ax_ec1, ax_ec2) = plt.subplots(1, 2, figsize=(14, 6),
                                                  facecolor="#111111",
                                                  num="Error Correlation")
        fig_ec.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.18,
                                wspace=0.05)
        fig_ec.suptitle("Error Correlation — Red=Pred Error  Blue=Struct Δ  Purple=Both",
                        color="white", fontsize=12, fontweight="bold")

        for ax in [ax_ec1, ax_ec2]:
            ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])

        ax_ec1.set_title("T1w" if has_pred_t1w else "T1w (no pred)",
                         color="#cc88ff", fontsize=11, pad=4)
        ax_ec2.set_title("T1c" if has_pred_t1c else "T1c (no pred)",
                         color="#cc88ff", fontsize=11, pad=4)

        im_ec1 = ax_ec1.imshow(np.zeros((10, 10, 4)), aspect="equal")
        im_ec2 = ax_ec2.imshow(np.zeros((10, 10, 4)), aspect="equal")

        # Slider synced
        ax_ec_slider = fig_ec.add_axes([0.12, 0.06, 0.45, 0.025], facecolor="#333")
        ec_slider = Slider(ax_ec_slider, 'Slice', 0, axis_max(t1w, initial_axis),
                            valinit=initial_slice, valstep=1, color="#cc44ff")

        # Axis radio
        ax_ec_radio = fig_ec.add_axes([0.65, 0.01, 0.10, 0.12], facecolor="#111111")
        ec_radio = RadioButtons(ax_ec_radio, ("axial", "coronal", "sagittal"),
                                 active=["axial", "coronal", "sagittal"].index(initial_axis))
        for lb in ec_radio.labels:
            lb.set_color("white"); lb.set_fontsize(9)

        # Metrics
        ax_ec_text = fig_ec.add_axes([0.01, 0.01, 0.60, 0.04], facecolor="#111111")
        ax_ec_text.set_xticks([]); ax_ec_text.set_yticks([])
        for sp in ax_ec_text.spines.values(): sp.set_visible(False)
        ec_metrics = ax_ec_text.text(0.02, 0.5, "", color="#cc88ff", fontsize=9,
                                      fontfamily="monospace",
                                      transform=ax_ec_text.transAxes,
                                      verticalalignment="center")

        # Yellow rectangles for zoom ROI
        ec_zoom_rects = []
        from matplotlib.patches import Rectangle as RectEC
        for ax in [ax_ec1, ax_ec2]:
            rect = RectEC((0, 0), 1, 1, linewidth=1.5, edgecolor="#ffcc00",
                           facecolor="none", visible=False)
            ax.add_patch(rect)
            ec_zoom_rects.append(rect)

    # ===================================================================
    # WINDOW 6 — Error Correlation Zoom popout
    # ===================================================================
    fig_ecz = None
    ax_ecz = None
    im_ecz = None
    ec_zoom_center = [None]

    if has_error_corr:
        fig_ecz = plt.figure(figsize=(8, 5), facecolor="#111111",
                              num="Zoom — Error Correlation")
        ax_ecz = fig_ecz.add_axes([0.02, 0.10, 0.96, 0.82])
        ax_ecz.set_facecolor("black")
        ax_ecz.set_xticks([]); ax_ecz.set_yticks([])
        ax_ecz.set_title("T1w err | T1c err  (click error correlation window)",
                         color="#cc88ff", fontsize=10)
        im_ecz = ax_ecz.imshow(np.zeros((10, 10, 4)), aspect="equal")

        # +/- buttons
        ax_eczin = fig_ecz.add_axes([0.35, 0.01, 0.12, 0.06])
        ax_eczout = fig_ecz.add_axes([0.53, 0.01, 0.12, 0.06])
        btn_eczin = Button(ax_eczin, '+ Zoom In', color="#333", hovercolor="#555")
        btn_eczout = Button(ax_eczout, '− Zoom Out', color="#333", hovercolor="#555")
        btn_eczin.label.set_color("white"); btn_eczin.label.set_fontsize(9)
        btn_eczout.label.set_color("white"); btn_eczout.label.set_fontsize(9)

    # ===================================================================
    # Crosshair system — two independent sync groups
    # Group 1: Window 1 (Intensity) ↔ Window 2 (Labels)
    # Group 2: Window 3 (Zoom Intensity) ↔ Window 4 (Zoom Labels)
    # ===================================================================
    crosshair_color = "#00ffff"
    crosshair_alpha = 0.6
    crosshair_lw = 0.8

    # --- Group 1: Main panels ---
    main_image_axes = [ax1, ax2, ax3]
    if has_label_window:
        main_image_axes += [lax1, lax2, lax3]
    if has_error_corr:
        main_image_axes += [ax_ec1, ax_ec2]

    main_crosshairs = {}
    for ax in main_image_axes:
        hline = ax.axhline(y=0, color=crosshair_color, alpha=crosshair_alpha,
                           linewidth=crosshair_lw, visible=False)
        vline = ax.axvline(x=0, color=crosshair_color, alpha=crosshair_alpha,
                           linewidth=crosshair_lw, visible=False)
        main_crosshairs[id(ax)] = (hline, vline)

    def update_main_crosshairs(x_disp, y_disp):
        for ax in main_image_axes:
            h, v = main_crosshairs[id(ax)]
            h.set_ydata([y_disp, y_disp]); h.set_visible(True)
            v.set_xdata([x_disp, x_disp]); v.set_visible(True)

    def hide_main_crosshairs():
        for ax in main_image_axes:
            h, v = main_crosshairs[id(ax)]
            h.set_visible(False); v.set_visible(False)

    def display_to_voxel(x_disp, y_disp, axis, idx, h, w):
        """Convert display coords back to volume (i, j, k) indices."""
        # Undo the flips from get_slice
        if axis == "axial":
            # get_slice: vol[:,:,idx].T -> flipud -> fliplr
            col = w - 1 - int(x_disp)   # undo fliplr
            row = int(y_disp)            # undo flipud (y is already from bottom)
            return (col, row, idx)       # (i, j, k)
        elif axis == "coronal":
            col = w - 1 - int(x_disp)
            row = int(y_disp)
            return (col, idx, row)
        elif axis == "sagittal":
            col = int(x_disp)
            row = int(y_disp)
            return (idx, col, row)

    def get_label_info_at_cursor(x_disp, y_disp, panel_ax):
        """Look up GT and Pred label names at cursor position."""
        axis = current_axis[0]
        idx = int(slider.val)
        sl = get_slice(t1w, axis, idx)
        h, w = sl.shape

        # Clamp to valid range
        cx = int(np.clip(x_disp, 0, w - 1))
        cy_disp = int(np.clip(y_disp, 0, h - 1))

        vox = display_to_voxel(cx, cy_disp, axis, idx, h, w)
        if vox is None:
            return ""
        i, j, k = vox
        # Bounds check
        shape = t1w.shape
        if not (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
            return ""

        parts = []
        if panel_ax in (lax1,):
            # T1w: GT vs Pred
            if labels_t1w_vol is not None:
                gt = int(labels_t1w_vol[i, j, k])
                parts.append(f"GT: {label_name(gt)}")
            if pred_t1w_vol is not None:
                pr = int(pred_t1w_vol[i, j, k])
                parts.append(f"Pred: {label_name(pr)}")
        elif panel_ax in (lax2,):
            # T1c: GT vs Pred
            if labels_t1c_vol is not None:
                gt = int(labels_t1c_vol[i, j, k])
                parts.append(f"GT: {label_name(gt)}")
            if pred_t1c_vol is not None:
                pr = int(pred_t1c_vol[i, j, k])
                parts.append(f"Pred: {label_name(pr)}")
        elif panel_ax in (lax3,):
            # Pred T1w vs Pred T1c
            if pred_t1w_vol is not None:
                p1 = int(pred_t1w_vol[i, j, k])
                parts.append(f"PredT1w: {label_name(p1)}")
            if pred_t1c_vol is not None:
                p2 = int(pred_t1c_vol[i, j, k])
                parts.append(f"PredT1c: {label_name(p2)}")

        return "   ".join(parts)

    def on_motion_main(event):
        if event.inaxes not in main_image_axes:
            hide_main_crosshairs()
            if hover_text:
                hover_text.set_text("")
            fig1.canvas.draw_idle()
            if fig2: fig2.canvas.draw_idle()
            if fig_ec: fig_ec.canvas.draw_idle()
            return
        if event.xdata is None or event.ydata is None:
            return
        update_main_crosshairs(event.xdata, event.ydata)

        # Label hover annotation on Window 2
        if hover_text and has_label_window:
            label_panels = [lax1, lax2, lax3]
            if event.inaxes in label_panels:
                info = get_label_info_at_cursor(event.xdata, event.ydata,
                                                 event.inaxes)
                hover_text.set_text(info)
            else:
                hover_text.set_text("")

        fig1.canvas.draw_idle()
        if fig2: fig2.canvas.draw_idle()
        if fig_ec: fig_ec.canvas.draw_idle()

    # --- Group 2: Zoom panels with multiple vertical crosshairs per panel ---
    # Intensity zoom has 2 sub-panels: T1w | T1c
    # Label zoom has 3 sub-panels: T1w+lbl | T1c+lbl | Disagree
    # Each sub-panel gets its own vertical crosshair line.
    # One horizontal line per zoom window (shared across sub-panels).

    # Intensity zoom: 1 hline + 3 vlines (T1w | T1c | Diff)
    zh_line = ax_z.axhline(y=0, color=crosshair_color, alpha=crosshair_alpha,
                            linewidth=crosshair_lw, visible=False)
    zv_lines = []
    for _ in range(3):
        vl = ax_z.axvline(x=0, color=crosshair_color, alpha=crosshair_alpha,
                           linewidth=crosshair_lw, visible=False)
        zv_lines.append(vl)

    # Label zoom: 1 hline + 3 vlines
    lzh_line = None
    lzv_lines = []
    lz_hover_text = None
    if ax_lz is not None:
        lzh_line = ax_lz.axhline(y=0, color=crosshair_color, alpha=crosshair_alpha,
                                  linewidth=crosshair_lw, visible=False)
        for _ in range(3):
            vl = ax_lz.axvline(x=0, color=crosshair_color, alpha=crosshair_alpha,
                                linewidth=crosshair_lw, visible=False)
            lzv_lines.append(vl)
        lz_hover_text = fig_lz.text(0.5, 0.95, "", color="#00ffcc", fontsize=9,
                                     fontfamily="monospace", ha="center", va="bottom")

    zoom_axes_list = [ax_z]
    if ax_lz is not None:
        zoom_axes_list.append(ax_lz)

    def get_zoom_rw():
        """Get the actual ROI pixel width from last zoom update."""
        return current_zoom_rw[0]

    def local_x_in_panel(x_disp, rw, gap, n_panels):
        """
        Given an x position in a composite image, figure out which
        sub-panel it's in and the local x within that panel.
        Returns (panel_index, local_x) or (None, None) if in a gap.
        """
        for i in range(n_panels):
            panel_start = i * (rw + gap)
            panel_end = panel_start + rw
            if panel_start <= x_disp < panel_end:
                return i, x_disp - panel_start
        return None, None

    def panel_x_to_composite(local_x, panel_idx, rw, gap):
        """Convert local x + panel index back to composite x."""
        return panel_idx * (rw + gap) + local_x

    def update_zoom_crosshairs(x_disp, y_disp, source_ax):
        """
        Update crosshairs on both zoom windows.
        Each sub-panel in each composite gets its own vertical line
        at the corresponding local x position.
        """
        rw = get_zoom_rw()
        if rw == 0:
            return
        gap = 2

        # Figure out which panel and local_x from the source
        # Both zoom windows now have 3 panels
        src_panel, src_local_x = local_x_in_panel(x_disp, rw, gap, 3)

        if src_panel is None or src_local_x is None:
            # Cursor is in a gap — hide everything
            hide_zoom_crosshairs()
            return

        # The local_x is the same for ALL panels (same spatial position)
        # so every vertical line goes at that same local_x within its panel

        # --- Intensity zoom: 2 vlines ---
        zh_line.set_ydata([y_disp, y_disp]); zh_line.set_visible(True)
        for i, vl in enumerate(zv_lines):
            cx = panel_x_to_composite(src_local_x, i, rw, gap)
            vl.set_xdata([cx, cx]); vl.set_visible(True)

        # --- Label zoom: 3 vlines ---
        if lzh_line is not None:
            lzh_line.set_ydata([y_disp, y_disp]); lzh_line.set_visible(True)
            for i, vl in enumerate(lzv_lines):
                cx = panel_x_to_composite(src_local_x, i, rw, gap)
                vl.set_xdata([cx, cx]); vl.set_visible(True)

    def hide_zoom_crosshairs():
        zh_line.set_visible(False)
        for vl in zv_lines:
            vl.set_visible(False)
        if lzh_line is not None:
            lzh_line.set_visible(False)
            for vl in lzv_lines:
                vl.set_visible(False)

    def on_motion_zoom(event):
        if event.inaxes not in zoom_axes_list:
            hide_zoom_crosshairs()
            if lz_hover_text:
                lz_hover_text.set_text("")
            fig_z.canvas.draw_idle()
            if fig_lz: fig_lz.canvas.draw_idle()
            return
        if event.xdata is None or event.ydata is None:
            return
        update_zoom_crosshairs(event.xdata, event.ydata, event.inaxes)

        # Label hover on label zoom window
        if lz_hover_text and event.inaxes == ax_lz and label_zoom_center[0] is not None:
            rw = get_zoom_rw()
            gap = 2
            if rw > 0:
                panel_idx, local_x = local_x_in_panel(event.xdata, rw, gap, 3)
                if panel_idx is not None and local_x is not None:
                    # Map zoom local coords back to volume
                    axis = current_axis[0]
                    idx = int(slider.val)
                    sl = get_slice(t1w, axis, idx)
                    h, w = sl.shape
                    cy, cx = label_zoom_center[0]
                    r = zoom_radius[0]
                    y0 = max(0, int(cy - r))
                    x0 = max(0, int(cx - r))

                    # Display coords in main image space
                    main_x = x0 + local_x
                    main_y = y0 + event.ydata  # y is in ROI coords already

                    vox = display_to_voxel(main_x, main_y, axis, idx, h, w)
                    if vox:
                        i, j, k = vox
                        shape = t1w.shape
                        if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                            parts = []
                            if panel_idx == 0:
                                # T1w GT vs Pred
                                if labels_t1w_vol is not None:
                                    parts.append(f"GT:{label_name(int(labels_t1w_vol[i,j,k]))}")
                                if pred_t1w_vol is not None:
                                    parts.append(f"Pred:{label_name(int(pred_t1w_vol[i,j,k]))}")
                            elif panel_idx == 1:
                                # T1c GT vs Pred
                                if labels_t1c_vol is not None:
                                    parts.append(f"GT:{label_name(int(labels_t1c_vol[i,j,k]))}")
                                if pred_t1c_vol is not None:
                                    parts.append(f"Pred:{label_name(int(pred_t1c_vol[i,j,k]))}")
                            elif panel_idx == 2:
                                # Pred T1w vs Pred T1c
                                if pred_t1w_vol is not None:
                                    parts.append(f"PredT1w:{label_name(int(pred_t1w_vol[i,j,k]))}")
                                if pred_t1c_vol is not None:
                                    parts.append(f"PredT1c:{label_name(int(pred_t1c_vol[i,j,k]))}")
                            lz_hover_text.set_text("  ".join(parts))
                        else:
                            lz_hover_text.set_text("")
                else:
                    lz_hover_text.set_text("")
        elif lz_hover_text:
            lz_hover_text.set_text("")

        fig_z.canvas.draw_idle()
        if fig_lz: fig_lz.canvas.draw_idle()

    # ===================================================================
    # Zoom update helpers
    # ===================================================================

    def update_intensity_zoom(sl_t1w, sl_t1c, sl_diff, h, w):
        """Update intensity zoom popout: T1w | T1c | Diff overlay."""
        if zoom_center[0] is None:
            for rect in zoom_rects:
                rect.set_visible(False)
            return

        cy, cx = zoom_center[0]
        r = zoom_radius[0]
        y0, y1 = max(0, int(cy - r)), min(h, int(cy + r))
        x0, x1 = max(0, int(cx - r)), min(w, int(cx + r))

        roi_t1w = sl_t1w[y0:y1, x0:x1]
        roi_t1c = sl_t1c[y0:y1, x0:x1]
        roi_diff = sl_diff[y0:y1, x0:x1]
        rh, rw = roi_t1w.shape
        if rh == 0 or rw == 0:
            return
        current_zoom_rw[0] = rw  # store for crosshair use

        # Panel 3: T1w + diff overlay (same as main window panel 3)
        panel_overlay = make_overlay_rgba(roi_t1w, roi_diff)

        gap = 2
        total_w = rw * 3 + gap * 2
        composite = np.zeros((rh, total_w, 4))

        # Panel 1: T1w grayscale
        for c in range(3):
            composite[:, :rw, c] = roi_t1w
        composite[:, :rw, 3] = 1.0

        # Gap
        composite[:, rw:rw+gap, :3] = [1.0, 0.8, 0.0]
        composite[:, rw:rw+gap, 3] = 1.0

        # Panel 2: T1c grayscale
        for c in range(3):
            composite[:, rw+gap:rw*2+gap, c] = roi_t1c
        composite[:, rw+gap:rw*2+gap, 3] = 1.0

        # Gap
        composite[:, rw*2+gap:rw*2+gap*2, :3] = [1.0, 0.8, 0.0]
        composite[:, rw*2+gap:rw*2+gap*2, 3] = 1.0

        # Panel 3: Diff overlay
        composite[:, rw*2+gap*2:, :] = panel_overlay

        im_z.set_data(np.clip(composite, 0, 1))
        im_z.set_extent([0, total_w, 0, rh])
        ax_z.set_xlim(0, total_w)
        ax_z.set_ylim(0, rh)

        local_cos = cosine_similarity_2d(roi_t1w, roi_t1c)
        local_ssim = compute_ssim_2d(roi_t1w, roi_t1c)
        ax_z.set_title(
            f"T1w | T1c | Δ   CosSim:{local_cos:.3f}  SSIM:{local_ssim:.3f}  [{2*r}px]",
            color="#ffcc00", fontsize=11
        )
        fig_z.canvas.draw_idle()

        rect_x, rect_y = x0, h - y1
        rect_w, rect_h = x1 - x0, y1 - y0
        for rect in zoom_rects:
            rect.set_xy((rect_x, rect_y))
            rect.set_width(rect_w); rect.set_height(rect_h)
            rect.set_visible(True)

    def update_label_zoom(sl_t1w, sl_t1c, sl_mask,
                           axis, idx, h, w):
        """Update label/disagreement zoom popout with 3-panel composite."""
        if label_zoom_center[0] is None:
            for rect in label_zoom_rects:
                rect.set_visible(False)
            return

        cy, cx = label_zoom_center[0]
        r = zoom_radius[0]
        y0, y1 = max(0, int(cy - r)), min(h, int(cy + r))
        x0, x1 = max(0, int(cx - r)), min(w, int(cx + r))

        roi_t1w = sl_t1w[y0:y1, x0:x1]
        roi_t1c = sl_t1c[y0:y1, x0:x1]
        roi_mask = sl_mask[y0:y1, x0:x1]
        rh, rw = roi_t1w.shape
        if rh == 0 or rw == 0:
            return

        # Panel 1: T1w GT vs Pred or GT boundaries
        if has_pred_t1w:
            roi_dis = get_slice(disagree_t1w_vol, axis, idx)[y0:y1, x0:x1]
            panel1 = make_disagreement_overlay(roi_t1w, roi_dis, roi_mask,
                                                color=(1.0, 0.6, 0.0), alpha=0.65)
        elif has_labels:
            roi_bnd = get_slice(bnd_t1w, axis, idx)[y0:y1, x0:x1]
            panel1 = make_scan_with_boundaries(roi_t1w, roi_bnd,
                                                color=(0.0, 1.0, 0.4), alpha=0.8)
        else:
            g = np.clip(roi_t1w, 0, 1)
            panel1 = np.stack([g, g, g, np.ones_like(g)], axis=-1)

        # Panel 2: T1c GT vs Pred or GT boundaries
        if has_pred_t1c:
            roi_dis = get_slice(disagree_t1c_vol, axis, idx)[y0:y1, x0:x1]
            panel2 = make_disagreement_overlay(roi_t1c, roi_dis, roi_mask,
                                                color=(1.0, 0.6, 0.0), alpha=0.65)
        elif has_labels:
            roi_bnd = get_slice(bnd_t1c, axis, idx)[y0:y1, x0:x1]
            panel2 = make_scan_with_boundaries(roi_t1c, roi_bnd,
                                                color=(0.0, 1.0, 0.4), alpha=0.8)
        else:
            g = np.clip(roi_t1c, 0, 1)
            panel2 = np.stack([g, g, g, np.ones_like(g)], axis=-1)

        # Panel 3: Pred T1w vs Pred T1c
        if has_pred_cross:
            roi_dis_cross = get_slice(disagree_pred_cross_vol, axis, idx)[y0:y1, x0:x1]
            panel3 = make_disagreement_overlay(roi_t1w, roi_dis_cross, roi_mask,
                                                color=(1.0, 0.15, 0.15), alpha=0.65)
        else:
            g = np.clip(roi_t1w, 0, 1)
            panel3 = np.stack([g, g, g, np.ones_like(g)], axis=-1)

        gap = 2
        total_w = rw * 3 + gap * 2
        composite = np.zeros((rh, total_w, 4))
        composite[:, :rw, :] = panel1
        composite[:, rw:rw+gap, :3] = [1.0, 0.8, 0.0]
        composite[:, rw:rw+gap, 3] = 1.0
        composite[:, rw+gap:rw*2+gap, :] = panel2
        composite[:, rw*2+gap:rw*2+gap*2, :3] = [1.0, 0.8, 0.0]
        composite[:, rw*2+gap:rw*2+gap*2, 3] = 1.0
        composite[:, rw*2+gap*2:, :] = panel3

        im_lz.set_data(np.clip(composite, 0, 1))
        im_lz.set_extent([0, total_w, 0, rh])
        ax_lz.set_xlim(0, total_w)
        ax_lz.set_ylim(0, rh)

        ax_lz.set_title(
            f"T1w err | T1c err | GT disagree   [{2*r}px]",
            color="#ffcc00", fontsize=10
        )
        fig_lz.canvas.draw_idle()

        rect_x, rect_y = x0, h - y1
        rect_w, rect_h = x1 - x0, y1 - y0
        for rect in label_zoom_rects:
            rect.set_xy((rect_x, rect_y))
            rect.set_width(rect_w); rect.set_height(rect_h)
            rect.set_visible(True)

    def update_ec_zoom(sl_t1w, sl_t1c, sl_diff, sl_mask, axis, idx, h, w,
                        struct_mask, alpha_overlay):
        """Update error correlation zoom: T1w err | T1c err side by side."""
        if not has_error_corr or ec_zoom_center[0] is None:
            if has_error_corr:
                for rect in ec_zoom_rects:
                    rect.set_visible(False)
            return

        cy, cx = ec_zoom_center[0]
        r = zoom_radius[0]
        y0, y1 = max(0, int(cy - r)), min(h, int(cy + r))
        x0, x1 = max(0, int(cx - r)), min(w, int(cx + r))

        roi_t1w = sl_t1w[y0:y1, x0:x1]
        roi_t1c = sl_t1c[y0:y1, x0:x1]
        roi_mask = sl_mask[y0:y1, x0:x1]
        roi_struct = struct_mask[y0:y1, x0:x1]
        rh, rw = roi_t1w.shape
        if rh == 0 or rw == 0:
            return

        def build_ec_roi(gray, pred_err, struct_m, brain_m, alpha_ov):
            g = np.clip(gray, 0, 1)
            out = np.stack([g, g, g, np.ones_like(g)], axis=-1)
            pm = (pred_err > 0) & (brain_m > 0)
            ro = pm & ~struct_m
            bo = struct_m & ~pm
            bth = pm & struct_m
            out[ro, 0] = out[ro, 0]*(1-alpha_ov) + 1.0*alpha_ov
            out[ro, 1] = out[ro, 1]*(1-alpha_ov) + 0.15*alpha_ov
            out[ro, 2] = out[ro, 2]*(1-alpha_ov) + 0.15*alpha_ov
            out[bo, 0] = out[bo, 0]*(1-alpha_ov) + 0.15*alpha_ov
            out[bo, 1] = out[bo, 1]*(1-alpha_ov) + 0.3*alpha_ov
            out[bo, 2] = out[bo, 2]*(1-alpha_ov) + 1.0*alpha_ov
            out[bth, 0] = out[bth, 0]*(1-alpha_ov) + 0.8*alpha_ov
            out[bth, 1] = out[bth, 1]*(1-alpha_ov) + 0.1*alpha_ov
            out[bth, 2] = out[bth, 2]*(1-alpha_ov) + 0.9*alpha_ov
            return np.clip(out, 0, 1)

        # Panel 1: T1w
        if has_pred_t1w:
            roi_dis = get_slice(disagree_t1w_vol, axis, idx)[y0:y1, x0:x1]
            panel1 = build_ec_roi(roi_t1w, roi_dis, roi_struct, roi_mask, alpha_overlay)
        else:
            g = np.clip(roi_t1w, 0, 1)
            panel1 = np.stack([g, g, g, np.ones_like(g)], axis=-1)

        # Panel 2: T1c
        if has_pred_t1c:
            roi_dis = get_slice(disagree_t1c_vol, axis, idx)[y0:y1, x0:x1]
            panel2 = build_ec_roi(roi_t1c, roi_dis, roi_struct, roi_mask, alpha_overlay)
        else:
            g = np.clip(roi_t1c, 0, 1)
            panel2 = np.stack([g, g, g, np.ones_like(g)], axis=-1)

        gap = 2
        total_w = rw * 2 + gap
        composite = np.zeros((rh, total_w, 4))
        composite[:, :rw, :] = panel1
        composite[:, rw:rw+gap, :3] = [1.0, 0.8, 0.0]
        composite[:, rw:rw+gap, 3] = 1.0
        composite[:, rw+gap:, :] = panel2

        im_ecz.set_data(np.clip(composite, 0, 1))
        im_ecz.set_extent([0, total_w, 0, rh])
        ax_ecz.set_xlim(0, total_w)
        ax_ecz.set_ylim(0, rh)
        ax_ecz.set_title(f"T1w err | T1c err   [{2*r}px]",
                         color="#cc88ff", fontsize=10)
        fig_ecz.canvas.draw_idle()

        rect_x, rect_y = x0, h - y1
        rect_w, rect_h = x1 - x0, y1 - y0
        for rect in ec_zoom_rects:
            rect.set_xy((rect_x, rect_y))
            rect.set_width(rect_w); rect.set_height(rect_h)
            rect.set_visible(True)

    # ===================================================================
    # Update function
    # ===================================================================

    def update(val=None):
        axis = current_axis[0]
        idx = int(slider.val)

        sl_t1w = get_slice(t1w, axis, idx)
        sl_t1c = get_slice(t1c, axis, idx)
        sl_mask = get_slice(brain_mask, axis, idx)

        # Structural edge-based difference (not intensity-based)
        sl_diff = compute_structural_diff_slice(sl_t1w, sl_t1c, sl_mask,
                                                 zscore_threshold=thresh_slider.val)

        h, w = sl_t1w.shape
        extent = [0, w, 0, h]

        # --- Window 1 ---
        sl_overlay = make_overlay_rgba(sl_t1w, sl_diff)

        im1.set_data(sl_t1w); im1.set_extent(extent)
        im2.set_data(sl_t1c); im2.set_extent(extent)
        im3.set_data(sl_overlay); im3.set_extent(extent)

        cossim = cosine_similarity_2d(sl_t1w, sl_t1c, sl_mask)
        ssim_val = compute_ssim_2d(sl_t1w, sl_t1c)
        mean_diff = sl_diff[sl_mask > 0].mean() if sl_mask.any() else 0.0
        metrics_text.set_text(
            f"CosSim:{cossim:.4f}\n"
            f"SSIM:  {ssim_val:.4f}\n"
            f"MeanΔ: {mean_diff:.4f}"
        )

        # Intensity zoom
        update_intensity_zoom(sl_t1w, sl_t1c, sl_diff, h, w)

        fig1.canvas.draw_idle()

        # --- Window 2 (disagreement) ---
        if has_label_window:
            # Panel 1: T1w GT vs Prediction (or GT boundaries if no pred)
            if has_pred_t1w:
                sl_dis_t1w = get_slice(disagree_t1w_vol, axis, idx)
                lim1.set_data(make_disagreement_overlay(sl_t1w, sl_dis_t1w, sl_mask,
                              color=(1.0, 0.6, 0.0), alpha=0.65))
            elif has_labels:
                sl_bnd_t1w = get_slice(bnd_t1w, axis, idx)
                lim1.set_data(make_scan_with_boundaries(sl_t1w, sl_bnd_t1w,
                              color=(0.0, 1.0, 0.4), alpha=0.8))
            else:
                lim1.set_data(np.stack([sl_t1w]*3 + [np.ones_like(sl_t1w)], axis=-1))
            lim1.set_extent(extent)

            # Panel 2: T1c GT vs Prediction (or GT boundaries if no pred)
            if has_pred_t1c:
                sl_dis_t1c = get_slice(disagree_t1c_vol, axis, idx)
                lim2.set_data(make_disagreement_overlay(sl_t1c, sl_dis_t1c, sl_mask,
                              color=(1.0, 0.6, 0.0), alpha=0.65))
            elif has_labels:
                sl_bnd_t1c = get_slice(bnd_t1c, axis, idx)
                lim2.set_data(make_scan_with_boundaries(sl_t1c, sl_bnd_t1c,
                              color=(0.0, 1.0, 0.4), alpha=0.8))
            else:
                lim2.set_data(np.stack([sl_t1c]*3 + [np.ones_like(sl_t1c)], axis=-1))
            lim2.set_extent(extent)

            # Panel 3: Pred T1w vs Pred T1c (cross-contrast prediction agreement)
            if has_pred_cross:
                sl_disagree_cross = get_slice(disagree_pred_cross_vol, axis, idx)
                lim3.set_data(make_disagreement_overlay(sl_t1w, sl_disagree_cross, sl_mask,
                              color=(1.0, 0.15, 0.15), alpha=0.65))
            else:
                lim3.set_data(np.stack([sl_t1w]*3 + [np.ones_like(sl_t1w)], axis=-1))
            lim3.set_extent(extent)

            # Metrics string
            parts = []
            brain_voxels = sl_mask.sum()
            if brain_voxels > 0:
                if has_pred_t1w:
                    d = (sl_dis_t1w[sl_mask > 0] > 0).sum()
                    parts.append(f"T1w GT vs Pred: {100*d/brain_voxels:.1f}%")
                if has_pred_t1c:
                    d = (sl_dis_t1c[sl_mask > 0] > 0).sum()
                    parts.append(f"T1c GT vs Pred: {100*d/brain_voxels:.1f}%")
                if has_pred_cross:
                    d = (sl_disagree_cross[sl_mask > 0] > 0).sum()
                    parts.append(f"Pred T1w vs T1c: {100*d/brain_voxels:.1f}%")
            label_metrics.set_text("    ".join(parts))

            # Label zoom
            update_label_zoom(sl_t1w, sl_t1c, sl_mask,
                              axis, idx, h, w)

            fig2.canvas.draw_idle()

        # --- Window 5 (error correlation: T1w + T1c) ---
        if has_error_corr:
            struct_thresh = 0.15
            struct_mask = (sl_diff > struct_thresh) & (sl_mask > 0)
            alpha_overlay = 0.6

            def build_error_corr_rgba(gray_sl, pred_err_sl, struct_msk, brain_msk, alpha_ov):
                """Build dual-colour overlay: red=pred err, blue=struct Δ, purple=both."""
                g = np.clip(gray_sl, 0, 1)
                out = np.stack([g, g, g, np.ones_like(g)], axis=-1)
                pred_mask = (pred_err_sl > 0) & (brain_msk > 0)
                red_only = pred_mask & ~struct_msk
                blue_only = struct_msk & ~pred_mask
                both_mask = pred_mask & struct_msk

                out[red_only, 0] = out[red_only, 0]*(1-alpha_ov) + 1.0*alpha_ov
                out[red_only, 1] = out[red_only, 1]*(1-alpha_ov) + 0.15*alpha_ov
                out[red_only, 2] = out[red_only, 2]*(1-alpha_ov) + 0.15*alpha_ov

                out[blue_only, 0] = out[blue_only, 0]*(1-alpha_ov) + 0.15*alpha_ov
                out[blue_only, 1] = out[blue_only, 1]*(1-alpha_ov) + 0.3*alpha_ov
                out[blue_only, 2] = out[blue_only, 2]*(1-alpha_ov) + 1.0*alpha_ov

                out[both_mask, 0] = out[both_mask, 0]*(1-alpha_ov) + 0.8*alpha_ov
                out[both_mask, 1] = out[both_mask, 1]*(1-alpha_ov) + 0.1*alpha_ov
                out[both_mask, 2] = out[both_mask, 2]*(1-alpha_ov) + 0.9*alpha_ov

                return np.clip(out, 0, 1), pred_mask, red_only, blue_only, both_mask

            # Panel 1: T1w error correlation
            parts = []
            if has_pred_t1w:
                sl_dis_t1w_ec = get_slice(disagree_t1w_vol, axis, idx)
                ec1_rgba, _, r1, b1, p1 = build_error_corr_rgba(
                    sl_t1w, sl_dis_t1w_ec, struct_mask, sl_mask, alpha_overlay)
                im_ec1.set_data(ec1_rgba)
                bv = sl_mask.sum()
                if bv > 0:
                    n_err = (sl_dis_t1w_ec[sl_mask > 0] > 0).sum()
                    ov = 100.0 * p1.sum() / n_err if n_err > 0 else 0
                    parts.append(f"T1w: {ov:.1f}% overlap")
            else:
                im_ec1.set_data(np.stack([sl_t1w]*3 + [np.ones_like(sl_t1w)], axis=-1))
            im_ec1.set_extent(extent)

            # Panel 2: T1c error correlation
            if has_pred_t1c:
                sl_dis_t1c_ec = get_slice(disagree_t1c_vol, axis, idx)
                ec2_rgba, _, r2, b2, p2 = build_error_corr_rgba(
                    sl_t1c, sl_dis_t1c_ec, struct_mask, sl_mask, alpha_overlay)
                im_ec2.set_data(ec2_rgba)
                bv = sl_mask.sum()
                if bv > 0:
                    n_err = (sl_dis_t1c_ec[sl_mask > 0] > 0).sum()
                    ov = 100.0 * p2.sum() / n_err if n_err > 0 else 0
                    parts.append(f"T1c: {ov:.1f}% overlap")
            else:
                im_ec2.set_data(np.stack([sl_t1c]*3 + [np.ones_like(sl_t1c)], axis=-1))
            im_ec2.set_extent(extent)

            ec_metrics.set_text("    ".join(parts))

            # Error correlation zoom
            update_ec_zoom(sl_t1w, sl_t1c, sl_diff, sl_mask, axis, idx, h, w,
                           struct_mask, alpha_overlay)

            fig_ec.canvas.draw_idle()

    # ===================================================================
    # Event handlers with slider/radio sync between windows
    # ===================================================================
    _syncing = [False]  # guard against infinite recursion

    def on_click_intensity(event):
        """Click on Window 1 panels — update all zoom windows."""
        if event.inaxes not in [ax1, ax2, ax3]:
            return
        if event.xdata is None or event.ydata is None:
            return
        sl = get_slice(t1w, current_axis[0], int(slider.val))
        h, w = sl.shape
        click_pos = (int(h - event.ydata), int(event.xdata))
        zoom_center[0] = click_pos
        label_zoom_center[0] = click_pos
        ec_zoom_center[0] = click_pos
        update(None)

    def on_click_labels(event):
        """Click on Window 2 panels — update all zoom windows."""
        if not has_label_window:
            return
        if event.inaxes not in [lax1, lax2, lax3]:
            return
        if event.xdata is None or event.ydata is None:
            return
        sl = get_slice(t1w, current_axis[0], int(slider.val))
        h, w = sl.shape
        click_pos = (int(h - event.ydata), int(event.xdata))
        zoom_center[0] = click_pos
        label_zoom_center[0] = click_pos
        ec_zoom_center[0] = click_pos
        update(None)

    def on_click_ec(event):
        """Click on Error Correlation panels — update ec zoom."""
        if not has_error_corr:
            return
        if event.inaxes not in [ax_ec1, ax_ec2]:
            return
        if event.xdata is None or event.ydata is None:
            return
        sl = get_slice(t1w, current_axis[0], int(slider.val))
        h, w = sl.shape
        click_pos = (int(h - event.ydata), int(event.xdata))
        ec_zoom_center[0] = click_pos
        # Also sync other zoom windows
        zoom_center[0] = click_pos
        label_zoom_center[0] = click_pos
        update(None)

    def on_slider1_changed(val):
        """Main slider changed — sync all other sliders and update."""
        if _syncing[0]:
            return
        _syncing[0] = True
        if has_label_window:
            slider2.set_val(int(val))
        if has_error_corr:
            ec_slider.set_val(int(val))
        _syncing[0] = False
        update()

    def on_slider2_changed(val):
        """Disagreement slider changed — sync others and update."""
        if _syncing[0]:
            return
        _syncing[0] = True
        slider.set_val(int(val))
        if has_error_corr:
            ec_slider.set_val(int(val))
        _syncing[0] = False
        update()

    def on_ec_slider_changed(val):
        """Error correlation slider changed — sync others and update."""
        if _syncing[0]:
            return
        _syncing[0] = True
        slider.set_val(int(val))
        if has_label_window:
            slider2.set_val(int(val))
        _syncing[0] = False
        update()

    def on_axis_change(label):
        """Main radio changed — sync all radios and update."""
        if _syncing[0]:
            return
        _syncing[0] = True
        current_axis[0] = label
        new_max = axis_max(t1w, label)
        ax_idx = ["axial", "coronal", "sagittal"].index(label)
        slider.valmax = new_max
        slider.ax.set_xlim(0, new_max)
        slider.set_val(min(int(slider.val), new_max))
        if has_label_window:
            radio2.set_active(ax_idx)
            slider2.valmax = new_max
            slider2.ax.set_xlim(0, new_max)
            slider2.set_val(int(slider.val))
        if has_error_corr:
            ec_radio.set_active(ax_idx)
            ec_slider.valmax = new_max
            ec_slider.ax.set_xlim(0, new_max)
            ec_slider.set_val(int(slider.val))
        zoom_center[0] = None
        label_zoom_center[0] = None
        ec_zoom_center[0] = None
        _syncing[0] = False
        update()

    def on_axis_change2(label):
        """Disagreement radio changed — sync others and update."""
        if _syncing[0]:
            return
        _syncing[0] = True
        current_axis[0] = label
        new_max = axis_max(t1w, label)
        ax_idx = ["axial", "coronal", "sagittal"].index(label)
        slider.valmax = new_max
        slider.ax.set_xlim(0, new_max)
        slider.set_val(min(int(slider.val), new_max))
        radio.set_active(ax_idx)
        slider2.valmax = new_max
        slider2.ax.set_xlim(0, new_max)
        slider2.set_val(int(slider.val))
        if has_error_corr:
            ec_radio.set_active(ax_idx)
            ec_slider.valmax = new_max
            ec_slider.ax.set_xlim(0, new_max)
            ec_slider.set_val(int(slider.val))
        zoom_center[0] = None
        label_zoom_center[0] = None
        ec_zoom_center[0] = None
        _syncing[0] = False
        update()

    def on_ec_axis_change(label):
        """Error correlation radio changed — sync others and update."""
        if _syncing[0]:
            return
        _syncing[0] = True
        current_axis[0] = label
        new_max = axis_max(t1w, label)
        ax_idx = ["axial", "coronal", "sagittal"].index(label)
        slider.valmax = new_max
        slider.ax.set_xlim(0, new_max)
        slider.set_val(min(int(slider.val), new_max))
        radio.set_active(ax_idx)
        if has_label_window:
            radio2.set_active(ax_idx)
            slider2.valmax = new_max
            slider2.ax.set_xlim(0, new_max)
            slider2.set_val(int(slider.val))
        ec_slider.valmax = new_max
        ec_slider.ax.set_xlim(0, new_max)
        ec_slider.set_val(int(slider.val))
        zoom_center[0] = None
        label_zoom_center[0] = None
        ec_zoom_center[0] = None
        _syncing[0] = False
        update()

    def on_scroll(event):
        if event.button == 'up':
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.button == 'down':
            slider.set_val(max(slider.val - 1, slider.valmin))

    def on_key(event):
        if event.key in ('up', 'right'):
            slider.set_val(min(slider.val + 1, slider.valmax))
        elif event.key in ('down', 'left'):
            slider.set_val(max(slider.val - 1, slider.valmin))
        elif event.key == 'escape':
            zoom_center[0] = None
            label_zoom_center[0] = None
            ec_zoom_center[0] = None
            for rect in zoom_rects:
                rect.set_visible(False)
            for rect in label_zoom_rects:
                rect.set_visible(False)
            if has_error_corr:
                for rect in ec_zoom_rects:
                    rect.set_visible(False)
            fig1.canvas.draw_idle()
            if fig2: fig2.canvas.draw_idle()
            if fig_ec: fig_ec.canvas.draw_idle()
        elif event.key == '+' or event.key == '=':
            zoom_radius[0] = max(10, zoom_radius[0] - 5)
            if zoom_center[0] is not None or label_zoom_center[0] is not None:
                update()
        elif event.key == '-':
            zoom_radius[0] = min(100, zoom_radius[0] + 5)
            if zoom_center[0] is not None or label_zoom_center[0] is not None:
                update()

    def on_zoom_in(event):
        zoom_radius[0] = max(10, zoom_radius[0] - 5)
        if zoom_center[0] is not None or label_zoom_center[0] is not None:
            update()

    def on_zoom_out(event):
        zoom_radius[0] = min(100, zoom_radius[0] + 5)
        if zoom_center[0] is not None or label_zoom_center[0] is not None:
            update()

    # Connect events — Window 1
    slider.on_changed(on_slider1_changed)
    thresh_slider.on_changed(lambda val: update())  # threshold change triggers redraw
    radio.on_clicked(on_axis_change)
    fig1.canvas.mpl_connect('scroll_event', on_scroll)
    fig1.canvas.mpl_connect('key_press_event', on_key)
    fig1.canvas.mpl_connect('button_press_event', on_click_intensity)
    fig1.canvas.mpl_connect('motion_notify_event', on_motion_main)

    # Intensity zoom window
    fig_z.canvas.mpl_connect('scroll_event', on_scroll)
    fig_z.canvas.mpl_connect('key_press_event', on_key)
    fig_z.canvas.mpl_connect('motion_notify_event', on_motion_zoom)
    btn_zin.on_clicked(on_zoom_in)
    btn_zout.on_clicked(on_zoom_out)

    # Disagreement window + zoom
    if fig2 is not None:
        slider2.on_changed(on_slider2_changed)
        radio2.on_clicked(on_axis_change2)
        fig2.canvas.mpl_connect('scroll_event', on_scroll)
        fig2.canvas.mpl_connect('key_press_event', on_key)
        fig2.canvas.mpl_connect('button_press_event', on_click_labels)
        fig2.canvas.mpl_connect('motion_notify_event', on_motion_main)
    if fig_lz is not None:
        fig_lz.canvas.mpl_connect('scroll_event', on_scroll)
        fig_lz.canvas.mpl_connect('key_press_event', on_key)
        fig_lz.canvas.mpl_connect('motion_notify_event', on_motion_zoom)
        btn_lzin.on_clicked(on_zoom_in)
        btn_lzout.on_clicked(on_zoom_out)

    # Error correlation window + zoom
    if fig_ec is not None:
        ec_slider.on_changed(on_ec_slider_changed)
        ec_radio.on_clicked(on_ec_axis_change)
        fig_ec.canvas.mpl_connect('scroll_event', on_scroll)
        fig_ec.canvas.mpl_connect('key_press_event', on_key)
        fig_ec.canvas.mpl_connect('button_press_event', on_click_ec)
        fig_ec.canvas.mpl_connect('motion_notify_event', on_motion_main)
    if fig_ecz is not None:
        fig_ecz.canvas.mpl_connect('scroll_event', on_scroll)
        fig_ecz.canvas.mpl_connect('key_press_event', on_key)
        btn_eczin.on_clicked(on_zoom_in)
        btn_eczout.on_clicked(on_zoom_out)

    update()
    plt.show()


# ---------------------------------------------------------------------------
# GUI file picker
# ---------------------------------------------------------------------------

def launch_file_picker():
    """
    Tkinter GUI with upload buttons for each file.
    T1w and T1c are required; GT labels and predictions are optional.
    Returns dict of file paths.
    """
    import tkinter as tk
    from tkinter import filedialog

    filetypes = [("NIfTI files", "*.nii *.nii.gz"), ("All files", "*.*")]
    paths = {"t1w": None, "t1c": None,
             "labels_t1w": None, "labels_t1c": None,
             "pred_t1w": None, "pred_t1c": None}

    root = tk.Tk()
    root.title("Registration Quality Viewer — Load Files")
    root.configure(bg="#1a1a1a")
    root.geometry("550x520")
    root.resizable(False, False)

    # Title
    tk.Label(root, text="Registration Quality Viewer",
             font=("Helvetica", 16, "bold"), fg="white", bg="#1a1a1a"
             ).pack(pady=(15, 5))
    tk.Label(root, text="Select your NIfTI files below",
             font=("Helvetica", 10), fg="#888888", bg="#1a1a1a"
             ).pack(pady=(0, 10))

    frame = tk.Frame(root, bg="#1a1a1a")
    frame.pack(padx=20, fill="x")

    path_labels = {}

    def make_row(parent, key, display_name, required=True):
        row = tk.Frame(parent, bg="#1a1a1a")
        row.pack(fill="x", pady=3)

        suffix = " *" if required else "  (optional)"
        tk.Label(row, text=display_name + suffix, font=("Helvetica", 10),
                 fg="white", bg="#1a1a1a", width=26, anchor="w"
                 ).pack(side="left")

        path_lbl = tk.Label(row, text="No file selected", font=("Helvetica", 9),
                            fg="#666666", bg="#222222", anchor="w", width=24,
                            relief="sunken", padx=5)
        path_lbl.pack(side="left", padx=(5, 5))
        path_labels[key] = path_lbl

        def pick():
            fp = filedialog.askopenfilename(title=f"Select {display_name}",
                                             filetypes=filetypes)
            if fp:
                paths[key] = fp
                import os
                fname = os.path.basename(fp)
                path_lbl.config(text=fname, fg="#00ff88")
                update_launch_btn()

        tk.Button(row, text="Browse", command=pick, width=8,
                  bg="#333333", fg="white", activebackground="#555555",
                  activeforeground="white", relief="flat", cursor="hand2"
                  ).pack(side="right")

    # Scans (required)
    tk.Label(frame, text="Scans", font=("Helvetica", 10, "bold"),
             fg="#4a9eff", bg="#1a1a1a").pack(anchor="w", pady=(5, 2))
    make_row(frame, "t1w", "T1w scan", required=True)
    make_row(frame, "t1c", "T1c scan", required=True)

    # Ground truth labels
    tk.Frame(frame, height=1, bg="#333333").pack(fill="x", pady=8)
    tk.Label(frame, text="Ground truth labels",
             font=("Helvetica", 10, "bold"), fg="#00ff66", bg="#1a1a1a"
             ).pack(anchor="w", pady=(0, 2))
    tk.Label(frame, text="Enables GT vs GT disagreement view",
             font=("Helvetica", 9, "italic"), fg="#888888", bg="#1a1a1a"
             ).pack(anchor="w")
    make_row(frame, "labels_t1w", "GT labels (T1w space)", required=False)
    make_row(frame, "labels_t1c", "GT labels (T1c space)", required=False)

    # Predictions
    tk.Frame(frame, height=1, bg="#333333").pack(fill="x", pady=8)
    tk.Label(frame, text="Model predictions",
             font=("Helvetica", 10, "bold"), fg="#ffcc00", bg="#1a1a1a"
             ).pack(anchor="w", pady=(0, 2))
    tk.Label(frame, text="Enables GT vs Prediction disagreement per contrast",
             font=("Helvetica", 9, "italic"), fg="#888888", bg="#1a1a1a"
             ).pack(anchor="w")
    make_row(frame, "pred_t1w", "Prediction (T1w)", required=False)
    make_row(frame, "pred_t1c", "Prediction (T1c)", required=False)

    # Status / launch
    status_var = tk.StringVar(value="Select T1w and T1c to continue")
    status_lbl = tk.Label(root, textvariable=status_var, font=("Helvetica", 9),
                          fg="#ff6666", bg="#1a1a1a")
    status_lbl.pack(pady=(10, 5))

    launched = [False]

    def update_launch_btn():
        if paths["t1w"] and paths["t1c"]:
            launch_btn.config(state="normal", bg="#2a6e2a")
            status_var.set("Ready to launch")
            status_lbl.config(fg="#00ff88")
        else:
            launch_btn.config(state="disabled", bg="#333333")
            missing = []
            if not paths["t1w"]: missing.append("T1w")
            if not paths["t1c"]: missing.append("T1c")
            status_var.set(f"Still need: {', '.join(missing)}")
            status_lbl.config(fg="#ff6666")

    def on_launch():
        launched[0] = True
        root.destroy()

    launch_btn = tk.Button(root, text="Launch Viewer", command=on_launch,
                           font=("Helvetica", 12, "bold"), width=20, height=1,
                           bg="#333333", fg="white", activebackground="#3a8e3a",
                           activeforeground="white", relief="flat",
                           state="disabled", cursor="hand2")
    launch_btn.pack(pady=(5, 15))

    update_launch_btn()
    root.mainloop()

    if not launched[0] or not paths["t1w"] or not paths["t1c"]:
        print("No files selected. Exiting.")
        sys.exit(0)

    return paths


# ---------------------------------------------------------------------------
# Loading + launch
# ---------------------------------------------------------------------------

def load_and_launch(paths):
    """Load files from paths dict and launch the viewer."""

    print(f"Loading T1w: {paths['t1w']}")
    t1w_raw = load_nifti(paths["t1w"])
    print(f"Loading T1c: {paths['t1c']}")
    t1c_raw = load_nifti(paths["t1c"])

    if t1w_raw.shape != t1c_raw.shape:
        print(f"ERROR: Shape mismatch — T1w {t1w_raw.shape} vs T1c {t1c_raw.shape}")
        sys.exit(1)

    print(f"Volume shape: {t1w_raw.shape}")

    t1w = normalise_01(t1w_raw)
    t1c = normalise_01(t1c_raw)

    mask_t1w = create_brain_mask(t1w, threshold_pct=5)
    mask_t1c = create_brain_mask(t1c, threshold_pct=5)
    brain_mask = ((mask_t1w + mask_t1c) > 0).astype(np.uint8)

    print("Computing structural edge-based difference...")
    n_slices = t1w.shape[2]
    start_z = n_slices // 4
    end_z = 3 * n_slices // 4
    edge_diffs = []
    for z in range(start_z, end_z):
        sl_t1w = t1w[:, :, z]
        sl_t1c = t1c[:, :, z]
        sl_mask = brain_mask[:, :, z]
        if sl_mask.sum() < 100:
            continue
        ed = compute_structural_diff_slice(sl_t1w, sl_t1c, sl_mask)
        edge_diffs.append(ed[sl_mask > 0].mean())
    edge_diffs = np.array(edge_diffs)
    print(f"\nStructural edge difference (brain-masked, middle axial slices):")
    print(f"  Mean edge Δ  = {edge_diffs.mean():.4f}")
    print(f"  Median       = {np.median(edge_diffs):.4f}")
    print(f"  Max slice Δ  = {edge_diffs.max():.4f}")

    cossim_vol = cosine_similarity_2d(t1w[brain_mask > 0], t1c[brain_mask > 0])
    print(f"\nVolume-level CosSim: {cossim_vol:.4f}")

    # GT labels
    labels_t1w_vol, labels_t1c_vol = None, None
    bnd_t1w, bnd_t1c = None, None
    if paths.get("labels_t1w") and paths.get("labels_t1c"):
        print("\nLoading GT label maps...")
        labels_t1w_vol = load_nifti(paths["labels_t1w"]).astype(np.int32)
        labels_t1c_vol = load_nifti(paths["labels_t1c"]).astype(np.int32)
        bnd_t1w = extract_label_boundaries(labels_t1w_vol)
        bnd_t1c = extract_label_boundaries(labels_t1c_vol)

        bg = (labels_t1w_vol == 0) & (labels_t1c_vol == 0)
        brain_labels = brain_mask.astype(bool) & ~bg
        total_brain = brain_labels.sum()
        disagree_total = ((labels_t1w_vol != labels_t1c_vol) & brain_labels).sum()
        print(f"\nGT T1w vs GT T1c disagreement:")
        print(f"  {disagree_total:,} / {total_brain:,} brain voxels "
              f"({100*disagree_total/total_brain:.2f}%)")
    else:
        print("\nNo GT label maps provided.")

    # Predictions
    pred_t1w_vol, pred_t1c_vol = None, None
    if paths.get("pred_t1w"):
        print(f"Loading T1w prediction: {paths['pred_t1w']}")
        pred_t1w_vol = load_nifti(paths["pred_t1w"]).astype(np.int32)
    if paths.get("pred_t1c"):
        print(f"Loading T1c prediction: {paths['pred_t1c']}")
        pred_t1c_vol = load_nifti(paths["pred_t1c"]).astype(np.int32)

    # Print GT vs prediction stats if both available
    if labels_t1w_vol is not None and pred_t1w_vol is not None:
        bg = (labels_t1w_vol == 0) & (pred_t1w_vol == 0)
        valid = brain_mask.astype(bool) & ~bg
        dis = ((labels_t1w_vol != pred_t1w_vol) & valid).sum()
        tot = valid.sum()
        print(f"\nT1w GT vs Prediction: {dis:,}/{tot:,} ({100*dis/tot:.2f}%)")

    if labels_t1c_vol is not None and pred_t1c_vol is not None:
        bg = (labels_t1c_vol == 0) & (pred_t1c_vol == 0)
        valid = brain_mask.astype(bool) & ~bg
        dis = ((labels_t1c_vol != pred_t1c_vol) & valid).sum()
        tot = valid.sum()
        print(f"T1c GT vs Prediction: {dis:,}/{tot:,} ({100*dis/tot:.2f}%)")

    if pred_t1w_vol is not None and pred_t1c_vol is not None:
        bg = (pred_t1w_vol == 0) & (pred_t1c_vol == 0)
        valid = brain_mask.astype(bool) & ~bg
        dis = ((pred_t1w_vol != pred_t1c_vol) & valid).sum()
        tot = valid.sum()
        print(f"Pred T1w vs Pred T1c: {dis:,}/{tot:,} ({100*dis/tot:.2f}%)")

    initial_axis = "axial"
    init_slice = axis_max(t1w, initial_axis) // 2

    print(f"\nControls:")
    print(f"  Scroll / arrows : change slice (works in all windows)")
    print(f"  Click on panels : zoom into region")
    print(f"  +/- buttons     : adjust zoom level")
    print(f"  Escape          : reset zoom")
    print(f"\nLaunching viewer...")

    run_viewer(t1w, t1c, brain_mask,
               labels_t1w_vol, labels_t1c_vol, bnd_t1w, bnd_t1c,
               pred_t1w_vol, pred_t1c_vol,
               initial_axis, init_slice)


# ---------------------------------------------------------------------------
# Main — supports both GUI and CLI
# ---------------------------------------------------------------------------

def main():
    # If command-line args provided, use CLI mode
    # Otherwise, launch the GUI file picker
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="T1w vs T1c registration + label quality viewer",
        )
        parser.add_argument("t1w", help="Path to T1w NIfTI")
        parser.add_argument("t1c", help="Path to T1c NIfTI")
        parser.add_argument("--labels_t1w", default=None)
        parser.add_argument("--labels_t1c", default=None)
        parser.add_argument("--pred_t1w", default=None,
                            help="T1w prediction label map")
        parser.add_argument("--pred_t1c", default=None,
                            help="T1c prediction label map")
        parser.add_argument("--axis", default="axial",
                            choices=["axial", "coronal", "sagittal"])
        parser.add_argument("--slice", type=int, default=None)
        args = parser.parse_args()

        paths = {
            "t1w": args.t1w,
            "t1c": args.t1c,
            "labels_t1w": args.labels_t1w,
            "labels_t1c": args.labels_t1c,
            "pred_t1w": args.pred_t1w,
            "pred_t1c": args.pred_t1c,
        }
        load_and_launch(paths)
    else:
        # GUI mode
        paths = launch_file_picker()
        load_and_launch(paths)


if __name__ == "__main__":
    main()
