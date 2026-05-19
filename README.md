# Registration Viewer

A free, open-source viewer for comparing two co-registered 3D medical scans (NIfTI format) side-by-side, with quantitative similarity metrics (Cosine Similarity, SSIM), structural difference heatmaps, click-to-zoom inspection, and optional overlays for ground-truth labels and model predictions.

Designed for **neuroimaging registration QC** (e.g. T1w vs T1c, pre/post contrast, longitudinal scans, atlas registration), but works for any pair of co-registered 3D NIfTI volumes.

---

## Highlights

- **Side-by-side 3D viewing** of any two NIfTI scans (axial / coronal / sagittal).
- **Click anywhere → zoom into that region** in a dedicated zoom window.
- **Quantitative similarity scores** computed automatically:
  - **Cosine Similarity** across the brain-masked volume.
  - **SSIM** (Structural Similarity Index) per slice.
  - **Edge-based structural difference** map highlighting registration mismatches.
- **Synchronised crosshairs** across every panel and window.
- **Optional ground-truth label overlay** with green boundary contours and a per-slice disagreement metric.
- **Optional model-prediction overlay** to compare predicted segmentations against GT and across contrasts.
- **Error-correlation view** (red = prediction error, blue = structural difference, purple = both) for studying whether registration errors drive segmentation errors.
- **Two ways to launch**: a friendly **Tkinter file-picker GUI** (no command line needed) or a **CLI** for scripted workflows.
- **Radiological display convention** (patient's left on screen right), matching 3D Slicer.

---

## Screenshots

<img width="1920" height="1080" alt="Screenshot (661)" src="https://github.com/user-attachments/assets/b0d862d8-368b-46b5-b91c-b6ca0833dc90" />
<img width="1920" height="1080" alt="Screenshot (662)" src="https://github.com/user-attachments/assets/1b7d9015-c554-48f2-87f7-a9c520c4f5a3" />
<img width="1920" height="1080" alt="Screenshot (663)" src="https://github.com/user-attachments/assets/86ed495d-5974-47d6-bd99-03ea2ad2dd0c" />
<img width="1920" height="1080" alt="Screenshot (675)" src="https://github.com/user-attachments/assets/aed3ef8e-8d38-4444-8766-58ee4a6428fb" />
<img width="1920" height="1080" alt="Screenshot (676)" src="https://github.com/user-attachments/assets/436fc8c9-1c8f-4751-96ba-b7c35a02cb1b" />

---

## Requirements

- **Python 3.8 or newer**
- Tkinter (ships with the standard CPython installer on Windows / macOS; on Linux install via your package manager, e.g. `sudo apt install python3-tk`)
- The Python packages listed in [`requirements.txt`](requirements.txt):
  - `nibabel`
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `scikit-image`

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Registration-Viewer.git
cd Registration-Viewer

# 2. (Recommended) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. GUI mode (easiest — just upload two scans)

```bash
python registration_viewer.py
```

A small launcher window opens. Click **Browse** to pick:

| Field | Required | What it is |
| --- | --- | --- |
| T1w scan | yes | First NIfTI volume (e.g. `.nii` / `.nii.gz`) |
| T1c scan | yes | Second NIfTI volume (must be in the same shape / space as the first) |
| GT labels (T1w space) | optional | Ground-truth segmentation label map for scan 1 |
| GT labels (T1c space) | optional | Ground-truth segmentation label map for scan 2 |
| Prediction (T1w) | optional | Model prediction label map for scan 1 |
| Prediction (T1c) | optional | Model prediction label map for scan 2 |

Once T1w and T1c are selected, the **Launch Viewer** button activates.

> The fields are named **T1w** and **T1c** for convenience, but you can load *any* two co-registered 3D NIfTI volumes (e.g. pre/post contrast, baseline/follow-up, atlas/subject). The math is contrast-agnostic.

### 2. CLI mode (scripted / batch use)

```bash
# Minimum: two scans
python registration_viewer.py path/to/scan_a.nii.gz path/to/scan_b.nii.gz

# With ground-truth labels
python registration_viewer.py scan_a.nii.gz scan_b.nii.gz \
    --labels_t1w labels_a.nii.gz \
    --labels_t1c labels_b.nii.gz

# Full: labels + predictions
python registration_viewer.py scan_a.nii.gz scan_b.nii.gz \
    --labels_t1w labels_a.nii.gz --labels_t1c labels_b.nii.gz \
    --pred_t1w pred_a.nii.gz   --pred_t1c pred_b.nii.gz

# Pick starting axis / slice
python registration_viewer.py a.nii.gz b.nii.gz --axis coronal --slice 90
```

CLI options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `t1w` (positional) | required | First 3D NIfTI volume |
| `t1c` (positional) | required | Second 3D NIfTI volume |
| `--labels_t1w` | none | GT label map aligned to scan 1 |
| `--labels_t1c` | none | GT label map aligned to scan 2 |
| `--pred_t1w` | none | Model prediction aligned to scan 1 |
| `--pred_t1c` | none | Model prediction aligned to scan 2 |
| `--axis` | `axial` | Initial slicing axis: `axial`, `coronal`, or `sagittal` |
| `--slice` | middle | Initial slice index |

---

## What the viewer shows

Different windows open depending on which files you load. Loading just two scans is enough to get started.

### Window 1 — Intensity comparison *(always open)*
- Panel 1: scan A in greyscale.
- Panel 2: scan B in greyscale.
- Panel 3: scan A with a structural-difference heatmap overlay (edge mismatch + local z-score shift).
- A **Z-threshold slider** lets you tune the sensitivity of the rank-based difference component.

### Window 2 — Click-to-zoom (intensity)
Click any panel in Window 1 → that region is enlarged here (scan A | scan B). Use the `+ Zoom In` / `− Zoom Out` buttons to change ROI size.

### Window 3 — Disagreement analysis *(opens if labels and/or predictions provided)*
- Panel 1: scan A with GT-vs-Pred disagreement (orange).
- Panel 2: scan B with GT-vs-Pred disagreement (orange).
- Panel 3: cross-contrast prediction disagreement (Pred A vs Pred B, red).
- Hover any pixel to read the **FreeSurfer structure name** at that voxel (uses an embedded 106-region lookup table).

### Window 4 — Click-to-zoom (disagreement)
Same as Window 2 but for the disagreement panels.

### Window 5 — Error correlation *(if at least one prediction is provided)*
Two panels (one per scan) showing:
- 🟥 **Red** — prediction error (GT ≠ Pred)
- 🟦 **Blue** — structural difference (registration mismatch)
- 🟪 **Purple** — both overlap (the interesting case: where registration error and segmentation error co-occur)

### Window 6 — Click-to-zoom (error correlation)
Inspect overlap regions at full resolution.

---

## Similarity metrics

The viewer reports several quantitative similarity measures, all computed inside an automatically estimated brain mask:

| Metric | What it tells you |
| --- | --- |
| **Volume-level Cosine Similarity** | Global intensity-pattern agreement between the two scans (1.0 = identical direction in voxel space). Printed in the terminal at launch. |
| **Per-slice Cosine Similarity** | Same metric per displayed slice (shown in the on-screen metrics panel). |
| **Per-slice SSIM** | Structural Similarity Index (Wang et al. 2004) per slice — sensitive to luminance, contrast, and structure. |
| **Edge structural difference** | Mean of the brain-masked structural-difference map (edge mismatch + heavily-thresholded local z-score shift). Printed at launch over middle axial slices. |
| **Label disagreement %** | If GT / prediction maps are provided: percentage of brain voxels where the two label maps disagree. |

These metrics are intentionally complementary: cosine similarity is global and cheap, SSIM captures local structure, and the edge-difference map localises *where* the registration is off — not just *how much*.

---

## Controls reference

| Action | Effect |
| --- | --- |
| Scroll wheel | Next / previous slice |
| `↑` / `↓` arrow keys | Next / previous slice |
| Click on any image panel | Zoom into that region (sends it to the matching Zoom window) |
| `+ Zoom In` / `− Zoom Out` buttons | Adjust zoom ROI size |
| `Escape` | Reset zoom |
| Axis radio buttons | Switch between axial / coronal / sagittal |
| Z-thresh slider | Tune sensitivity of the rank-based structural difference |
| Hover over disagreement panels | Show GT / predicted FreeSurfer structure names |

The slice slider, axis selector, and crosshairs are **synchronised across every window** — moving in one window updates the others.

---

## Input file expectations

- Format: **NIfTI** (`.nii` or `.nii.gz`).
- Both scans must have the **same shape**. If they don't, the script exits with a clear error and prints the two shapes — you need to register / resample them first.
- Volumes are automatically reoriented to **RAS+ canonical** (`nibabel.as_closest_canonical`), so the displayed axes mean the same thing regardless of the source orientation.
- Display follows the **radiological convention** (patient's left on screen right), matching 3D Slicer's default.
- Labels and predictions should be **integer-valued** NIfTI volumes in the same space as the corresponding scan.

---

## Troubleshooting

**`Shape mismatch — T1w (X,Y,Z) vs T1c (A,B,C)`**
The two volumes are not in the same grid. Resample one to the other first (e.g. with `nibabel`, ANTs, or 3D Slicer's `Resample Scalar/Vector/DWI Volume`).

**`ModuleNotFoundError: No module named 'tkinter'`** (Linux only)
Install your distro's Tk package, e.g. `sudo apt install python3-tk` on Debian/Ubuntu.

**Windows appear blank / no GUI on a remote server**
Matplotlib needs a display. If you're on a headless server, use X11 forwarding (`ssh -X`) or run locally.

**Hover labels show `ID:1234` instead of a name**
The internal lookup table covers FreeSurfer's core + DKT parcellation (~106 structures). Unknown integer IDs are shown verbatim — they're not an error, just out-of-table values.

---

## Project structure

```
Registration-Viewer/
├── registration_viewer.py   # the viewer (single-file, no extra modules)
├── requirements.txt
├── README.md
├── LICENSE                  # MIT
├── CITATION.cff             # citation metadata (auto-detected by GitHub)
├── CONTRIBUTING.md
└── .gitignore
```

The viewer is intentionally a **single file** so it's easy to drop into any project, share, or run on a clean machine without packaging concerns.

---

## Citation

If this tool helps your work, please cite it via the metadata in [`CITATION.cff`](CITATION.cff), or use the "Cite this repository" button on the GitHub page.

```bibtex
@software{singh_registration_viewer,
  author  = {Singh, Pushpendra},
  title   = {Registration Viewer: an open-source NIfTI registration-quality viewer},
  year    = {2026},
  url     = {https://github.com/<your-username>/Registration-Viewer}
}
```

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE). You are free to use, modify, and redistribute it for academic or commercial purposes; attribution is appreciated.

---

## Author

**Pushpendra Singh** — built as part of PhD work on multi-contrast brain-MRI registration and segmentation quality control. Issues and pull requests are welcome.
