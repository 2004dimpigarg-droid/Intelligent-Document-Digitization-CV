# Intelligent Document Digitization and Quality Enhancement System

A Python/OpenCV desktop project for clear document photographs with all four page corners visible and a contrasting background. It detects the page outline, corrects perspective, enhances grayscale contrast, and optionally creates a black-and-white scan.

## Requirements
- Python 3.10 or newer
- Windows, macOS, or Linux
- Internet for installing packages

## Installation (Windows)
Open Command Prompt in this folder and run:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

If `py` is not recognized, use `python` instead. On macOS/Linux, create and activate the environment with `python3 -m venv .venv` and `source .venv/bin/activate`, then run `python -m pip install -r requirements.txt` and `python main.py`.

Tkinter is included in most standard Windows/macOS Python installations. On some Linux distributions it must be installed separately (for Ubuntu/Debian: `sudo apt install python3-tk`).

## How to use
1. Click **Upload Image** and choose a JPG, JPEG, PNG, BMP, or TIFF file.
2. Start with a page that has all four corners visible and clearly contrasts with the background.
3. Click **Detect + Correct + Enhance**.
4. Compare the original, perspective-corrected, and enhanced grayscale previews.
5. Click **Save Enhanced Result**. Optionally save the black-and-white scan.

## Computer Vision pipeline
1. Resize large images for faster detection.
2. Convert to grayscale and apply Gaussian blur.
3. Use Canny edge detection.
4. Find contours and approximate them as polygons.
5. Select a large convex four-corner contour.
6. Order corners and calculate a perspective transform.
7. Use `warpPerspective` to rectify the page.
8. Apply CLAHE for local contrast enhancement and mild denoising.
9. Apply adaptive thresholding for optional black-and-white output.

## Suggested test set
Use 5–10 photographs: two nearly straight pages, three moderately tilted pages, two or three with mild lighting variation, and one or two with different printed layouts. Keep all corners visible. Record whether detection succeeded, whether the page was rectified, and whether readability improved.

## Limitations
- This first version targets clear, high-contrast page boundaries.
- Automatic detection can fail if corners are hidden, page/background contrast is poor, or strong background edges distract the detector. It reports a failure rather than claiming success.
- Enhancement cannot recover text that is missing or severely blurred.
- Black-and-white thresholding may remove faint writing; inspect the result before using it.
- OCR and PDF export are not included in this initial version.

## Suggested project contribution
Accurately describe your contribution as implementation and integration of the OpenCV pipeline, tuning edge/contour detection, corner ordering and perspective correction, contrast enhancement, GUI and saving workflow, and evaluation on a test set. Do not claim to have invented established algorithms such as Canny or CLAHE.
