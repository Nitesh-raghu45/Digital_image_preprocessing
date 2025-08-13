#!/bin/bash
# ================================================
# Lab 2 – Bit Slicing & Histogram Equalization
# ================================================

# 📌 Description:
# This script demonstrates:
#   1. Bit-Plane Slicing
#   2. Histogram Equalization
# on a grayscale image using Python (OpenCV, NumPy, Matplotlib).

# ------------------------------------------------
# Bit Slicing:
# ------------------------------------------------
# - Technique to analyze specific bits of pixel intensity.
# - Useful for image compression, enhancement, feature extraction.
# - Each pixel in grayscale: 8 bits (0-255).
# - Bit slicing isolates each bit-plane.
# - Higher bits show main structures, lower bits show fine/noisy details.

# ------------------------------------------------
# Histogram Equalization:
# ------------------------------------------------
# - Contrast enhancement technique.
# - Redistributes pixel intensities for a more uniform histogram.
# - Enhances details in low-contrast regions.

# ================================================
# 🎯 Objectives
# ================================================
# Bit Slicing:
#   - Understand and implement bit-plane slicing.
#   - Extract all 8 bit-planes manually.
#   - Display contribution of each bit to image quality.
#
# Histogram Equalization:
#   - Understand image histogram and contrast enhancement.
#   - Implement equalization manually (no built-in functions).
#   - Compare original vs equalized images and histograms.

# ================================================
# ⚙️ Logic (Manual Implementation)
# ================================================

# ---------- Bit Slicing ----------
# 1. Read input image.
# 2. Convert to grayscale if RGB.
# 3. Get image dimensions.
# 4. Initialize array for 8 bit-planes.
# 5. For each pixel:
#    - Get intensity (0–255).
#    - Extract bits using bitwise ops.
#    - Store in corresponding bit-plane.
# 6. Display bit-planes (MSB bit-8 to LSB bit-1).

# ---------- Histogram Equalization ----------
# 1. Read grayscale image.
# 2. Flatten into 1D array.
# 3. Compute histogram (pixel intensity frequency).
# 4. Compute CDF (Cumulative Distribution Function).
# 5. Apply formula:
#      Equalized_Value = ((CDF - CDF_min) / ((M*N) - CDF_min)) * 255
# 6. Map old pixel values to equalized values.
# 7. Display:
#    - Original image & histogram.
#    - Equalized image & histogram.

# ================================================
# 🖼️ Sample Output
# ================================================
# Bit Slicing:
#   - Bit-plane 8 (MSB) → major image structures, edges.
#   - Bit-plane 1 (LSB) → fine, noise-like details.
#
# Histogram Equalization:
#   - Original: narrow histogram.
#   - Equalized: spread-out histogram.

# ================================================
# 💻 Requirements
# ================================================
# Install Python libraries before running:
pip install opencv-python matplotlib numpy
