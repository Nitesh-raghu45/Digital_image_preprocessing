import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------- Shannon Fano Functions ----------
def shannon_fano(symbols, codes, prefix=""):
    if len(symbols) == 1:
        codes[symbols[0][0]] = prefix
        return
    total = sum([s[1] for s in symbols])
    acc, split = 0, 0
    for i, s in enumerate(symbols):
        acc += s[1]
        if acc >= total / 2:
            split = i + 1
            break
    left, right = symbols[:split], symbols[split:]
    shannon_fano(left, codes, prefix + "0")
    shannon_fano(right, codes, prefix + "1")

def build_codes(pixels):
    unique, counts = np.unique(pixels, return_counts=True)
    symbols = sorted(list(zip(unique, counts)), key=lambda x: x[1], reverse=True)
    codes = {}
    shannon_fano(symbols, codes)
    return codes

def compress(img):
    pixels = img.flatten()
    codes = build_codes(pixels)
    encoded = "".join([codes[p] for p in pixels])
    return codes, encoded

def decompress(encoded, codes, shape):
    reverse_codes = {v: k for k, v in codes.items()}
    decoded_pixels, buffer = [], ""
    for bit in encoded:
        buffer += bit
        if buffer in reverse_codes:
            decoded_pixels.append(reverse_codes[buffer])
            buffer = ""
    return np.array(decoded_pixels, dtype=np.uint8).reshape(shape)

# ---------- Main Program ----------
# Input image (use full path if needed)
image_path = r"C:\Users\NITESH\OneDrive\Desktop\DIP\lab3\lena.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Compress + Decompress
codes, encoded = compress(img)
reconstructed = decompress(encoded, codes, img.shape)

# Save output image
output_path = "reconstructed.png"
cv2.imwrite(output_path, reconstructed)

# File sizes
input_size = os.path.getsize(image_path) / 1024   # KB
output_size = os.path.getsize(output_path) / 1024 # KB

# Dimensions
h, w = img.shape
dims = f"{w}×{h}"

# ---------- Overlay size + dims directly on images ----------
def add_info(image, size_kb, dims_text, label):
    img_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    text1 = f"{label}"
    text2 = f"Size: {size_kb:.2f} KB"
    text3 = f"Dims: {dims_text}"
    cv2.putText(img_copy, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img_copy, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img_copy, text3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return img_copy

img_info = add_info(img, input_size, dims, "Input Image")
reconstructed_info = add_info(reconstructed, output_size, dims, "Reconstructed Image")

# ---------- Show both ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(cv2.cvtColor(img_info, cv2.COLOR_BGR2RGB))
axs[0].axis("off")

axs[1].imshow(cv2.cvtColor(reconstructed_info, cv2.COLOR_BGR2RGB))
axs[1].axis("off")

plt.tight_layout()

# Save the figure to file
plt.savefig("comparison.png", dpi=300, bbox_inches="tight")

# Show on screen
plt.show()

