import cv2
import numpy as np
import matplotlib.pyplot as plt

def bit_slicing(image_path):
    # 1. Read image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # 2. Convert RGB to grayscale if needed
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Get dimensions
    rows, cols = img.shape

    # 4. Initialize 8 bit-planes
    bit_planes = np.zeros((rows, cols, 8), dtype=np.uint8)

    # 5. Extract each bit-plane
    for k in range(8):  # k = 0 (LSB) to 7 (MSB)
        bit_planes[:, :, k] = (img >> k) & 1

    # 6. Display bit planes
    plt.figure(figsize=(10, 5))
    for k in range(8):
        plt.subplot(2, 4, k + 1)

        # Weighted grayscale version like MATLAB's commented imshow(uint8(...))
        weighted_plane = (bit_planes[:, :, 7 - k] * (2 ** (7 - k))).astype(np.uint8)

        plt.imshow(weighted_plane, cmap='gray')
        plt.title(f'Bit Plane {8 - k}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
bit_slicing("lena.png")
