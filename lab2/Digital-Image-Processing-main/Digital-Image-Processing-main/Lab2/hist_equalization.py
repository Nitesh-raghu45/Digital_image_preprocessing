import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_equalization(image_path):
    # Step 1: Read image
    I = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert to grayscale if it's a color image
    if I.ndim == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    M, N = I.shape
    num_pixels = M * N

    # Step 2: Flatten image
    I_flat = I.flatten()

    # Step 3: Initialize histogram
    histogram = np.zeros(256, dtype=int)

    # Step 4: Count frequency of each pixel value
    for pixel_val in I_flat:
        histogram[pixel_val] += 1

    # Step 5: Compute CDF
    cdf = np.cumsum(histogram)
    cdf_min = np.min(cdf[cdf > 0])

    # Step 6: Apply Histogram Equalization formula
    equalized_map = np.round((cdf - cdf_min) / (num_pixels - cdf_min) * 255).astype(np.uint8)

    # Step 7: Map old pixel values to equalized values
    I_eq_flat = equalized_map[I_flat]

    # Step 8: Reshape to original dimensions
    I_eq = I_eq_flat.reshape(M, N)

    # Step 9: Plot results
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(I, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.bar(np.arange(256), histogram)
    plt.xlim(0, 255)
    plt.title('Histogram of Original Image')

    plt.subplot(2, 2, 3)
    plt.imshow(I_eq, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    eq_hist = np.zeros(256, dtype=int)
    for pixel_val in I_eq_flat:
        eq_hist[pixel_val] += 1
    plt.bar(np.arange(256), eq_hist)
    plt.xlim(0, 255)
    plt.title('Histogram of Equalized Image')

    plt.tight_layout()
    plt.show()


# Example usage
hist_equalization("lena.jpeg")
