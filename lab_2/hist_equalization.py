import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def hist_equalization(image_path):
    # Step 1: Read image
    I = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if I is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to grayscale if it's a color image
    if I.ndim == 3:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    M, N = I.shape
    num_pixels = M * N

    # Step 2: Flatten image
    I_flat = I.flatten()

    # Step 3: Compute histogram
    histogram = np.bincount(I_flat, minlength=256)

    # Step 4: Compute CDF
    cdf = np.cumsum(histogram)
    cdf_min = np.min(cdf[cdf > 0])

    # Step 5: Apply Histogram Equalization formula
    equalized_map = np.round((cdf - cdf_min) / (num_pixels - cdf_min) * 255).astype(np.uint8)

    # Step 6: Map old pixel values to equalized values
    I_eq_flat = equalized_map[I_flat]
    I_eq = I_eq_flat.reshape(M, N)

    # Step 7: Compute equalized histogram
    eq_hist = np.bincount(I_eq_flat, minlength=256)

    # Step 8: Create output folder
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Step 9: Save histograms as images
    # Original histogram
    plt.figure()
    plt.bar(np.arange(256), histogram, color='gray')
    plt.title('Histogram of Original Image')
    plt.xlim(0, 255)
    plt.savefig(os.path.join(output_folder, f"{base_name}_original_histogram.png"))
    plt.close()

    # Equalized histogram
    plt.figure()
    plt.bar(np.arange(256), eq_hist, color='gray')
    plt.title('Histogram of Equalized Image')
    plt.xlim(0, 255)
    plt.savefig(os.path.join(output_folder, f"{base_name}_equalized_histogram.png"))
    plt.close()

    # Step 10: Show all results together
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(I, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.bar(np.arange(256), histogram, color='gray')
    plt.xlim(0, 255)
    plt.title('Histogram of Original Image')

    plt.subplot(2, 2, 3)
    plt.imshow(I_eq, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.bar(np.arange(256), eq_hist, color='gray')
    plt.xlim(0, 255)
    plt.title('Histogram of Equalized Image')

    plt.tight_layout()
    plt.show()

# Example usage
hist_equalization("lena.png")
