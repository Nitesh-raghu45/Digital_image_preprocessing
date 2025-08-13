import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def bit_slicing(image_filename):
    # Get absolute path of the image (in same folder as script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, image_filename)

    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to grayscale if needed
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rows, cols = img.shape
    bit_planes = np.zeros((rows, cols, 8), dtype=np.uint8)

    # Extract bit planes
    for k in range(8):
        bit_planes[:, :, k] = (img >> k) & 1

    # Create output folder inside script directory
    output_folder = os.path.join(script_dir, "output")
    os.makedirs(output_folder, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_filename))[0]

    # Display & Save bit planes
    plt.figure(figsize=(10, 5))
    for k in range(8):
        plt.subplot(2, 4, k + 1)
        weighted_plane = (bit_planes[:, :, 7 - k] * (2 ** (7 - k))).astype(np.uint8)
        plt.imshow(weighted_plane, cmap='gray')
        plt.title(f'Bit Plane {8 - k}')
        plt.axis('off')

        # Save image
        save_path = os.path.join(output_folder, f"{base_name}_bitplane_{8 - k}.png")
        cv2.imwrite(save_path, weighted_plane)

    plt.tight_layout()
    plt.show()

# Example usage
bit_slicing("lena.png")
