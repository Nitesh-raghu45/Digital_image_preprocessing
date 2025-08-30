import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct


# -----------------------------
# JPEG Compression Functions
# -----------------------------

def load_grayscale_image(img_path):
    """Loads an image and converts it to grayscale if needed."""
    image = cv2.imread(img_path)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def extract_block(image, row_range, col_range):
    """Extracts a specific 8x8 block from the image and shifts values to [-128, 127]."""
    block = image[row_range[0]:row_range[1], col_range[0]:col_range[1]].astype(float)
    return block - 128  # Shift pixel values for DCT


def apply_2d_dct(block):
    """Applies 2D Discrete Cosine Transform to an 8x8 block."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def apply_2d_idct(block):
    """Applies 2D Inverse Discrete Cosine Transform to reconstruct an image block."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def get_quantization_table():
    """Returns standard JPEG luminance quantization table."""
    return np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])


def quantize_block(dct_block, quant_table):
    """Quantizes the DCT coefficients using JPEG quantization table."""
    return np.round(dct_block / quant_table)


def dequantize_block(quantized_block, quant_table):
    """Reverses the quantization step."""
    return quantized_block * quant_table


def reconstruct_block(dequantized_block):
    """Reconstructs pixel values from DCT coefficients and shifts back to [0,255]."""
    return np.round(apply_2d_idct(dequantized_block) + 128).astype(np.uint8)


def calculate_compression(block, quantized_block):
    """Calculates compression metrics."""
    original_bits = block.size * 8
    non_zero = np.count_nonzero(quantized_block)
    compressed_bits = non_zero * 16
    if non_zero < 64:
        compressed_bits += 4
    ratio = original_bits / compressed_bits
    return original_bits, compressed_bits, ratio, non_zero


def display_results(block, quantized_block, reconstructed_block, original_image, compressed_image, sizes):
    """Displays original, quantized, and reconstructed results using matplotlib."""
    original_bits, compressed_bits, ratio, non_zero = sizes

    plt.figure(figsize=(12, 6))
    plt.suptitle('JPEG DCT Compression on an 8x8 Block')

    # Original Block
    plt.subplot(2, 3, 1)
    plt.imshow(block, cmap='gray')
    plt.title('Original Block')
    plt.xlabel(f'Size: {original_bits} bits')

    # Quantized Coefficients
    plt.subplot(2, 3, 2)
    plt.imshow(quantized_block, cmap='viridis')
    plt.colorbar()
    plt.title('Quantized DCT')
    plt.xlabel(f'{non_zero} non-zero values')

    # Reconstructed Block
    plt.subplot(2, 3, 3)
    plt.imshow(reconstructed_block, cmap='gray')
    plt.title('Reconstructed Block')

    # Original Image
    plt.subplot(2, 3, 4)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    # Compressed Image
    plt.subplot(2, 3, 6)
    plt.imshow(compressed_image, cmap='gray')
    plt.title(f'Compressed Image\nRatio: {ratio:.1f}:1')

    plt.show()

    print("\n--- Compression Report ---")
    print(f"Original Size      : {original_bits} bits ({original_bits/8:.1f} bytes)")
    print(f"Compressed Estimate: {compressed_bits} bits ({compressed_bits/8:.1f} bytes)")
    print(f"Compression Ratio  : {ratio:.1f}:1")


# -----------------------------
# Main JPEG Compression Demo
# -----------------------------
def jpeg_compression_demo(img_path):
    image = load_grayscale_image(img_path)
    block_coords = (141, 149), (61, 69)

    block = extract_block(image, *block_coords)
    dct_block = apply_2d_dct(block)
    quant_table = get_quantization_table()
    quantized_block = quantize_block(dct_block, quant_table)
    dequantized_block = dequantize_block(quantized_block, quant_table)
    reconstructed_block = reconstruct_block(dequantized_block)

    compressed_image = image.copy()
    compressed_image[block_coords[0][0]:block_coords[0][1],
                     block_coords[1][0]:block_coords[1][1]] = reconstructed_block

    sizes = calculate_compression(block, quantized_block)
    display_results(block, quantized_block, reconstructed_block, image, compressed_image, sizes)


if __name__ == "__main__":
    jpeg_compression_demo("lena.png")
