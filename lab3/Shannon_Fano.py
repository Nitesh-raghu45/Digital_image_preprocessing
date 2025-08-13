import cv2
import numpy as np
from collections import Counter
import math

# Shannon–Fano Encoding Functions


def shannon_fano(symbols):
    """Recursively assign Shannon–Fano codes"""
    if len(symbols) == 1:
        return {symbols[0][0]: ""}

    # Find split point
    total = sum(freq for _, freq in symbols)
    acc = 0
    split_idx = 0
    for i, (_, freq) in enumerate(symbols):
        acc += freq
        if acc >= total / 2:
            split_idx = i
            break

    left = symbols[:split_idx+1]
    right = symbols[split_idx+1:]

    left_codes = shannon_fano(left)
    right_codes = shannon_fano(right)

    # Add '0' to left, '1' to right
    for k in left_codes:
        left_codes[k] = '0' + left_codes[k]
    for k in right_codes:
        right_codes[k] = '1' + right_codes[k]

    return {**left_codes, **right_codes}


# Image Compression

def compress_image(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Flatten to 1D array
    pixels = img.flatten()

    # Count frequency of each pixel value
    freq = Counter(pixels)

    # Sort by frequency (descending)
    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

    # Generate Shannon–Fano codes
    codes = shannon_fano(sorted_freq)

    # Encode pixels
    encoded_data = ''.join(codes[p] for p in pixels)

    return img.shape, codes, encoded_data


# Image Decompression

def decompress_image(shape, codes, encoded_data):
    # Reverse the code dictionary for decoding
    reverse_codes = {v: k for k, v in codes.items()}

    decoded_pixels = []
    current_code = ""

    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded_pixels.append(reverse_codes[current_code])
            current_code = ""

    # Convert back to image
    img_array = np.array(decoded_pixels, dtype=np.uint8).reshape(shape)
    return img_array


# Example Run

if __name__ == "__main__":
    # Path to your grayscale image
    image_path = "lena.png"

    # Compress
    shape, codes, encoded_data = compress_image(image_path)
    print("Shannon–Fano Codes:", codes)
    print("Original bits:", shape[0] * shape[1] * 8)
    print("Compressed bits:", len(encoded_data))

    # Decompress
    reconstructed_img = decompress_image(shape, codes, encoded_data)

    # Save output
    cv2.imwrite("reconstructed.png", reconstructed_img)
    print("Decompressed image saved as reconstructed.png")
