import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


# ------------------------------
# Arithmetic Coding Core
# ------------------------------
def get_intervals(prob_table):
    """
    Generates cumulative probability intervals for each symbol.
    Returns a dictionary {symbol: (low, high)}.
    """
    intervals = {}
    low = 0.0
    for symbol, prob in prob_table.items():
        high = low + prob
        intervals[symbol] = (low, high)
        low = high
    return intervals


def arithmetic_encode(data, prob_table):
    """
    Encodes data into a single floating-point number.
    """
    intervals = get_intervals(prob_table)
    low, high = 0.0, 1.0

    for symbol in data:
        sym_low, sym_high = intervals[symbol]
        range_ = high - low
        high = low + range_ * sym_high
        low = low + range_ * sym_low

    return (low + high) / 2


def arithmetic_decode(encoded_value, length, prob_table):
    """
    Decodes an arithmetic-coded value back into the original sequence.
    """
    intervals = get_intervals(prob_table)
    result = []
    low, high = 0.0, 1.0

    for _ in range(length):
        value_range = high - low
        for symbol, (sym_low, sym_high) in intervals.items():
            new_low = low + value_range * sym_low
            new_high = low + value_range * sym_high
            if new_low <= encoded_value < new_high:
                result.append(symbol)
                low, high = new_low, new_high
                break
    return result


# ------------------------------
# Image Compression Using Arithmetic Coding
# ------------------------------
def image_compression_arithmetic(img_path):
    """
    Compresses and decompresses an image using arithmetic coding.
    """
    # Step 1: Read grayscale image
    original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    rows, cols = original_image.shape

    # Step 2: Flatten to a 1D sequence
    data = original_image.flatten().tolist()

    # Step 3: Build probability model
    counts = Counter(data)
    total = len(data)
    prob_table = {symbol: count / total for symbol, count in counts.items()}

    # Step 4: Encode data
    print("Encoding image...")
    encoded_value = arithmetic_encode(data, prob_table)

    # Step 5: Decode back to original data
    print("Decoding image...")
    decoded_data = arithmetic_decode(encoded_value, len(data), prob_table)
    reconstructed_image = np.array(decoded_data, dtype=np.uint8).reshape((rows, cols))

    # Step 6: Verify reconstruction
    if np.array_equal(original_image, reconstructed_image):
        print("✅ Success: Reconstructed image is identical to the original.")
    else:
        print("❌ Error: Images differ.")

    # Step 7: Estimate compression size
    original_bits = rows * cols * 8
    min_prob = min(prob_table.values())
    compressed_bits = int(np.ceil(-np.log2(min_prob))) * len(data)
    compression_ratio = original_bits / compressed_bits if compressed_bits > 0 else 0

    # Step 8: Print report
    print("\n--- Compression Report ---")
    print(f"Original Size    : {original_bits / (8 * 1024):.2f} KB")
    print(f"Compressed Size  : {compressed_bits / (8 * 1024):.2f} KB")
    print(f"Compression Ratio: {compression_ratio:.2f} : 1")

    # Step 9: Display results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title(f"Original\n{original_bits / (8 * 1024):.2f} KB")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f"Reconstructed\n{compressed_bits / (8 * 1024):.2f} KB")
    plt.axis('off')

    plt.show()


# ------------------------------
# Main Runner
# ------------------------------
if __name__ == "__main__":
    image_compression_arithmetic("lena.png")

