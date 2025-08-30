Lab 4: Image Compression – Arithmetic Coding & JPEG
Objective
This experiment focuses on understanding and applying two essential techniques of image compression:
1. Lossless Compression through Arithmetic Coding.
2. Lossy Compression using the JPEG approach with Discrete Cosine Transform (DCT).

Overview
Image compression minimizes the number of bits required to represent an image by removing redundancy while maintaining visual quality. It can be performed using:

1. Lossless Compression (Arithmetic Coding)

Ensures the original image can be reconstructed perfectly.

Implements entropy-based encoding to represent sequences of symbols as fractional numbers between 0 and 1.

2. Lossy Compression (JPEG)

Provides greater compression by eliminating less significant visual details.

Utilizes DCT to convert image blocks into the frequency domain and applies quantization to discard high-frequency data.
