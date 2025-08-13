# Lab 2 – Bit Slicing & Histogram Equalization 
---

## 📌 Description

This experiment demonstrates **Bit-Plane Slicing** and **Histogram Equalization** on a grayscale image using **Python (OpenCV, NumPy, Matplotlib)**.

### **Bit Slicing**
Bit slicing is a technique used to analyze specific bits of pixel intensity values, which helps in **image compression**, **enhancement**, and **feature extraction**.  
Each pixel in a grayscale image is typically represented using 8 bits (values from 0 to 255). Bit slicing isolates each individual bit-plane from these 8-bit representations.

### **Histogram Equalization**
Histogram equalization is a contrast enhancement technique that redistributes the image’s pixel intensities to make the histogram more uniform. It enhances details in low-contrast regions.

---

## 🎯 Objectives

### **Bit Slicing**
- Understand and implement the concept of bit-plane slicing.
- Extract all **8 bit-planes** from a grayscale image manually using custom logic.
- Display the contribution of each bit to the image quality.

### **Histogram Equalization**
- Understand the concept of image histogram and contrast enhancement.
- Implement histogram equalization **manually** without using built-in equalization functions.
- Compare the original and equalized images along with their histograms.

---

## ⚙️ Logic (Manual Implementation)

### **Bit Slicing**
1. Read the input image.
2. Convert to grayscale if the image is RGB.
3. Get image dimensions.
4. Initialize an array of 8 matrices to store bit-planes.
5. For each pixel:
   - Get intensity value (0–255).
   - Extract each bit using bitwise operations.
   - Store bits in respective bit-plane matrices.
6. Display bit-planes from **MSB (bit 8)** to **LSB (bit 1)**.

### **Histogram Equalization**
1. Read the grayscale image.
2. Flatten it into a 1D array.
3. Create a histogram (count frequency of each pixel intensity).
4. Compute the **Cumulative Distribution Function (CDF)**.
5. Apply the histogram equalization formula:
   \[
   \text{Equalized Value} = \frac{\text{CDF} - \text{CDF}_{min}}{(MN) - \text{CDF}_{min}} \times 255
   \]
6. Map old pixel values to equalized values.
7. Display:
   - Original image & histogram.
   - Equalized image & histogram.

---

## 🖼️ Sample Output

### **Bit Slicing**
- **Bit-plane 8 (MSB)** → Shows major image structures and edges.
- **Bit-plane 1 (LSB)** → Shows fine noise-like details.

*(Example figure with 8 subplots, one for each bit-plane)*

### **Histogram Equalization**
- Original image with a narrow histogram.
- Equalized image with a more spread-out histogram.

*(Example figure showing original vs equalized images and histograms)*

---

## 💻 Requirements
Install required Python libraries before running:
```bash
pip install opencv-python matplotlib numpy

