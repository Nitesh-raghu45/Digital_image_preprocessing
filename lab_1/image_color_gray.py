import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def RGBProcessor(image_name, choice):
    """
    Digital Image Processing Task Handler

    Parameters:
    - image_name: Filename of the image in the same directory
    - choice:
        1 - Convert to Grayscale (weighted sum)
        2 - Show RGB Planes in subplots
        3 - Convert to Black and White (thresholded)
    """

    # Resolve absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, image_name)

    # Load image
    I = cv2.imread(image_path)
    if I is None:
        print("Error: Unable to read image.")
        return

    # Convert BGR to RGB
    I_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    if choice == 1:
        # Grayscale using weighted sum method
        R = I_rgb[:, :, 0].astype(float)
        G = I_rgb[:, :, 1].astype(float)
        B = I_rgb[:, :, 2].astype(float)
        Gray = (0.298936 * R + 0.587043 * G + 0.114021 * B).astype(np.uint8)

        # Save grayscale image
        grayscale_path = os.path.join(script_dir, "output_grayscale.png")
        cv2.imwrite(grayscale_path, Gray)
        print(f"Grayscale image saved as: {grayscale_path}")

        # Display
        plt.figure(figsize=(5, 5))
        plt.imshow(Gray, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')
        plt.show()

    elif choice == 2:
        # Red, Green, Blue Planes
        red_plane = I_rgb.copy()
        red_plane[:, :, 1] = 0
        red_plane[:, :, 2] = 0

        green_plane = I_rgb.copy()
        green_plane[:, :, 0] = 0
        green_plane[:, :, 2] = 0

        blue_plane = I_rgb.copy()
        blue_plane[:, :, 0] = 0
        blue_plane[:, :, 1] = 0

        # Save RGB planes
        cv2.imwrite(os.path.join(script_dir, "output_red_plane.png"), cv2.cvtColor(red_plane, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(script_dir, "output_green_plane.png"), cv2.cvtColor(green_plane, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(script_dir, "output_blue_plane.png"), cv2.cvtColor(blue_plane, cv2.COLOR_RGB2BGR))
        print("RGB planes saved as: output_red_plane.png, output_green_plane.png, output_blue_plane.png")

        # Plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(red_plane)
        plt.title("Red Plane")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(green_plane)
        plt.title("Green Plane")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(blue_plane)
        plt.title("Blue Plane")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    elif choice == 3:
        # Convert to grayscale then threshold
        R = I_rgb[:, :, 0].astype(float)
        G = I_rgb[:, :, 1].astype(float)
        B = I_rgb[:, :, 2].astype(float)
        Gray = (0.298936 * R + 0.587043 * G + 0.114021 * B).astype(np.uint8)
        BW = np.where(Gray < 128, 0, 255).astype(np.uint8)

        # Save B&W image
        bw_path = os.path.join(script_dir, "output_black_white.png")
        cv2.imwrite(bw_path, BW)
        print(f"Black & White image saved as: {bw_path}")

        # Plot
        plt.figure(figsize=(5, 5))
        plt.imshow(BW, cmap='gray')
        plt.title("Black and White Image")
        plt.axis('off')
        plt.show()

    else:
        print("Invalid choice. Choose 1, 2, or 3.")

# Example usages
RGBProcessor("lena.png", choice=1)
RGBProcessor("lena.png", choice=2)
RGBProcessor("lena.png", choice=3)
