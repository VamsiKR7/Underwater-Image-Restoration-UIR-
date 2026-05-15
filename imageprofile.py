import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import csv

def get_image_info(image_path):
    # Open the image using Pillow
    img = Image.open(image_path)

    # 1. Color Depth (bit depth)
    color_depth = img.bits  # Color depth in bits per channel

    # 2. Compression (this is usually related to the image format, e.g., PNG, JPEG, etc.)
    # We can check the format of the image
    image_format = img.format
    if image_format == "JPEG":
        compression = "Lossy (JPEG)"
    elif image_format == "PNG":
        compression = "Lossless (PNG)"
    else:
        compression = "Unknown or other format"

    # 3. RGB Values
    img_rgb = np.array(img.convert('RGB'))  # Convert to RGB (numpy array)
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]  # Extract R, G, B channels
    avg_rgb = (np.mean(r), np.mean(g), np.mean(b))  # Average RGB values for the image

    # 4. Brightness and Contrast
    enhancer = ImageEnhance.Brightness(img)
    brightness = enhancer.enhance(1)  # Enhances image by factor of 1 (no change)
    
    contrast_enhancer = ImageEnhance.Contrast(img)  # Correct variable name here
    contrast = contrast_enhancer.enhance(1)  # Enhances image by factor of 1 (no change)
    brightness_value = np.mean(np.array(brightness.convert('L')))  # Convert to grayscale and compute mean brightness
    contrast_value = np.std(np.array(contrast.convert('L')))  # Standard deviation of contrast in grayscale

    # 5. Saturation (Hue-Saturation-Lightness)
    enhancer_saturation = ImageEnhance.Color(img)
    saturated_img = enhancer_saturation.enhance(1)  # No change
    img_hsv = np.array(saturated_img.convert('HSV'))  # Convert to HSV
    saturation = np.mean(img_hsv[:, :, 1])  # Mean of saturation channel in HSV

    # Prepare results as a dictionary or tuple
    return {
        "image_name": os.path.basename(image_path),
        "color_depth": color_depth,
        "compression": compression,
        "avg_r": avg_rgb[0],
        "avg_g": avg_rgb[1],
        "avg_b": avg_rgb[2],
        "brightness": brightness_value,
        "contrast": contrast_value,
        "saturation": saturation
    }


def write_metrics_to_csv(input_dir, output_csv):
    # Prepare CSV headers
    headers = ["image_name", "color_depth", "compression", "avg_r", "avg_g", "avg_b", "brightness", "contrast", "saturation"]

    # Open CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        # List images in directory and sort them by name (alphabetical/numerical order)
        image_files = sorted(os.listdir(input_dir))

        # Process each image in the directory
        for filename in image_files:
            image_path = os.path.join(input_dir, filename)
            if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                metrics = get_image_info(image_path)
                writer.writerow(metrics)  # Write the metrics to CSV

        print(f"Metrics have been written to {output_csv}")

# Example usage:
input_dir = './Dataset/UIE/UIEBD/test/image' #'/media/cvblns/NS/SVC_342/unsuper 2022 6th/Dataset/UIE/UIEBD/test/image'  # Change this to your image folder path
output_csv = 'image_metrics.csv'  # Path where CSV will be saved
write_metrics_to_csv(input_dir, output_csv)

