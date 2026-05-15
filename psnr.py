import os
import cv2
import numpy as np
import csv
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(image1, image2):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.
    """
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # Images are identical
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_metrics_for_folders(target_folder, reference_folder, output_csv):
    """
    Calculate PSNR and SSIM for all images in a target folder against corresponding images in a reference folder.

    :param target_folder: Path to the folder containing target images to evaluate
    :param reference_folder: Path to the folder containing reference images
    :param output_csv: Path to save the results in CSV format
    """
    if not os.path.exists(reference_folder):
        print(f"Error: Reference folder not found at {reference_folder}")
        return
    if not os.path.exists(target_folder):
        print(f"Error: Target folder not found at {target_folder}")
        return

    results = []
    psnr_values = []
    ssim_values = []

    for filename in os.listdir(target_folder):
        target_file_path = os.path.join(target_folder, filename)
        reference_file_path = os.path.join(reference_folder, filename)

        # Check if corresponding reference image exists
        if not os.path.exists(reference_file_path):
            print(f"Skipping {filename}: No corresponding reference image found.")
            continue

        # Load images
        target_image = cv2.imread(target_file_path, cv2.IMREAD_COLOR)
        reference_image = cv2.imread(reference_file_path, cv2.IMREAD_COLOR)

        if target_image is None or reference_image is None:
            print(f"Skipping {filename}: Unable to load one of the images.")
            continue

        # Convert images to RGB and float32
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Check dimensions
        if target_image.shape != reference_image.shape:
            print(f"Skipping {filename}: Dimension mismatch between target and reference image.")
            continue

        # Calculate PSNR
        psnr_value = calculate_psnr(reference_image, target_image)
        psnr_values.append(psnr_value)

        # Calculate SSIM with appropriate window size
        min_dim = min(reference_image.shape[:2])  # Smallest dimension of the image
        win_size = min(7, min_dim)  # Use smallest dimension if less than 7
        win_size = win_size if win_size % 2 == 1 else win_size - 1  # Ensure it's odd

        if min_dim < 7:
            print(f"Warning: {filename} has small dimensions. Using smaller window size.")

        ssim_value = ssim(
            reference_image, 
            target_image, 
            win_size=win_size, 
            channel_axis=2, 
            data_range=target_image.max() - target_image.min()
        )
        ssim_values.append(ssim_value)

        results.append((filename, psnr_value, ssim_value))
    
    # Calculate averages
    avg_psnr = np.mean(psnr_values) if psnr_values else 0
    avg_ssim = np.mean(ssim_values) if ssim_values else 0

    # Print results and averages
    print("Results:")
    for filename, psnr_value, ssim_value in results:
        print(f"{filename}: PSNR = {psnr_value:.2f} dB, SSIM = {ssim_value:.4f}")
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # Save results to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "PSNR (dB)", "SSIM"])
        writer.writerows(results)
        writer.writerow([])
        writer.writerow(["Average", avg_psnr, avg_ssim])
    
    print(f"\nResults saved to {output_csv}")

# Example Usage
target_folder = r"C:\Users\buzo1\Desktop\PC\Computer vision\unsuper 2022 6th\unsuper 2022 6th\USUIR-uieb\prevresults\predict\J"
reference_folder = r"C:\Users\buzo1\Desktop\PC\Computer vision\unsuper 2022 6th\unsuper 2022 6th\Dataset\UIE\UIEBD\test\label"
output_csv = r"C:\Users\buzo1\Desktop\PC\Computer vision\unsuper 2022 6th\unsuper 2022 6th\psnr_ssim_results.csv"

calculate_metrics_for_folders(target_folder, reference_folder, output_csv)
