from __future__ import print_function
import argparse
import os
import torch.multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.utils.data import DataLoader
from net.net import net
from data import get_eval_set
from utils import *
from skimage.color import rgb2lab, rgb2gray, lab2rgb
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
import torch
import gc
from torchvision.transforms.functional import to_tensor
from PIL import Image
from kornia.metrics import psnr, ssim
from torchvision.datasets.folder import is_image_file
import pandas as pd
import torch.nn.functional as F
from scipy import stats

# ====================
# Argument Parser
# ====================
parser = argparse.ArgumentParser(description='PyTorch USUIR Multi-Epoch Evaluation')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--data_test', type=str, default='../Dataset/UIE/UIEBD/test/image')
parser.add_argument('--label_test', type=str, default='../Dataset/UIE/UIEBD/test/label')
parser.add_argument('--weights_folder', type=str, default='weights/', help='Folder containing all epoch weights')
parser.add_argument('--start_epoch', type=int, default=10, help='Start evaluation from this epoch')
parser.add_argument('--end_epoch', type=int, default=150, help='End evaluation at this epoch')
parser.add_argument('--output_folder', type=str, default='results/')
parser.add_argument('--use_log', action='store_false', help='whether print log to std output')
opt = parser.parse_args()

# ====================
# Helper Functions
# ====================
def apply_color_correction_to_tensor(input_tensor):
    batch_size = input_tensor.size(0)
    input_corrected_list = []
    for b in range(batch_size):
        img_rgb = input_tensor[b].cpu().permute(1, 2, 0).numpy()
        if img_rgb.max() > 1.0:
            img_rgb = img_rgb / 255.0

        img_lab = rgb2lab(img_rgb)
        img_gray = rgb2gray(img_rgb)
        img_gray[img_gray > 0.85] = 0
        img_gray[img_gray <= 0.85] = 1

        gray_gauss = cv2.GaussianBlur(img_gray, (0, 0), 5)
        ch1, ch2, ch3 = cv2.split(img_lab)
        ch2_gauss = cv2.GaussianBlur(ch2, (0, 0), 100)
        img_lab[:, :, 1] = np.subtract(ch2, np.multiply(gray_gauss, ch2_gauss))
        ch3_gauss = cv2.GaussianBlur(ch3, (0, 0), 100)
        img_lab[:, :, 2] = np.subtract(ch3, np.multiply(gray_gauss, ch3_gauss))

        corrected = lab2rgb(img_lab)
        corrected_tensor = torch.from_numpy(corrected).permute(2, 0, 1).float()
        input_corrected_list.append(corrected_tensor)
    input_corrected = torch.stack(input_corrected_list)
    return input_corrected

def laplacian_sharpen(img_tensor, strength):
    laplacian_filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=img_tensor.device).unsqueeze(0).unsqueeze(0)
    laplacian_filter = laplacian_filter.repeat(img_tensor.shape[1], 1, 1, 1)
    edge = F.conv2d(img_tensor, laplacian_filter, padding=1, groups=img_tensor.shape[1])
    sharpened = img_tensor - strength * edge
    return torch.clamp(sharpened, 0, 1)

def find_available_epochs(weights_folder, start_epoch, end_epoch):
    all_weight_files = glob.glob(os.path.join(weights_folder, "epoch_*.pth"))
    available_epochs = []
    for weight_file in all_weight_files:
        match = re.search(r'epoch_(\d+)\.pth', weight_file)
        if match:
            epoch_num = int(match.group(1))
            if start_epoch <= epoch_num <= end_epoch:
                available_epochs.append(epoch_num)
    return sorted(available_epochs)

"""
# ====================
# UIQM Metric Functions
# ====================
def sobel(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    return grad

def calculate_uism(img):
    # Calculate Underwater Image Sharpness Measure (UISM)
    if len(img.shape) < 3 or img.shape[2] < 3:
        return 0
    
    img = img.astype(np.float32)
    b, g, r = cv2.split(img)
    
    # Apply Sobel edge detection to each channel
    sobel_b = sobel(cv2.merge([b, b, b]))
    sobel_g = sobel(cv2.merge([g, g, g]))
    sobel_r = sobel(cv2.merge([r, r, r]))
    
    # Calculate Edge-Aware Local Contrast
    mean_b = np.mean(sobel_b)
    mean_g = np.mean(sobel_g)
    mean_r = np.mean(sobel_r)
    
    # Weight contributions from each channel
    uism = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b
    return uism

def calculate_uicm(img):
    # Calculate Underwater Image Colorfulness Measure (UICM)
    if len(img.shape) < 3 or img.shape[2] < 3:
        return 0
    
    img = img.astype(np.float32) / 255
    
    # Convert to lab color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Remove top and bottom 1% of values
    a_vec = a.flatten()
    b_vec = b.flatten()
    
    a_vec.sort()
    b_vec.sort()
    
    n = len(a_vec)
    trim_percent = int(n * 0.01)
    
    a_vec = a_vec[trim_percent:n-trim_percent]
    b_vec = b_vec[trim_percent:n-trim_percent]
    
    # Calculate mean and standard deviation
    a_mean = np.mean(a_vec)
    b_mean = np.mean(b_vec)
    a_std = np.std(a_vec)
    b_std = np.std(b_vec)
    
    # UICM calculation
    alpha_a = -10 * np.log(a_std)
    alpha_b = -10 * np.log(b_std)
    
    uicm = (a_mean**2 + b_mean**2)**0.5 + alpha_a + alpha_b
    return uicm

def calculate_uiconm(img):
    # Calculate Underwater Image Contrast Measure (UIConM)
    if len(img.shape) < 3 or img.shape[2] < 3:
        return 0
    
    img = img.astype(np.float32) / 255
    
    window_size = 8
    h, w, _ = img.shape
    
    b, g, r = cv2.split(img)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    entropy_sum = 0
    num_blocks = 0
    
    for i in range(0, h-window_size+1, window_size):
        for j in range(0, w-window_size+1, window_size):
            block = gray[i:i+window_size, j:j+window_size]
            hist, _ = np.histogram(block, bins=256, range=[0, 1])
            hist = hist / (window_size * window_size)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist)) if np.any(hist) else 0
            entropy_sum += entropy
            num_blocks += 1
    
    uiconm = entropy_sum / num_blocks if num_blocks > 0 else 0
    return uiconm

def calculate_uiqm(img):
    # Calculate Underwater Image Quality Measure (UIQM)
    if len(img.shape) < 3 or img.shape[2] < 3:
        return 0
    
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    
    uism = calculate_uism(img)
    uicm = calculate_uicm(img)
    uiconm = calculate_uiconm(img)
    
    uiqm = c1 * uicm + c2 * uiconm + c3 * uism
    return uiqm

# ====================
# UCIQE Metric Functions
# ====================
def calculate_uciqe(img):
    # Calculate Underwater Color Image Quality Evaluation (UCIQE)
    if len(img.shape) < 3 or img.shape[2] < 3:
        return 0
    
    img = img.astype(np.float32) / 255
    
    # Convert to lab color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Calculate chroma
    c_ab = np.sqrt(np.square(a) + np.square(b))
    
    # Calculate average of chroma
    mu_c = np.mean(c_ab)
    
    # Calculate standard deviation of chroma
    sigma_c = np.std(c_ab)
    
    # Calculate contrast of luminance
    l_sorted = np.sort(l, axis=None)
    n = len(l_sorted)
    contrast_l = l_sorted[int(0.99*n)] - l_sorted[int(0.01*n)]
    
    # Combine the metrics using the weights from the original paper
    w1 = 0.0282
    w2 = 0.2953
    w3 = 3.5753
    
    uciqe = w1 * contrast_l + w2 * mu_c + w3 * sigma_c
    return uciqe
"""

def evaluate_metrics(input_dir, refer_dir, output_dir, epoch_num):
    input_image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if is_image_file(f)])
    refer_image_paths = sorted([os.path.join(refer_dir, f) for f in os.listdir(refer_dir) if is_image_file(f)])

    if len(input_image_paths) != len(refer_image_paths):
        print(f"Error: Image count mismatch ({len(input_image_paths)} vs {len(refer_image_paths)})")
        return None, None, None, None

    psnr_values = []
    ssim_values = []
    # uiqm_values = []
    # uciqe_values = []
    metrics_file = os.path.join(output_dir, f'quantitive_eval_epoch_{epoch_num}.csv')

    with open(metrics_file, 'w') as f:
        f.write('img_name,psnr,ssim\n')  # Removed uiqm,uciqe from header
        for input_img_path, refer_img_path in zip(input_image_paths, refer_image_paths):
            try:
                input_img = Image.open(input_img_path)
                refer_img = Image.open(refer_img_path)

                # For PSNR and SSIM
                input_tensor = to_tensor(input_img).unsqueeze(0)
                refer_tensor = to_tensor(refer_img).unsqueeze(0)

                psnr_value = psnr(input_tensor, refer_tensor, max_val=1.0).item()
                ssim_value = ssim(input_tensor, refer_tensor, max_val=1.0, window_size=11).mean().item()

                """
                # For UIQM and UCIQE
                input_np = np.array(input_img)
                input_np_bgr = cv2.cvtColor(input_np, cv2.COLOR_RGB2BGR)
                
                uiqm_value = calculate_uiqm(input_np_bgr)
                uciqe_value = calculate_uciqe(input_np_bgr)
                """

                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)
                # uiqm_values.append(uiqm_value)
                # uciqe_values.append(uciqe_value)
                
                f.write(f"{os.path.basename(input_img_path)},{psnr_value:.3f},{ssim_value:.3f}\n")  # Removed uiqm and uciqe

                if opt.use_log:
                    print(f"{os.path.basename(input_img_path)}, PSNR: {psnr_value:.3f}, SSIM: {ssim_value:.3f}")  # Removed UIQM and UCIQE

            except Exception as e:
                print(f"Error processing {input_img_path}: {e}")

        avg_psnr = np.average(psnr_values) if psnr_values else 0
        avg_ssim = np.average(ssim_values) if ssim_values else 0
        # avg_uiqm = np.average(uiqm_values) if uiqm_values else 0
        # avg_uciqe = np.average(uciqe_values) if uciqe_values else 0
        
        f.write(f"average_value,{avg_psnr:.3f},{avg_ssim:.3f}")  # Removed uiqm and uciqe
        print(f"Epoch {epoch_num} - PSNR: {avg_psnr:.3f}, SSIM: {avg_ssim:.3f}")  # Removed UIQM and UCIQE

    return avg_psnr, avg_ssim, None, None  # Return None for UIQM and UCIQE

# def get_A(x):
#     """Estimate atmospheric light based on brightest pixels"""
#     b = x[:, 0, :, :]
#     g = x[:, 1, :, :]
#     r = x[:, 2, :, :]
    
#     img_gray = 0.299 * r + 0.587 * g + 0.114 * b
#     batch_size = x.shape[0]
#     a_list = []
    
#     for i in range(batch_size):
#         gray = img_gray[i].cpu().numpy()
#         flat_gray = gray.flatten()
        
#         # Get brightest pixels (top 0.1%)
#         num_pixels = flat_gray.size
#         num_brightest = max(1, int(num_pixels * 0.001))
#         brightest_indices = np.argsort(flat_gray)[-num_brightest:]
        
#         # Extract those pixels from original image
#         h, w = gray.shape
#         b_channel = b[i].cpu().numpy()
#         g_channel = g[i].cpu().numpy()
#         r_channel = r[i].cpu().numpy()
        
#         a_r = a_g = a_b = 0
#         for idx in brightest_indices:
#             row, col = idx // w, idx % w
#             a_b += b_channel[row, col]
#             a_g += g_channel[row, col]
#             a_r += r_channel[row, col]
        
#         a_b /= num_brightest
#         a_g /= num_brightest
#         a_r /= num_brightest
        
#         a_val = torch.ones_like(x[i]) * torch.tensor([a_b, a_g, a_r]).view(3, 1, 1).to(x.device)
#         a_list.append(a_val)
    
#     return torch.stack(a_list)

def eval_epoch(epoch_num):
    print(f'===> Loading datasets')
    test_set = get_eval_set(opt.data_test, opt.label_test)
    loader = DataLoader(test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    print('===> Building model')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    model_path = os.path.join(opt.weights_folder, f'epoch_{epoch_num}.pth')
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None, None, None, None

    try:
        model.load_state_dict(torch.load(model_path))
        print(f'Loaded model for epoch {epoch_num}')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None

    model.eval()
    torch.set_grad_enabled(False)

    output_folder = os.path.join(opt.output_folder, f'epoch_{epoch_num}')
    os.makedirs(output_folder, exist_ok=True)
    j_folder = os.path.join(output_folder, 'J')
    a_folder = os.path.join(output_folder, 'A')
    t_folder = os.path.join(output_folder, 'T')
    os.makedirs(j_folder, exist_ok=True)
    os.makedirs(a_folder, exist_ok=True)
    os.makedirs(t_folder, exist_ok=True)

    for batch in loader:
        try:
            input, label, name = batch[0].to(device), batch[1], batch[2]
            print(f"Processing {name[0]} (original: {input.shape})")

            original_height = input.shape[2]
            original_width = input.shape[3]
            # new_height = 512
            # new_width = int(original_width * (new_height / original_height))
            # input_resized = F.interpolate(input, size=(new_height, new_width), mode='bilinear', align_corners=False)

            j_out, t_out = model(input)
            a_out = get_A(input)

            j_np = np.clip(torch_to_np(j_out.cpu()), 0, 1)
            t_np = np.clip(torch_to_np(t_out.cpu()), 0, 1)
            a_np = np.clip(torch_to_np(a_out.cpu()), 0, 1)

            my_save_image(name[0], j_np, j_folder + '/')
            my_save_image(name[0], t_np, t_folder + '/')
            my_save_image(name[0], a_np, a_folder + '/')
            print(f"{name[0]} processed.")

            del j_out, t_out, a_out, input
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"Error processing {name[0]}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue

    return evaluate_metrics(j_folder, opt.label_test, output_folder, epoch_num)

def plot_results(results):
    epochs = sorted(results.keys())
    psnr_values = [results[e]['psnr'] for e in epochs]
    ssim_values = [results[e]['ssim'] for e in epochs]
    # uiqm_values = [results[e]['uiqm'] for e in epochs]
    # uciqe_values = [results[e]['uciqe'] for e in epochs]

    # Plot PSNR and SSIM
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs, psnr_values, 'b-o', label='PSNR')
    ax1.set_ylabel('PSNR (dB)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlabel('Epoch')

    ax2 = ax1.twinx()
    ax2.plot(epochs, ssim_values, 'r-o', label='SSIM')
    ax2.set_ylabel('SSIM', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    plt.title('PSNR and SSIM vs Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(opt.output_folder, 'psnr_ssim_plot.png'))
    plt.close()
    
    """
    # Plot UIQM and UCIQE
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(epochs, uiqm_values, 'g-o', label='UIQM')
    ax3.set_ylabel('UIQM', color='green')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.set_xlabel('Epoch')

    ax4 = ax3.twinx()
    ax4.plot(epochs, uciqe_values, 'm-o', label='UCIQE')
    ax4.set_ylabel('UCIQE', color='magenta')
    ax4.tick_params(axis='y', labelcolor='magenta')

    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper center')

    plt.title('UIQM and UCIQE vs Epochs')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(opt.output_folder, 'uiqm_uciqe_plot.png'))
    plt.close()
    """
    
    print("Metrics plots saved.")

def save_metrics_to_csv(results):
    csv_path = os.path.join(opt.output_folder, 'all_epochs_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write("Epoch,PSNR,SSIM\n")  # Removed UIQM and UCIQE
        for epoch in sorted(results.keys()):
            f.write(f"{epoch},{results[epoch]['psnr']:.4f},{results[epoch]['ssim']:.4f}\n")  # Removed UIQM and UCIQE
    print(f"Metrics saved to {csv_path}")

def evaluate_all_epochs():
    print(f"Evaluating epochs from {opt.start_epoch} to {opt.end_epoch}")
    os.makedirs(opt.output_folder, exist_ok=True)
    available_epochs = find_available_epochs(opt.weights_folder, opt.start_epoch, opt.end_epoch)

    if not available_epochs:
        print("No valid weights found.")
        return

    results = {}
    for epoch in available_epochs:
        print(f"\nEvaluating epoch {epoch}")
        avg_psnr, avg_ssim, _, _ = eval_epoch(epoch)  # Ignore UIQM and UCIQE
        if avg_psnr is not None and avg_ssim is not None:
            results[epoch] = {'psnr': avg_psnr, 'ssim': avg_ssim}

    plot_results(results)
    save_metrics_to_csv(results)

    print("\nTop Epochs by PSNR:")
    for i, (ep, m) in enumerate(sorted(results.items(), key=lambda x: x[1]['psnr'], reverse=True)[:5]):
        print(f"Rank {i+1}: Epoch {ep} - PSNR: {m['psnr']:.4f}, SSIM: {m['ssim']:.4f}")

    print("\nTop Epochs by SSIM:")
    for i, (ep, m) in enumerate(sorted(results.items(), key=lambda x: x[1]['ssim'], reverse=True)[:5]):
        print(f"Rank {i+1}: Epoch {ep} - PSNR: {m['psnr']:.4f}, SSIM: {m['ssim']:.4f}")
        
    """
    print("\nTop Epochs by UIQM:")
    for i, (ep, m) in enumerate(sorted(results.items(), key=lambda x: x[1]['uiqm'], reverse=True)[:5]):
        print(f"Rank {i+1}: Epoch {ep} - PSNR: {m['psnr']:.4f}, SSIM: {m['ssim']:.4f}, UIQM: {m['uiqm']:.4f}, UCIQE: {m['uciqe']:.4f}")
        
    print("\nTop Epochs by UCIQE:")
    for i, (ep, m) in enumerate(sorted(results.items(), key=lambda x: x[1]['uciqe'], reverse=True)[:5]):
        print(f"Rank {i+1}: Epoch {ep} - PSNR: {m['psnr']:.4f}, SSIM: {m['ssim']:.4f}, UIQM: {m['uiqm']:.4f}, UCIQE: {m['uciqe']:.4f}")
    """

if __name__ == '__main__':
    mp.freeze_support()
    evaluate_all_epochs()