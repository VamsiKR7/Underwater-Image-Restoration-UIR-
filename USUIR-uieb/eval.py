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
import numpy as np
import torch
import gc
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch UIE')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--data_test', type=str, default='../Dataset/UIE/UIEBD/test/image')
parser.add_argument('--label_test', type=str, default='../Dataset/UIE/UIEBD/test/label')
parser.add_argument('--model', default='weights/epoch_84.pth', help='Pretrained base model')
parser.add_argument('--output_folder', type=str, default='results/predict/')
# parser.add_argument('--max_size', type=int, default=152, help='Maximum image dimension for processing')
parser.add_argument('--use_log', action='store_false', help='whether print log to std output')

opt = parser.parse_args()

print('===> Loading datasets')
test_set = get_eval_set(opt.data_test, opt.label_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model')

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() and opt.gpu_mode else "cpu")
model = net().to(device)
model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained model is loaded.')

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

def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')

    # Create output directories
    os.makedirs(opt.output_folder, exist_ok=True)
    j_folder = os.path.join(opt.output_folder, 'J')
    a_folder = os.path.join(opt.output_folder, 'A')
    t_folder = os.path.join(opt.output_folder, 'T')
    os.makedirs(j_folder, exist_ok=True)
    os.makedirs(a_folder, exist_ok=True)
    os.makedirs(t_folder, exist_ok=True)

    for batch in testing_data_loader:
        try:
            with torch.no_grad():
                # Get input and move to device
                input, label, name = batch[0].to(device), batch[1], batch[2]
                print(f"Processing {name[0]} (original size: {input.shape})")
                
                # Resize input to standard height while preserving aspect ratio
                original_height = input.shape[2]
                original_width = input.shape[3]
                # new_height = opt.max_size
                # new_width = int(original_width * (new_height / original_height))

                
                
                # # Don't resize if image is already smaller than max_size
                # # if original_height <= opt.max_size and original_width <= opt.max_size:
                # #     input_resized = input
                # #     print(f"  Image already within size limit: {input_resized.shape[2:]}")
                # # else:
                # input_resized = F.interpolate(input, size=(new_height, new_width), mode='bilinear', align_corners=False)
                # print(f"  Resized to: {input_resized.shape[2:]}")
                
                # Apply color correction - already handled on GPU
                # input_corrected = apply_color_correction_to_tensor(input_resized.cpu()).to(device)
                
                # Process the entire image at once
                j_out, t_out = model(input)
                
                # Get the atmospheric light
                a_out = get_A(input.cpu())
                
                # Optional sharpening (uncomment if needed)
                # j_out = laplacian_sharpen(j_out, strength=0.4)
                # t_out = laplacian_sharpen(t_out, strength=0.4)
                
                # Resize outputs back to original dimensions if needed
                # if original_height != new_height or original_width != new_width:
                #     j_out = F.interpolate(j_out, size=(original_height, original_width), mode='bilinear', align_corners=False)
                #     t_out = F.interpolate(t_out, size=(original_height, original_width), mode='bilinear', align_corners=False)
                #     a_out = F.interpolate(a_out, size=(original_height, original_width), mode='bilinear', align_corners=False)
                
                # Convert to numpy and save
                j_out_np = np.clip(torch_to_np(j_out.cpu()), 0, 1)
                t_out_np = np.clip(torch_to_np(t_out.cpu()), 0, 1)
                a_out_np = np.clip(torch_to_np(a_out), 0, 1)
                
                # Save results
                my_save_image(name[0], j_out_np, j_folder + '/')
                my_save_image(name[0], t_out_np, t_folder + '/')
                my_save_image(name[0], a_out_np, a_folder + '/')
                
                print(f"{name[0]} successfully processed!")
                
                # Clean up to avoid memory issues
                del j_out, t_out, a_out, input#, input_resized #, input_corrected
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"Error processing {name[0]}: {str(e)}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()
            continue

if __name__ == '__main__':
    mp.freeze_support()
    eval()