from torchvision import transforms
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def resize(data, shape_in, shape_out):
    resize_transform = transforms.Resize((shape_out[0], shape_out[1]))
    resized_data = resize_transform(data)
    
    return resized_data


def tensor_to_frame(tensor):                
    np_image = tensor.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    return np_image


def downscaling_for_send(img, feature, sf, rf):
    img = img.squeeze(0)
    feature = feature.squeeze(0)
    
    N = feature.shape[-1]
    R = img.shape[-1]
    N_ds = int(N * N * (1-sf))
    patch_size = R // N

    flattened_features = feature.view(-1)
    _, indices = torch.topk(flattened_features, N_ds, largest=False)  
    
    patch_indices = []
    for idx in indices:
        row = idx // N
        col = idx % N
        patch_indices.append((row, col))
        
    origin_patches = []
    downscaled_patches = []
    location_info = []
    for i in range(N):
        for j in range(N):
            patch_start_row = i * patch_size
            patch_end_row = (i + 1) * patch_size if i < N - 1 else R  
            patch_start_col = j * patch_size
            patch_end_col = (j + 1) * patch_size if j < N - 1 else R  

            patch = img[..., patch_start_row:patch_end_row, patch_start_col:patch_end_col]
            
            if (i, j) in patch_indices:
                downscaled_patch = F.interpolate(patch.unsqueeze(0), scale_factor=rf, mode='bilinear', align_corners=False).squeeze(0)
                downscaled_patches.append(downscaled_patch)
                location_info.append(0)
            else:
                origin_patches.append(patch)
                location_info.append(1)
    
    return R, N, location_info, origin_patches, downscaled_patches


def downscaling_for_send_with_dummy(img, feature, sf, rf):
    len_origin_patches, len_downscaled_patches, origin_patches_shape, downscaled_patches_shape = downscaling(img, feature, sf, rf)
    
    origin_pixels = np.prod(origin_patches_shape) * len_origin_patches
    downscaled_pixels = np.prod(downscaled_patches_shape) * len_downscaled_patches
    total_pixels = origin_pixels + downscaled_pixels
    
    dummy_shape = (1, 1, 1, total_pixels)
    return dummy_shape
    

def merge_patches(origin_patches, downscaled_patches, location_info, R, N):
    patch_size = R // N
    downscaled_image = torch.zeros((origin_patches[0].shape[0], R, R), dtype=origin_patches[0].dtype, device=origin_patches[0].device)

    for i in range(N):
        for j in range(N):
            # final row and col
            if i == N - 1: 
                end_row = R
            else:
                end_row = (i + 1) * patch_size
            if j == N - 1:  
                end_col = R
            else:
                end_col = (j + 1) * patch_size
            
            start_row = i * patch_size
            start_col = j * patch_size
            
            if location_info[i * N + j] == 0:  
                downscaled_patch = downscaled_patches.pop(0) 
                downscaled_patch_resized = F.interpolate(downscaled_patch.unsqueeze(0), size=(end_row - start_row, end_col - start_col), mode='bilinear', align_corners=False).squeeze(0)
                downscaled_image[..., start_row:end_row, start_col:end_col] = downscaled_patch_resized
            else:  
                origin_patch = origin_patches.pop(0)  
                downscaled_image[..., start_row:end_row, start_col:end_col] = origin_patch

    return downscaled_image


# def downscaling(img, feature, sf, rf):
#     img = img.squeeze(0)
#     feature = feature.squeeze(0)
    
#     N = feature.shape[-1]
#     R = img.shape[-1]
#     N_ds = int(N * N * (1-sf))
#     N_orign = N * N - N_ds
    
#     flattened_features = feature.view(-1)
#     _, indices = torch.topk(flattened_features, N_ds, largest=False)  
    
#     patch_size = R // N
#     origin_patch_shape = (1, 3, patch_size, patch_size)
    
#     for idx in indices:
#         row = idx // N
#         col = idx % N
        
#         patch_start_row = row * patch_size
#         patch_end_row = (row + 1) * patch_size if row < N - 1 else R  
#         patch_start_col = col * patch_size
#         patch_end_col = (col + 1) * patch_size if col < N - 1 else R  

#         patch = img[..., patch_start_row:patch_end_row, patch_start_col:patch_end_col]
#         downscaled_patch = F.interpolate(patch.unsqueeze(0), scale_factor=rf, mode='bilinear', align_corners=False).squeeze(0)
#         downscaled_patch_shape = downscaled_patch.shape

#     return N_orign, N_ds, origin_patch_shape, downscaled_patch_shape


def downscaling(img, feature, sf, rf):
    img = img.squeeze(0)
    feature = feature.squeeze(0)

    N = feature.shape[-1]
    R = img.shape[-1]
    N_ds = int(N * N * (1 - sf))
    N_orign = N * N - N_ds

    # Flatten feature and select indices for downscaling
    flattened_features = feature.view(-1)
    _, indices = torch.topk(flattened_features, N_ds, largest=False)

    # Compute patch size
    patch_size = R // N
    origin_patch_shape = (1, 3, patch_size, patch_size)

    # Initialize a tensor for downscaled patches
    downscaled_patches = []
    
    for idx in indices:
        row = idx // N
        col = idx % N
        patch_start_row = row * patch_size
        patch_end_row = (row + 1) * patch_size
        patch_start_col = col * patch_size
        patch_end_col = (col + 1) * patch_size

        # Extract and downscale the patch
        patch = img[..., patch_start_row:patch_end_row, patch_start_col:patch_end_col]
        downscaled_patch = F.interpolate(patch.unsqueeze(0), scale_factor=rf, mode='bilinear', align_corners=False).squeeze(0)
        downscaled_patches.append(downscaled_patch)

    downscaled_patch_shape = downscaled_patches[0].shape if downscaled_patches else origin_patch_shape

    return N_orign, N_ds, origin_patch_shape, downscaled_patch_shape