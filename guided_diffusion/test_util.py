import torch
import torch.nn.functional as F
import numpy as np
import os
import glob


def select_patch(ref_img, size_min=8, size_max=25, bg_threshold=0.2):
    """
    Select a patch from the image based on the index.
    """
    # Augment measurement to generate intrinsic hallucination
    while True:
        #idx_lst = [100, 108, 20, 28]#[55,87,115,147] #[55,105,115,165]
        idx_size = np.random.randint(size_min, size_max, 2)
        idx_lst = np.random.randint(0, 256-idx_size[1], 2)
        
        assert ref_img.min() == 0.0, f"ref_img min: {ref_img.min()}"
        gt_patch = ref_img[:, :, idx_lst[0]:idx_lst[0]+idx_size[0], idx_lst[1]:idx_lst[1]+idx_size[1]]
        #Count the number of pixels in the patch equal to 0
        background = (gt_patch == 0.).sum()
        total = torch.numel(gt_patch)
        # Calculate the percentage of pixels equal to 0
        percentage = background.item() / total
        print(f"Percentage of pixels equal to 0: {percentage:.2%}")
        if percentage > bg_threshold:
            print("Patch has too many 0 pixels, skipping...")
            continue
        else:
            return idx_lst, idx_size