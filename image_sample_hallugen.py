"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from guided_diffusion import logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import yaml
import torch
import tqdm
import glob
from torch.utils.data import DataLoader
from guided_diffusion.image_datasets import IQTDataset
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.condition_methods import get_conditioning_method

import matplotlib.pyplot as plt
import time
from torchvision.transforms import Resize
from PIL import Image
from guided_diffusion.test_util import select_patch


torch.backends.cudnn.enabled = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data_custom(data_loader):
    while True:
        yield from data_loader

def compute_patch_entropy(patch: torch.Tensor, num_bins: int = 256, eps: float = 1e-10, zero2two: bool = True) -> torch.Tensor:
    """
    Computes the entropy of a grayscale image patch.

    Args:
        patch (torch.Tensor): A 2D tensor (H, W) with values in [0, 1].
        num_bins (int): Number of bins to discretize the values.
        eps (float): Small value to avoid log(0).

    Returns:
        torch.Tensor: A scalar tensor representing entropy.
    """
    if zero2two:
        patch /= 2.0
    assert patch.ndim == 2, "Patch must be a 2D tensor"
    assert patch.min() >= 0 and patch.max() <= 1, f"Patch values must be in range [0, 1] but got MAX: {patch.max()} MIN: {patch.min()}"

    # Flatten the patch and bin the values
    flat = patch.view(-1)
    bin_idx = torch.clamp((flat * (num_bins - 1)).long(), 0, num_bins - 1)

    # Histogram
    hist = torch.bincount(bin_idx, minlength=num_bins).float()

    # Convert to probability distribution
    probs = hist / hist.sum()

    # Compute entropy (avoid log(0) by masking)
    entropy = -torch.sum(probs * torch.log(probs + eps))
    return entropy
        
def main():
    
    set_seed(42)

    with open('./configs.yaml') as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(configs = configs,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        torch.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print('Using device:', device)

    # ---- Added lines ----
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")   # formatted with commas

    #import sys
    #sys.exit()   # terminate the program

    save_path = '/SAN/medic/IQT_ScoreMatching/skim/HalluBench_final_full/final/Intrinsic_hallu_0.002_interpolation_t200_multi_1.4entropy_hvm_final'

    #lst_files = ['994273', '993675', '992774', '992673', '991267', '990366', '989987', '987983', '987074', '984472', '983773', '979984', '978578', '973770', '972566', '971160', '970764', '969476', '966975', '965771', '965367', '962058', '969574', '958976', '957974', '955465', '953764', '947668', '943862', '942658', '937160', '933253', '932554', '930449', '929464', '927359', '926862', '923755', '922854', '919966', '917558', '917255', '912477', '911849', '910443', '910241', '908860', '907656', '905147', '904044']
    #lst_files = lst_files[:10]
    lst_files = ['996782', '995174', '994273', '993675', '992774', '992673', '991267', '990366', '989987', '987983', '987074', '984472', '983773', '979984', '978578', '973770', '972566', '971160', '970764', '969476', '966975', '965771', '965367', '962058', '959574', '958976', '957974', '955465', '953764','952863', '951457', '947668', '943862', '942658', '937160', '933253', '932554', '930449', '929464', '927359', '926862', '923755', '922854', '919966', '917558', '917255', '912447', '911849', '910443', '910241', '908860', '907656', '905147', '904044', '902242', '901442', '901139', '901038']
    lst_files = lst_files#[50:]
    #print(lst_files)
    
    save_files_pred = {i: [] for i in lst_files}
    save_files_gt = {i: [] for i in lst_files}
    save_files_lr = {i: [] for i in lst_files}
    
    data_dir = '/SAN/medic/IQT_ScoreMatching/skim/HalluBench_final_full/final/synth_gt/' 
    files = glob.glob(data_dir + '/*/gt*.npy')
    #print(files[:5])
    files_new = []
    lst_files_nohallu = os.listdir('/SAN/medic/IQT_ScoreMatching/skim/HalluBench_final_full/final/dps_nohallu_t200/')
    for f in files:
        if (f.split('/')[-2] in lst_files_nohallu) and (f.split('/')[-2] in lst_files):
            files_new.append(f)
    files = files_new
    #iles = files[:10]
    print(len(files), files)

    # Check if already done
    try:
        files_exist = os.listdir(save_path)
        print(f"Files exist: {len(files_exist)}")
        files_new = []
        for i in files:
            fname = i.split('/')[-3]
            existing = True if fname in files_exist else False
            if existing:
                print(f"Skipping {fname}...")
            else:
                files_new.append(i)
                print(fname)

        files = files_new
        print(f"Files after checking existing: {len(files)}")
        print(files)
    except:
        print("No file exist so run all")

    dataset = IQTDataset(files, configs = configs, return_id=configs['data']['return_id'])
    print(f"Files: {len(files)} Dataset size: {len(dataset)}")
    data = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    try:
        ref_img, data_dict = next(iter(data))
        print(f"Batch: ref_img shape: {ref_img.shape}, data_dict: {data_dict}")
    except Exception as e:
        print(f"Error in batch: {e}")       
 
    # Prepare Operator and noise
    measure_config = configs['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")
 
     # Working directory
    save_dir = '/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/guided-diffusion/results/'
    out_path = os.path.join(save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
           
    # Prepare conditioning method
    cond_config = configs['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {configs['conditioning']['method']}")

    logger.log("sampling...")
    all_images = []
    ys = []
    refs = []
    time_lst = []
    
    for i, (ref_img, data_dict) in tqdm.tqdm(enumerate(data)):
        #for j in range(args.batch_size):
        if os.path.exists(f'{save_path}/{data_dict["file_id"][0]}/pred_{data_dict["slice_idx"][0]}_axial.npy'):
            print(f'Skipping -> {save_path}/{data_dict["file_id"][0]}/pred_{data_dict["slice_idx"][0]}_axial.npy')
            continue
        hvm_flag = False
        print(f"{i}/{len(data)}")
        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
            )
            model_kwargs["y"] = classes
            
        assert ref_img.shape[0] == 1, f"batch size must be 1 but got {ref_img.shape[0]}"
        cond_img = np.load(f'/SAN/medic/IQT_ScoreMatching/skim/HalluBench_final_full/final/dps_nohallu_t200/{data_dict["file_id"][0]}/pred_{data_dict["slice_idx"][0]}_axial.npy')
        cond_img = torch.tensor(cond_img, device=device).unsqueeze(0)    
        ref_img = ref_img.to(device)
        # Load U-Net output
        print(data_dict['file_id'][0], data_dict['slice_idx'].numpy()[0])
        fname_curr, slice_curr = str(data_dict['file_id'][0]), str(data_dict['slice_idx'].numpy()[0])
        print(fname_curr, slice_curr)
        
        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img)
        y_n = noiser(y)

        #y_n = y_n.clamp(min=0., max=2.)

        #Inject noise
        if configs['skip_timestep']:
            skip_x0 = cond_img.clone().to(device) #y_n.clone().to(device) #ref_img.clone().to(device) #torch.tensor(data).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
        else:
            skip_x0 = None

        # Augment measurement to generate intrinsic hallucination
        repeat_cnt = 0
        num_hallu = np.random.randint(1, 4) # number of hallucinated patches in the image
        while hvm_flag == False: # Repeat until hvm is passed
            if repeat_cnt > 1 and num_hallu > 1:
                num_hallu = min(1, num_hallu)
            #num_hallu = np.random.randint(1, 4) # number of hallucinated patches in the image
            patch_idx_lst = []
            
            for j in range(num_hallu):
                cnt = 0
                entropy_threshold = 1.4 #1.4
                while True:
                    idx_lst, idx_size = select_patch(ref_img, size_min=16, size_max=25, bg_threshold=0.05) #24, 33
                    
                    # --- REPLACE THE OLD 'IF' BLOCK WITH THIS ---
                    valid = True
                    buffer = 5 # Set this to the minimum pixel gap you want between patches

                    # 1. We only need to check for overlaps if we've already saved some patches
                    if len(patch_idx_lst) > 0:
                        # Define boundaries for the NEW proposed patch
                        new_y_start, new_y_stop = idx_lst[0], idx_lst[0] + idx_size[0]
                        new_x_start, new_x_stop = idx_lst[1], idx_lst[1] + idx_size[1]

                        for k in range(len(patch_idx_lst)):
                            # Get boundaries for EXISTING patch [k]
                            old_y_start = patch_idx_lst[k][0].start
                            old_y_stop  = patch_idx_lst[k][0].stop
                            old_x_start = patch_idx_lst[k][1].start
                            old_x_stop  = patch_idx_lst[k][1].stop

                            # Check for ANY overlap (including the buffer)
                            # This logic says: "If the patches are NOT completely separated, they must overlap"
                            overlap = not (new_x_stop + buffer < old_x_start or 
                                           new_x_start - buffer > old_x_stop or 
                                           new_y_stop + buffer < old_y_start or 
                                           new_y_start - buffer > old_y_stop)
    
                            if overlap:
                                valid = False
                                break
                    #if selected idx range is already in the list, skip
                    #valid = True
                    #if len(patch_idx_lst) > 0:
                    #    for k in range(len(patch_idx_lst)):
                    #        if (idx_lst[0] >= patch_idx_lst[k][0].start and idx_lst[0] <= patch_idx_lst[k][0].stop) and (idx_lst[1] >= patch_idx_lst[k][1].start and idx_lst[1] <= patch_idx_lst[k][1].stop):
                    #            valid = False
                    #            break 
                    if valid:
                        patch_selected = ref_img[0,0,idx_lst[0]:idx_lst[0]+idx_size[0], idx_lst[1]:idx_lst[1]+idx_size[1]].cpu()
                        if patch_selected.max() > 1.0:
                            patch_selected /= 2.0
                        if compute_patch_entropy(patch_selected, num_bins=16, zero2two=False) > entropy_threshold: ######################### >
                            patch_idx = [slice(idx_lst[0], idx_lst[0]+idx_size[0]), slice(idx_lst[1], idx_lst[1]+idx_size[1])]
                            patch_idx_lst.append(patch_idx)
                            print("Patch Selected!")
                            break
                        else:
                            cnt +=1
                            if cnt == 10:
                                entropy_threshold -= 0.1
                                cnt = 0
            
            #patch_idx = [slice(30, 30+168), slice(20, 20+168]
            #if num_hallu != 0:
            #    patch_idx_lst.append(patch_idx)
            assert num_hallu == len(patch_idx_lst), f"Num hallu: {num_hallu}, Patch_idx_lst: {len(patch_idx_lst)}"
            mask = torch.zeros_like(ref_img)
            for j in range(num_hallu):
                mask[:, :, patch_idx_lst[j][0], patch_idx_lst[j][1]] = 1.0
                logger.info(f"{np.unique(mask.cpu().numpy(), return_counts=True)}")
                if (configs['extrinsic'] is not None) and (skip_x0 is not None):
                    skip_x0[:, :, patch_idx_lst[j][0], patch_idx_lst[j][1]] = y_n[:, :, patch_idx_lst[j][0], patch_idx_lst[j][1]]
                if configs['perturb_measurement']:
                    y_n[:, :, patch_idx_lst[j][0], patch_idx_lst[j][1]] = y_n[:, :, patch_idx_lst[j][0], patch_idx_lst[j][1]] + 0.1*torch.randn_like(y_n[:, :, patch_idx_lst[j][0], patch_idx_lst[j][1]], device=y_n.device)
                
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            start = time.time()
            sample, hvm_flag = sample_fn(
                model,
                (args.batch_size, 1, args.image_size, args.image_size),
                measurement=y_n.to(torch.float32),
                measurement_cond_fn = measurement_cond_fn,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                skip_timesteps=configs['skip_timestep'],
                skip_x0=skip_x0.to(torch.float32),
                line_search=configs['line_search'],
                patch_idx=patch_idx_lst,
                cond_img=cond_img if configs['interpolate_nohallu'] else None,
                extrinsic=ref_img if configs['extrinsic'] else None,
                semantic=ref_img if configs['semantic'] else None
            )
            end = time.time()
            print("Inf time: ", end-start)
            time_diff = end-start
            time_lst.append(time_diff)

            if hvm_flag == False:
                print("Hallucination verification failed! Skipping saving...")
                repeat_cnt += 1

        #sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        all_images.append(sample.cpu().numpy()) 
        refs.append(ref_img.cpu().numpy())
        ys.append(y_n.cpu().numpy())
        print("One image done!")
         
        if data_dict is not None:
            #pass
            # # Save the images
            for j in range(args.batch_size):
                if not os.path.exists(f'{save_path}/{data_dict["file_id"][j]}'):        
                    os.makedirs(f'{save_path}/{data_dict["file_id"][j]}')
                np.save(f'{save_path}/{data_dict["file_id"][j]}/pred_{data_dict["slice_idx"][j]}_axial.npy', sample[j].cpu().numpy())
                #np.save(f'{save_path}/{data_dict["file_id"][j]}/gt_{data_dict["slice_idx"][j]}_axial.npy', ref_img[j].cpu().numpy())
                np.save(f'{save_path}/{data_dict["file_id"][j]}/lr_{data_dict["slice_idx"][j]}_axial.npy', y_n[j].cpu().numpy())
                np.save(f'{save_path}/{data_dict["file_id"][j]}/hallu_mask_{data_dict["slice_idx"][j]}_axial.npy', mask[j].cpu().numpy())
                 
     
    time_lst = np.array(time_lst)
    print("Mean time: ", np.mean(time_lst))
    print("Std time: ", np.std(time_lst))
   
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/guided-diffusion/logs_large_zero2two_HCPMoreSlice2025/model360000.pt",)
#120000.pt",    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
