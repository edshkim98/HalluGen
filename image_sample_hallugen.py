"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
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
        th.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print('Using device:', device)

    save_path = '/cluster/project0/IQT_Nigeria/skim/DPS_hallu/'
    #data_dir = '/cluster/project0/IQT_Nigeria/skim/HCP_Kim_x4_2D/Sig1.0Gam0.7DS04/test/test_small/*'#HCP_t1t2_ALL/sim/901038' #996782
    #files = glob.glob(data_dir + '/gt.npy')

    lst_files = ['901038','901139','901442','902242','904044','905147'] #['116120', '116221', '116423', '116524', '116726', '117021', '117122', '117324', '117728', '117930', '118023', '118124', '118225', '118528', '118730', '118831', '118932', '119025', '119126', '119732']
    lst_files = lst_files[0:1]
    
    save_files_pred = {i: [] for i in lst_files}
    save_files_gt = {i: [] for i in lst_files}
    save_files_lr = {i: [] for i in lst_files}
    
    data_dir = '/cluster/project0/IQT_Nigeria/HCP_t1t2_ALL/sim/9*'
    files = glob.glob(data_dir + '/T1w/T1w_acpc_dc_restore_brain.nii.gz')
    print(len(files))
    #print(files[:5])
    files_new = []
    for f in files:
        if f.split('/')[-3] in lst_files:
            files_new.append(f)
    files = files_new

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
        print(f"{i}/{len(data)}")
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
            )
            model_kwargs["y"] = classes
            
            
        ref_img = ref_img.to(device)
        # Load U-Net output
        print(data_dict['file_id'][0], data_dict['slice_idx'].numpy()[0])
        fname_curr, slice_curr = int(data_dict['file_id'][0]), str(data_dict['slice_idx'].numpy()[0])
        print(fname_curr, slice_curr)
        #data = np.load(f'/cluster/project0/IQT_Nigeria/skim/diffusion_inverse/guided-diffusion/cond_results/unet/ind/{fname_curr}/pred_{slice_curr}_axial.npy')[0]
        #data = np.load(f'/cluster/project0/IQT_Nigeria/skim/sss/pred_all_{fname_curr}.npy')[int(slice_curr)]

        # Denormalize
        #data = data * std + mean
        #data = np.clip(data, 0., 1.0)
        #print("DATA shape")
        #print(data.min(), data.max())
        
        #Inject noise
        if configs['skip_timestep']:
            skip_x0 = ref_img #torch.tensor(data).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
            #np.save("skip_x0.npy", skip_x0.cpu().numpy())
        else:
            skip_x0 = None

        # Forward measurement model (Ax + n)
        y = operator.forward(ref_img)
        y_n = noiser(y)
        #y_n = y_n.clmap(min=0., max=2.)

        # Augment measurement to generate intrinsic hallucination
        idx_lst = [100, 164, 20, 84]#[55,87,115,147] #[55,105,115,165]
        
        y_n_clone = y_n.clone()
        
        measurement_patch = y_n[:, :, idx_lst[0]: idx_lst[1], idx_lst[2]:idx_lst[3]].cpu()
        #Downsample the measurement
        down_ratio = 1
        trans_pixel = 0
        measurement_patch_down = Resize((measurement_patch.shape[2]//down_ratio, measurement_patch.shape[3]//down_ratio), interpolation=Image.BILINEAR)(measurement_patch)
        #Upsample the measurement
        measurement_patch = Resize((measurement_patch.shape[2], measurement_patch.shape[3]), interpolation=Image.BILINEAR)(measurement_patch_down)
        #measurement_patch = measurement_patch + 0.05 * torch.randn_like(measurement_patch)
        #measurement_patch[:] = torch.rot90(measurement_patch,1,[2,3])
        meausrement_patch = measurement_patch.clone()
        measurement_patch = measurement_patch.to(device)
        y_n_clone[:, :, idx_lst[0]: idx_lst[1], idx_lst[2]:idx_lst[3]] = measurement_patch
        #skip_x0 = y_n_clone
            
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        start = time.time()
        sample = sample_fn(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            measurement=y_n_clone.to(torch.float32),
            measurement_cond_fn = measurement_cond_fn,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            skip_timesteps=configs['skip_timestep'],
            skip_x0=skip_x0,
            line_search=configs['line_search']
        )
        end = time.time()
        print("Inf time: ", end-start)
        time_diff = end-start
        time_lst.append(time_diff)
        #print("Inf time: ", time_diff)
        
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
                np.save(f'{save_path}/{data_dict["file_id"][j]}/gt_{data_dict["slice_idx"][j]}_axial.npy', ref_img[j].cpu().numpy())
                np.save(f'{save_path}/{data_dict["file_id"][j]}/lr_{data_dict["slice_idx"][j]}_axial.npy', y_n[j].cpu().numpy())
                np.save(f'{save_path}/{data_dict["file_id"][j]}/lr_hallu_{data_dict["slice_idx"][j]}_axial.npy', y_n_clone[j].cpu().numpy())
 
            #for j in range(args.batch_size):
            #    key = str(int(data_dict['file_id'][j].numpy()))
            #    save_files_pred[key].append((sample[j].cpu().numpy()))
            #    save_files_gt[key].append((ref_img[j].cpu().numpy()))
            #    save_files_lr[key].append((y[j].cpu().numpy()))
                 
     
    time_lst = np.array(time_lst)
    print("Mean time: ", np.mean(time_lst))
    print("Std time: ", np.std(time_lst))
    print("Saving the results in Numpy")
    mini, maxi = 0.0, 1.0 
   
    # Concatenate the results for each file and save
    '''
    for k, v in save_files_pred.items():
        v = np.array(v)
        v = np.clip(v, mini, maxi)
        if not os.path.exists(f'{save_path}/{k}'):
            os.makedirs(f'{save_path}/{k}')
        np.save(f'{save_path}/{k}/pred_axial.npy', v)
    for k, v in save_files_gt.items():
        v = np.array(v)
        np.save(f'{save_path}/{k}/gt_axial.npy', v)
    for k, v in save_files_lr.items():
        v = np.array(v)
        np.save(f'{save_path}/{k}/lr_axial.npy', v)
    '''
    # concatenate all the images into a single numpy array
    # arr = np.array(all_images)
    # arr_ys = np.array(ys)
    # arr_refs = np.array(refs)
    
    #arr = np.clip(arr, mini, maxi)
    
    # save the samples in a numpy file
    # np.savez("samples_pred", arr)
    # np.savez("samples_ys", arr_ys)
    # np.savez("samples_refs", arr_refs)

    # dist.barrier()
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
