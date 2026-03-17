import sys
import numpy as np
import torch.nn as nn
from timm import create_model
import torch
from torch.nn.functional import interpolate
import csv
import torchvision.transforms as transforms
import pandas as pd
import warnings
from torchvision.transforms import Resize
from PIL import Image
import nibabel as nib
from types import SimpleNamespace
from torchvision import transforms
import glob 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Callable, Literal, Optional
# Apply PCA or use fewer feature dimensions
from sklearn.metrics import roc_auc_score
import torch.fft

# Compute Spearman correlation
from scipy.stats import spearmanr

sys.path.append('/SAN/medic/IQT_ScoreMatching/SAM-Med2D')

from segment_anything import sam_model_registry
import time
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings("ignore")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ps = 32
k = 5
temperature = 0.005
radius = 50
multiscale = False

import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Callable, Literal, Optional, Tuple
from types import SimpleNamespace

# ============================================================
# 1. SAM-Med2D Setup (unchanged)
# ============================================================

args = SimpleNamespace(
    image_size=256,
    sam_checkpoint="/SAN/medic/IQT_ScoreMatching/SAM-Med2D/sam-med2d_b.pth",
    encoder_adapter=True
)

builder = sam_model_registry["vit_b"]
sam = builder(args)
sam.to(device).eval()

# ============================================================
# 2. Feature Extractor (MODIFIED for batch input)
# ============================================================

class SAMMed2DFeatExtractor(torch.nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()
        self.encoder = vision_encoder

    def forward(self, x):
        """
        x: [B, 3, H, W] - batch of images
        returns: [B, C, H', W'] - spatial feature maps
        """
        out = self.encoder(x)
        # Handle dict vs tensor outputs
        if isinstance(out, dict):
            fmap = out.get('feat', out.get('x', out))
        else:
            fmap = out
        return fmap  # [B, C, H', W']

feat_extractor = SAMMed2DFeatExtractor(sam.image_encoder).to(device)

# Preprocessing for SAM-Med2D
sam_preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Adjust if needed
])

# ============================================================
# 3. Feature Extraction Functions
# ============================================================

def get_sammed2d_features_batch(images: torch.Tensor) -> torch.Tensor:
    """
    Extract features for a batch of images.
    
    Args:
        images: [B, 3, H, W] float tensor in [0, 1]
    
    Returns:
        features: [B, C, H', W'] spatial feature maps
    """
    # Preprocess for SAM-Med2D (resize to 256x256)
    if images.shape[-2:] != (256, 256):
        images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
    
    # Normalize (adjust based on your SAM-Med2D training)
    # If your model expects [0,1], skip normalization
    # If it expects ImageNet normalization, use sam_preprocess
    x = images  # or: x = sam_preprocess(images)
    
    with torch.no_grad():
        fmap = feat_extractor(x)  # [B, C, H', W']
    
    return fmap


def get_sammed2d_features(patch: torch.Tensor) -> torch.Tensor:
    """
    LEGACY: Extract features for a single patch (for backward compatibility).
    
    Args:
        patch: [3, 32, 32] float tensor in [0, 1]
    
    Returns:
        features: [C, H', W'] feature map
    """
    # Convert to PIL for compatibility with old code
    to_pil = transforms.ToPILImage()
    prep = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    pil = to_pil((patch.clamp(0, 1) * 255).to(torch.uint8))
    x = prep(pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        fmap = feat_extractor(x)
    
    return fmap.squeeze(0)

# ============================================================
# 4. Distance Metrics
# ============================================================

class SpatialFeatureMetric:
    """
    Feature-based metric supporting both patch and whole-image modes.
    """
    def __init__(
        self,
        sigma: float = 10.0,
        calc_sigma: bool = False,
        distance: str = 'cosine',
        topk: int = 64,
        use_patches: bool = True
    ):
        self.sigma = sigma
        self.calc_sigma = calc_sigma
        self.distance = distance
        self.topk = topk
        self.reg_eps = 1e-6
        self.use_patches = use_patches

    def __call__(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distance between pred and gt.
        
        Args:
            pred: [B, 3, H, W] if use_patches=False, else [1, 3, patch, patch]
            gt: Same shape as pred
        
        Returns:
            distance: Scalar tensor or [B, H', W'] spatial map
        """
        if self.use_patches:
            return self._compute_patch_distance(pred, gt)
        else:
            return self._compute_spatial_distance_map(pred, gt)

    def _compute_patch_distance(
        self,
        pred_patch: torch.Tensor,
        gt_patch: torch.Tensor
    ) -> torch.Tensor:
        """Legacy patch-based distance (single patch)."""
        f_pred = get_sammed2d_features(pred_patch)  # [C, H', W']
        f_gt = get_sammed2d_features(gt_patch)
        
        C, H, W = f_pred.shape
        x = f_pred.view(C, H * W).transpose(0, 1)  # [H'*W', C]
        y = f_gt.view(C, H * W).transpose(0, 1)
        
        return self._compute_distance_vectors(x, y)

    def _compute_spatial_distance_map(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor
    ) -> torch.Tensor:
        """
        NEW: Whole-image distance computation.
        
        Args:
            pred: [B, 3, H, W]
            gt: [B, 3, H, W]
        
        Returns:
            dist_map: [B, H', W'] spatial distance map
        """
        # Extract features for entire images
        f_pred = get_sammed2d_features_batch(pred)  # [B, C, H', W']
        f_gt = get_sammed2d_features_batch(gt)
        
        B, C, H, W = f_pred.shape
        
        if self.distance == 'cosine':
            # Cosine distance per spatial location
            f_pred_norm = F.normalize(f_pred, dim=1)  # [B, C, H', W']
            f_gt_norm = F.normalize(f_gt, dim=1)
            sim = (f_pred_norm * f_gt_norm).sum(dim=1)  # [B, H', W']
            dist_map = 1 - sim  # Higher = worse
            
        elif self.distance == 'euclidean':
            diff = f_pred - f_gt
            dist_map = torch.sqrt((diff ** 2).sum(dim=1) + 1e-8)  # [B, H', W']
            
        elif self.distance == 'l2':
            diff = f_pred - f_gt
            dist_map = torch.norm(diff, p=2, dim=1)  # [B, H', W']
            
        elif self.distance == 'energy':
            # Energy distance per spatial location
            x = f_pred.permute(0, 2, 3, 1).reshape(B * H * W, C)
            y = f_gt.permute(0, 2, 3, 1).reshape(B * H * W, C)
            
            d_xy = torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0).mean(dim=1)
            d_xx = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0).mean(dim=1)
            d_yy = torch.cdist(y.unsqueeze(0), y.unsqueeze(0)).squeeze(0).mean(dim=1)
            
            energy = 2 * d_xy - d_xx - d_yy
            dist_map = energy.view(B, H, W)
            
        elif self.distance == 'mahalanobis':
            dist_maps = []
            for i in range(B):
                x = f_pred[i].view(C, -1).T  # [H'*W', C]
                y = f_gt[i].view(C, -1).T
                
                # Fit Gaussian on GT
                mu = y.mean(dim=0, keepdim=True)
                y_centered = y - mu
                cov = (y_centered.T @ y_centered) / (y.size(0) - 1)
                cov += torch.eye(C, device=cov.device) * self.reg_eps
                inv_cov = torch.inverse(cov)
                
                # Compute distance
                diff = x - mu
                m2 = (diff @ inv_cov * diff).sum(dim=1)
                dist_maps.append(m2.view(H, W))
            
            dist_map = torch.stack(dist_maps, dim=0)
            
        elif self.distance == 'mmd':
            # MMD per spatial location (expensive!)
            dist_maps = []
            for i in range(B):
                x = f_pred[i].view(C, -1).T  # [H'*W', C]
                y = f_gt[i].view(C, -1).T
                
                if self.calc_sigma:
                    self.sigma = compute_sigma_median(x, y)
                
                mmd_val = compute_mmd(x, y, self.sigma)
                # For MMD, return scalar per image (not spatial)
                dist_maps.append(torch.full((H, W), mmd_val.item(), device=pred.device))
            
            dist_map = torch.stack(dist_maps, dim=0)
        
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
        
        return dist_map  # [B, H', W']

    def _compute_distance_vectors(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Helper for patch-based mode."""
        if self.distance == 'euclidean':
            return torch.sqrt(torch.sum((x - y) ** 2))
            
        elif self.distance == 'cosine':
            dist = 1 - F.cosine_similarity(x, y)
            return dist.mean()
            
        elif self.distance == 'l2':
            return torch.linalg.norm(x - y)
            
        elif self.distance == 'energy':
            d_xy = torch.cdist(x, y, p=2).mean()
            d_xx = torch.cdist(x, x, p=2).mean()
            d_yy = torch.cdist(y, y, p=2).mean()
            return 2 * d_xy - d_xx - d_yy
            
        elif self.distance == 'mahalanobis':
            mu = y.mean(dim=0, keepdim=True)
            y_centered = y - mu
            cov = (y_centered.T @ y_centered) / (y.size(0) - 1)
            cov += torch.eye(cov.size(0), device=cov.device) * self.reg_eps
            inv_cov = torch.inverse(cov)
            
            diff = x - mu
            m2 = (diff @ inv_cov * diff).sum(dim=1)
            top_vals = torch.topk(m2, min(self.topk, m2.numel()), largest=True).values
            return top_vals.mean()
            
        elif self.distance == 'mmd':
            if self.calc_sigma:
                self.sigma = compute_sigma_median(x, y)
            return compute_mmd(x, y, self.sigma)
        
        else:
            raise ValueError(f"Unknown distance: {self.distance}")


# ============================================================
# 5. Aggregation Function
# ============================================================

def patchwise_metric(
    ref: torch.Tensor,
    test: torch.Tensor,
    metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    patch_size: int = 32,
    stride: Optional[int] = None,
    agg: Literal['mean', 'worstk', 'max', 'min', 'softmax'] = 'mean',
    temperature: float = 0.005,
    worst_k: int = 5,
    return_map: bool = False,
    map_mode: Literal['coarse', 'upsampled'] = 'coarse',
    invert: bool = False,
    use_patches: bool = False  # NEW parameter
):
    """
    Compute metric with optional patching.
    
    Args:
        ref: [B, C, H, W]
        test: [B, C, H, W]
        metric_fn: Distance function
        patch_size: Patch size (only used if use_patches=True)
        stride: Stride (default: patch_size // 2)
        agg: Aggregation method
        temperature: Temperature for softmax
        worst_k: K for worst-k
        return_map: Return spatial heatmap
        map_mode: 'coarse' or 'upsampled'
        invert: Whether higher is worse
        use_patches: If False, use whole-image features
    
    Returns:
        image_scores: [B] aggregated scores
        Optional: heatmap [B, 1, H, W] if return_map=True
    """
    if ref.shape != test.shape:
        raise ValueError("ref and test must have same shape")

    B, C, H, W = ref.shape
    
    # ============== Whole-image mode (NEW) ==============
    if not use_patches:
        with torch.no_grad():
            dist_map = metric_fn(test, ref)  # [B, H', W']
        
        B, H_feat, W_feat = dist_map.shape
        patch_scores = dist_map.view(B, -1)  # [B, H'*W']
        
        # Aggregate
        image_scores = _aggregate_scores(
            patch_scores, agg, temperature, worst_k, invert
        )
        
        if not return_map:
            return image_scores
        
        # Return heatmap
        heatmap = dist_map.unsqueeze(1)  # [B, 1, H', W']
        if map_mode == 'upsampled':
            heatmap = F.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=False)
        
        return image_scores, heatmap
    
    # ============== Patch-based mode (ORIGINAL) ==============
    stride = stride or patch_size // 2
    
    ref_p = F.unfold(ref, kernel_size=patch_size, stride=stride)
    tst_p = F.unfold(test, kernel_size=patch_size, stride=stride)
    
    P = ref_p.shape[-1]
    n_h = (H - patch_size) // stride + 1
    n_w = (W - patch_size) // stride + 1
    
    ref_p = ref_p.transpose(1, 2).reshape(B * P, C, patch_size, patch_size)
    tst_p = tst_p.transpose(1, 2).reshape_as(ref_p)
    
    with torch.no_grad():
        patch_scores = metric_fn(tst_p, ref_p).view(B, P)  # [B, P]
    
    # Aggregate
    image_scores = _aggregate_scores(
        patch_scores, agg, temperature, worst_k, invert
    )
    
    if not return_map:
        return image_scores
    
    # Heatmap
    patch_map = patch_scores.view(B, 1, n_h, n_w)
    
    if map_mode == 'coarse':
        return image_scores, patch_map
    
    if map_mode == 'upsampled':
        heatmap = F.interpolate(patch_map, size=(H, W), mode='nearest')
        return image_scores, heatmap
    
    raise ValueError("map_mode must be 'coarse' or 'upsampled'")


def _aggregate_scores(
    patch_scores: torch.Tensor,
    agg: str,
    temperature: float,
    worst_k: int,
    invert: bool
) -> torch.Tensor:
    """Aggregate patch scores into image scores."""
    B, P = patch_scores.shape
    
    if agg == 'mean':
        return patch_scores.mean(dim=1)
    
    elif agg == 'max':
        return patch_scores.max(dim=1).values
    
    elif agg == 'min':
        return patch_scores.min(dim=1).values
    
    elif agg == 'worstk':
        sorted_sc, _ = patch_scores.sort(dim=1, descending=not invert)
        k = min(worst_k, P)
        return sorted_sc[:, :k].mean(dim=1)
    
    elif agg == 'softmax':
        # ETH: attention-based aggregation
        if invert:
            weights = F.softmax(-patch_scores / temperature, dim=1)
        else:
            weights = F.softmax(patch_scores / temperature, dim=1)
        
        return (weights * patch_scores).sum(dim=1)
    
    else:
        raise ValueError(f"Unknown aggregation: {agg}")


# ============================================================
# 6. Helper Functions (unchanged)
# ============================================================

def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    diff2 = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-diff2 / (2 * sigma ** 2))


def compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    N, M = x.size(0), y.size(0)
    K_xx = rbf_kernel(x, x, sigma)
    K_yy = rbf_kernel(y, y, sigma)
    K_xy = rbf_kernel(x, y, sigma)
    
    K_xx_sum = (K_xx.sum() - K_xx.diag().sum()) / (N * (N - 1))
    K_yy_sum = (K_yy.sum() - K_yy.diag().sum()) / (M * (M - 1))
    K_xy_mean = K_xy.mean()
    
    return K_xx_sum + K_yy_sum - 2 * K_xy_mean


def compute_sigma_median(U: torch.Tensor, V: torch.Tensor, exclude_self: bool = True) -> float:
    d_uu = torch.cdist(U, U).view(-1)
    d_vv = torch.cdist(V, V).view(-1)
    d_uv = torch.cdist(U, V).view(-1)
    
    if exclude_self:
        d_uu = d_uu[d_uu > 0]
        d_vv = d_vv[d_vv > 0]
    
    all_d = torch.cat([d_uu, d_vv, d_uv])
    return all_d.median().item()


# # ============================================================
# # 7. Backward-compatible wrapper
# # ============================================================

# # Legacy PatchMMD class for backward compatibility
# class PatchMMD(SpatialFeatureMetric):
#     """Backward-compatible wrapper."""
#     def __init__(self, sigma: float = 10.0, calc_sigma: bool = False, 
#                  distance: str = 'mmd', topk: int = 64):
#         super().__init__(
#             sigma=sigma,
#             calc_sigma=calc_sigma,
#             distance=distance,
#             topk=topk,
#             use_patches=True  # Legacy uses patches
#         )


# def cphs_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     """
#     LEGACY: Compute distance between two batches using patches.
    
#     For new whole-image version, use:
#         metric = SpatialFeatureMetric(distance='cosine', use_patches=False)
#         scores = patchwise_metric(a, b, metric, use_patches=False, agg='softmax')
#     """
#     mmd = PatchMMD(sigma=1.5, calc_sigma=False, distance='cosine', topk=96)
#     mmd_score = [mmd(x.unsqueeze(0), y.unsqueeze(0)) for x, y in zip(a, b)]
#     return torch.tensor(mmd_score, dtype=torch.float32, device=a.device)


def standardized_effect_size(hallu_vals, clean_vals):
    """Compute effect size after z-scoring"""
    # Pool all data to get normalization parameters
    all_vals = np.concatenate([hallu_vals, clean_vals])
    mean_all = np.mean(all_vals)
    std_all = np.std(all_vals)
    
    # Z-score normalize
    hallu_z = (hallu_vals - mean_all) / std_all
    clean_z = (clean_vals - mean_all) / std_all
    
    # Now compute Cohen's d on normalized data
    pooled_std = np.sqrt((np.var(hallu_z) + np.var(clean_z)) / 2)
    d_all = (hallu_z - np.mean(clean_z)) / pooled_std
    d = (np.mean(hallu_z) - np.mean(clean_z)) / pooled_std
    d_all_std = np.std(np.abs(d_all))
    
    return d, d_all, d_all_std
'''
def standardized_effect_size(hallu_vals, clean_vals):
    """Compute Cohen's d, per-sample standardized deviations (d_all),
       and the standard error (se_d) after z-scoring."""

    # Pool for normalization
    all_vals = np.concatenate([hallu_vals, clean_vals])
    mean_all = np.mean(all_vals)
    std_all = np.std(all_vals)

    # Z-score normalize
    hallu_z = (hallu_vals - mean_all) / std_all
    clean_z = (clean_vals - mean_all) / std_all

    # Pooled std on z-scored values
    var_h = np.var(hallu_z, ddof=1)
    var_c = np.var(clean_z, ddof=1)
    pooled_std = np.sqrt((var_h + var_c) / 2)

    # Cohen's d (global)
    d = (np.mean(hallu_z) - np.mean(clean_z)) / pooled_std

    # Per-sample standardized deviations
    d_all = (hallu_z - np.mean(clean_z)) / pooled_std

    # ---- Standard error of Cohen's d ----
    n1 = len(hallu_vals)
    n2 = len(clean_vals)
    # Hedges & Olkin variance of d
    var_d = (n1 + n2) / (n1 * n2) + (d**2) / (2 * (n1 + n2 - 2))
    se_d = np.sqrt(var_d)

    return d, d_all, se_d
'''
def evaluate_metrics(preds, gts, lrs, blur=False, noise=False):
    score_lst = []
    times = []
    metric = SpatialFeatureMetric(
        distance='cosine',
        use_patches=False
    )
    for i in range(len(preds)):
        pred = torch.from_numpy(np.load(preds[i])).to(torch.float32).unsqueeze(0).to(device)
        gt   = torch.from_numpy(np.load(gts[i].replace('pred', 'gt'))).to(torch.float32).unsqueeze(0).to(device)
        lr   = torch.from_numpy(np.load(lrs[i])).to(torch.float32).unsqueeze(0).to(device)

        pred[pred < 0] = 0
        pred[pred > 1] = 1
        gt[gt<0] = 0
        gt[gt>1] = 1

        if blur:
            pred = transforms.GaussianBlur(kernel_size=5, sigma=(1.5, 1.5))(pred)
        if noise:
            pred = pred + torch.randn_like(pred) * 0.01

        pred = pred.repeat(1,3,1,1)
        gt   = gt.repeat(1,3,1,1)

        pred = apply_lowpass_filter(pred.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)
        gt = apply_lowpass_filter(gt.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)
        '''
        pred = transforms.GaussianBlur(kernel_size=11, sigma=(1.5, 1.5))(pred)
        gt = transforms.GaussianBlur(kernel_size=11, sigma=(1.5, 1.5))(gt)
        '''
        
        # NEW: Whole-image ETH-Feature
        start = time.time()
        mmd_score = patchwise_metric(
            gt, pred,
            metric_fn=metric,
            use_patches=False,
            agg='softmax',
            invert=False,
            temperature=0.005
        )
        # mmd_score, mmd_scores = patchwise_metric(
        #     gt, pred, cphs_batch, patch=ps, agg='softmax', return_map=True, map_mode='upsampled', invert=True, worst_k=k, temperature=temperature, patch=False
        # )

        # if multiscale:
        #     ps2, temperature2 = 64, 0.005
        #     mmd_score2, _ = patchwise_metric(
        #         gt, pred, cphs_batch, patch=ps2, agg='softmax', return_map=True, map_mode='upsampled', invert=True, worst_k=k, temperature=temperature2, patch=False
        #     )
        #     mmd_score = (mmd_score + mmd_score2) / 2.0

        end = time.time()
        times.append(end-start)
        score_lst.append(mmd_score.cpu().numpy())
        # torch.cuda.empty_cache()

    score_lst = np.array(score_lst).squeeze()
    times = np.array(times)
    print(f"Average Inference time per image: {times.mean()}")
    print(f"Average MMD over {len(preds)} images: {score_lst.mean():.4f} std {score_lst.std():.4f}")
    return score_lst

def z_score_sensitivity(preds, gts, mean, std):
    metric = SpatialFeatureMetric(
        distance='cosine',
        use_patches=False
    )
    z_scores_lst = []
    for i in range(len(preds)):
        pred = torch.from_numpy(np.load(preds[i])).to(torch.float32).unsqueeze(0).to(device)
        gt   = torch.from_numpy(np.load(gts[i])).to(torch.float32).unsqueeze(0).to(device)

        pred[pred < 0] = 0
        pred[pred > 1] = 1
        gt[gt<0] = 0
        gt[gt>1] = 1

        pred = pred.repeat(1,3,1,1)
        gt   = gt.repeat(1,3,1,1)

        pred = apply_lowpass_filter(pred.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)
        gt = apply_lowpass_filter(gt.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)
        '''
        pred = transforms.GaussianBlur(kernel_size=11, sigma=(1.5, 1.5))(pred)
        gt = transforms.GaussianBlur(kernel_size=11, sigma=(1.5, 1.5))(gt)
        '''

        mmd_score = patchwise_metric(
            gt, pred,
            metric_fn=metric,
            use_patches=False,
            agg='softmax',
            invert=False,
            temperature=0.005
        )

        # mmd_score, mmd_scores = patchwise_metric(
        #     gt, pred, cphs_batch, patch=ps, agg='softmax', return_map=True, map_mode='upsampled', invert=True, worst_k=k, temperature=temperature
        # )

        z_scores = -1*(mmd_score.cpu().numpy() - mean) / (std + 1e-6)  # avoid division by zero
        z_scores_lst.append(z_scores)

    return z_scores_lst

def auc_score_test(preds_nohallu, preds_intrinsc, preds_extrinsic, gts):
    metric = SpatialFeatureMetric(
        distance='cosine',
        use_patches=False
    )
    intrinsic_scores, extrinsic_scores, nohallu_scores = [], [], []

    for i in range(len(preds_nohallu)):
        pred_nohallu = torch.from_numpy(np.load(preds_nohallu[i])).to(torch.float32).unsqueeze(0).to(device)
        pred_intrinsic = torch.from_numpy(np.load(preds_intrinsc[i])).to(torch.float32).unsqueeze(0).to(device)
        pred_extrinsic = torch.from_numpy(np.load(preds_extrinsic[i])).to(torch.float32).unsqueeze(0).to(device)
        gt   = torch.from_numpy(np.load(gts[i])).to(torch.float32).unsqueeze(0).to(device)

        pred_nohallu[pred_nohallu < 0] = 0
        pred_nohallu[pred_nohallu > 1] = 1
        pred_intrinsic[pred_intrinsic < 0] = 0
        pred_intrinsic[pred_intrinsic > 1] = 1
        pred_extrinsic[pred_extrinsic < 0] = 0
        pred_extrinsic[pred_extrinsic > 1] = 1
        gt[gt<0] = 0
        gt[gt>1] = 1

        pred_nohallu = pred_nohallu.repeat(1,3,1,1)
        pred_intrinsic = pred_intrinsic.repeat(1,3,1,1)
        pred_extrinsic = pred_extrinsic.repeat(1,3,1,1)
        gt   = gt.repeat(1,3,1,1)


        pred_nohallu = apply_lowpass_filter(pred_nohallu.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)
        pred_intrinsic = apply_lowpass_filter(pred_intrinsic.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)
        pred_extrinsic = apply_lowpass_filter(pred_extrinsic.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)
        gt = apply_lowpass_filter(gt.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)

        '''
        pred_nohallu = transforms.GaussianBlur(kernel_size=11, sigma=(1.5, 1.5))(pred_nohallu)
        pred_intrinsic = transforms.GaussianBlur(kernel_size=11, sigma=(1.5, 1.5))(pred_intrinsic)
        pred_extrinsic = transforms.GaussianBlur(kernel_size=11, sigma=(1.5, 1.5))(pred_extrinsic)
        gt = transforms.GaussianBlur(kernel_size=11, sigma=(1.5, 1.5))(gt)
        '''

        mmd_score_nohallu = patchwise_metric(
            gt, pred_nohallu,
            metric_fn=metric,
            use_patches=False,
            agg='softmax',
            invert=False,
            temperature=0.005
        )

        mmd_score_intrinsic = patchwise_metric(
            gt, pred_intrinsic,
            metric_fn=metric,
            use_patches=False,
            agg='softmax',
            invert=False,
            temperature=0.005
        )

        mmd_score_extrinsic = patchwise_metric(
            gt, pred_extrinsic,
            metric_fn=metric,
            use_patches=False,
            agg='softmax',
            invert=False,
            temperature=0.005
        )

        intrinsic_scores.append(mmd_score_intrinsic.cpu().numpy())
        extrinsic_scores.append(mmd_score_extrinsic.cpu().numpy())
        nohallu_scores.append(mmd_score_nohallu.cpu().numpy())

    intrinsic_scores = np.array(intrinsic_scores).squeeze()
    extrinsic_scores = np.array(extrinsic_scores).squeeze()
    nohallu_scores = np.array(nohallu_scores).squeeze()

    intrinsic_auc = roc_auc_score(np.concatenate([np.ones(len(intrinsic_scores)), np.zeros(len(nohallu_scores))]), np.concatenate([intrinsic_scores, nohallu_scores]))
    extrinsic_auc = roc_auc_score(np.concatenate([np.ones(len(extrinsic_scores)), np.zeros(len(nohallu_scores))]), np.concatenate([extrinsic_scores, nohallu_scores]))

    return intrinsic_auc, extrinsic_auc

def auc_score_test_real(preds, gts, labels):
    metric = SpatialFeatureMetric(
        distance='cosine',
        use_patches=False
    )
    shafe_lst = []
    labels_lst = []

    #create dataframe to save mmd_score and mmd_scores and file_name and label
    mmd_scores_df = pd.DataFrame(columns=['pred_path', 'gt_path', 'label', 'mmd_score', 'mmd_scores'])

    for i in range(len(labels)):
        pred = np.load(labels['pred_path'].iloc[i])
        gt = np.load(labels['gt_path'].iloc[i])
        file_name = labels['gt_path'].iloc[i]
        label = labels['has_hallucination'].iloc[i]
        labels_lst.append(label)

        pred = torch.from_numpy(pred).to(torch.float32).unsqueeze(0).to(device)
        gt   = torch.from_numpy(gt).to(torch.float32).unsqueeze(0).to(device)

        pred[pred < 0] = 0
        pred[pred > 1] = 1
        gt[gt<0] = 0
        gt[gt>1] = 1

        pred = pred.repeat(1,3,1,1)
        gt   = gt.repeat(1,3,1,1)

        pred = apply_lowpass_filter(pred.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)
        gt = apply_lowpass_filter(gt.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)

        mmd_score = patchwise_metric(
            gt, pred,
            metric_fn=metric,
            use_patches=False,
            agg='softmax',
            invert=False,
            temperature=0.005
        )

        shafe_lst.append(mmd_score.cpu().numpy())

        # Append to dataframe
        mmd_scores_df = mmd_scores_df.append({'pred_path': labels['pred_path'].iloc[i], 'gt_path': labels['gt_path'].iloc[i], 'label': label, 'mmd_score': mmd_score.cpu().numpy(), 'mmd_scores': mmd_score.cpu().numpy()}, ignore_index=True) 

    shafe_lst = np.array(shafe_lst).squeeze()
    labels_lst = np.array(labels_lst).squeeze()

    auc = roc_auc_score(labels_lst, shafe_lst)

    # Save dataframe to CSV
    mmd_scores_df.to_csv('mmd_scores.csv', index=False)

    return auc

def quality_correctness_tradeoff(preds_nohallu, preds_intrinsc, preds_extrinsic, gts):
    metric = SpatialFeatureMetric(
        distance='cosine',
        use_patches=False
    )
    intrinsic_lst, extrinsic_lst = [], []
    intrinsic_win, extrinsic_win = [], []
    nohallu_lst = []
    freq_lst = [128, 96, 64, 48, 32, 24, 16, 12, 4]
    intrinsic_alive, extrinsic_alive = True, True

    for freq_idx, freq in enumerate(freq_lst):
        for i in range(len(preds_intrinsc)):
            pred_intrinsic = torch.from_numpy(np.load(preds_intrinsc[i])).to(torch.float32).unsqueeze(0).to(device)
            pred_extrinsic = torch.from_numpy(np.load(preds_extrinsic[i])).to(torch.float32).unsqueeze(0).to(device)
            pred_nohallu = torch.from_numpy(np.load(preds_nohallu[i])).to(torch.float32).unsqueeze(0).to(device)
            gt   = torch.from_numpy(np.load(gts[i])).to(torch.float32).unsqueeze(0).to(device)

            pred_intrinsic[pred_intrinsic < 0] = 0
            pred_intrinsic[pred_intrinsic > 1] = 1
            pred_extrinsic[pred_extrinsic < 0] = 0
            pred_extrinsic[pred_extrinsic > 1] = 1
            pred_nohallu[pred_nohallu < 0] = 0
            pred_nohallu[pred_nohallu > 1] = 1
            gt[gt<0] = 0
            gt[gt>1] = 1

            pred_nohallu = apply_lowpass_filter(pred_nohallu.squeeze(0), cutoff_radius=freq, device=device).unsqueeze(0) 
            pred_nohallu = apply_lowpass_filter(pred_nohallu.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)

            pred_intrinsic = apply_lowpass_filter(pred_intrinsic.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)
            pred_extrinsic = apply_lowpass_filter(pred_extrinsic.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)
            gt = apply_lowpass_filter(gt.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)

            pred_intrinsic = pred_intrinsic.repeat(1,3,1,1)
            pred_extrinsic = pred_extrinsic.repeat(1,3,1,1)
            pred_nohallu = pred_nohallu.repeat(1,3,1,1)
            gt   = gt.repeat(1,3,1,1)

            if intrinsic_alive:
                mmd_score_intrinsic = patchwise_metric(
                    gt, pred_intrinsic,
                    metric_fn=metric,
                    use_patches=False,
                    agg='softmax',
                    invert=False,
                    temperature=0.005
                )

                intrinsic_lst.append(mmd_score_intrinsic.cpu().numpy())
            if extrinsic_alive:
                mmd_score_extrinsic = patchwise_metric(
                    gt, pred_extrinsic,
                    metric_fn=metric,
                    use_patches=False,
                    agg='softmax',
                    invert=False,
                    temperature=0.005
                )
                extrinsic_lst.append(mmd_score_extrinsic.cpu().numpy())
            mmd_score_nohallu = patchwise_metric(
                gt, pred_nohallu,
                metric_fn=metric,
                use_patches=False,
                agg='softmax',
                invert=False,
                temperature=0.005
            )
            nohallu_lst.append(mmd_score_nohallu.cpu().numpy())

        if intrinsic_alive:
            intrinsic_scores = np.array(intrinsic_lst).squeeze()
        if extrinsic_alive:
            extrinsic_scores = np.array(extrinsic_lst).squeeze()
        nohallu_scores = np.array(nohallu_lst).squeeze()

        #Win rate
        if intrinsic_alive:
            win_intrinsic = (intrinsic_scores > nohallu_scores).sum() / len(nohallu_scores) # Nohallu better than intrinsic
        if extrinsic_alive:
            win_extrinsic = (extrinsic_scores > nohallu_scores).sum() / len(nohallu_scores) # Nohallu better than extrinsic

        intrinsic_win.append(win_intrinsic)
        extrinsic_win.append(win_extrinsic)

        #if win_intrinsic < 0.5:
        #    intrinsic_alive = False
        #if win_extrinsic < 0.5:
        #    extrinsic_alive = False

        #print(f"Cutoff freq {freq}: Intrinsic win rate {win_intrinsic if intrinsic_alive else 'N/A'}, Extrinsic win rate {win_extrinsic if extrinsic_alive else 'N/A'}")

        #if intrinsic_alive == False and extrinsic_alive == False:
        #    break
    intrinsic_win = np.array(intrinsic_win)
    extrinsic_win = np.array(extrinsic_win)
    # Original blur frequencies (higher = sharper)
    freq_lst = np.array([128, 96, 64, 48, 32, 24, 16, 12, 4], dtype=float)

    # Sort ascending for integration (low freq first)
    freq_sorted = np.sort(freq_lst)
    # Normalize to [0, 1]
    freq_norm = (freq_sorted - freq_sorted.min()) / (freq_sorted.max() - freq_sorted.min())
    #Intrinsic
    y_sorted = np.array(intrinsic_win)[np.argsort(freq_lst)]
    intrinsic_auc = np.trapz(y_sorted, freq_norm)
    #extrinsic
    y_sorted = np.array(extrinsic_win)[np.argsort(freq_lst)]
    extrinsic_auc = np.trapz(y_sorted, freq_norm)
    print(f"SBC AUC -> Intrinsic: {intrinsic_auc} Extrinsic: {extrinsic_auc}")

def severity_correlation(preds_hallu, gts, hallu_masks):
    metric = SpatialFeatureMetric(
        distance='cosine',
        use_patches=False
    )
    score_severity = []
    severity_labels = []

    for i in range(len(preds_hallu)):
        pred_hallu = torch.from_numpy(np.load(preds_hallu[i])).to(torch.float32).unsqueeze(0).to(device)
        gt = torch.from_numpy(np.load(gts[i])).to(torch.float32).unsqueeze(0).to(device)
        hallu_mask = np.load(hallu_masks[i])
        hallu_mask_bool = torch.from_numpy((hallu_mask > 0).astype(np.float32)).to(device)

        # Count severity as the percentage of the hallu mask that is active
        severity = torch.sum(((gt - pred_hallu)**2) * hallu_mask_bool)
        severity_mse = torch.mean(((gt - pred_hallu)**2) * hallu_mask_bool) # <--- This is the hallu region
        severity_not_mse = torch.mean(((gt - pred_hallu)**2) * (1 - hallu_mask_bool)) # <--- This is the non-hallu region
        assert severity != 0, f'Severity is zero for {preds_hallu[i]}'
        #if severity_mse < severity_not_mse: # If pred_nohallu is worse than pred_hallu in the hallu region and pred_hallu is better in the hallu region than non-hallu region, skip
        #    print(f'Skipping {preds_hallu[i]} due to severity check')
        #    continue
        hallu_count = np.sum(hallu_mask)
        if hallu_count <= 20**2: # 16x16 pixels
            severity_multiplier = 1
        elif hallu_count <= 2*20**2: # 2 x 16x16 pixels
            severity_multiplier = 1
        else:
            severity_multiplier = 3
        severity_level = severity * severity_multiplier

        pred_hallu[pred_hallu < 0] = 0
        pred_hallu[pred_hallu > 1] = 1
        gt[gt<0] = 0
        gt[gt>1] = 1

        pred_hallu = pred_hallu.repeat(1,3,1,1)
        gt   = gt.repeat(1,3,1,1)

        pred_hallu = apply_lowpass_filter(pred_hallu.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)
        gt = apply_lowpass_filter(gt.squeeze(0), cutoff_radius=radius, device=device).unsqueeze(0)

        mmd_score = patchwise_metric(
            gt, pred_hallu,
            metric_fn=metric,
            use_patches=False,
            agg='softmax',
            invert=False,
            temperature=0.005
        )

        score_severity.append(mmd_score.cpu().numpy())
        severity_labels.append(severity_level.cpu().numpy())

    score_severity = np.array(score_severity).squeeze()
    severity_labels = np.array(severity_labels).squeeze()

    # Compute distribution of severity levels
    unique, counts = np.unique(severity_labels, return_counts=True)
    print(f"Severity level distribution: {dict(zip(unique, counts))}")

    spearman_corr, _ = spearmanr(score_severity, severity_labels)
    print(f"Spearman correlation between SHAFE score and severity levels: {spearman_corr:.4f}")

    return spearman_corr

def create_rectangular_lowpass_mask(shape, cutoff_height, cutoff_width, device='cpu'):
    """
    Create a rectangular low-pass filter mask in frequency domain.
    
    Args:
        shape: Tuple (H, W) - image dimensions
        cutoff_height: Height of the pass band (in pixels from center)
        cutoff_width: Width of the pass band (in pixels from center)
        device: torch device
    
    Returns:
        mask: Binary mask of shape (H, W)
    """
    H, W = shape
    mask = torch.zeros((H, W), dtype=torch.float32, device=device)
    
    # Center coordinates
    center_h, center_w = H // 2, W // 2
    
    # Define rectangular region
    h_start = max(0, center_h - cutoff_height // 2)
    h_end = min(H, center_h + cutoff_height // 2)
    w_start = max(0, center_w - cutoff_width // 2)
    w_end = min(W, center_w + cutoff_width // 2)
    
    # Set rectangular region to 1 (pass these frequencies)
    mask[h_start:h_end, w_start:w_end] = 1.0
    
    return mask

def create_circular_lowpass_mask(shape, cutoff_radius, device='cpu'):
    """
    Create a circular low-pass filter mask in frequency domain.
    
    Args:
        shape: Tuple (H, W) - image dimensions
        cutoff_radius: Radius of the pass band (in pixels from center)
        device: torch device
    
    Returns:
        mask: Binary mask of shape (H, W)
    """
    H, W = shape
    
    # Center coordinates
    center_h, center_w = H // 2, W // 2
    
    # Create coordinate grids
    y = torch.arange(H, dtype=torch.float32, device=device) - center_h
    x = torch.arange(W, dtype=torch.float32, device=device) - center_w
    
    # Create meshgrid
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Calculate distance from center
    distance = torch.sqrt(X**2 + Y**2)
    
    # Create circular mask (1 inside circle, 0 outside)
    mask = (distance <= cutoff_radius).float()
    
    return mask


def apply_lowpass_filter(image, cutoff_height=None, cutoff_width=None, cutoff_radius=None, device='cpu'):
    """
    Apply rectangular or circular low-pass filter to an image using FFT.
    
    Args:
        image: torch.Tensor of shape (C, H, W) or (H, W)
        cutoff_height: Height of the rectangular pass band (for rectangular mask)
        cutoff_width: Width of the rectangular pass band (for rectangular mask)
        cutoff_radius: Radius of the circular pass band (for circular mask)
        device: torch device
    
    Returns:
        filtered_image: torch.Tensor of same shape as input
        
    Note: Provide either (cutoff_height, cutoff_width) for rectangular mask
          or cutoff_radius for circular mask.
    """
    # Handle different input shapes
    if image.dim() == 2:
        image = image.unsqueeze(0)  # Add channel dimension
        squeeze_output = True
    else:
        squeeze_output = False
    
    image = image.to(device)
    C, H, W = image.shape
    
    # Create the appropriate mask
    if cutoff_radius is not None:
        # Use circular mask
        mask = create_circular_lowpass_mask((H, W), cutoff_radius, device)
    elif cutoff_height is not None and cutoff_width is not None:
        # Use rectangular mask
        mask = create_rectangular_lowpass_mask((H, W), cutoff_height, cutoff_width, device)
    else:
        raise ValueError("Must provide either cutoff_radius (for circular) or both cutoff_height and cutoff_width (for rectangular)")
    
    # Apply filter to each channel
    filtered_channels = []
    for c in range(C):
        # Forward FFT
        freq = torch.fft.fft2(image[c])
        
        # Shift zero frequency to center
        freq_shifted = torch.fft.fftshift(freq)
        
        # Apply mask
        freq_filtered = freq_shifted * mask
        
        # Shift back and inverse FFT
        freq_filtered = torch.fft.ifftshift(freq_filtered)
        filtered = torch.fft.ifft2(freq_filtered)
        
        # Take real part (imaginary part should be ~0 for real input)
        filtered_channels.append(filtered.real)
    
    # Stack channels
    filtered_image = torch.stack(filtered_channels, dim=0)
    
    if squeeze_output:
        filtered_image = filtered_image.squeeze(0)
    
    return filtered_image

# Example usage
if __name__ == "__main__":

    fname = 'Intrinsic_hallu_0.005_interpolation_t200_multi_1.4entropy_hvm'
    fname2 = 'Extrinsic_hallu_0.007_medsam_0.007_interpolation_t200_multi_1.4entropy_hvm'
    fname3 = 'dps_nohallu_t200'
    np.random.seed(0)
    idx = np.random.choice(500, size=100, replace=False)
    print(f"SOFTMAX SELECTION and temperature {temperature}, radois {radius}, patch size {ps}, multiscale {multiscale}")
    gts  = np.array(glob.glob(f'/SAN/medic/IQT_ScoreMatching/skim/HalluBench_final_full/final/synth_gt/*/gt*.npy'))[idx]
    pred_intrinsics = np.array(glob.glob(f'/SAN/medic/IQT_ScoreMatching/skim/HalluBench_final_full/final/{fname}/*/pred*.npy'))[idx]
    pred_extrinsics = np.array(glob.glob(f'/SAN/medic/IQT_ScoreMatching/skim/HalluBench_final_full/final/{fname2}/*/pred*.npy'))[idx]
    pred_dps_nohallus = np.array(glob.glob(f'/SAN/medic/IQT_ScoreMatching/skim/HalluBench_final_full/final/{fname3}/*/pred*.npy'))[idx]
    lrs = np.array(glob.glob(f'/SAN/medic/IQT_ScoreMatching/skim/HalluBench_final_full/final/{fname3}/*/lr*.npy'))[idx]
    hallu_masks_intrinsic = np.array(glob.glob(f'/SAN/medic/IQT_ScoreMatching/skim/HalluBench_final_full/final/{fname}/*/hallu_mask*.npy'))[idx]
    hallu_masks_extrinsic = np.array(glob.glob(f'/SAN/medic/IQT_ScoreMatching/skim/HalluBench_final_full/final/{fname2}/*/hallu_mask*.npy'))[idx]
    print(len(gts), len(pred_intrinsics), len(pred_extrinsics))
     
    pred_intrinsics = []
    for i in range(len(pred_extrinsics)):
        pred_intrinsics.append(pred_extrinsics[i].replace(fname2, fname))
    pred_dps_nohallus = []
    for i in range(len(pred_intrinsics)):
        pred_dps_nohallus.append(pred_intrinsics[i].replace(fname, fname3))
    gts = []
    for i in range(len(pred_intrinsics)):
        gts.append(pred_intrinsics[i].replace('pred', 'gt').replace(f'{fname}','synth_gt'))
    hallu_masks_intrinsic = []
    for i in range(len(pred_intrinsics)):
        hallu_masks_intrinsic.append(pred_intrinsics[i].replace('pred', 'hallu_mask'))
    hallu_masks_extrinsic = []
    for i in range(len(pred_extrinsics)):
        hallu_masks_extrinsic.append(pred_extrinsics[i].replace('pred', 'hallu_mask'))


    print(len(pred_intrinsics), len(pred_extrinsics), len(pred_dps_nohallus), len(gts), len(lrs), len(hallu_masks_intrinsic))
    
    print("Sample intrinsic file paths:")
    print(pred_intrinsics[:10])
    print("Sample extrinsic file paths:")
    print(pred_extrinsics[:10])
    print("Sample DPS no-hallu file paths:")
    print(pred_dps_nohallus[:10])
    
    # print("Evaluating Dummy...")
    # scores_dummy = evaluate_metrics(gts[:5], gts[:5], lrs, blur=False)
    
    print("Evaluating Intrinsic...")
    scores_intrinsic = evaluate_metrics(pred_intrinsics, gts, lrs, blur=False)
    intrinsic_mean, intrinsic_std = scores_intrinsic.mean(), scores_intrinsic.std()
    
    print("Evaluating Extrinsic...")
    scores_extrinsic = evaluate_metrics(pred_extrinsics, gts, lrs, blur=False)
    extrinsic_mean, extrinsic_std = scores_extrinsic.mean(), scores_extrinsic.std()
    print("Evaluating DPS no-hallu...")
    scores_dps_nohallu = evaluate_metrics(pred_dps_nohallus, gts, lrs, blur=False)
    dps_nohallu_mean, dps_nohallu_std = scores_dps_nohallu.mean(), scores_dps_nohallu.std()
    

    print("Effect size analysis(Normalized)...")
    effect_intrinsic, effect_intrinsic_all, intrinsic_std = standardized_effect_size(scores_intrinsic, scores_dps_nohallu)
    effect_extrinsic, effect_extrinsic_all, extrinsic_std = standardized_effect_size(scores_extrinsic, scores_dps_nohallu)
    print(f"Effect size for intrinsic: {effect_intrinsic} with std: {intrinsic_std}, extrinsic: {effect_extrinsic} with std: {extrinsic_std}")
    
    
    intrinsic_pooled_std = (intrinsic_std**2 + dps_nohallu_std**2) / 2
    extrinsic_pooled_std = (intrinsic_std**2 + dps_nohallu_std**2) / 2

    print("FNR/FPR...")
    metric_intrinsic = scores_intrinsic
    metric_extrinsic = scores_extrinsic
    metric_nohallu = scores_dps_nohallu

    y_true = np.array([1]*len(metric_intrinsic) + [1]*len(metric_extrinsic) + [0]*len(metric_nohallu))
    y_score = np.concatenate([metric_intrinsic, metric_extrinsic, metric_nohallu])
    y_score = y_score

    # AUC (threshold-independent)
    auc = roc_auc_score(y_true, y_score)

    # Choose an operating point, e.g. threshold that maximizes Youden's J (TPR - FPR)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = j.argmax()
    thr = thresholds[idx]

    # FNR at that threshold
    fnr = 1 - tpr[idx]
    print(f"FNR/FPR: ", "AUC:", auc, "Threshold:", thr, "FNR:", fnr, "FPR:", fpr[idx])

    #print("Z-score Sensitivity Analysis...")
    #z_intrinsic = z_score_sensitivity(pred_intrinsics, gts, dps_nohallu_mean, intrinsic_pooled_std)
    #z_extrinsic = z_score_sensitivity(pred_extrinsics, gts, dps_nohallu_mean, extrinsic_pooled_std)

    #print(f"Intrinsic - Z-score mean: {np.mean(z_intrinsic):.4f}, std: {np.std(z_intrinsic):.4f}")
    #print(f"Extrinsic - Z-score mean: {np.mean(z_extrinsic):.4f}, std: {np.std(z_extrinsic):.4f}")
    
    #print("Win Rate Analysis...")
    #win_intrinsic = (scores_intrinsic > scores_dps_nohallu).sum() / len(scores_intrinsic)
    #win_extrinsic = (scores_extrinsic > scores_dps_nohallu).sum() / len(scores_extrinsic)

    # print("Evaluating DPS no-hallu with Blur...")
    # scores_dps_nohallu_blur = evaluate_metrics(pred_dps_nohallus, gts, lrs, blur=True)
    # win_intrinsic_blur = (scores_intrinsic > scores_dps_nohallu_blur).sum() / len(scores_intrinsic)
    # win_extrinsic_blur = (scores_extrinsic > scores_dps_nohallu_blur).sum() / len(scores_extrinsic)

    # print(f"Intrinsic - Win rate: {win_intrinsic:.4f}, Win rate with Blur: {win_intrinsic_blur:.4f}")
    # print(f"Extrinsic - Win rate: {win_extrinsic:.4f}, Win rate with Blur: {win_extrinsic_blur:.4f}")
   
    print("Win Quality-Correctness Trade off...")
    quality_correctness_tradeoff(pred_dps_nohallus, pred_intrinsics, pred_extrinsics, gts)
    scores_dps_nohallu_noise = evaluate_metrics(pred_dps_nohallus, gts, lrs, blur=False, noise=True)
    win_intrinsic_noise = (scores_intrinsic > scores_dps_nohallu_noise).sum() / len(scores_intrinsic)
    win_extrinsic_noise = (scores_extrinsic > scores_dps_nohallu_noise).sum() / len(scores_extrinsic)

    print(f"Intrinsic - Win rate with Noise: {win_intrinsic_noise:.4f}")
    print(f"Extrinsic - Win rate with Noise: {win_extrinsic_noise:.4f}")
    
    
    print("AUC Score Analysis...")
    intrinsic_auc, extrinsic_auc = auc_score_test(pred_dps_nohallus, pred_intrinsics, pred_extrinsics, gts)
    print(f"Intrinsic AUC: {intrinsic_auc:.4f}, Extrinsic AUC: {extrinsic_auc:.4f}")
    
        
    #start = time.time() 
    
    #print("AUC Score Analysis on Real Predictions...")
    #preds = glob.glob(rf'/SAN/medic/IQT_ScoreMatching/HalluBench/real_predictions/hallucinations/*/*/pred_*.npy')
    #gts = glob.glob(rf'/SAN/medic/IQT_ScoreMatching/HalluBench/real_predictions/hallucinations/*/*/gt_*.npy')
    #labels = pd.read_csv(rf'/SAN/medic/IQT_ScoreMatching/HalluBench/real_predictions/hallucinations/hallucination_labels_real.csv')

    #auc_real = auc_score_test_real(preds, gts, labels)
    #print(f"Real Predictions AUC: {auc_real:.4f}")
    #end = time.time()
    #print(f"Time: {end-start}s")
    
    
    print("Severity Correlation Analysis...")
    print("Evaluating Intrinsic...")
    spearman_corr = severity_correlation(pred_intrinsics, gts, hallu_masks_intrinsic)
    print(f"Intrinsic Spearman correlation: {spearman_corr:.4f}")
    print("Evaluating Extrinsic...")
    spearman_corr = severity_correlation(pred_extrinsics, gts, hallu_masks_extrinsic)
    print(f"Extrinsic Spearman correlation: {spearman_corr:.4f}")
    
    print("Done.")
