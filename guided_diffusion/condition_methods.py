from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance as FID

import numpy as np
import torch.nn as nn
from timm import create_model
from torch.nn.functional import interpolate
import csv
import torchvision.transforms as transforms
import pandas as pd
import warnings
from torch.optim import LBFGS
from torchvision.transforms import Resize
from PIL import Image
import nibabel as nib

import sys
from types import SimpleNamespace
from torchvision import transforms
import timm
from scipy.ndimage import label

sys.path.append('/SAN/medic/IQT_ScoreMatching/SAM-Med2D')
#sys.path.append('/SAN/medic/IQT_ScoreMatching/segment-anything')

from segment_anything import sam_model_registry

warnings.filterwarnings("ignore")

__CONDITIONING_METHOD__ = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(42)

class TimmFeatExtractor(nn.Module):
    """
    Feature extractor using Timm pre-trained models.
    """

    def __init__(self,
                model_name: str = "vit_base_patch16_dinov3.lvd1689m"):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        # Load a pre-trained model from Timm
        # Initialize feature extractor using timm
        if 'sam' in self.model_name.lower() or 'dinov2' in self.model_name.lower():
            self.model = timm.create_model(
                self.model_name,
                pretrained=True,
                num_classes=0,      # No classification head
            ).to(device).eval()
        elif 'resnet' in self.model_name.lower():            
            self.model = timm.create_model(
                self.model_name,
                pretrained=True,
                features_only=True,  # Return intermediate feature maps
                out_indices=[1, 2]     # Use last layer features 1,2, 3
            ).to(device).eval()
        elif 'dinov3' in self.model_name.lower():            
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                features_only=True,  # Return intermediate feature maps
                out_indices=[1, 2, 3]     # Use last layer features 1,2, 3
            ).to(device).eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps from images.
        
        Args:
            x: Input images (N, C, H, W)
        
        Returns:
            features: Spatial feature maps (N, C', H', W')
        """
        with torch.enable_grad():  # Allow gradients
            if 'sam' in self.model_name.lower() or 'dinov2' in self.model_name.lower():
                N, C, H, W = x.shape
                features = self.model.forward_features(x)  # (N, C', H', W')
                # Check if features are tokens or spatial
                if features.dim() == 3:  # (N, num_patches, C)
                    N, num_patches, C = features.shape
                    H = W = int(num_patches ** 0.5)
                    features = features.transpose(1, 2).reshape(N, C, H, W)
            else:
                features = self.model(x)
            if isinstance(features, (list, tuple)):
                #features = features[-1]  # Take last layer
                # Upsample and concatenate all features
                if features[0].shape[2] != features[-1].shape[2] or features[0].shape[3] != features[-1].shape[3]:
                    feats = []
                else:
                    feats = torch.zeros_like(features[-1]) 
                for feat in features:
                    if isinstance(feats, list):
                        feat = F.interpolate(feat, size=(features[-1].shape[2], features[-1].shape[3]), mode='bilinear', align_corners=False)
                        feats.append(feat)
                    else:
                        feats += feat
                if isinstance(feats, list):
                    features = torch.cat(feats, dim=1)
                else:
                    features = feats / len(features)
        return features

class SAMFeatExtractor(nn.Module):
    """
    Feature extractor using original SAM model.
    - Uses SAM's image encoder only
    - forward expects x: [B, 3, 1024, 1024] (SAM's native resolution)
    - Includes preprocessing helper for resizing to 1024x1024
    """
    def __init__(self,
                 model_type: str = "vit_b",
                 image_size: int = 1024,  # SAM uses 1024x1024
                 sam_checkpoint: str = "sam_vit_b_01ec64.pth"):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size

        # --- Build original SAM model ---
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam = sam.to(self.device).eval()
        self.encoder = sam.image_encoder

        # Enable gradients for training (SAM freezes encoder by default)
        for param in self.encoder.parameters():
            param.requires_grad = True

        # Preprocessing transforms
        self.to_pil = transforms.ToPILImage()
        self.prep = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        
        # SAM normalization: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        # These are in BGR order but applied to RGB, scaled to [0,1] range
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats (SAM uses these)
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 1024, 1024] (already batched, resized & normalized)
        Returns: feature map [B, 256, 64, 64] for vit_b
        """
        x = x.to(self.device)
        fmap = self.encoder(x)
        return fmap

    def preprocess(self, x: torch.Tensor, normalize: bool = True, use_pil: bool = False) -> torch.Tensor:
        """
        Resize and optionally normalize for SAM.
        x: [B, 3, H, W] in [0,1] or [0,255]
        Returns: [B, 3, 1024, 1024] ready for SAM
        
        Args:
            use_pil: If True, uses PIL (breaks gradients). If False, uses pure PyTorch (preserves gradients)
        """
        # Handle grayscale
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError("preprocess expects [B,3,H,W]")

        if use_pil:
            # PIL path - breaks gradients but sometimes needed for inference
            xs = []
            for i in range(x.size(0)):
                xi = x[i]
                if xi.dtype.is_floating_point:
                    xi_01 = xi.clamp(0, 1)
                    pil = self.to_pil((xi_01 * 255).byte().cpu())
                else:
                    pil = self.to_pil(xi.cpu())
                xs.append(self.prep(pil))
            x_proc = torch.stack(xs, dim=0).to(self.device)
        else:
            # Pure PyTorch path - preserves gradients for training
            # Ensure input is in [0, 1] range
            if not x.dtype.is_floating_point:
                x = x.float() / 255.0
            x = x.clamp(0, 1)
            
            # Resize using bilinear interpolation (differentiable)
            x_proc = F.interpolate(x, size=(self.image_size, self.image_size), 
                                  mode='bilinear', align_corners=False)
            x_proc = x_proc.to(self.device)
        
        # Apply SAM normalization
        if normalize:
            x_proc = self.normalize(x_proc)
        
        return x_proc
    
class SAMMed2DFeatExtractor(nn.Module):
    """
    Single-class version of your original code.
    - Builds SAM-Med2D (vit_b by default) and keeps only the image encoder.
    - forward expects x: [B, 3, 256, 256] (already preprocessed/normalized upstream).
    - If you want a quick PIL-based resize to 256 and tensorize, call .preprocess().
    """
    def __init__(self,
                 model_type: str = "vit_b",
                 image_size: int = 256,
                 sam_checkpoint: str = "/SAN/medic/IQT_ScoreMatching/SAM-Med2D/sam-med2d_b.pth",
                 encoder_adapter: bool = True):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size

        # --- Build args for SAM-Med2D (kept as in your original) ---
        args = SimpleNamespace(
            image_size=image_size,
            sam_checkpoint=sam_checkpoint,
            encoder_adapter=encoder_adapter
        )

        # --- Instantiate SAM-Med2D model and grab the encoder ---
        builder = sam_model_registry[model_type]
        sam = builder(args)                  # keeps your original builder(args) pattern
        sam = sam.to(self.device).eval()
        self.encoder = sam.image_encoder     # <- what your original wrapper wrapped

        # (Optional) simple PIL-based preprocessor similar to your old snippet
        self.to_pil = transforms.ToPILImage()
        self.prep = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),           # -> [3,256,256] in [0,1]
        ])

    #@torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 256, 256]  (already batched & resized)
        Returns: feature map [B, C, Hf, Wf]
        """
        x = x.to(self.device)
        out = self.encoder(x)

        # Handle dict vs tensor outputs (kept exactly as you had)
        if isinstance(out, dict):
            fmap = out.get('feat', out.get('x', out))
        else:
            fmap = out
        return fmap

    def preprocess(self, x: torch.Tensor, normalize: bool = False, use_pil: bool = False) -> torch.Tensor:
        """
        Resize and optionally normalize for SAM.
        x: [B, 3, H, W] in [0,1] or [0,255]
        Returns: [B, 3, 1024, 1024] ready for SAM
        
        Args:
            use_pil: If True, uses PIL (breaks gradients). If False, uses pure PyTorch (preserves gradients)
        """
        # Handle grayscale
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)

        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError("preprocess expects [B,3,H,W]")

        if use_pil:
            # PIL path - breaks gradients but sometimes needed for inference
            xs = []
            for i in range(x.size(0)):
                xi = x[i]
                if xi.dtype.is_floating_point:
                    xi_01 = xi.clamp(0, 1)
                    pil = self.to_pil((xi_01 * 255).byte().cpu())
                else:
                    pil = self.to_pil(xi.cpu())
                xs.append(self.prep(pil))
            x_proc = torch.stack(xs, dim=0).to(self.device)
        else:
            # Pure PyTorch path - preserves gradients for training
            # Ensure input is in [0, 1] range
            if not x.dtype.is_floating_point:
                x = x.float() / 255.0
            x = x.clamp(0, 1)
            
            # Resize using bilinear interpolation (differentiable)
            x_proc = F.interpolate(x, size=(self.image_size, self.image_size), 
                                  mode='bilinear', align_corners=False)
            x_proc = x_proc.to(self.device)
        
        # Apply SAM normalization
        #if normalize:
        #    x_proc = self.normalize(x_proc)
        
        return x_proc


class PatchSSIMLoss(torch.nn.Module):
    def __init__(self, patch_size=32, eps=1e-6):
        """
        Initializes the Patch-based Mutual Information Loss module.
        Args:
            patch_size (int): Size of each patch (square patches are assumed).
            num_bins (int): Number of bins for the histogram.
            eps (float): Small value to avoid division by zero.
        """
        super(PatchSSIMLoss, self).__init__()
        self.patch_size = patch_size
        self.eps = eps
        self.SSIM = StructuralSimilarityIndexMeasure(data_range=1.0, win_size=11, win_sigma=1.5, K=(0.01, 0.03))

    def compute_patch_ssim(self, patch_x, patch_y):
        """
        Computes ssim for a single patch.
        Args:
            patch_x (Tensor): Patch from image X (batch_size, 1, H, W).
            patch_y (Tensor): Patch from image Y (batch_size, 1, H, W).
        Returns:
            mi (Tensor): Mutual information for the patch (scalar).
        """
        ssim_val = self.SSIM(patch_x, patch_y)
        return ssim_val

    def forward(self, x, y):
        """
        Computes the patch-based mutual information loss between two images.
        Args:
            x (Tensor): Image 1 (batch_size, 1, H, W), normalized to [0, 1].
            y (Tensor): Image 2 (batch_size, 1, H, W), normalized to [0, 1].
        Returns:
            loss (Tensor): Patch-based mutual information loss (scalar).
        """
        batch_size, _, height, width = x.size()
        ssim_loss = []
        num_patches = 0

        for i in range(0, height, self.patch_size):
            for j in range(0, width, self.patch_size):
                patch_x = x[:, :, i:i+self.patch_size, j:j+self.patch_size]
                patch_y = y[:, :, i:i+self.patch_size, j:j+self.patch_size]

                if patch_x.size(2) == self.patch_size and patch_x.size(3) == self.patch_size:
                    ssim_loss.append(1.0 - self.compute_patch_ssim(patch_x, patch_y))
                    num_patches += 1

        ssim_loss = torch.stack(ssim_loss)
        ssim_loss = torch.linalg.norm(ssim_loss)
        return ssim_loss

class CannyEdgeLoss(torch.nn.Module):
    def __init__(self, low_threshold=0.1, high_threshold=0.3):
        """
        Initializes the EdgeLoss module with thresholds suitable for normalized images.
        Args:
            low_threshold (float): Lower threshold for Canny edge detection (normalized scale 0–1).
            high_threshold (float): Higher threshold for Canny edge detection (normalized scale 0–1).
        """
        super(CannyEdgeLoss, self).__init__()
        self.low_threshold = int(low_threshold * 255)  # Scale for OpenCV (expects 0-255)
        self.high_threshold = int(high_threshold * 255)  # Scale for OpenCV (expects 0-255)

        # Example 3x3 Sobel kernels:
        self.sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).reshape(1,1,3,3).to(device)
        self.sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32).reshape(1,1,3,3).to(device)

    def sobel_edge_magnitude(self,image):
        """Compute a differentiable approximation of edge magnitude via Sobel."""
        #print data type
        grad_x = F.conv2d(image, self.sobel_x, padding=1)
        grad_y = F.conv2d(image, self.sobel_y, padding=1)
        # Edge magnitude
        edges = torch.sqrt(grad_x**2 + grad_y**2 + 1e-7)
        return edges

    def forward(self, image_A, image_B):
        """
        Compute edge loss between two normalized images.
        Input:
            image_A: PyTorch tensor (batch_size, C, H, W), normalized to [0, 1].
            image_B: PyTorch tensor (batch_size, C, H, W), normalized to [0, 1].
        Output:
            loss: Scalar edge loss.
        """
        
        # Check if the input images are normalized to [0, 1]
        if image_A.max() > 1:
            image_A = image_A / 2.0
        if image_B.max() > 1:
            image_B = image_B / 2.0

        edges_A = self.sobel_edge_magnitude(image_A.to(torch.float32))
        edges_B = self.sobel_edge_magnitude(image_B.to(torch.float32))
        difference = edges_A - edges_B
        loss = torch.linalg.norm(difference)  # sum of all differences
        return loss

class TotalVariationLoss(nn.Module):
    def __init__(self):
        """
        Initialize the Total Variation Loss module.
        """
        super(TotalVariationLoss, self).__init__()

    def forward(self, image):
        """
        Compute the Total Variation (TV) Loss for an image.
        
        Args:
            image (torch.Tensor): Input image of shape (B, C, H, W),
                                  where B is batch size, C is the number of channels,
                                  H is the height, and W is the width.
        
        Returns:
            tv_loss (torch.Tensor): Scalar tensor representing the total variation loss.
        """
        # Compute horizontal and vertical differences
        diff_h = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])  # Horizontal differences
        diff_w = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])  # Vertical differences
        
        # Sum the absolute differences
        tv_loss = diff_h.sum() + diff_w.sum()
        
        return tv_loss
    
class PerceptualLoss(nn.Module):
    def __init__(self, model_name="resnet18", layers=("layer3",), device='cuda'):
        """
        Perceptual Loss using intermediate features of a Timm pre-trained model.
        Args:
            model_name: Name of the model to use from Timm (e.g., "resnet18").
            layers: Tuple of layer names from which to extract intermediate features.
        """
        super(PerceptualLoss, self).__init__()
        self.layers = layers
        self.device = device

        # replace your create_model call with:
        def _to_idx(tag):
            if isinstance(tag, int):
                return tag
            if isinstance(tag, str) and tag.startswith("layer"):
                return int(tag.replace("layer", ""))
            raise ValueError(f"Unsupported layer spec: {tag}")

        self.indices = tuple(_to_idx(x) for x in layers)  # e.g., ("layer3",) -> (3,)
        # Load a pre-trained model from Timm
        self.feature_extractor = create_model(model_name, pretrained=True, features_only=True, out_indices=self.indices)
        self.layer_names = [f"layer{i}" for i in range(len(self.feature_extractor.feature_info))]

        # Freeze the feature extractor's parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval().to(self.device)
        self.feature_extractor.float()

    def forward(self, input_image, target_image):
        """
        Calculate perceptual loss between input and target images.
        Args:
            input_image: Tensor of shape (B, 1, H, W), normalized to [0, 1].
            target_image: Tensor of shape (B, 1, H, W), normalized to [0, 1].
        Returns:
            loss: Scalar perceptual loss.
        """
        input_image = input_image.to(dtype=torch.float32)
        target_image = target_image.to(dtype=torch.float32)
        # Convert grayscale (1-channel) images to 3-channel by repeating
        input_image = input_image.repeat(1, 3, 1, 1)  # Shape: (B, 3, H, W)
        target_image = target_image.repeat(1, 3, 1, 1)  # Shape: (B, 3, H, W)
        
        # Normalize images using ImageNet statistics
        #input_image = (input_image - 0.485) / 0.229
        #target_image = (target_image - 0.485) / 0.229

        # Ensure input and target are resized to match the model's expected input size
        input_image = interpolate(input_image, size=(256, 256), mode="bilinear", align_corners=False)
        target_image = interpolate(target_image, size=(256, 256), mode="bilinear", align_corners=False)

        # Extract intermediate features
        input_features = self.feature_extractor(input_image)
        target_features = self.feature_extractor(target_image)
       
        feat_shape = input_features[0]
        print(feat_shape.shape)

        # Calculate perceptual loss using L2 norm of feature differences
        #loss = 0.0
        #for input_feat, target_feat in zip(input_features, target_features):
        #    loss += torch.mean((input_feat - target_feat) ** 2) #loss += torch.linalg.norm(input_feat - target_feat)
        diff = target_features[0] - input_features[0]

        return diff, feat_shape

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
        self.ssim = PatchSSIMLoss(patch_size=32)
        self.perceptual_loss = PerceptualLoss(device=device)
        self.tv = TotalVariationLoss()
        self.edge_ls = CannyEdgeLoss(low_threshold=0.05, high_threshold=0.1)
        self.cnt = 0
        self.apply_mask = True
        self.hvm_flag = False
        if self.apply_mask == False:
            self.hvm_flag = True
        self.checkpoint = "/SAN/medic/IQT_ScoreMatching/SAM-Med2D/sam-med2d_b.pth" #"/SAN/medic/IQT_ScoreMatching/SAM-Med2D/sam-med2d_b.pth" 
#"/SAN/medic/IQT_ScoreMatching/segment-anything/sam_vit_b_01ec64.pth" #"/SAN/medic/IQT_ScoreMatching/SAM-Med2D/sam-med2d_b.pth" 
        self.imagenet_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.imagenet_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

        #self.feat_extractor = TimmFeatExtractor(
        #    model_name=self.checkpoint
        #)

        #self.feat_extractor = SAMFeatExtractor(
        #    model_type="vit_b",
        #    image_size=1024,
        #    sam_checkpoint=self.checkpoint)

        self.feat_extractor = SAMMed2DFeatExtractor(
            model_type="vit_b",
            image_size=256,
            sam_checkpoint=self.checkpoint,
            encoder_adapter=True)

    def edge_loss(self, image1, image2):
        """
        Compute edge loss between two 1-channel images using Sobel filters.
        The loss is the mean squared error (MSE) between the edge maps of the two images.

        Args:
            image1 (torch.Tensor): First image, shape (B, 1, H, W), values in [0, 1].
            image2 (torch.Tensor): Second image, shape (B, 1, H, W), values in [0, 1].

        Returns:
            torch.Tensor: Scalar edge loss.
        """

        def sobel_filter(image):
            # Sobel filters for edge detection
            sobel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], device=image.device).view(1, 1, 3, 3)
            sobel_x = sobel_x.to(image.dtype)
            sobel_y = torch.tensor([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]], device=image.device).view(1, 1, 3, 3)
            sobel_y = sobel_y.to(image.dtype)
            # Apply Sobel filters in x and y directions
            grad_x = F.conv2d(image, sobel_x, padding=1)  # Gradient in x-direction
            grad_y = F.conv2d(image, sobel_y, padding=1)  # Gradient in y-direction
            
            # Compute gradient magnitude
            grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            return grad_magnitude

        # Compute edge maps for both images
        edge_map1 = sobel_filter(image1.to(torch.float32))
        edge_map2 = sobel_filter(image2.to(torch.float32))

        # Compute mean squared error between edge maps
        difference = edge_map1 - edge_map2
        loss = torch.linalg.norm(difference)
        #loss = F.mse_loss(edge_map1, edge_map2)
        return loss    

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)

    def downsample_mask(self, mask, size):
        """
        mask: [B,1,H,W] (binary 0/1)
        size: (Hf, Wf) target feature map size
        returns: [B,1,Hf,Wf] (binary 0/1)
        """
        mask_ds = F.interpolate(mask.float(), size=size, mode="nearest")
        return mask_ds

    def mask_to_token_binary(self, mask, feat_hw):
        Hf, Wf = feat_hw
        B, _, H, W = mask.shape
        kH = (H + Hf - 1) // Hf  # ceil kernel size if not divisible
        kW = (W + Wf - 1) // Wf
        binmask = F.max_pool2d(mask.float(), kernel_size=(kH,kW), stride=(H//Hf, W//Wf), padding=0)
        # Fallback to adaptive if shapes are awkward:
        binmask = (F.adaptive_max_pool2d(mask.float(), (Hf, Wf)) > 0).float()
        return binmask
    
    def standardized_effect_size(self, hallu_vals, clean_vals):
        """Compute effect size after z-scoring"""
        # Pool all data to get normalization parameters
        all_vals = np.concatenate([hallu_vals, clean_vals])
        # Calculate mean and std excluding background if needed
        mean_all = np.mean(all_vals)
        std_all = np.std(all_vals)
        
        # Z-score normalize
        hallu_z = (hallu_vals - mean_all) / std_all
        clean_z = (clean_vals - mean_all) / std_all
        
        # Now compute Cohen's d on normalized data
        pooled_std = np.sqrt((np.var(hallu_z) + np.var(clean_z)) / 2)
        d = (np.mean(hallu_z) - np.mean(clean_z)) / pooled_std
        
        return np.abs(d)
    
    def hallucination_verification_legacy(self, pred,  pred_measurement, measurement, gt, hallu_mask, extrinsic, **kwargs):
        # Compute metrics to verify hallucination quality: For intrinsic we compute effect size on measurement consistency and for extrinsic we compute image-space metrics

        # Extract hallucinated regions from predicted measurement and ground truth
        pred_measurement_hallu = pred_measurement[hallu_mask == 1]
        measurement_hallu = measurement[hallu_mask == 1]

        if extrinsic is None:
            # Intrinsic hallucination verification
            effect = self.standardized_effect_size(pred_measurement_hallu.detach().cpu().numpy(), measurement_hallu.detach().cpu().numpy())
            qualified = effect > 0.1 # Threshold for significant effect size
        else:
            # Extrinsic hallucination verification
            gt_hallu = gt[hallu_mask == 1]
            pred_hallu = pred[hallu_mask == 1]
            effect_measurement = self.standardized_effect_size(pred_measurement_hallu.detach().cpu().numpy(), measurement_hallu.detach().cpu().numpy())
            effect_image = self.standardized_effect_size(pred_hallu.detach().cpu().numpy(), gt_hallu.detach().cpu().numpy())
            qualified = (effect_measurement < 0.1) and (effect_image > 0.05) # Measurement should be consistent while image should have significant change
            if qualified == False:
                print(f"HVM pass failed, measurement: {effect_measurement} image: {effect_image}")
            else:
                print(f"HVM passed, measurement: {effect_measurement} image: {effect_image}")

        return qualified

    def hallucination_verification(self, pred, pred_measurement, measurement, gt, hallu_mask, extrinsic, **kwargs):
        # Identify individual boxes in the mask
        # label() assigns a unique integer (1, 2, 3...) to each non-overlapping box
        labeled_mask, num_features = label(hallu_mask.detach().cpu().numpy())
    
        if num_features == 0:
            return False

        # Iterate through each individual box
        for i in range(1, num_features + 1):
            # Create a mask specifically for the current box
            current_box_mask = (labeled_mask == i)
        
            # Extract regions for the specific box
            pm_box = pred_measurement.detach().cpu().numpy()[current_box_mask]
            m_box = measurement.detach().cpu().numpy()[current_box_mask]
        
            if extrinsic is None:
                # Intrinsic verification per box
                effect = self.standardized_effect_size(pm_box, m_box)
                if not (effect > 0.1):
                    print(f"Box {i} failed Intrinsic HVM: effect {effect:.4f}")
                    return False
            else:
                # Extrinsic verification per box
                gt_box = gt.detach().cpu().numpy()[current_box_mask]
                p_box = pred.detach().cpu().numpy()[current_box_mask]
            
                eff_m = self.standardized_effect_size(pm_box, m_box)
                eff_i = self.standardized_effect_size(p_box, gt_box)
            
                # Condition: Measurement consistency AND significant image-space change
                qualified_box = (eff_m < 0.1) and (eff_i > 0.05)
            
                if not qualified_box:
                    print(f"Box {i} failed Extrinsic HVM, meas: {eff_m:.4f} img: {eff_i:.4f}")
                    return False
                else:
                    print(f"Box {i} passed, meas: {eff_m:.4f} img: {eff_i:.4f}")

        # If the loop completes without returning False, all boxes passed
        return True

    def grad_and_value(self, x_prev, x_0_hat, measurement, t, hallu_mask, extrinsic, semantic, hallu_weight, **kwargs):
        if self.noiser.__name__ == 'gaussian':

            x_0_hat = x_0_hat.clamp(min=0., max=2.)  ############# Ensure x_0_hat is clamped safely
            #tv_loss = self.tv(x_0_hat)
            # Augment measurement to generate intrinsic hallucination

            pred_measurement = self.operator.forward(x_0_hat, **kwargs)          

            if t == 0 and self.apply_mask == True:
                self.hvm_flag = self.hallucination_verification(pred=x_0_hat, pred_measurement=pred_measurement, measurement=measurement, gt=extrinsic, hallu_mask=hallu_mask, extrinsic=extrinsic, **kwargs)
                print(f"Step {self.cnt}: Hallucination verification passed?: {self.hvm_flag}")                   
                
            difference = measurement - pred_measurement
            # 2. Calculate the norm PER SAMPLE (dim=1, 2, 3)
            # Reshape to [B, -1] makes it easy to take the norm across all non-batch dims
            #norm = torch.linalg.norm(difference.view(difference.shape[0], -1), dim=1)
            norm = torch.linalg.norm(difference)

            if self.apply_mask == True: 
                # #Soft thresholding with 0.1
                nohallu_mask = 1.0 - hallu_mask
                difference_nohallu = nohallu_mask * difference
                difference_hallu = hallu_mask * difference

                norm_hallu = torch.linalg.norm(difference_hallu)
                norm_nohallu = torch.linalg.norm(difference_nohallu)           
            else:
                norm_hallu = norm.clone()
                norm_nohallu = norm.clone()            

            if extrinsic is not None:
                extrinsic_difference = extrinsic - x_0_hat
                extrinsic_difference_hallu = hallu_mask * extrinsic_difference
                extrinsic_loss = torch.linalg.norm(extrinsic_difference_hallu)
            # Extrinsic loss
            if semantic is not None:
                # Feature loss

                # --- Tensor-only preprocessing to keep graph ---
                # (1) resize to 256x256
                x_cat = torch.cat((x_0_hat, semantic), dim=0)             # [2B,1,H,W] or [2,1,H,W]
                x_res = torch.nn.functional.interpolate(x_cat, size=(256, 256), mode='bilinear', align_corners=False)

                if x_res.shape[1] == 1:
                    # (2) to 3-channel
                    x_rgb = x_res.repeat(1, 3, 1, 1)                              # [2B,3,256,256]
                else:
                    x_rgb = x_res

                # (3) (optional) SAM-style pixel-space normalization if your encoder expects it
                # mean/std in pixel space; if your inputs are 0..1, scale to 0..255 first.
                # Uncomment if needed and ensure consistency with your encoder init.
                if x_rgb.max() > 1.0:
                    x_rgb /= 2.0

                # (4) extract features
                if 'dinov3' in self.checkpoint:
                    x_rgb = F.interpolate(x_rgb, size=(224, 224), mode='bilinear', align_corners=False)  # [2B,3,224,224]
                    x_norm = (x_rgb - self.imagenet_mean) / self.imagenet_std  # [2B,3,256,256]
                elif 'med2d' in self.checkpoint:
                    x_norm = x_rgb
                elif 'sam' in self.checkpoint:
                    x_norm = self.feat_extractor.preprocess(x_rgb, normalize=True)  # [2B,3,1024,1024]
                else:
                    # Raise error if unknown checkpoint
                    raise ValueError(f"Unknown checkpoint type: {self.checkpoint}")
                    
                feats  = self.feat_extractor(x_norm.to(dtype=torch.float32))
                assert feats != None, f"feats is None: {feats}"

                feat_pred, feat_gt = torch.chunk(feats, 2, dim=0)
 
                # 3) Downsample the mask to feature resolution (don’t hardcode 16x16)
                Hf, Wf = feat_pred.shape[-2], feat_pred.shape[-1]
                H, W = x_0_hat.shape[-2], x_0_hat.shape[-1]
                feat_pred = F.interpolate(feat_pred, size=(H, W), mode='bilinear', align_corners=False)
                feat_gt = F.interpolate(feat_gt, size=(H, W), mode='bilinear', align_corners=False)
                #mask_ds = self.mask_to_token_binary(hallu_mask.float(), feat_hw=(Hf, Wf)) #torch.nn.functional.interpolate(hallu_mask.float(),size=(Hf, Wf),mode='nearest')             # [B,1,Hf,Wf]
               
                if hallu_mask.shape[1] == 3:
                    hallu_mask = hallu_mask[:,[0]]
                #assert feats.shape[1] != 2, f"mask only has {torch.unique(mask_ds)}"
                #feat_pred = feat_pred * mask_ds
                #feat_gt = feat_gt * mask_ds
                #feats = feat_pred - feat_gt
                feats = feat_pred - feat_gt
                semantic_loss = feats * hallu_mask #torch.linalg.norm(feat_diff)
                semantic_loss = torch.linalg.norm(feats) #torch.mean((semantic_loss)**2)
                assert semantic_loss > 0, f"Loss is zero: {torch.unique(mask_ds)}, {semantic_loss}"
            
            if (extrinsic is not None) and self.apply_mask: # Extrinsic hallucination
                self.retain_graph=True
                norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, retain_graph=self.retain_graph)[0]
                self.retain_graph=False
                extrinsic_grad = torch.autograd.grad(outputs=extrinsic_loss, inputs=x_0_hat, retain_graph=self.retain_graph)[0]
                self.retain_graph=False
                semantic_grad = torch.autograd.grad(outputs=semantic_loss, inputs=x_prev, retain_graph=self.retain_graph)[0] if semantic is not None else None
                norm_grad_hallu = None
            
            elif (extrinsic is None) and self.apply_mask: # Intrinsic hallucination
                self.retain_graph=True
                norm_grad = torch.autograd.grad(outputs=norm_nohallu, inputs=x_prev, retain_graph=self.retain_graph)[0]
                norm_grad_hallu = torch.autograd.grad(outputs=norm_hallu, inputs=x_prev, retain_graph=self.retain_graph)[0]
                self.retain_graph=False
                semantic_grad = torch.autograd.grad(outputs=semantic_loss, inputs=x_prev, retain_graph=self.retain_graph)[0] if semantic is not None else None
                extrinsic_grad = None
            else:   # No hallucination
                norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
                semantic_grad = None
                norm_grad_hallu = None
                extrinsic_grad = None
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
        
        return [norm_grad, semantic_grad, norm_grad_hallu, extrinsic_grad], [norm_nohallu, norm_hallu, pred_measurement]

   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        self.scale_original = self.scale
        self.alpha = self.scale_original
        self.best_ls = 1000
        self.loss_df = pd.DataFrame(columns=['Time', 'Loss'])
        self.cnt = 0
        self.csv_file = "./line_search_stepsize.csv"
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Step Size"])
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, t, patch_idx, extrinsic, semantic, **kwargs):
        
        self.idx_lst = patch_idx
        self.mu = 0.1
        self.hallu_weight = 0.007 #0.08 #05 #0.06 #0.01 # if self.mu < 1 else self.sclale * self.mu * 1.
        self.semantic_weight = 0.007
        self.tv_coeff = 5e-5
        if t == 999:
            self.v_momentum = 0                      # velocity image, same shape as x
            self.beta_momentum = 0.9                 # momentum factor

        self.hallu_mask = torch.zeros_like(measurement).to(device)
        for i in range(len(self.idx_lst)):
            self.hallu_mask[0,:,self.idx_lst[i][0], self.idx_lst[i][1]] = 1.0
        
        # Compute initial gradient and norm
        #x_prev.requires_grad_()  ###############
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, t=t, hallu_mask = self.hallu_mask, extrinsic=extrinsic, hallu_weight=self.hallu_weight, semantic=semantic, **kwargs)

        if (self.apply_mask == True):
            norm_grad, semantic_grad, norm_grad_hallu, extrinsic_grad = norm_grad[0], norm_grad[1], norm_grad[2], norm_grad[3]
            norm_nohallu, norm_hallu, pred_measurement = norm[0], norm[1], norm[2]
        else:
            norm_grad = norm_grad[0]
            norm_nohallu = norm[1]
            norm_hallu = norm[2]
        #Add t and norm to dataframe
        self.loss_df = self.loss_df.append({'Time': t.cpu().numpy()[0], 'NoHalluLoss': norm_nohallu.detach().cpu().numpy(), 'HalluLoss': norm_hallu.detach().cpu().numpy()}, ignore_index=True)
        extrinsic_flag = True if extrinsic is not None else False
        self.loss_df.to_csv(f'measurement_loss_timestep_extrinsic_{extrinsic_flag}_{self.hallu_weight}_final.csv')
       
        #Reduce step size to avoid artifact 
        if t[0] <= 5:
            self.scale = 0.2
        else:
            self.scale = self.scale_original


        if (self.apply_mask) and (extrinsic is None):# and (t <= 800): #Intrinsic hallucination
            #t=10 for mri t=30 for mvtec
            if t[0] <= 5:
                self.hallu_weight = 0.0
                self.mu = 0.0
                self.tv_grad = 0.0
                self.semantic_weight = 0.0
                #print("APPLY MASK INVOKED!")

            self.nohallu_mask = 1.0 - self.hallu_mask
            assert len(torch.unique(self.nohallu_mask)) == 2, f"There is no hallucination mask unique: {torch.unique(self.nohallu_mask,return_counts=True)}"

            if semantic is None:
                self.semantic_weight = 0.0
                x_t = x_t - norm_grad * self.scale * self.nohallu_mask + norm_grad_hallu * self.hallu_weight * self.hallu_mask
            else:
                x_t = x_t - norm_grad * self.scale * self.nohallu_mask + norm_grad_hallu * self.hallu_weight * self.hallu_mask + semantic_grad * self.semantic_weight * self.hallu_mask 
        
        elif (extrinsic is not None) and (self.apply_mask): #Extrinsic hallucination

            self.nohallu_mask = 1.0 - self.hallu_mask
            #t=10 for mri t=30 for brain
            if t[0] <= 10:
                self.hallu_weight = 0.0
                self.semantic_weight = 0.0
                # Apply soft thresholding to reduce step size in hallucination regions
                self.nohallu_mask[self.nohallu_mask == 0.0] = 0.1   
         
            assert len(torch.unique(self.nohallu_mask)) == 2, f"There is no hallucination mask unique: {torch.unique(self.nohallu_mask,return_counts=True)}"
            #norm_grad_hallu = norm_grad / (norm_grad.norm() + 1e-12)       # unit-norm to control scale
            #self.v_momentum = self.beta_momentum * self.v_momentum + (1 - self.beta_momentum) * norm_grad_hallu    # momentum update
            if t[0] <= 10:
                x_t = x_t - norm_grad * self.scale * self.nohallu_mask
            else:
                if semantic is None:
                    self.semantic_weight = 0.0
                    semantic_grad = 0.0
                x_t = x_t - norm_grad * self.scale + semantic_grad * self.semantic_weight * self.hallu_mask +extrinsic_grad * self.hallu_weight * self.hallu_mask 
        
        else:
            x_t -= norm_grad * self.scale

        return x_t, norm, self.hvm_flag

@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
