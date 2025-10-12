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

sys.path.append('/SAN/medic/IQT_ScoreMatching/SAM-Med2D')

from segment_anything import sam_model_registry

warnings.filterwarnings("ignore")

__CONDITIONING_METHOD__ = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(42)

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

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optional helper if you still want the old PIL path.
        Accepts x: [B, 3, H, W] in [0,1] or [0,255]; returns [B, 3, 256, 256] in [0,1].
        Note: SAM-Med2D typically expects further normalization upstream if required.
        """

        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # Convert grayscale to 3-channel by repeating

        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError("preprocess expects [B,3,H,W]")

        # Convert each sample via PIL -> resize -> ToTensor (kept simple like your original)
        xs = []
        for i in range(x.size(0)):
            xi = x[i]
            # Ensure uint8 for PIL if input is float
            if xi.dtype.is_floating_point:
                xi_01 = xi.clamp(0, 1)
                pil = self.to_pil((xi_01 * 255).byte().cpu())
            else:
                pil = self.to_pil(xi.cpu())
            xs.append(self.prep(pil))
        x_proc = torch.stack(xs, dim=0).to(self.device)  # [B,3,256,256]
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

        self.feat_extractor = SAMMed2DFeatExtractor(
            model_type="vit_b",
            image_size=256,
            sam_checkpoint="/SAN/medic/IQT_ScoreMatching/SAM-Med2D/sam-med2d_b.pth",
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
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, t, hallu_mask, extrinsic, hallu_weight, **kwargs):
        if self.noiser.__name__ == 'gaussian':

            x_0_hat = x_0_hat.clamp(min=0., max=2.)  ############# Ensure x_0_hat is clamped safely
            tv_loss = self.tv(x_0_hat)
            # Augment measurement to generate intrinsic hallucination

            pred_measurement = self.operator.forward(x_0_hat, **kwargs)             
                
            difference = measurement - pred_measurement
            
            # #Soft thresholding with 0.1
            nohallu_mask = 1.0 - hallu_mask
            difference_nohallu = nohallu_mask * difference
            difference_hallu = hallu_mask * difference
            
            norm = torch.linalg.norm(difference)
            norm_hallu = torch.linalg.norm(difference_hallu)
            norm_nohallu = torch.linalg.norm(difference_nohallu)

            # Extrinsic loss
            if extrinsic is not None:
                extrinsic_difference = extrinsic - x_0_hat
                extrinsic_difference_hallu = hallu_mask * extrinsic_difference
                extrinsic_loss = torch.linalg.norm(extrinsic_difference_hallu)

                # Feature loss
                # ----- semantic (feature) loss over masked region -----
                #coords = torch.nonzero(hallu_mask[0,0], as_tuple=False)  # [N,2], each row is (y,x)
                #ymin, xmin = coords.min(dim=0).values
                #ymax, xmax = coords.max(dim=0).values

                #x_0_hat_mask = x_0_hat[:, :, ymin:ymax+1, xmin:xmax+1]     # keep it as an op on x_0_hat (no requires_grad_ here)
                #gt_mask     = extrinsic[:, :, ymin:ymax+1, xmin:xmax+1]

                # --- Tensor-only preprocessing to keep graph ---
                # (1) resize to 256x256
                x_cat = torch.cat((x_0_hat, extrinsic), dim=0)             # [2B,1,H,W] or [2,1,H,W]
                x_res = torch.nn.functional.interpolate(x_cat, size=(256, 256), mode='bilinear', align_corners=False)

                # (2) to 3-channel
                x_rgb = x_res.repeat(1, 3, 1, 1)                              # [2B,3,256,256]

                # (3) (optional) SAM-style pixel-space normalization if your encoder expects it
                # mean/std in pixel space; if your inputs are 0..1, scale to 0..255 first.
                # Uncomment if needed and ensure consistency with your encoder init.
                x_pix = x_rgb * 255.0
                mean = torch.tensor([123.675, 116.28, 103.53], device=x_rgb.device).view(1,3,1,1)
                std  = torch.tensor([58.395, 57.12, 57.375], device=x_rgb.device).view(1,3,1,1)
                x_norm = (x_pix - mean) / std
                feats  = self.feat_extractor(x_norm.to(dtype=torch.float32))

                #feats, feat_shape = self.perceptual_loss(x_0_hat, extrinsic)
                feat_pred, feat_gt = torch.chunk(feats, 2, dim=0)
 
                # 3) Downsample the mask to feature resolution (don’t hardcode 16x16)
                Hf, Wf = feat_pred.shape[-2], feat_pred.shape[-1]
                mask_ds = self.mask_to_token_binary(hallu_mask.float(), feat_hw=(Hf, Wf)) #torch.nn.functional.interpolate(hallu_mask.float(),size=(Hf, Wf),mode='nearest')             # [B,1,Hf,Wf]
                assert len(torch.unique(mask_ds)) == 2, f"mask only has {torch.unique(mask_ds)}"
                #feat_pred = feat_pred * mask_ds
                #feat_gt = feat_gt * mask_ds
                #feat_diff = feat_pred - feat_gt
                feats = feat_pred - feat_gt
                semantic_loss = feats * mask_ds #torch.linalg.norm(feat_diff)
                semantic_loss = torch.linalg.norm(feats) #torch.mean((semantic_loss)**2)

            if (extrinsic is not None) and self.apply_mask: # Extrinsic hallucination
                self.retain_graph=True
                norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, retain_graph=self.retain_graph)[0]
                self.retain_graph=False
                extrinsic_grad = torch.autograd.grad(outputs=extrinsic_loss, inputs=x_0_hat, retain_graph=self.retain_graph)[0]
                self.retain_graph=False
                semantic_grad = torch.autograd.grad(outputs=semantic_loss, inputs=x_prev, retain_graph=self.retain_graph)[0]
                norm_grad_hallu = None
            elif (extrinsic is None) and self.apply_mask: # Intrinsic hallucination
                self.retain_graph=True
                norm_grad = torch.autograd.grad(outputs=norm_nohallu, inputs=x_prev, retain_graph=self.retain_graph)[0]
                norm_grad_hallu = torch.autograd.grad(outputs=norm_hallu, inputs=x_prev, retain_graph=self.retain_graph)[0]
                self.retain_graph=False
                semantic_grad = None #torch.autograd.grad(outputs=tv_loss, inputs=x_prev, retain_graph=self.retain_graph)[0]
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
        self.c1 = 1e-3      # Sufficient decrease parameter
        self.c2 = 0.8       # Curvature parameter
        self.max_line_search = 5 #10  # Max iterations for line search
        self.alpha = self.scale_original
        self.best_ls = 1000
        self.loss_df = pd.DataFrame(columns=['Time', 'Loss'])
        self.cnt = 0
        self.csv_file = "./line_search_stepsize.csv"
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Step Size"])
        
    def line_search(self, x_prev, x_t, x_0_hat, measurement, norm_grad, norm, t, **kwargs):
        """
        Perform line search to find step size (alpha) satisfying Wolfe conditions.
        """
        #self.alpha = self.scale_original  # Initial step size
        alpha_min = 0.3 #1e-8  # Minimum step size
        alpha_max = 1.0 #4.0*self.scale_original  # Maximum step size
        scale = self.scale

        # Original function and gradient values
        norm_orig = norm #torch.linalg.norm(measurement - self.operator.forward(x_0_hat, **kwargs))
        grad_orig = -1*norm_grad.view(-1).dot(norm_grad.view(-1))  # Norm of the gradient (directional derivative)

        for _ in range(self.max_line_search):
            # Apply step size to get new x_t
            x_t_new = x_t - self.alpha * norm_grad
            #x_t_new.requires_grad_()  # Ensure gradient tracking #############
            
            # Compute new norm and gradient
            #with torch.no_grad():            
            if t > 0:
                x_0_hat_new = kwargs['func'](kwargs['model'], x_t_new, t-1, kwargs['clip_denoised'], kwargs['denoised_fn'], kwargs['cond_fn'], kwargs['model_kwargs'])['pred_xstart']
            
            # Ensure x_prev tracks gradients
            #x_prev = x_prev.clone().detach().requires_grad_()  #############
            
            norm_grad_new, norm_grad_hallu_new, norm_new = self.grad_and_value(x_prev=x_t_new, x_0_hat=x_0_hat_new, measurement=measurement, t=t, **kwargs)
           # assert len(torch.unique(norm_grad_new.cpu().detach())) > 1, f"Norm grad is zero: {torch.unique(norm_grad_new.cpu().detach())}"

            # Check Wolfe conditions
            # 1. Sufficient decrease (Armijo condition)
            #print("NORM New, Norm, Grad")
            #print(norm_new.detach().cpu(), norm_orig + self.c1 * self.alpha * grad_orig, grad_orig)
           
            if norm_new > norm_orig + self.c1 * self.alpha * grad_orig:
                #print("Armjiho condition not met")
                self.alpha *= 0.75  # Reduce step size
                # print(norm_new, norm_orig + self.c1 * alpha * grad_orig)
                if self.alpha < alpha_min:
                    self.alpha = alpha_min
                    break   # Break if minimum step size is reached
                continue

            # 2. Curvature condition
            grad_new = torch.abs(-1*norm_grad_new.view(-1).dot(norm_grad.view(-1)))
            #grad_new = torch.abs(-1*norm_grad_new.view(-1).dot(norm_grad.view(-1)))
            #print("Grad NEW, Grad_Orig")
            #print(torch.abs(grad_new), torch.abs(grad_orig))
            if grad_new < self.c2 * torch.abs(grad_orig):
                #print("Curvature condition not met")
                self.alpha *= 1.5  # Increase step size
                if self.alpha > alpha_max:
                    self.alpha = alpha_max
                    break   # Break if maximum step size is reached
                continue

            # If both conditions are satisfied, return the step size
            return self.alpha

        # If no suitable step size is found, return the minimum step size
        return self.alpha #alpha_min
    
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, t, patch_idx, extrinsic, **kwargs):
        
        self.idx_lst = patch_idx
        self.mu = 0.1
        self.hallu_weight = 0.001 #0.08 #05 #0.06 #0.01 # if self.mu < 1 else self.sclale * self.mu * 1.
        self.semantic_weight = 0.1
        self.tv_coeff = 5e-5
        if t == 999:
            self.v_momentum = 0                      # velocity image, same shape as x
            self.beta_momentum = 0.9                 # momentum factor

        #if (self.apply_mask is True):
        #    assert self.hallu_weight >= self.scale * self.mu, f"Hallu Step {self.hallu_weight} must be bigger than {self.scale * self.mu}"

        self.hallu_mask = torch.zeros_like(measurement).to(device)
        for i in range(len(self.idx_lst)):
            self.hallu_mask[0,0,self.idx_lst[i][0], self.idx_lst[i][1]] = 1.0
        
        # Compute initial gradient and norm
        #x_prev.requires_grad_()  ###############
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, t=t, hallu_mask = self.hallu_mask, extrinsic=extrinsic, hallu_weight=self.hallu_weight, **kwargs)

        if (self.apply_mask == True):
            norm_grad, semantic_grad, norm_grad_hallu, extrinsic_grad = norm_grad[0], norm_grad[1], norm_grad[2], norm_grad[3]
            norm_nohallu, norm_hallu, pred_measurement = norm[0], norm[1], norm[2]
        else:
            norm_grad = norm_grad[0]
            norm_nohallu = norm_grad.clone()
            norm_hallu = norm_grad.clone()
        #Add t and norm to dataframe
        self.loss_df = self.loss_df.append({'Time': t.cpu().numpy()[0], 'NoHalluLoss': norm_nohallu.detach().cpu().numpy(), 'HalluLoss': norm_hallu.detach().cpu().numpy()}, ignore_index=True)
        extrinsic_flag = True if extrinsic is not None else False
        self.loss_df.to_csv(f'measurement_loss_timestep_extrinsic_{extrinsic_flag}.csv')
        
        # #there are multiple patches, so we need to loop through them
        # self.hallu_weight = -0.1
        # for i in range(len(idx_lst)):
        #     #print("idx_lst: ", idx_lst[i][0].start, idx_lst[i][0].stop, idx_lst[i][1].start, idx_lst[i][1].stop)
        #     hallu_mask[0,0,idx_lst[i][0], idx_lst[i][1]] = self.hallu_weight #-0.1
        # #hallu_mask[0,0,idx_lst[0], idx_lst[1]] = -0.05 #-0.1

        if (self.apply_mask) and (extrinsic is None):# and (t <= 800): #Intrinsic hallucination
            if t <= 10:
                self.hallu_weight = 0.0
                self.mu = 0.0
                self.tv_grad = 0.0
                print("APPLY MASK INVOKED!")

            self.nohallu_mask = 1.0 - self.hallu_mask
            assert len(torch.unique(self.nohallu_mask)) == 2, f"There is no hallucination mask unique: {torch.unique(self.nohallu_mask,return_counts=True)}"
            # #Soft thresholding with 0.1
            #self.nohallu_mask = torch.where(self.nohallu_mask == 0.0, self.mu, self.nohallu_mask)
            #norm_grad_hallu = norm_grad / (norm_grad.norm() + 1e-12)       # unit-norm to control scale
            #self.v_momentum = self.beta_momentum * self.v_momentum + (1 - self.beta_momentum) * norm_grad_hallu    # momentum update
            x_t = x_t - norm_grad * self.scale * self.nohallu_mask + norm_grad_hallu * self.hallu_weight * self.hallu_mask
        elif (extrinsic is not None) and (self.apply_mask): #Extrinsic hallucination
            if t <= 10:
                self.hallu_weight = 0.0
                self.semantic_weight = 0.0
                
           
            #    self.mu = 0.0
            #    self.tv_grad = 0.0
            #    print("APPLY MASK INVOKED!")
            self.nohallu_mask = 1.0 - self.hallu_mask
            if t<=10:
                # Apply soft thresholding to reduce step size in hallucination regions
                self.nohallu_mask[self.nohallu_mask == 0.0] = 0.1   
         
            assert len(torch.unique(self.nohallu_mask)) == 2, f"There is no hallucination mask unique: {torch.unique(self.nohallu_mask,return_counts=True)}"
            #norm_grad_hallu = norm_grad / (norm_grad.norm() + 1e-12)       # unit-norm to control scale
            #self.v_momentum = self.beta_momentum * self.v_momentum + (1 - self.beta_momentum) * norm_grad_hallu    # momentum update
            if t <= 10:
                x_t = x_t - norm_grad * self.scale + semantic_grad * self.semantic_weight * self.hallu_mask +extrinsic_grad * self.hallu_weight * self.hallu_mask 
            else:
                x_t = x_t - norm_grad * self.scale * self.nohallu_mask
        else:
            x_t -= norm_grad * self.scale
        #if t == 0:
            #np.save(f"pred_measurement_{self.cnt}.npy", pred_measurement.detach().cpu().numpy())
         #   self.cnt += 1
        #if t > 0:
        return x_t, norm
        
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
