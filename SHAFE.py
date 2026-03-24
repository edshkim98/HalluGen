import torch
import torch.nn as nn
import torch.fft
from torch.functional import F
import timm

from typing import Literal, Optional, Tuple


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

class ETHFeature:
    """
    ETH-Feature metric using whole-image feature extraction.
    No patching required - uses spatial feature maps directly.
    """
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',  # or your custom model
        distance: str = 'cosine',
        aggregation: str = 'softmax',
        temperature: float = 0.005,
        topk: int = 64,
        device: str = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model_name: timm model name (e.g., 'vit_base_patch16_224', 'resnet50')
            distance: Distance metric ('cosine', 'euclidean', 'l2', 'energy', 'mahalanobis')
            aggregation: Aggregation method ('softmax', 'mean', 'max', 'min', 'worstk')
            temperature: Temperature for softmax aggregation (lower = more focus on worst)
            topk: Number of worst patches for 'worstk' aggregation or mahalanobis
            device: Device to run on
        """
        self.distance = distance
        self.aggregation = aggregation
        self.temperature = temperature
        self.model_name = model_name
        self.topk = topk
        self.device = device
        self.reg_eps = 1e-6
        self.cosine = nn.CosineSimilarity(dim=1)

        # Initialize feature extractor using timm
        if 'sam' in model_name.lower() or 'dinov2' in model_name.lower():
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,      # No classification head
            ).to(device).eval()
        else:            
            self.model = timm.create_model(
                model_name,
                pretrained=True,
                features_only=True,  # Return intermediate feature maps
                out_indices=[1,2]     # Use last layer features 1,2, 3
            ).to(device).eval()
        
        # For custom models (e.g., MedSAM), replace with:
        # self.model = load_medsam_encoder().to(device).eval()
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps from images.
        
        Args:
            x: Input images (N, C, H, W)
        
        Returns:
            features: Spatial feature maps (N, C', H', W')
        """
        with torch.no_grad():
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
                    feats = torch.zeros_like(features[-1]) #[]
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
    
    def compute_distance_map(
        self, 
        feat_pred: torch.Tensor, 
        feat_gt: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial distance map between feature maps.
        
        Args:
            feat_pred: Prediction features (N, C, H', W')
            feat_gt: Ground truth features (N, C, H', W')
        
        Returns:
            distance_map: Spatial distances (N, H', W')
        """
        N, C, H, W = feat_pred.shape
        
        if self.distance == 'cosine':
            # Cosine distance per spatial location
            feat_pred_norm = F.normalize(feat_pred, dim=1)  # (N, C, H', W')
            feat_gt_norm = F.normalize(feat_gt, dim=1)
            # Cosine similarity
            sim = (feat_pred_norm * feat_gt_norm).sum(dim=1)  # (N, H', W')
            dist_map = 1 - sim  # Cosine distance (higher = worse)
            
        elif self.distance == 'euclidean':
            # Euclidean distance per spatial location
            diff = feat_pred - feat_gt
            dist_map = torch.sqrt((diff ** 2).sum(dim=1) + 1e-8)  # (N, H', W')
            
        elif self.distance == 'l2':
            # L2 norm per spatial location
            diff = feat_pred - feat_gt
            dist_map = torch.norm(diff, p=2, dim=1)  # (N, H', W')
            
        elif self.distance == 'energy':
            # Energy distance per spatial location
            # Reshape for cdist: (N*H'*W', C)
            x = feat_pred.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H'*W', C)
            y = feat_gt.permute(0, 2, 3, 1).reshape(-1, C)
            
            # Compute pairwise distances (expensive!)
            d_xy = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2).squeeze(0).mean(dim=1)
            d_xx = torch.cdist(x.unsqueeze(0), x.unsqueeze(0), p=2).squeeze(0).mean(dim=1)
            d_yy = torch.cdist(y.unsqueeze(0), y.unsqueeze(0), p=2).squeeze(0).mean(dim=1)
            
            energy = 2 * d_xy - d_xx - d_yy
            dist_map = energy.view(N, H, W)
            
        elif self.distance == 'mahalanobis':
            # Mahalanobis distance per spatial location
            # Fit distribution on GT features
            x = feat_pred.permute(0, 2, 3, 1).reshape(N, -1, C)  # (N, H'*W', C)
            y = feat_gt.permute(0, 2, 3, 1).reshape(N, -1, C)
            
            dist_maps = []
            for i in range(N):
                # Compute mean and covariance from GT
                mu = y[i].mean(dim=0, keepdim=True)  # (1, C)
                y_centered = y[i] - mu
                cov = (y_centered.t() @ y_centered) / (y[i].size(0) - 1)  # (C, C)
                cov += torch.eye(C, device=cov.device) * self.reg_eps
                inv_cov = torch.inverse(cov)
                
                # Compute Mahalanobis distance for pred
                diff = x[i] - mu  # (H'*W', C)
                m2 = (diff @ inv_cov * diff).sum(dim=1)  # (H'*W',)
                dist_maps.append(m2.view(H, W))
            
            dist_map = torch.stack(dist_maps, dim=0)  # (N, H', W')
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
        
        return dist_map
    
    def aggregate_distances(
        self, 
        dist_map: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Aggregate spatial distances into scalar scores.
        
        Args:
            dist_map: Spatial distance map (N, H', W')
        
        Returns:
            scores: Aggregated scores per image (N,)
            weights: Attention weights if using softmax, else None (N, H', W')
        """
        N, H, W = dist_map.shape
        dist_flat = dist_map.view(N, -1)  # (N, H'*W')
        
        if self.aggregation == 'mean':
            scores = dist_flat.mean(dim=1)
            weights = None
            
        elif self.aggregation == 'max':
            scores = dist_flat.max(dim=1).values
            weights = None
            
        elif self.aggregation == 'min':
            scores = dist_flat.min(dim=1).values
            weights = None
            
        elif self.aggregation == 'worstk':
            # Take mean of top-k worst (highest distance) locations
            k = min(self.topk, dist_flat.size(1))
            topk_vals = torch.topk(dist_flat, k, dim=1, largest=True).values
            scores = topk_vals.mean(dim=1)
            weights = None
            
        elif self.aggregation == 'softmax':
            # Attention-based aggregation (ETH)
            # Higher distance = worse, so use positive temperature
            weights = F.softmax(dist_flat / self.temperature, dim=1)  # (N, H'*W')
            scores = (weights * dist_flat).sum(dim=1)  # (N,)
            weights = weights.view(N, H, W)  # Reshape for visualization
            
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return scores, weights
    
    def __call__(
        self, 
        pred: torch.Tensor, 
        gt: torch.Tensor,
        return_map: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute ETH-Feature metric.
        
        Args:
            pred: Prediction images (N, C, H, W)
            gt: Ground truth images (N, C, H, W)
            return_map: Whether to return distance map and attention weights
        
        Returns:
            scores: ETH scores per image (N,)
            dist_map: Distance map if return_map=True (N, H', W')
            weights: Attention weights if return_map=True and agg='softmax' (N, H', W')
        """
        if pred.shape != gt.shape:
            raise ValueError("pred and gt must have same shape")
    
        if 'sam' in self.model_name.lower():
            # Interpoalte to 1024x1024 for SAM-based models
            pred = F.interpolate(pred, size=(1024, 1024), mode='bilinear', align_corners=False)
            gt = F.interpolate(gt, size=(1024, 1024), mode='bilinear', align_corners=False)
        elif 'dino' in self.model_name.lower() or 'resnetaa' in self.model_name.lower():
            # Interpolate to 224x224 for DINOv2-based models
            pred = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)
            gt = F.interpolate(gt, size=(224, 224), mode='bilinear', align_corners=False)

        pred = pred.to(self.device)
        gt = gt.to(self.device)
        
        # Extract features
        feat_pred = self.extract_features(pred)  # (N, C', H', W')
        feat_gt = self.extract_features(gt)
        
        # Compute spatial distance map
        dist_map = self.compute_distance_map(feat_pred, feat_gt)  # (N, H', W')
        
        # Aggregate into scalar scores
        scores, weights = self.aggregate_distances(dist_map)  # (N,), (N, H', W') or None
        
        if return_map:
            return scores, dist_map, weights
        else:
            return scores
        

class SHAFE(nn.Module):
    def __init__(self, model_name='resnetaa50d.d_in12k', device='cpu'):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self.eth_feature = ETHFeature(
            model_name=self.model_name,
            distance='cosine',
            aggregation='softmax',
            temperature=0.02,
            device=device
        )
        if self.model_name == 'resnetaa50d.d_in12k':
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        else:
            imagenet_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.register_buffer('mean', imagenet_mean)
        self.register_buffer('std', imagenet_std)

        # Cache lowpass masks keyed by (H, W) to avoid recomputation
        self._mask_cache: dict = {}

    def _get_lowpass_mask(self, H: int, W: int, cutoff_radius: float) -> torch.Tensor:
        key = (H, W, cutoff_radius)
        if key not in self._mask_cache:
            self._mask_cache[key] = create_circular_lowpass_mask((H, W), cutoff_radius, self.device)
        return self._mask_cache[key]

    def _apply_lowpass_batched(self, images: torch.Tensor, cutoff_radius: float = 60) -> torch.Tensor:
        """Vectorized lowpass filter over (B, C, H, W) — no per-channel loop."""
        B, C, H, W = images.shape
        mask = self._get_lowpass_mask(H, W, cutoff_radius)  # (H, W)
        freq = torch.fft.fft2(images)                        # (B, C, H, W)
        freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))
        freq_filtered = freq_shifted * mask                  # broadcasts over B, C
        freq_filtered = torch.fft.ifftshift(freq_filtered, dim=(-2, -1))
        return torch.fft.ifft2(freq_filtered).real

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        gt = torch.clamp(gt, 0.0, 1.0)
        pred = torch.clamp(pred, 0.0, 1.0)

        gt = self._apply_lowpass_batched(gt, cutoff_radius=60)
        pred = self._apply_lowpass_batched(pred, cutoff_radius=60)

        gt = gt.repeat(1, 3, 1, 1)
        pred = pred.repeat(1, 3, 1, 1)

        # Apply ImageNet normalisation before feature extraction
        gt = (gt - self.mean) / self.std
        pred = (pred - self.mean) / self.std

        return self.eth_feature(pred, gt, return_map=False)


