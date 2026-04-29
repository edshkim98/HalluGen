import sys
import glob
import time
import warnings
from types import SimpleNamespace
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.fft
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision import transforms

# SAM-Med2D must be on the Python path; set SAM_MED2D_PATH env var or adjust below.
sys.path.append('/SAN/medic/IQT_ScoreMatching/SAM-Med2D')
from segment_anything import sam_model_registry

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Evaluation hyperparameters
_TEMPERATURE = 0.005
_LOWPASS_RADIUS = 50

# ---------------------------------------------------------------------------
# SAM-Med2D feature extractor (initialised once at module load)
# ---------------------------------------------------------------------------

_sam_args = SimpleNamespace(
    image_size=256,
    sam_checkpoint="/SAN/medic/IQT_ScoreMatching/SAM-Med2D/sam-med2d_b.pth",
    encoder_adapter=True,
)
_sam = sam_model_registry["vit_b"](_sam_args)
_sam.to(device).eval()


class SAMMed2DFeatExtractor(torch.nn.Module):
    def __init__(self, vision_encoder):
        super().__init__()
        self.encoder = vision_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W] -> [B, C, H', W']"""
        out = self.encoder(x)
        if isinstance(out, dict):
            return out.get('feat', out.get('x', out))
        return out


feat_extractor = SAMMed2DFeatExtractor(_sam.image_encoder).to(device)


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def get_sammed2d_features_batch(images: torch.Tensor) -> torch.Tensor:
    """Extract features for a batch [B, 3, H, W] -> [B, C, H', W']."""
    if images.shape[-2:] != (256, 256):
        images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
    with torch.no_grad():
        return feat_extractor(images)


def get_sammed2d_features(patch: torch.Tensor) -> torch.Tensor:
    """Legacy single-patch feature extraction [3, 32, 32] -> [C, H', W']."""
    to_pil = transforms.ToPILImage()
    prep = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    pil = to_pil((patch.clamp(0, 1) * 255).to(torch.uint8))
    x = prep(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        return feat_extractor(x).squeeze(0)


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

class SpatialFeatureMetric:
    """Feature-based similarity metric supporting patch and whole-image modes."""

    def __init__(
        self,
        sigma: float = 10.0,
        calc_sigma: bool = False,
        distance: str = 'cosine',
        topk: int = 64,
        use_patches: bool = True,
    ):
        self.sigma = sigma
        self.calc_sigma = calc_sigma
        self.distance = distance
        self.topk = topk
        self.reg_eps = 1e-6
        self.use_patches = use_patches

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        if self.use_patches:
            return self._compute_patch_distance(pred, gt)
        return self._compute_spatial_distance_map(pred, gt)

    def _compute_patch_distance(self, pred_patch: torch.Tensor, gt_patch: torch.Tensor) -> torch.Tensor:
        f_pred = get_sammed2d_features(pred_patch)
        f_gt = get_sammed2d_features(gt_patch)
        C, H, W = f_pred.shape
        return self._compute_distance_vectors(f_pred.view(C, H * W).T, f_gt.view(C, H * W).T)

    def _compute_spatial_distance_map(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Returns [B, H', W'] spatial distance map."""
        f_pred = get_sammed2d_features_batch(pred)
        f_gt = get_sammed2d_features_batch(gt)
        B, C, H, W = f_pred.shape

        if self.distance == 'cosine':
            sim = (F.normalize(f_pred, dim=1) * F.normalize(f_gt, dim=1)).sum(dim=1)
            return 1.0 - sim

        if self.distance == 'euclidean':
            return torch.sqrt(((f_pred - f_gt) ** 2).sum(dim=1) + 1e-8)

        if self.distance == 'l2':
            return torch.norm(f_pred - f_gt, p=2, dim=1)

        if self.distance == 'energy':
            x = f_pred.permute(0, 2, 3, 1).reshape(B * H * W, C)
            y = f_gt.permute(0, 2, 3, 1).reshape(B * H * W, C)
            d_xy = torch.cdist(x.unsqueeze(0), y.unsqueeze(0)).squeeze(0).mean(dim=1)
            d_xx = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0).mean(dim=1)
            d_yy = torch.cdist(y.unsqueeze(0), y.unsqueeze(0)).squeeze(0).mean(dim=1)
            return (2 * d_xy - d_xx - d_yy).view(B, H, W)

        if self.distance == 'mahalanobis':
            dist_maps = []
            for i in range(B):
                x = f_pred[i].view(C, -1).T
                y = f_gt[i].view(C, -1).T
                mu = y.mean(dim=0, keepdim=True)
                cov = (y - mu).T @ (y - mu) / (y.size(0) - 1)
                cov += torch.eye(C, device=cov.device) * self.reg_eps
                diff = x - mu
                m2 = (diff @ torch.inverse(cov) * diff).sum(dim=1)
                dist_maps.append(m2.view(H, W))
            return torch.stack(dist_maps)

        if self.distance == 'mmd':
            dist_maps = []
            for i in range(B):
                x = f_pred[i].view(C, -1).T
                y = f_gt[i].view(C, -1).T
                if self.calc_sigma:
                    self.sigma = compute_sigma_median(x, y)
                val = compute_mmd(x, y, self.sigma)
                dist_maps.append(torch.full((H, W), val.item(), device=pred.device))
            return torch.stack(dist_maps)

        raise ValueError(f"Unknown distance: {self.distance}")

    def _compute_distance_vectors(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.distance == 'euclidean':
            return torch.sqrt(torch.sum((x - y) ** 2))
        if self.distance == 'cosine':
            return (1.0 - F.cosine_similarity(x, y)).mean()
        if self.distance == 'l2':
            return torch.linalg.norm(x - y)
        if self.distance == 'energy':
            d_xy = torch.cdist(x, y, p=2).mean()
            d_xx = torch.cdist(x, x, p=2).mean()
            d_yy = torch.cdist(y, y, p=2).mean()
            return 2 * d_xy - d_xx - d_yy
        if self.distance == 'mahalanobis':
            mu = y.mean(dim=0, keepdim=True)
            y_c = y - mu
            cov = y_c.T @ y_c / (y.size(0) - 1)
            cov += torch.eye(cov.size(0), device=cov.device) * self.reg_eps
            diff = x - mu
            m2 = (diff @ torch.inverse(cov) * diff).sum(dim=1)
            return torch.topk(m2, min(self.topk, m2.numel()), largest=True).values.mean()
        if self.distance == 'mmd':
            if self.calc_sigma:
                self.sigma = compute_sigma_median(x, y)
            return compute_mmd(x, y, self.sigma)
        raise ValueError(f"Unknown distance: {self.distance}")


# ---------------------------------------------------------------------------
# Aggregation and metric computation
# ---------------------------------------------------------------------------

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
    use_patches: bool = False,
):
    """Compute spatial feature metric with optional patch extraction.

    When ``use_patches=False`` (default), features are extracted from the full
    image and aggregated. When ``use_patches=True``, the image is tiled into
    overlapping patches before extraction.
    """
    if ref.shape != test.shape:
        raise ValueError("ref and test must have the same shape")

    B, C, H, W = ref.shape

    if not use_patches:
        with torch.no_grad():
            dist_map = metric_fn(test, ref)           # [B, H', W']
        patch_scores = dist_map.view(B, -1)
        image_scores = _aggregate_scores(patch_scores, agg, temperature, worst_k, invert)
        if not return_map:
            return image_scores
        heatmap = dist_map.unsqueeze(1)
        if map_mode == 'upsampled':
            heatmap = F.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=False)
        return image_scores, heatmap

    stride = stride or patch_size // 2
    ref_p = F.unfold(ref, kernel_size=patch_size, stride=stride)
    tst_p = F.unfold(test, kernel_size=patch_size, stride=stride)
    P = ref_p.shape[-1]
    n_h = (H - patch_size) // stride + 1
    n_w = (W - patch_size) // stride + 1

    ref_p = ref_p.transpose(1, 2).reshape(B * P, C, patch_size, patch_size)
    tst_p = tst_p.transpose(1, 2).reshape_as(ref_p)

    with torch.no_grad():
        patch_scores = metric_fn(tst_p, ref_p).view(B, P)

    image_scores = _aggregate_scores(patch_scores, agg, temperature, worst_k, invert)
    if not return_map:
        return image_scores

    patch_map = patch_scores.view(B, 1, n_h, n_w)
    if map_mode == 'coarse':
        return image_scores, patch_map
    if map_mode == 'upsampled':
        return image_scores, F.interpolate(patch_map, size=(H, W), mode='nearest')
    raise ValueError("map_mode must be 'coarse' or 'upsampled'")


def _aggregate_scores(
    patch_scores: torch.Tensor,
    agg: str,
    temperature: float,
    worst_k: int,
    invert: bool,
) -> torch.Tensor:
    B, P = patch_scores.shape
    if agg == 'mean':
        return patch_scores.mean(dim=1)
    if agg == 'max':
        return patch_scores.max(dim=1).values
    if agg == 'min':
        return patch_scores.min(dim=1).values
    if agg == 'worstk':
        sorted_sc, _ = patch_scores.sort(dim=1, descending=not invert)
        return sorted_sc[:, :min(worst_k, P)].mean(dim=1)
    if agg == 'softmax':
        sign = -1 if invert else 1
        weights = F.softmax(sign * patch_scores / temperature, dim=1)
        return (weights * patch_scores).sum(dim=1)
    raise ValueError(f"Unknown aggregation: {agg}")


# ---------------------------------------------------------------------------
# Kernel / MMD helpers
# ---------------------------------------------------------------------------

def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-torch.cdist(x, y, p=2) ** 2 / (2 * sigma ** 2))


def compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    N, M = x.size(0), y.size(0)
    K_xx = rbf_kernel(x, x, sigma)
    K_yy = rbf_kernel(y, y, sigma)
    K_xy = rbf_kernel(x, y, sigma)
    return (
        (K_xx.sum() - K_xx.diag().sum()) / (N * (N - 1))
        + (K_yy.sum() - K_yy.diag().sum()) / (M * (M - 1))
        - 2 * K_xy.mean()
    )


def compute_sigma_median(U: torch.Tensor, V: torch.Tensor, exclude_self: bool = True) -> float:
    d_uu = torch.cdist(U, U).view(-1)
    d_vv = torch.cdist(V, V).view(-1)
    d_uv = torch.cdist(U, V).view(-1)
    if exclude_self:
        d_uu = d_uu[d_uu > 0]
        d_vv = d_vv[d_vv > 0]
    return torch.cat([d_uu, d_vv, d_uv]).median().item()


# ---------------------------------------------------------------------------
# Low-pass filter helpers
# ---------------------------------------------------------------------------

def create_circular_lowpass_mask(shape, cutoff_radius: float, device='cpu') -> torch.Tensor:
    H, W = shape
    y = torch.arange(H, dtype=torch.float32, device=device) - H // 2
    x = torch.arange(W, dtype=torch.float32, device=device) - W // 2
    Y, X = torch.meshgrid(y, x, indexing='ij')
    return (torch.sqrt(X ** 2 + Y ** 2) <= cutoff_radius).float()


def create_rectangular_lowpass_mask(shape, cutoff_height: int, cutoff_width: int, device='cpu') -> torch.Tensor:
    H, W = shape
    mask = torch.zeros((H, W), dtype=torch.float32, device=device)
    ch, cw = H // 2, W // 2
    mask[
        max(0, ch - cutoff_height // 2):min(H, ch + cutoff_height // 2),
        max(0, cw - cutoff_width // 2):min(W, cw + cutoff_width // 2),
    ] = 1.0
    return mask


def apply_lowpass_filter(
    image: torch.Tensor,
    cutoff_height: Optional[int] = None,
    cutoff_width: Optional[int] = None,
    cutoff_radius: Optional[float] = None,
    device: str = 'cpu',
) -> torch.Tensor:
    """Apply rectangular or circular low-pass filter via FFT.

    Args:
        image: (C, H, W) or (H, W) tensor.
        cutoff_radius: radius for circular mask.
        cutoff_height / cutoff_width: dimensions for rectangular mask.
    """
    squeeze_output = image.dim() == 2
    if squeeze_output:
        image = image.unsqueeze(0)

    image = image.to(device)
    C, H, W = image.shape

    if cutoff_radius is not None:
        mask = create_circular_lowpass_mask((H, W), cutoff_radius, device)
    elif cutoff_height is not None and cutoff_width is not None:
        mask = create_rectangular_lowpass_mask((H, W), cutoff_height, cutoff_width, device)
    else:
        raise ValueError("Provide cutoff_radius or both cutoff_height and cutoff_width")

    filtered = []
    for c in range(C):
        freq = torch.fft.fftshift(torch.fft.fft2(image[c]))
        filtered.append(torch.fft.ifft2(torch.fft.ifftshift(freq * mask)).real)

    result = torch.stack(filtered, dim=0)
    return result.squeeze(0) if squeeze_output else result


# ---------------------------------------------------------------------------
# Shared preprocessing and scoring helpers
# ---------------------------------------------------------------------------

def _to_rgb_filtered(t: torch.Tensor, cutoff_radius: int = _LOWPASS_RADIUS) -> torch.Tensor:
    """Expand single-channel image to RGB and apply circular low-pass filter.

    Input:  [1, 1, H, W] clamped to [0, 1].
    Output: [1, 3, H, W].
    """
    t = t.repeat(1, 3, 1, 1)
    return apply_lowpass_filter(t.squeeze(0), cutoff_radius=cutoff_radius, device=device).unsqueeze(0)


def _make_metric() -> SpatialFeatureMetric:
    return SpatialFeatureMetric(distance='cosine', use_patches=False)


def _shafe_score(gt: torch.Tensor, pred: torch.Tensor, metric: SpatialFeatureMetric) -> torch.Tensor:
    return patchwise_metric(
        gt, pred, metric_fn=metric,
        use_patches=False, agg='softmax', invert=False, temperature=_TEMPERATURE,
    )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def standardized_effect_size(hallu_vals: np.ndarray, clean_vals: np.ndarray):
    """Compute Cohen's d and per-sample standardized deviations after z-scoring."""
    all_vals = np.concatenate([hallu_vals, clean_vals])
    mean_all, std_all = np.mean(all_vals), np.std(all_vals)
    hallu_z = (hallu_vals - mean_all) / std_all
    clean_z = (clean_vals - mean_all) / std_all
    pooled_std = np.sqrt((np.var(hallu_z) + np.var(clean_z)) / 2)
    d_all = (hallu_z - np.mean(clean_z)) / pooled_std
    d = (np.mean(hallu_z) - np.mean(clean_z)) / pooled_std
    return d, d_all, np.std(np.abs(d_all))


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def evaluate_metrics(preds, gts, lrs, blur: bool = False, noise: bool = False) -> np.ndarray:
    metric = _make_metric()
    score_lst, times = [], []

    for pred_path, gt_path in zip(preds, gts):
        pred = torch.from_numpy(np.load(pred_path)).float().unsqueeze(0).to(device)
        gt = torch.from_numpy(np.load(gt_path.replace('pred', 'gt'))).float().unsqueeze(0).to(device)

        pred = pred.clamp(0., 1.)
        gt = gt.clamp(0., 1.)

        if blur:
            pred = transforms.GaussianBlur(kernel_size=5, sigma=(1.5, 1.5))(pred)
        if noise:
            pred = pred + torch.randn_like(pred) * 0.01

        pred = _to_rgb_filtered(pred)
        gt = _to_rgb_filtered(gt)

        t0 = time.time()
        score = _shafe_score(gt, pred, metric)
        times.append(time.time() - t0)
        score_lst.append(score.cpu().numpy())

    scores = np.array(score_lst).squeeze()
    times = np.array(times)
    print(f"Average inference time: {times.mean():.3f}s")
    print(f"Average SHAFE over {len(preds)} images: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores


def z_score_sensitivity(preds, gts, mean: float, std: float) -> list:
    metric = _make_metric()
    z_scores_lst = []

    for pred_path, gt_path in zip(preds, gts):
        pred = torch.from_numpy(np.load(pred_path)).float().unsqueeze(0).to(device)
        gt = torch.from_numpy(np.load(gt_path)).float().unsqueeze(0).to(device)

        pred = _to_rgb_filtered(pred.clamp(0., 1.))
        gt = _to_rgb_filtered(gt.clamp(0., 1.))

        score = _shafe_score(gt, pred, metric)
        z_scores_lst.append(-1.0 * (score.cpu().numpy() - mean) / (std + 1e-6))

    return z_scores_lst


def auc_score_test(preds_nohallu, preds_intrinsic, preds_extrinsic, gts):
    metric = _make_metric()
    nohallu_scores, intrinsic_scores, extrinsic_scores = [], [], []

    for i in range(len(preds_nohallu)):
        pred_nohallu = _to_rgb_filtered(
            torch.from_numpy(np.load(preds_nohallu[i])).float().unsqueeze(0).to(device).clamp(0., 1.)
        )
        pred_intr = _to_rgb_filtered(
            torch.from_numpy(np.load(preds_intrinsic[i])).float().unsqueeze(0).to(device).clamp(0., 1.)
        )
        pred_extr = _to_rgb_filtered(
            torch.from_numpy(np.load(preds_extrinsic[i])).float().unsqueeze(0).to(device).clamp(0., 1.)
        )
        gt = _to_rgb_filtered(
            torch.from_numpy(np.load(gts[i])).float().unsqueeze(0).to(device).clamp(0., 1.)
        )

        nohallu_scores.append(_shafe_score(gt, pred_nohallu, metric).cpu().numpy())
        intrinsic_scores.append(_shafe_score(gt, pred_intr, metric).cpu().numpy())
        extrinsic_scores.append(_shafe_score(gt, pred_extr, metric).cpu().numpy())

    intrinsic_scores = np.array(intrinsic_scores).squeeze()
    extrinsic_scores = np.array(extrinsic_scores).squeeze()
    nohallu_scores = np.array(nohallu_scores).squeeze()

    labels_intr = np.concatenate([np.ones(len(intrinsic_scores)), np.zeros(len(nohallu_scores))])
    labels_extr = np.concatenate([np.ones(len(extrinsic_scores)), np.zeros(len(nohallu_scores))])
    intrinsic_auc = roc_auc_score(labels_intr, np.concatenate([intrinsic_scores, nohallu_scores]))
    extrinsic_auc = roc_auc_score(labels_extr, np.concatenate([extrinsic_scores, nohallu_scores]))
    return intrinsic_auc, extrinsic_auc


def auc_score_test_real(preds, gts, labels) -> float:
    metric = _make_metric()
    shafe_lst, labels_lst, records = [], [], []

    for i in range(len(labels)):
        pred = _to_rgb_filtered(
            torch.from_numpy(np.load(labels['pred_path'].iloc[i])).float().unsqueeze(0).to(device).clamp(0., 1.)
        )
        gt = _to_rgb_filtered(
            torch.from_numpy(np.load(labels['gt_path'].iloc[i])).float().unsqueeze(0).to(device).clamp(0., 1.)
        )
        label = labels['has_hallucination'].iloc[i]

        score = _shafe_score(gt, pred, metric)
        shafe_lst.append(score.cpu().numpy())
        labels_lst.append(label)
        records.append({
            'pred_path': labels['pred_path'].iloc[i],
            'gt_path': labels['gt_path'].iloc[i],
            'label': label,
            'shafe_score': float(score.cpu().numpy()),
        })

    pd.DataFrame(records).to_csv('shafe_scores.csv', index=False)
    auc = roc_auc_score(np.array(labels_lst).squeeze(), np.array(shafe_lst).squeeze())
    return auc


def quality_correctness_tradeoff(preds_nohallu, preds_intrinsic, preds_extrinsic, gts):
    metric = _make_metric()
    freq_lst = [128, 96, 64, 48, 32, 24, 16, 12, 4]
    intrinsic_win, extrinsic_win = [], []

    for freq in freq_lst:
        intrinsic_lst, extrinsic_lst, nohallu_lst = [], [], []

        for i in range(len(preds_intrinsic)):
            pred_nohallu_raw = torch.from_numpy(np.load(preds_nohallu[i])).float().unsqueeze(0).to(device)
            pred_intr_raw = torch.from_numpy(np.load(preds_intrinsic[i])).float().unsqueeze(0).to(device)
            pred_extr_raw = torch.from_numpy(np.load(preds_extrinsic[i])).float().unsqueeze(0).to(device)
            gt_raw = torch.from_numpy(np.load(gts[i])).float().unsqueeze(0).to(device)

            # Apply frequency-specific blur before the standard low-pass
            pred_nohallu_freq = apply_lowpass_filter(
                pred_nohallu_raw.clamp(0., 1.).squeeze(0),
                cutoff_radius=freq, device=device,
            ).unsqueeze(0)

            pred_nohallu = _to_rgb_filtered(pred_nohallu_freq)
            pred_intr = _to_rgb_filtered(pred_intr_raw.clamp(0., 1.))
            pred_extr = _to_rgb_filtered(pred_extr_raw.clamp(0., 1.))
            gt = _to_rgb_filtered(gt_raw.clamp(0., 1.))

            intrinsic_lst.append(_shafe_score(gt, pred_intr, metric).cpu().numpy())
            extrinsic_lst.append(_shafe_score(gt, pred_extr, metric).cpu().numpy())
            nohallu_lst.append(_shafe_score(gt, pred_nohallu, metric).cpu().numpy())

        intrinsic_scores = np.array(intrinsic_lst).squeeze()
        extrinsic_scores = np.array(extrinsic_lst).squeeze()
        nohallu_scores = np.array(nohallu_lst).squeeze()
        intrinsic_win.append((intrinsic_scores > nohallu_scores).mean())
        extrinsic_win.append((extrinsic_scores > nohallu_scores).mean())

    freq_arr = np.array(freq_lst, dtype=float)
    order = np.argsort(freq_arr)
    freq_sorted = freq_arr[order]
    freq_norm = (freq_sorted - freq_sorted.min()) / (freq_sorted.max() - freq_sorted.min())
    intrinsic_auc = np.trapz(np.array(intrinsic_win)[order], freq_norm)
    extrinsic_auc = np.trapz(np.array(extrinsic_win)[order], freq_norm)
    print(f"SBC AUC -> Intrinsic: {intrinsic_auc:.4f}  Extrinsic: {extrinsic_auc:.4f}")


def severity_correlation(preds_hallu, gts, hallu_masks):
    metric = _make_metric()
    score_severity, severity_labels = [], []

    for pred_path, gt_path, mask_path in zip(preds_hallu, gts, hallu_masks):
        pred_raw = torch.from_numpy(np.load(pred_path)).float().unsqueeze(0).to(device)
        gt_raw = torch.from_numpy(np.load(gt_path)).float().unsqueeze(0).to(device)
        hallu_mask = np.load(mask_path)
        hallu_mask_bool = torch.from_numpy((hallu_mask > 0).astype(np.float32)).to(device)

        # Severity computed on raw (unclamped) values to capture full error magnitude
        severity = torch.sum(((gt_raw - pred_raw) ** 2) * hallu_mask_bool)
        assert severity != 0, f"Severity is zero for {pred_path}"

        hallu_count = np.sum(hallu_mask)
        severity_multiplier = 3 if hallu_count > 2 * 20 ** 2 else 1
        severity_level = severity * severity_multiplier

        pred = _to_rgb_filtered(pred_raw.clamp(0., 1.))
        gt = _to_rgb_filtered(gt_raw.clamp(0., 1.))

        score = _shafe_score(gt, pred, metric)
        score_severity.append(score.cpu().numpy())
        severity_labels.append(severity_level.cpu().numpy())

    score_severity = np.array(score_severity).squeeze()
    severity_labels = np.array(severity_labels).squeeze()

    unique, counts = np.unique(severity_labels, return_counts=True)
    print(f"Severity distribution: {dict(zip(unique, counts))}")

    corr, _ = spearmanr(score_severity, severity_labels)
    print(f"Spearman correlation: {corr:.4f}")
    return corr


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fname = 'Intrinsic_hallu_0.005_interpolation_t200_multi_1.4entropy_hvm'
    fname2 = 'Extrinsic_hallu_0.007_medsam_0.007_interpolation_t200_multi_1.4entropy_hvm'
    fname3 = 'dps_nohallu_t200'
    BASE_DIR = '/SAN/medic/IQT_ScoreMatching/skim/HalluBench_final_full/final'

    np.random.seed(0)
    idx = np.random.choice(500, size=100, replace=False)
    print(f"SOFTMAX selection | temperature={_TEMPERATURE}, radius={_LOWPASS_RADIUS}")

    pred_extrinsics = np.array(glob.glob(f'{BASE_DIR}/{fname2}/*/pred*.npy'))[idx]
    pred_intrinsics = [p.replace(fname2, fname) for p in pred_extrinsics]
    pred_dps_nohallus = [p.replace(fname, fname3) for p in pred_intrinsics]
    gts = [p.replace('pred', 'gt').replace(fname, 'synth_gt') for p in pred_intrinsics]
    lrs = np.array(glob.glob(f'{BASE_DIR}/{fname3}/*/lr*.npy'))[idx]
    hallu_masks_intrinsic = [p.replace('pred', 'hallu_mask') for p in pred_intrinsics]
    hallu_masks_extrinsic = [p.replace('pred', 'hallu_mask') for p in pred_extrinsics]

    print(f"Files: intrinsic={len(pred_intrinsics)}, extrinsic={len(pred_extrinsics)}, nohallu={len(pred_dps_nohallus)}")

    print("Evaluating Intrinsic...")
    scores_intrinsic = evaluate_metrics(pred_intrinsics, gts, lrs)
    print("Evaluating Extrinsic...")
    scores_extrinsic = evaluate_metrics(pred_extrinsics, gts, lrs)
    print("Evaluating DPS no-hallu...")
    scores_dps_nohallu = evaluate_metrics(pred_dps_nohallus, gts, lrs)

    effect_i, _, std_i = standardized_effect_size(scores_intrinsic, scores_dps_nohallu)
    effect_e, _, std_e = standardized_effect_size(scores_extrinsic, scores_dps_nohallu)
    print(f"Effect size -> intrinsic: {effect_i:.4f} (±{std_i:.4f}), extrinsic: {effect_e:.4f} (±{std_e:.4f})")

    y_true = np.array([1] * len(scores_intrinsic) + [1] * len(scores_extrinsic) + [0] * len(scores_dps_nohallu))
    y_score = np.concatenate([scores_intrinsic, scores_extrinsic, scores_dps_nohallu])
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_idx = (tpr - fpr).argmax()
    print(f"AUC: {auc:.4f}  FNR: {1 - tpr[j_idx]:.4f}  FPR: {fpr[j_idx]:.4f}  threshold: {thresholds[j_idx]:.4f}")

    print("Quality-Correctness Tradeoff...")
    quality_correctness_tradeoff(pred_dps_nohallus, pred_intrinsics, pred_extrinsics, gts)
    scores_noisy = evaluate_metrics(pred_dps_nohallus, gts, lrs, noise=True)
    print(f"Noise robustness -> intrinsic win: {(scores_intrinsic > scores_noisy).mean():.4f}, "
          f"extrinsic win: {(scores_extrinsic > scores_noisy).mean():.4f}")

    print("AUC Score Analysis...")
    auc_i, auc_e = auc_score_test(pred_dps_nohallus, pred_intrinsics, pred_extrinsics, gts)
    print(f"Intrinsic AUC: {auc_i:.4f}  Extrinsic AUC: {auc_e:.4f}")

    print("Severity Correlation...")
    corr_i = severity_correlation(pred_intrinsics, gts, hallu_masks_intrinsic)
    corr_e = severity_correlation(pred_extrinsics, gts, hallu_masks_extrinsic)
    print(f"Intrinsic Spearman: {corr_i:.4f}  Extrinsic: {corr_e:.4f}")
    print("Done.")
