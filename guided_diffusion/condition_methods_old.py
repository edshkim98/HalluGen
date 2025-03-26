from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
import torch.nn as nn
from timm import create_model
from torch.nn.functional import interpolate


__CONDITIONING_METHOD__ = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, model_name="resnet18", layers=("layer2", "layer3"), device='cuda'):
        """
        Perceptual Loss using intermediate features of a Timm pre-trained model.
        Args:
            model_name: Name of the model to use from Timm (e.g., "resnet18").
            layers: Tuple of layer names from which to extract intermediate features.
        """
        super(PerceptualLoss, self).__init__()
        self.layers = layers
        self.device = device

        # Load a pre-trained model from Timm
        self.feature_extractor = create_model(model_name, pretrained=True, features_only=True, out_indices=tuple(range(len(layers))))
        self.layer_names = [f"layer{i}" for i in range(len(self.feature_extractor.feature_info))]

        # Freeze the feature extractor's parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.to(self.device)

    def forward(self, input_image, target_image):
        """
        Calculate perceptual loss between input and target images.
        Args:
            input_image: Tensor of shape (B, 1, H, W), normalized to [0, 1].
            target_image: Tensor of shape (B, 1, H, W), normalized to [0, 1].
        Returns:
            loss: Scalar perceptual loss.
        """
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

        # Calculate perceptual loss using L2 norm of feature differences
        loss = 0.0
        for input_feat, target_feat in zip(input_features, target_features):
            loss += torch.linalg.norm(input_feat - target_feat)

        return loss


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
        self.ssim = StructuralSimilarityIndexMeasure(window_size=31)
        self.perceptual_loss = PerceptualLoss(device=device)
        self.tv = TotalVariationLoss()
        self.cnt = 0
        #self.v = 0
        #self.beta = 0.9
        #self.tau = 100.0
        #self.total_time = 1000

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
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, t, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            #if t < 10:
            #    print(f"New beta: {self.beta}, Time: {t}")
            #if t == 999:
            #    self.beta = 0.9
            #    print("Beta initialized: ", self.beta)
            x_0_hat[x_0_hat < 0.] = 0.
            pred_measurement = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - pred_measurement
            norm = torch.linalg.norm(difference)
            edge_ls = self.edge_loss(measurement, pred_measurement)
            tv = self.tv(pred_measurement)
            #ssim = self.ssim(pred_measurement.type(torch.DoubleTensor), measurement.type(torch.DoubleTensor))
            # pred_measurement[pred_measurement < 0.] = 0.
            # measurement[measurement < 0.] = 0.
            #percept = self.perceptual_loss(pred_measurement.type(torch.float32), measurement.type(torch.float32))
            norm_total = norm #+ 0.1*edge_ls + 0.001*tv
            norm_grad = torch.autograd.grad(outputs=norm_total, inputs=x_prev)[0]
            #self.v = self.beta * self.v + norm_grad
            #self.t = 1. - (t/self.total_time)
            #self.beta = self.beta*torch.exp(-self.t/self.tau)
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
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

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, t, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, t=t, **kwargs)
        x_t -= norm_grad * self.scale
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
