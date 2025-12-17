"""
Adaptive Integrated Grad-CAM
=============================
Novel attribution method that dynamically allocates integration steps
based on gradient variance and attribution stability.

Author: [Your Name]
Date: December 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import cv2


class AdaptiveIntegratedGradCAM:
    """
    Adaptive Integrated Grad-CAM combines:
    1. Integrated Gradients for faithful attribution
    2. Grad-CAM for class-discriminative localization  
    3. Adaptive sampling for computational efficiency
    
    Key Innovation: Dynamically allocate integration steps where needed most.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: nn.Module,
        min_steps: int = 10,
        max_steps: int = 100,
        variance_threshold: float = 0.1,
        convergence_threshold: float = 0.05
    ):
        """
        Args:
            model: Pretrained CNN model
            target_layer: Layer to extract gradients from (e.g., model.layer4)
            min_steps: Minimum integration steps (for efficiency)
            max_steps: Maximum integration steps (for quality)
            variance_threshold: Threshold for gradient variance
            convergence_threshold: Threshold for attribution convergence
        """
        self.model = model
        self.target_layer = target_layer
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.variance_threshold = variance_threshold
        self.convergence_threshold = convergence_threshold
        
        self.gradients = None
        self.activations = None
        self.step_allocation_history = []
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def _compute_gradient_variance(
        self, 
        gradients_list: List[torch.Tensor]
    ) -> float:
        """
        Compute variance of gradients across integration steps.
        High variance → need more steps.
        """
        if len(gradients_list) < 2:
            return float('inf')
        
        grad_stack = torch.stack(gradients_list)
        variance = torch.var(grad_stack, dim=0).mean().item()
        return variance
    
    def _compute_attribution_change(
        self,
        prev_attribution: torch.Tensor,
        curr_attribution: torch.Tensor
    ) -> float:
        """
        Compute relative change in attribution map.
        Large change → need more steps.
        """
        diff = torch.abs(curr_attribution - prev_attribution)
        relative_change = (diff.sum() / (prev_attribution.abs().sum() + 1e-10)).item()
        return relative_change
    
    def _determine_adaptive_steps(
        self,
        image: torch.Tensor,
        target_class: int,
        baseline: Optional[torch.Tensor] = None
    ) -> int:
        """
        Adaptively determine number of integration steps needed.
        
        Strategy:
        1. Start with min_steps
        2. Check gradient variance
        3. Check attribution convergence
        4. Allocate more steps if needed
        """
        if baseline is None:
            baseline = torch.zeros_like(image)
        
        # Initial coarse sampling with min_steps
        coarse_steps = self.min_steps
        gradients_list = []
        attributions_list = []
        
        for i in range(coarse_steps):
            alpha = i / (coarse_steps - 1) if coarse_steps > 1 else 0
            interpolated = baseline + alpha * (image - baseline)
            interpolated.requires_grad = True
            
            # Forward pass
            output = self.model(interpolated)
            self.model.zero_grad()
            
            # Backward pass
            target_score = output[0, target_class]
            target_score.backward(retain_graph=True)
            
            # Store gradients
            gradients_list.append(self.gradients.clone())
            
            # Compute current attribution
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            attributions_list.append(cam.clone())
        
        # Compute metrics
        grad_variance = self._compute_gradient_variance(gradients_list)
        
        # Check convergence if we have enough samples
        if len(attributions_list) >= 2:
            attr_change = self._compute_attribution_change(
                attributions_list[-2], 
                attributions_list[-1]
            )
        else:
            attr_change = float('inf')
        
        # Adaptive allocation logic
        if grad_variance > self.variance_threshold or attr_change > self.convergence_threshold:
            # Need more steps - use max_steps
            allocated_steps = self.max_steps
        else:
            # Converged quickly - use fewer steps
            allocated_steps = coarse_steps + (self.max_steps - coarse_steps) // 2
        
        # Store for analysis
        self.step_allocation_history.append({
            'gradient_variance': grad_variance,
            'attribution_change': attr_change,
            'allocated_steps': allocated_steps
        })
        
        return allocated_steps
    
    def generate_cam(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        use_adaptive: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Generate Adaptive Integrated Grad-CAM.
        
        Args:
            image: Input image tensor [1, C, H, W]
            target_class: Target class index (if None, use predicted class)
            baseline: Baseline image for integration (if None, use zeros)
            use_adaptive: Whether to use adaptive step allocation
            
        Returns:
            cam: Attribution map [H, W]
            num_steps: Number of integration steps used
        """
        self.model.eval()
        
        if baseline is None:
            baseline = torch.zeros_like(image)
        
        # Get predicted class if not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()
        
        # Determine number of steps
        if use_adaptive:
            num_steps = self._determine_adaptive_steps(image, target_class, baseline)
        else:
            num_steps = self.max_steps
        
        # Integrated gradients with allocated steps
        integrated_gradients = torch.zeros_like(self.activations)
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1) if num_steps > 1 else 0
            interpolated = baseline + alpha * (image - baseline)
            interpolated.requires_grad = True
            
            # Forward pass
            output = self.model(interpolated)
            self.model.zero_grad()
            
            # Backward pass
            target_score = output[0, target_class]
            target_score.backward(retain_graph=(i < num_steps - 1))
            
            # Accumulate gradients
            integrated_gradients += self.gradients / num_steps
        
        # Compute weighted CAM
        weights = torch.mean(integrated_gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)  # ReLU for positive attributions
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        
        return cam, num_steps
    
    def visualize(
        self,
        image: torch.Tensor,
        cam: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay CAM on original image.
        
        Args:
            image: Original image tensor [1, C, H, W]
            cam: Attribution map [H, W]
            alpha: Overlay transparency
            
        Returns:
            Overlayed image as numpy array
        """
        # Convert image to numpy
        img = image.squeeze().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        
        # Denormalize if needed (assumes ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        # Resize CAM to image size
        h, w = img.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap / 255.0
        
        # Overlay
        overlayed = alpha * heatmap + (1 - alpha) * img
        overlayed = np.clip(overlayed, 0, 1)
        
        return overlayed


class BaselineIntegratedGradCAM:
    """
    Baseline Integrated Grad-CAM with fixed number of steps.
    For comparison purposes.
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, num_steps: int = 50):
        self.model = model
        self.target_layer = target_layer
        self.num_steps = num_steps
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """Generate standard Integrated Grad-CAM with fixed steps."""
        self.model.eval()
        
        if baseline is None:
            baseline = torch.zeros_like(image)
        
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()
        
        integrated_gradients = torch.zeros_like(self.activations) if self.activations is not None else None
        
        for i in range(self.num_steps):
            alpha = i / (self.num_steps - 1) if self.num_steps > 1 else 0
            interpolated = baseline + alpha * (image - baseline)
            interpolated.requires_grad = True
            
            output = self.model(interpolated)
            self.model.zero_grad()
            
            target_score = output[0, target_class]
            target_score.backward(retain_graph=(i < self.num_steps - 1))
            
            if integrated_gradients is None:
                integrated_gradients = torch.zeros_like(self.gradients)
            integrated_gradients += self.gradients / self.num_steps
        
        weights = torch.mean(integrated_gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        
        return cam
