#!/usr/bin/env python3
# Verification script for hybrid Grad-CAM + LayerCAM method
# Testing if fusion improves over individual methods

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Tuple
import time

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
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
    def generate_cam(self, image, target_class=None):
        self.model.eval()
        image = image.clone().requires_grad_(True)
        output = self.model(image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        return cam

class LayerCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
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
    def generate_cam(self, image, target_class=None):
        self.model.eval()
        image = image.clone().requires_grad_(True)
        output = self.model(image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()
        positive_gradients = torch.relu(self.gradients)
        cam = torch.sum(positive_gradients * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        return cam

class HybridGradLayerCAM:
    # Combines Grad-CAM and LayerCAM using multiplicative fusion
    # alpha controls the balance between methods
    def __init__(self, model, target_layer, alpha=0.7):
        self.model = model
        self.target_layer = target_layer
        self.alpha = alpha
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

    def generate_cam(self, image, target_class=None):
        self.model.eval()
        image = image.clone().requires_grad_(True)
        output = self.model(image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # compute Grad-CAM (global average pooling)
        weights_gradcam = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam_gradcam = torch.sum(weights_gradcam * self.activations, dim=1, keepdim=True)

        # compute LayerCAM (element-wise)
        positive_gradients = torch.relu(self.gradients)
        cam_layercam = torch.sum(positive_gradients * self.activations, dim=1, keepdim=True)

        # normalize both
        cam_gradcam_norm = cam_gradcam / (cam_gradcam.max() + 1e-10)
        cam_layercam_norm = cam_layercam / (cam_layercam.max() + 1e-10)

        # multiplicative fusion
        cam_hybrid = (cam_gradcam_norm ** self.alpha) * (cam_layercam_norm ** (1 - self.alpha))

        cam_hybrid = torch.relu(cam_hybrid)
        cam_hybrid = cam_hybrid.squeeze().cpu().numpy()
        cam_hybrid = (cam_hybrid - cam_hybrid.min()) / (cam_hybrid.max() - cam_hybrid.min() + 1e-10)

        return cam_hybrid

class AttributionEvaluator:
    # evaluates attribution maps using insertion metric
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def insertion_metric(self, image, attribution, target_class, steps=10):
        # progressively add important pixels and measure score increase
        image = image.to(self.device)
        h, w = image.shape[2], image.shape[3]
        attribution_flat = cv2.resize(attribution, (w, h)).copy().flatten()
        sorted_indices = np.argsort(attribution_flat)[::-1].copy()
        scores = []
        num_pixels = len(sorted_indices)
        modified_image = torch.zeros_like(image)
        with torch.no_grad():
            output = self.model(modified_image)
            baseline_score = torch.softmax(output, dim=1)[0, target_class].item()
            scores.append(baseline_score)
        pixels_per_step = num_pixels // steps
        for step in range(1, steps + 1):
            end_idx = min(step * pixels_per_step, num_pixels)
            pixels_to_insert = sorted_indices[:end_idx].copy()
            mask = torch.zeros((h * w,), device=self.device, dtype=torch.float32)
            pixels_to_insert_tensor = torch.from_numpy(pixels_to_insert).long().to(self.device)
            mask[pixels_to_insert_tensor] = 1
            mask = mask.reshape(1, 1, h, w)
            modified_image = image * mask
            with torch.no_grad():
                output = self.model(modified_image)
                score = torch.softmax(output, dim=1)[0, target_class].item()
                scores.append(score)
        scores = np.array(scores)
        auc = np.trapezoid(scores, dx=1.0 / steps)
        return scores, auc

print("\nTesting: GradCAM vs LayerCAM vs Hybrid\n")

device = 'cpu'
model = models.resnet50(weights='IMAGENET1K_V1')
model.eval().to(device)

gradcam = GradCAM(model, model.layer4[-1])
layercam = LayerCAM(model, model.layer4[-1])
hybrid = HybridGradLayerCAM(model, model.layer4[-1], alpha=0.7)
evaluator = AttributionEvaluator(model, device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_files = list(Path('input_images').glob('*.jpg'))[:10]

results = {'GradCAM': [], 'LayerCAM': [], 'Hybrid': []}

for image_path in image_files:
    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_class = model(image).argmax(dim=1).item()

    # GradCAM
    t0 = time.time()
    cam_gc = gradcam.generate_cam(image, pred_class)
    time_gc = time.time() - t0
    _, ins_gc = evaluator.insertion_metric(image, cam_gc, pred_class, steps=10)

    # LayerCAM
    t0 = time.time()
    cam_lc = layercam.generate_cam(image, pred_class)
    time_lc = time.time() - t0
    _, ins_lc = evaluator.insertion_metric(image, cam_lc, pred_class, steps=10)

    # Hybrid
    t0 = time.time()
    cam_hy = hybrid.generate_cam(image, pred_class)
    time_hy = time.time() - t0
    _, ins_hy = evaluator.insertion_metric(image, cam_hy, pred_class, steps=10)

    results['GradCAM'].append({'ins': ins_gc, 'time': time_gc})
    results['LayerCAM'].append({'ins': ins_lc, 'time': time_lc})
    results['Hybrid'].append({'ins': ins_hy, 'time': time_hy})

    print(f"{image_path.name}:")
    print(f"  GradCAM  : Ins={ins_gc:.4f}, Time={time_gc:.3f}s")
    print(f"  LayerCAM : Ins={ins_lc:.4f}, Time={time_lc:.3f}s")
    print(f"  Hybrid   : Ins={ins_hy:.4f}, Time={time_hy:.3f}s")

    best_baseline = max(ins_gc, ins_lc)
    best_name = "GradCAM" if ins_gc > ins_lc else "LayerCAM"

    if ins_hy > best_baseline:
        print(f"  -> Hybrid is {((ins_hy-best_baseline)/best_baseline)*100:.1f}% better than {best_name}")
    else:
        print(f"  -> Hybrid is {((best_baseline-ins_hy)/best_baseline)*100:.1f}% worse than {best_name}")
    print()

avg_ins_gc = np.mean([r['ins'] for r in results['GradCAM']])
avg_ins_lc = np.mean([r['ins'] for r in results['LayerCAM']])
avg_ins_hy = np.mean([r['ins'] for r in results['Hybrid']])

print("\nResults Summary")
print(f"GradCAM  : Insertion AUC = {avg_ins_gc:.4f}")
print(f"LayerCAM : Insertion AUC = {avg_ins_lc:.4f}")
print(f"Hybrid   : Insertion AUC = {avg_ins_hy:.4f}")

best_baseline_avg = max(avg_ins_gc, avg_ins_lc)
best_baseline_name = "GradCAM" if avg_ins_gc > avg_ins_lc else "LayerCAM"

hybrid_vs_best = ((avg_ins_hy - best_baseline_avg) / best_baseline_avg) * 100

if avg_ins_hy > best_baseline_avg:
    print(f"\nSUCCESS: Hybrid is {hybrid_vs_best:.1f}% better than {best_baseline_name}")
    exit(0)
else:
    print(f"\nHybrid is {-hybrid_vs_best:.1f}% worse than {best_baseline_name}")
    print(f"GradCAM ({avg_ins_gc:.4f}) performs best on insertion metric")
    exit(1)
