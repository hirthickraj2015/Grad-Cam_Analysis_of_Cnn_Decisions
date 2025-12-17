"""
Evaluation Metrics for Attribution Methods
==========================================
Implements quantitative metrics to evaluate explanation quality:
- Deletion/Insertion curves
- Average Drop/Increase
- Faithfulness metrics
- Computational efficiency

These metrics are CRITICAL for PhD-level work - they separate
your project from undergraduate visualizations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict
from tqdm import tqdm
import time


class AttributionEvaluator:
    """
    Comprehensive evaluation suite for attribution methods.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def deletion_metric(
        self,
        image: torch.Tensor,
        attribution: np.ndarray,
        target_class: int,
        steps: int = 20,
        pixel_batch_size: int = 224
    ) -> Tuple[np.ndarray, float]:
        """
        Deletion metric: progressively delete most important pixels
        and measure drop in prediction score.
        
        Better attribution → faster drop in score
        
        Returns:
            scores: Array of prediction scores at each deletion step
            auc: Area under the deletion curve (lower is better)
        """
        image = image.to(self.device)
        h, w = image.shape[2], image.shape[3]
        
        # Flatten and sort attribution
        attribution_flat = cv2.resize(attribution, (w, h)).flatten()
        sorted_indices = np.argsort(attribution_flat)[::-1].copy()  # Descending order
        
        scores = []
        num_pixels = len(sorted_indices)
        
        # Get baseline score
        with torch.no_grad():
            output = self.model(image)
            baseline_score = torch.softmax(output, dim=1)[0, target_class].item()
            scores.append(baseline_score)
        
        # Progressively delete pixels
        modified_image = image.clone()
        pixels_per_step = num_pixels // steps
        
        for step in range(1, steps + 1):
            # Delete pixels
            end_idx = min(step * pixels_per_step, num_pixels)
            pixels_to_delete = sorted_indices[:end_idx]
            
            # Create mask
            mask = torch.ones((h * w,), device=self.device)
            mask[pixels_to_delete] = 0
            mask = mask.reshape(1, 1, h, w)
            
            # Apply mask (set to mean value)
            modified_image = image * mask
            
            # Measure score
            with torch.no_grad():
                output = self.model(modified_image)
                score = torch.softmax(output, dim=1)[0, target_class].item()
                scores.append(score)
        
        scores = np.array(scores)
        auc = np.trapz(scores, dx=1.0 / steps)
        
        return scores, auc
    
    def insertion_metric(
        self,
        image: torch.Tensor,
        attribution: np.ndarray,
        target_class: int,
        steps: int = 20
    ) -> Tuple[np.ndarray, float]:
        """
        Insertion metric: progressively insert most important pixels
        into blank image and measure increase in prediction score.
        
        Better attribution → faster increase in score
        
        Returns:
            scores: Array of prediction scores at each insertion step
            auc: Area under the insertion curve (higher is better)
        """
        image = image.to(self.device)
        h, w = image.shape[2], image.shape[3]

        # Flatten and sort attribution
        attribution_flat = cv2.resize(attribution, (w, h)).flatten()
        sorted_indices = np.argsort(attribution_flat)[::-1].copy()

        scores = []
        num_pixels = len(sorted_indices)

        # Start with blank image
        modified_image = torch.zeros_like(image)
        
        # Get baseline score
        with torch.no_grad():
            output = self.model(modified_image)
            baseline_score = torch.softmax(output, dim=1)[0, target_class].item()
            scores.append(baseline_score)
        
        # Progressively insert pixels
        pixels_per_step = num_pixels // steps
        
        for step in range(1, steps + 1):
            # Insert pixels
            end_idx = min(step * pixels_per_step, num_pixels)
            pixels_to_insert = sorted_indices[:end_idx]
            
            # Create mask
            mask = torch.zeros((h * w,), device=self.device)
            mask[pixels_to_insert] = 1
            mask = mask.reshape(1, 1, h, w)
            
            # Apply mask
            modified_image = image * mask
            
            # Measure score
            with torch.no_grad():
                output = self.model(modified_image)
                score = torch.softmax(output, dim=1)[0, target_class].item()
                scores.append(score)
        
        scores = np.array(scores)
        auc = np.trapz(scores, dx=1.0 / steps)
        
        return scores, auc
    
    def average_drop(
        self,
        image: torch.Tensor,
        attribution: np.ndarray,
        target_class: int,
        percentile: float = 0.1
    ) -> float:
        """
        Average Drop (AD): Remove top k% most important pixels
        and measure relative drop in score.
        
        Lower is better.
        """
        image = image.to(self.device)
        h, w = image.shape[2], image.shape[3]
        
        # Get original score
        with torch.no_grad():
            output = self.model(image)
            original_score = torch.softmax(output, dim=1)[0, target_class].item()
        
        # Remove top percentile pixels
        attribution_flat = cv2.resize(attribution, (w, h)).flatten()
        threshold = np.percentile(attribution_flat, (1 - percentile) * 100)
        mask = (attribution_flat < threshold).astype(float)
        mask = torch.from_numpy(mask).reshape(1, 1, h, w).float().to(self.device)
        
        # Apply mask
        masked_image = image * mask
        
        # Get new score
        with torch.no_grad():
            output = self.model(masked_image)
            masked_score = torch.softmax(output, dim=1)[0, target_class].item()
        
        # Compute relative drop
        avg_drop = max(0, (original_score - masked_score) / original_score)
        
        return avg_drop
    
    def average_increase(
        self,
        image: torch.Tensor,
        attribution: np.ndarray,
        target_class: int,
        percentile: float = 0.1
    ) -> float:
        """
        Average Increase (AI): Keep only top k% most important pixels
        and measure relative increase compared to blank image.
        
        Higher is better.
        """
        image = image.to(self.device)
        h, w = image.shape[2], image.shape[3]
        
        # Get blank image score
        blank_image = torch.zeros_like(image)
        with torch.no_grad():
            output = self.model(blank_image)
            blank_score = torch.softmax(output, dim=1)[0, target_class].item()
        
        # Keep top percentile pixels
        attribution_flat = cv2.resize(attribution, (w, h)).flatten()
        threshold = np.percentile(attribution_flat, (1 - percentile) * 100)
        mask = (attribution_flat >= threshold).astype(float)
        mask = torch.from_numpy(mask).reshape(1, 1, h, w).float().to(self.device)
        
        # Apply mask
        masked_image = image * mask
        
        # Get new score
        with torch.no_grad():
            output = self.model(masked_image)
            masked_score = torch.softmax(output, dim=1)[0, target_class].item()
        
        # Compute relative increase
        avg_increase = (masked_score - blank_score) / (1 - blank_score + 1e-10)
        
        return avg_increase
    
    def faithfulness_correlation(
        self,
        image: torch.Tensor,
        attribution: np.ndarray,
        target_class: int,
        num_samples: int = 100
    ) -> float:
        """
        Faithfulness correlation: Measure correlation between
        attribution magnitude and actual impact on prediction.
        
        Higher correlation = more faithful attribution.
        """
        image = image.to(self.device)
        h, w = image.shape[2], image.shape[3]
        attribution_resized = cv2.resize(attribution, (w, h))
        
        # Get original score
        with torch.no_grad():
            output = self.model(image)
            original_score = torch.softmax(output, dim=1)[0, target_class].item()
        
        attribution_values = []
        impact_values = []
        
        # Sample random regions
        for _ in range(num_samples):
            # Random region
            region_size = np.random.randint(10, min(h, w) // 4)
            y = np.random.randint(0, h - region_size)
            x = np.random.randint(0, w - region_size)
            
            # Get attribution in this region
            region_attr = attribution_resized[y:y+region_size, x:x+region_size].mean()
            attribution_values.append(region_attr)
            
            # Mask this region and measure impact
            masked_image = image.clone()
            masked_image[:, :, y:y+region_size, x:x+region_size] = 0
            
            with torch.no_grad():
                output = self.model(masked_image)
                masked_score = torch.softmax(output, dim=1)[0, target_class].item()
            
            impact = original_score - masked_score
            impact_values.append(impact)
        
        # Compute correlation
        correlation = np.corrcoef(attribution_values, impact_values)[0, 1]
        
        return correlation
    
    def computational_efficiency(
        self,
        attribution_method,
        image: torch.Tensor,
        target_class: int,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Measure computational cost of attribution method.
        
        Returns:
            Dictionary with timing statistics and memory usage
        """
        times = []
        
        for _ in range(num_runs):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            start_time = time.time()
            _ = attribution_method.generate_cam(image, target_class)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
    
    def comprehensive_evaluation(
        self,
        image: torch.Tensor,
        attribution: np.ndarray,
        target_class: int,
        attribution_method=None
    ) -> Dict[str, float]:
        """
        Run all evaluation metrics and return comprehensive results.
        
        This is what you'll use to compare methods in your paper!
        """
        results = {}
        
        print("Computing deletion metric...")
        _, deletion_auc = self.deletion_metric(image, attribution, target_class)
        results['deletion_auc'] = deletion_auc
        
        print("Computing insertion metric...")
        _, insertion_auc = self.insertion_metric(image, attribution, target_class)
        results['insertion_auc'] = insertion_auc
        
        print("Computing average drop...")
        avg_drop = self.average_drop(image, attribution, target_class)
        results['average_drop'] = avg_drop
        
        print("Computing average increase...")
        avg_increase = self.average_increase(image, attribution, target_class)
        results['average_increase'] = avg_increase
        
        print("Computing faithfulness correlation...")
        faithfulness = self.faithfulness_correlation(image, attribution, target_class)
        results['faithfulness_correlation'] = faithfulness
        
        if attribution_method is not None:
            print("Computing computational efficiency...")
            efficiency = self.computational_efficiency(attribution_method, image, target_class)
            results.update(efficiency)
        
        return results


def compare_methods(
    evaluator: AttributionEvaluator,
    image: torch.Tensor,
    target_class: int,
    methods_dict: Dict[str, tuple]
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple attribution methods side-by-side.
    
    Args:
        evaluator: AttributionEvaluator instance
        image: Input image
        target_class: Target class for attribution
        methods_dict: Dict of {method_name: (method_instance, attribution_map)}
    
    Returns:
        Dictionary of results for each method
    """
    results = {}
    
    for method_name, (method_instance, attribution) in methods_dict.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {method_name}")
        print('='*50)
        
        method_results = evaluator.comprehensive_evaluation(
            image, attribution, target_class, method_instance
        )
        results[method_name] = method_results
    
    return results


# Import for visualization
import cv2
