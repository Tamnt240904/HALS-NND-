import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional
import os
from .utils import *
from .gradcam import GradCAM

# ============================================================================
# Optimized L1 Coding with Batch Support
# ============================================================================

def l1_code_batch(A_batch, D_atoms, lam=1e-2, iters=100, nonneg=True):
    """
    Batch version of L1 coding for multiple activation maps.
    
    Args:
        A_batch: (batch_size, d, d) - batch of activation maps
        D_atoms: (m, d, d) - dictionary atoms
        lam: L1 regularization
        iters: iterations
        nonneg: non-negativity constraint
        
    Returns:
        W: (batch_size, m) - sparse codes for all maps
    """
    batch_size = A_batch.shape[0]
    d = A_batch.shape[1]
    m = D_atoms.shape[0]
    
    # Flatten everything: D: (m, d^2), A: (batch, d^2)
    D = D_atoms.reshape(m, -1).T  # (d^2, m)
    A = A_batch.reshape(batch_size, -1)  # (batch, d^2)
    
    # Precompute Gram matrix and correlation (shared across batch)
    G = D.T @ D  # (m, m)
    B = A @ D  # (batch, m)
    
    diagG = np.diag(G)  # (m,)
    W = np.zeros((batch_size, m))
    
    # Coordinate descent with vectorized batch operations
    for _ in range(iters):
        for j in range(m):
            # Vectorized residual computation for all samples
            residual = B[:, j] - (W @ G[:, j] - W[:, j] * G[j, j])
            
            if nonneg:
                W[:, j] = np.maximum(0.0, (residual - lam) / (diagG[j] + 1e-12))
            else:
                v = np.sign(residual) * np.maximum(np.abs(residual) - lam, 0.0)
                W[:, j] = v / (diagG[j] + 1e-12)
    
    return W


def l1_code(A, D_atoms, lam=1e-2, iters=100, nonneg=False):
    """Single map version (kept for compatibility)"""
    D = np.stack([Di.reshape(-1) for Di in D_atoms], axis=1)
    a = A.reshape(-1)
    G = D.T @ D
    b = D.T @ a
    m = G.shape[0]
    w = np.zeros(m)
    for _ in range(iters):
        for j in range(m):
            zj = b[j] - (G[j] @ w - G[j, j] * w[j])
            if nonneg:
                w[j] = max(0.0, (zj - lam) / (G[j, j] + 1e-12))
            else:
                v = np.sign(zj) * max(abs(zj) - lam, 0.0)
                w[j] = v / (G[j, j] + 1e-12)
    return w


def l1_code_nonneg(A, D_atoms, lam=1e-2, iters=100, tol=1e-6):
    """FISTA version (kept for compatibility)"""
    D = np.stack([Di.reshape(-1) for Di in D_atoms], axis=1)
    a = A.reshape(-1)
    DTD = D.T @ D
    DTa = D.T @ a
    L = np.linalg.norm(DTD, ord=2) + 1e-8
    step_size = 1.0 / L
    m = DTD.shape[0]
    w = np.zeros(m)
    w_old = w.copy()
    t = 1.0
    
    for iter_num in range(iters):
        y = w + ((t - 1) / (t + 2)) * (w - w_old)
        grad = DTD @ y - DTa
        z = y - step_size * grad
        w_new = np.maximum(0.0, z - step_size * lam)
        change = np.linalg.norm(w_new - w)
        if change < tol:
            w = w_new
            break
        w_old = w
        w = w_new
        t += 1
    return w


# ============================================================================
# Optimized TopKActivationAnalyzer
# ============================================================================

class TopKActivationAnalyzer:
    """Optimized analyzer with batch processing support."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module, 
                 dictionary: torch.Tensor, device: torch.device):
        self.model = model
        self.target_layer = target_layer
        self.dictionary = dictionary.cpu().numpy()
        self.device = device
        self.gradcam = GradCAM(model, target_layer)
    
    def get_top_k_maps(self, image: torch.Tensor, k: int = 5, 
                       class_idx: Optional[int] = None) -> Tuple[List[torch.Tensor], List[int], torch.Tensor]:
        image = image.to(self.device)
        weights, _, pred_class = self.gradcam.forward(image, class_idx=class_idx)
        activations = self.gradcam.activations.squeeze().cpu()
        top_k_indices = torch.argsort(weights, descending=True)[:k].cpu().numpy()
        top_maps = [activations[idx] for idx in top_k_indices]
        return top_maps, top_k_indices.tolist(), weights
    
    def analyze_sparse_codes(self, activation_maps: List[torch.Tensor], 
                            lam: float = 1e-2, iters: int = 100, 
                            nonneg: bool = True) -> List[np.ndarray]:
        sparse_codes = []
        for A in activation_maps:
            A_np = A.numpy() if torch.is_tensor(A) else A
            w = l1_code(A_np, self.dictionary, lam=lam, iters=iters, nonneg=nonneg)
            sparse_codes.append(w)
        return sparse_codes
    
    def visualize_analysis(self, image: torch.Tensor, k: int = 5, 
                          class_idx: Optional[int] = None, 
                          lam: float = 1e-2) -> dict:
        top_maps, top_indices, all_weights = self.get_top_k_maps(image, k, class_idx)
        sparse_codes = self.analyze_sparse_codes(top_maps, lam=lam)
        
        results = {
            'top_maps': top_maps,
            'top_indices': top_indices,
            'all_weights': all_weights.cpu().numpy(),
            'sparse_codes': sparse_codes,
            'top_weights': all_weights[top_indices].cpu().numpy(),
            'input_image': image
        }
        
        print(f"\n{'='*70}")
        print(f"Top-{k} Most Influential Activation Maps Analysis")
        print(f"{'='*70}")
        
        for i, (idx, w, code) in enumerate(zip(top_indices, results['top_weights'], sparse_codes)):
            print(f"\nRank {i+1}: Channel {idx}")
            print(f"  GradCAM Weight: {w:.6f}")
            print(f"  Sparse Code Stats:")
            print(f"    - Non-zero atoms: {np.sum(np.abs(code) > 1e-6)}/{len(code)}")
            print(f"    - L1 norm: {np.sum(np.abs(code)):.6f}")
            print(f"    - Max coefficient: {np.max(np.abs(code)):.6f}")
            print(f"    - Top 5 atom indices: {np.argsort(np.abs(code))[-5:][::-1].tolist()}")
            print(f"    - Top 5 coefficients: {code[np.argsort(np.abs(code))[-5:][::-1]]}")
        
        print(f"\n{'='*70}\n")
        return results
    
    def visualize_channel_decomposition(self, results: dict, channel_rank: int = 0, 
                                       figsize: Tuple[int, int] = (15, 10),
                                       save_path: Optional[str] = None):
        if channel_rank >= len(results['top_maps']):
            raise ValueError(f"channel_rank {channel_rank} out of range")
        
        input_image = results['input_image']
        activation_map = results['top_maps'][channel_rank]
        sparse_code = results['sparse_codes'][channel_rank]
        channel_idx = results['top_indices'][channel_rank]
        gradcam_weight = results['top_weights'][channel_rank]
        
        top_atom_indices = np.argsort(np.abs(sparse_code))[-4:][::-1]
        top_atom_weights = sparse_code[top_atom_indices]
        
        act_map_np = activation_map.numpy() if torch.is_tensor(activation_map) else activation_map
        target_size = act_map_np.shape
        
        input_np = input_image.squeeze(0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        input_denorm = input_np * std + mean
        input_denorm = np.clip(input_denorm, 0, 1)
        input_denorm = np.transpose(input_denorm, (1, 2, 0))
        
        input_pil = Image.fromarray((input_denorm * 255).astype(np.uint8))
        input_downscaled = input_pil.resize(target_size[::-1], Image.BILINEAR)
        input_downscaled = np.array(input_downscaled) / 255.0
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(f'Channel {channel_idx} Decomposition (Rank {channel_rank + 1})\n'
                    f'GradCAM Weight: {gradcam_weight:.4f}', 
                    fontsize=16, fontweight='bold')
        
        axes[0, 0].imshow(input_downscaled)
        axes[0, 0].set_title('Original Image\n(Downscaled)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(act_map_np, cmap='gray')
        axes[0, 1].set_title(f'Activation Map\n(Channel {channel_idx})', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        dict_np = self.dictionary
        for i, (atom_idx, weight) in enumerate(zip(top_atom_indices, top_atom_weights)):
            row = (i // 2) + 1
            col = i % 2
            atom = dict_np[atom_idx]
            axes[row, col].imshow(atom, cmap='gray')
            axes[row, col].set_title(f'Atom {atom_idx}\nWeight: {weight:.4f}', 
                                     fontsize=11, fontweight='bold')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        plt.show()
    
    def visualize_all_top_channels(self, results: dict, save_dir: Optional[str] = None):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for i in range(len(results['top_maps'])):
            save_path = None
            if save_dir:
                channel_idx = results['top_indices'][i]
                save_path = os.path.join(save_dir, f'channel_{channel_idx}_rank_{i+1}.png')
            self.visualize_channel_decomposition(results, channel_rank=i, save_path=save_path)
    
    def analyze_image_batch(self, images: torch.Tensor, k: int = 5, 
                           lam: float = 1e-2, verbose: bool = False) -> List[dict]:
        """
        Batch analyze multiple images efficiently.
        
        Args:
            images: (batch_size, 3, H, W) - batch of images
            k: top-k channels
            lam: L1 regularization
            verbose: print debug info
            
        Returns:
            List of analysis results for each image
        """
        batch_size = images.shape[0]
        results_list = []
        
        images = images.to(self.device)
        
        # Process each image (GradCAM requires individual processing)
        for i in range(batch_size):
            img = images[i:i+1]
            
            # Get weights and activations
            weights, _, pred_class = self.gradcam.forward(img)
            activations = self.gradcam.activations.squeeze().cpu()  # (C, H, W)
            n_channels = activations.shape[0]
            
            # Convert all activations to numpy for batch processing
            activations_np = activations.numpy()  # (C, H, W)
            
            # Batch compute sparse codes for ALL channels at once
            sparse_codes = l1_code_batch(activations_np, self.dictionary, 
                                        lam=lam, iters=100, nonneg=True)
            
            # Get top-k indices
            top_k_indices = torch.argsort(weights, descending=True)[:k].cpu().numpy()
            
            results_list.append({
                'weights': weights.cpu().numpy(),
                'sparse_codes': sparse_codes,
                'top_k_indices': top_k_indices.tolist(),
                'pred_class': pred_class
            })
            
            if verbose and (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{batch_size} images")
        
        return results_list
    
    def analyze_image(self, image: torch.Tensor, k: int = 5, 
                     lam: float = 1e-2, verbose: bool = False) -> dict:
        """Backward compatible single image analysis."""
        results = self.analyze_image_batch(image.unsqueeze(0) if image.dim() == 3 else image, 
                                          k=k, lam=lam, verbose=verbose)
        return results[0]


# ============================================================================
# Optimized similarity computation
# ============================================================================

def compute_similarity_matrix_vectorized(results_list: List[dict], k: int = 5) -> dict:
    """
    Vectorized computation of similarity matrices for all pairs.
    
    Returns:
        dict with keys 'weight', 'sparse', 'topk' containing similarity matrices
    """
    n = len(results_list)
    
    # Stack all data
    weights_matrix = np.stack([r['weights'] for r in results_list])  # (n, C)
    sparse_matrix = np.stack([r['sparse_codes'] for r in results_list])  # (n, C, m)
    
    # 1. Weight similarity (cosine similarity matrix)
    from sklearn.metrics.pairwise import cosine_similarity
    weight_sim_matrix = cosine_similarity(weights_matrix)  # (n, n)
    
    # 2. Sparse code similarity (average cosine over channels)
    n_channels = sparse_matrix.shape[1]
    sparse_sim_matrix = np.zeros((n, n))
    
    for ch in range(n_channels):
        sparse_ch = sparse_matrix[:, ch, :]  # (n, m)
        sparse_sim_matrix += cosine_similarity(sparse_ch)
    
    sparse_sim_matrix /= n_channels  # Average over channels
    
    # 3. Top-K Jaccard similarity
    topk_sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            setA = set(results_list[i]['top_k_indices'])
            setB = set(results_list[j]['top_k_indices'])
            jaccard = len(setA & setB) / max(1, len(setA | setB))
            topk_sim_matrix[i, j] = jaccard
            topk_sim_matrix[j, i] = jaccard
    
    return {
        'weight': weight_sim_matrix,
        'sparse': sparse_sim_matrix,
        'topk': topk_sim_matrix
    }


# ============================================================================
# OPTIMIZED run_complete_analysis
# ============================================================================

def run_complete_analysis(model, target_layer, D,
                         subset_paths: List[Tuple[str, int]], 
                         k: int = 5, lam: float = 1e-2,
                         batch_size: int = 8):
    """
    OPTIMIZED: Run complete analysis with batch processing.
    
    Args:
        model: Pre-trained model
        target_layer: Target layer for analysis
        D: Dictionary atoms
        subset_paths: List of (image_path, label) tuples
        k: Number of top channels
        lam: L1 regularization
        batch_size: Batch size for processing (adjust based on GPU memory)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nLoading model and dictionary...")
    analyzer = TopKActivationAnalyzer(model, target_layer, D, device)
    
    # Batch load and process images
    print(f"\nAnalyzing {len(subset_paths)} images in batches of {batch_size}...")
    
    all_results = []
    labels = []
    
    n_batches = (len(subset_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(subset_paths))
        batch_paths = subset_paths[start_idx:end_idx]
        
        # Load batch of images
        images_batch = []
        batch_labels = []
        
        for path, label in batch_paths:
            img_tensor = load_image(path)
            images_batch.append(img_tensor)
            batch_labels.append(label)
        
        # Stack into batch tensor
        images_batch = torch.cat(images_batch, dim=0)  # (batch, 3, H, W)
        
        # Batch analyze
        batch_results = analyzer.analyze_image_batch(images_batch, k=k, lam=lam)
        
        all_results.extend(batch_results)
        labels.extend(batch_labels)
        
        print(f"Processed batch {batch_idx + 1}/{n_batches} ({end_idx}/{len(subset_paths)} images)")
    
    # Wrap results with metadata
    results = [{'path': subset_paths[i][0], 'label': labels[i], 'analysis': all_results[i]} 
               for i in range(len(all_results))]
    
    # Compute similarity matrices (vectorized)
    print("\nComputing similarity matrices (vectorized)...")
    sim_matrices = compute_similarity_matrix_vectorized(all_results, k=k)
    
    # Extract intra vs inter class similarities
    print("\nExtracting intra/inter-class similarities...")
    intra = {'weight': [], 'sparse': [], 'topk': []}
    inter = {'weight': [], 'sparse': [], 'topk': []}
    
    n = len(labels)
    for i in range(n):
        for j in range(i+1, n):
            for metric in ['weight', 'sparse', 'topk']:
                sim_value = sim_matrices[metric][i, j]
                
                if labels[i] == labels[j]:
                    intra[metric].append(sim_value)
                else:
                    inter[metric].append(sim_value)
    
    # Print summary
    print("\n" + "="*70)
    print("INTRA-CLASS SIMILARITY (same class)")
    print("="*70)
    for metric, values in intra.items():
        print(f"{metric:10s}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
    
    print("\n" + "="*70)
    print("INTER-CLASS SIMILARITY (different classes)")
    print("="*70)
    for metric, values in inter.items():
        print(f"{metric:10s}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
    
    print("\n" + "="*70)
    print("DIFFERENCE (Intra - Inter)")
    print("="*70)
    for metric in intra.keys():
        diff = np.mean(intra[metric]) - np.mean(inter[metric])
        print(f"{metric:10s}: {diff:+.4f}")
    
    # Visualize
    visualize_similarity_distributions(intra, inter)
    
    return {
        'intra': intra,
        'inter': inter,
        'results': results,
        'similarity_matrices': sim_matrices
    }