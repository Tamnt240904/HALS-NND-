# FILE: src/analysis.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional
import os

# Import từ các module nội bộ
from .utils import *
from .gradcam import GradCAM

# ============================================================================
# Optimized L1 Coding with Batch Support
# ============================================================================

def l1_code_batch(A_batch, D_atoms, lam=1e-2, iters=100, nonneg=True):
    """Batch version of L1 coding for multiple activation maps."""
    batch_size = A_batch.shape[0]
    d = A_batch.shape[1]
    m = D_atoms.shape[0]
    
    D = D_atoms.reshape(m, -1).T  # (d^2, m)
    A = A_batch.reshape(batch_size, -1)  # (batch, d^2)
    
    G = D.T @ D
    B = A @ D
    
    diagG = np.diag(G)
    W = np.zeros((batch_size, m))
    
    for _ in range(iters):
        for j in range(m):
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
    """FISTA version"""
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
    
    def analyze_image_batch(self, images: torch.Tensor, k: int = 5, 
                           lam: float = 1e-2, verbose: bool = False) -> List[dict]:
        """Batch analyze multiple images efficiently."""
        batch_size = images.shape[0]
        results_list = []
        images = images.to(self.device)
        
        for i in range(batch_size):
            img = images[i:i+1]
            weights, _, pred_class = self.gradcam.forward(img)
            activations = self.gradcam.activations.squeeze().cpu()
            activations_np = activations.numpy()
            
            sparse_codes = l1_code_batch(activations_np, self.dictionary, 
                                        lam=lam, iters=100, nonneg=True)
            
            top_k_indices = torch.argsort(weights, descending=True)[:k].cpu().numpy()
            
            results_list.append({
                'weights': weights.cpu().numpy(),
                'sparse_codes': sparse_codes,
                'top_k_indices': top_k_indices.tolist(),
                'pred_class': pred_class
            })
        return results_list

# ============================================================================
# Optimized similarity computation
# ============================================================================

def compute_similarity_matrix_vectorized(results_list: List[dict], k: int = 5) -> dict:
    """Vectorized computation of similarity matrices."""
    n = len(results_list)
    weights_matrix = np.stack([r['weights'] for r in results_list])
    sparse_matrix = np.stack([r['sparse_codes'] for r in results_list])
    
    from sklearn.metrics.pairwise import cosine_similarity
    weight_sim_matrix = cosine_similarity(weights_matrix)
    
    n_channels = sparse_matrix.shape[1]
    sparse_sim_matrix = np.zeros((n, n))
    for ch in range(n_channels):
        sparse_ch = sparse_matrix[:, ch, :]
        sparse_sim_matrix += cosine_similarity(sparse_ch)
    sparse_sim_matrix /= n_channels
    
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
# MAIN ANALYSIS FUNCTIONS
# ============================================================================

def run_complete_analysis(model, target_layer, D, subset_paths, k=5, lam=1e-2, batch_size=8):
    """Run complete analysis with batch processing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    analyzer = TopKActivationAnalyzer(model, target_layer, D, device)
    all_results = []
    labels = []
    
    n_batches = (len(subset_paths) + batch_size - 1) // batch_size
    print(f"\nAnalyzing {len(subset_paths)} images in batches of {batch_size}...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(subset_paths))
        batch_paths = subset_paths[start_idx:end_idx]
        
        images_batch = []
        batch_labels = []
        for path, label in batch_paths:
            img_tensor = load_image(path)
            images_batch.append(img_tensor)
            batch_labels.append(label)
            
        images_batch = torch.cat(images_batch, dim=0)
        batch_results = analyzer.analyze_image_batch(images_batch, k=k, lam=lam)
        
        all_results.extend(batch_results)
        labels.extend(batch_labels)
        print(f"Processed batch {batch_idx + 1}/{n_batches}")
    
    results = [{'path': subset_paths[i][0], 'label': labels[i], 'analysis': all_results[i]} 
               for i in range(len(all_results))]
    
    print("\nComputing similarity matrices...")
    sim_matrices = compute_similarity_matrix_vectorized(all_results, k=k)
    
    intra = {'weight': [], 'sparse': [], 'topk': []}
    inter = {'weight': [], 'sparse': [], 'topk': []}
    
    n = len(labels)
    for i in range(n):
        for j in range(i+1, n):
            for metric in ['weight', 'sparse', 'topk']:
                val = sim_matrices[metric][i, j]
                if labels[i] == labels[j]:
                    intra[metric].append(val)
                else:
                    inter[metric].append(val)
                    
    visualize_similarity_distributions(intra, inter)
    
    return {
        'intra': intra, 'inter': inter,
        'results': results,
        'similarity_matrices': sim_matrices
    }

# ============================================================================
# ATOM CHARACTERISTIC ANALYSIS (Moved from Notebook)
# ============================================================================

def visualize_atom_overlap_matrix(overlap_matrix: np.ndarray, class_names: List[str]):
    """Visualize the atom overlap matrix as a heatmap."""
    plt.figure(figsize=(5, 4))
    
    mask = np.eye(len(class_names), dtype=bool)
    overlap_masked = np.ma.masked_where(mask, overlap_matrix)
    
    plt.imshow(overlap_masked, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(label='Jaccard Similarity')
    
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right', fontsize=8)
    plt.yticks(range(len(class_names)), class_names, fontsize=8)
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and overlap_matrix[i, j] > 0:
                plt.text(j, i, f'{overlap_matrix[i, j]:.2f}', 
                        ha='center', va='center', fontsize=6)
    
    plt.title('Top Characteristic Atoms Overlap', fontsize=10, fontweight='bold')
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/atom_overlap_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_class_characteristic_atoms(results: List[dict], n_atoms: int, 
                                       top_k_atoms: int = 10, threshold: float = 0.1,
                                       top_channels: int = 10):
    """Analyze which atoms are characteristic for each class."""
    
    # 1. Extract class names
    label_to_class_name = {}
    for r in results:
        path = r['path']
        label = r['label']
        class_name = path.split('/')[-2]
        label_to_class_name[label] = class_name
    
    sorted_labels = sorted(label_to_class_name.keys())
    class_names = [label_to_class_name[label] for label in sorted_labels]
    n_classes = len(class_names)
    
    # 2. Aggregate stats
    class_codes = {i: [] for i in range(n_classes)}
    for r in results:
        label = r['label']
        sparse_codes = r['analysis']['sparse_codes']
        weights = r['analysis']['weights']
        top_channel_indices = np.argsort(weights)[-top_channels:]
        top_codes = sparse_codes[top_channel_indices]
        avg_code = np.mean(np.abs(top_codes), axis=0)
        class_codes[label].append(avg_code)
    
    # 3. Compute characteristics
    class_characteristic_atoms = {}
    
    print(f"\n{'='*60}\nCLASS-CHARACTERISTIC ATOMS ANALYSIS\n{'='*60}")
    
    for class_idx in range(n_classes):
        if not class_codes[class_idx]: continue
        
        codes = np.array(class_codes[class_idx])
        mean_act = np.mean(codes, axis=0)
        usage = np.mean(codes > 1e-6, axis=0)
        scores = mean_act * usage
        
        top_indices = np.argsort(scores)[-top_k_atoms:][::-1]
        
        characteristic = []
        for atom_idx in top_indices:
            if scores[atom_idx] >= threshold:
                characteristic.append({
                    'atom_idx': int(atom_idx),
                    'score': float(scores[atom_idx]),
                    'mean_activation': float(mean_act[atom_idx]),
                    'usage_rate': float(usage[atom_idx])
                })
        class_characteristic_atoms[class_idx] = characteristic

    # 4. Cross-class analysis
    top_atoms_per_class = {}
    for class_idx, atoms in class_characteristic_atoms.items():
        if atoms:
            top_atoms_per_class[class_idx] = [a['atom_idx'] for a in atoms[:5]]
            
    atom_class_map = {}
    for class_idx, atom_list in top_atoms_per_class.items():
        for atom_idx in atom_list:
            if atom_idx not in atom_class_map: atom_class_map[atom_idx] = []
            atom_class_map[atom_idx].append(class_idx)
            
    unique_atoms = {atom: classes[0] for atom, classes in atom_class_map.items() if len(classes) == 1}
    
    # 5. Overlap Matrix
    overlap_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if i in top_atoms_per_class and j in top_atoms_per_class:
                atoms_i = set(top_atoms_per_class[i])
                atoms_j = set(top_atoms_per_class[j])
                # Sử dụng hàm jaccard_sim từ utils
                overlap_matrix[i, j] = jaccard_sim(list(atoms_i), list(atoms_j))
    
    visualize_atom_overlap_matrix(overlap_matrix, class_names)
    
    return {
        'characteristic_atoms': class_characteristic_atoms,
        'unique_atoms': unique_atoms,
        'class_names': class_names,
        'overlap_matrix': overlap_matrix
    }

def visualize_unique_atoms_grid(atom_analysis: dict, D: torch.Tensor, figsize=(6, 6)):
    """Visualize the first unique atom for each class."""
    unique_atoms = atom_analysis['unique_atoms']
    class_names = atom_analysis['class_names']
    
    class_unique_atoms = {}
    for atom_idx, class_idx in unique_atoms.items():
        if class_idx not in class_unique_atoms:
            class_unique_atoms[class_idx] = []
        class_unique_atoms[class_idx].append(atom_idx)
    
    classes_with_unique = sorted(class_unique_atoms.keys())
    
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, class_idx in enumerate(classes_with_unique):
        if idx >= 9: break
        ax = axes[idx]
        
        first_atom_idx = class_unique_atoms[class_idx][0]
        atom = D[first_atom_idx]
        if torch.is_tensor(atom): atom = atom.cpu().numpy()
            
        ax.imshow(atom, cmap='gray')
        ax.set_title(f"{class_names[class_idx]}\nAtom {first_atom_idx}", fontsize=8)
        ax.axis('off')
        
    for idx in range(len(classes_with_unique), 9):
        axes[idx].axis('off')
        
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/unique_atoms_grid.png', dpi=150)
    print("Saved visualization to: figures/unique_atoms_grid.png")
    plt.show()