from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import matplotlib.pyplot as plt 
import torch
import torchvision.transforms as T
from PIL import Image

# ============================================================================
# Similarity Metrics
# ============================================================================

def cosine_sim(a, b):
    """Compute cosine similarity between two vectors."""
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


def jaccard_sim(a, b):
    """Compute Jaccard similarity between two sets."""
    A, B = set(a), set(b)
    return len(A & B) / max(1, len(A | B))


def compare_images(imgA: dict, imgB: dict, k: int = 5) -> dict:
    """
    Compare similarity between two analyzed images.
    
    Returns:
        dict with keys: 'weight', 'sparse', 'topk'
    """
    wA, wB = imgA['weights'], imgB['weights']
    cA, cB = imgA['sparse_codes'], imgB['sparse_codes']
    tA, tB = imgA['top_k_indices'], imgB['top_k_indices']
    
    # 1. Weight (GradCAM importance) similarity
    weight_sim = cosine_sim(wA, wB)
    
    # 2. Sparse-code similarity: average over all channels
    n_channels = min(len(cA), len(cB))
    code_sims = [cosine_sim(cA[ch], cB[ch]) for ch in range(n_channels)]
    code_sim = float(np.mean(code_sims))
    
    # 3. Top-K channel overlap (Jaccard similarity)
    topk_sim = jaccard_sim(tA, tB)
    
    return {
        'weight': weight_sim,
        'sparse': code_sim,
        'topk': topk_sim
    }



def visualize_similarity_distributions(intra: dict, inter: dict):
    """Visualize similarity distributions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ['weight', 'sparse', 'topk']
    titles = ['GradCAM Weight Similarity', 'Sparse Code Similarity', 'Top-K Channel Overlap']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        ax.hist(intra[metric], bins=40, alpha=0.6, label='Intra-class', color='blue')
        ax.hist(inter[metric], bins=40, alpha=0.6, label='Inter-class', color='red')
        
        ax.axvline(np.mean(intra[metric]), color='blue', linestyle='--', 
                   label=f'Intra mean: {np.mean(intra[metric]):.3f}')
        ax.axvline(np.mean(inter[metric]), color='red', linestyle='--',
                   label=f'Inter mean: {np.mean(inter[metric]):.3f}')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('similarity_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved visualization to: similarity_analysis.png")
    plt.show()




def load_subset(subset_root="data/subset"):
    """
    Load subset from existing directory structure.
    Expected structure: subset_root/class_name/*.png
    
    Args:
        subset_root: Root directory containing class folders
        
    Returns:
        List of (image_path, label_index) tuples
    """
    class_names = ['cassette_player', 'chain_saw', 'church', 'English_springer', 'French_horn',
               'garbage_truck', 'gas_pump', 'golf_ball', 'parachute', 'tench']
    
    # Create class name to index mapping
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    subset_paths = []
    
    print(f"Loading subset from: {subset_root}")
    
    # Iterate through each class folder
    for class_name in class_names:
        class_dir = os.path.join(subset_root, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
        
        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Add to subset_paths with correct label
        label_idx = class_to_idx[class_name]
        for img_file in sorted(image_files):  # Sort for consistency
            img_path = os.path.join(class_dir, img_file)
            subset_paths.append((img_path, label_idx))
        
        print(f"  {class_name:12s}: {len(image_files):3d} images (label={label_idx})")
    
    print(f"\nTotal: {len(subset_paths)} images loaded")
    return subset_paths

def load_image(image_path: str) -> torch.Tensor:
    """Load and preprocess image."""
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


