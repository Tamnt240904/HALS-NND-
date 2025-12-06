"""
Batch Feature Analysis Script

Analyze multiple features và tạo summary report.

Usage:
    # Analyze specific features
    python batch_feature_analysis.py --features 137 256 512 1024 2048 4096
    
    # Analyze random features
    python batch_feature_analysis.py --num_random 20
    
    # Analyze top-k most important features (based on activation frequency)
    python batch_feature_analysis.py --top_important 10 --dataset_path X_activations.pt
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import sys

sys.path.append('.')
from analyze_feature import FeatureAnalyzer


def find_most_important_features(csae_model_path, 
                                 activation_data_path,
                                 top_k=20):
    """
    Tìm features quan trọng nhất dựa trên activation frequency.
    
    Args:
        csae_model_path: Path to model
        activation_data_path: Path to activation data (X tensor)
        top_k: Number of top features
    
    Returns:
        List of (feature_id, importance_score)
    """
    print(f"Finding top-{top_k} most important features...")
    
    # Load model
    csae_model = joblib.load(csae_model_path)
    csae_model.eval()
    
    # Load activation data
    print(f"Loading activation data from {activation_data_path}...")
    X = torch.load(activation_data_path)  # [N, 1024, 14, 14]
    print(f"  Data shape: {X.shape}")
    
    # Pass through encoder
    print("Computing feature activations...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csae_model = csae_model.to(device)
    
    feature_importance = torch.zeros(csae_model.hidden_dim)
    
    batch_size = 32
    for i in tqdm(range(0, X.shape[0], batch_size)):
        batch = X[i:i+batch_size].to(device)
        
        with torch.no_grad():
            _, sparse_features = csae_model(batch, use_topk=True)
        
        # Sum activations across batch and spatial dimensions
        importance = sparse_features.sum(dim=(0, 2, 3)).cpu()
        feature_importance += importance
    
    # Get top-k
    top_k_values, top_k_indices = torch.topk(feature_importance, k=top_k)
    
    results = [(idx.item(), val.item()) 
               for idx, val in zip(top_k_indices, top_k_values)]
    
    print(f"\nTop-{top_k} features by importance:")
    for i, (fid, score) in enumerate(results[:10]):
        print(f"  #{i+1}: Feature {fid:4d} | Score: {score:.2f}")
    
    return results


def analyze_multiple_features(feature_ids,
                              image_path,
                              csae_model_path,
                              output_dir='batch_analysis',
                              create_pdf=True):
    """
    Analyze multiple features và save results.
    
    Args:
        feature_ids: List of feature IDs
        image_path: Sample image path
        csae_model_path: Model path
        output_dir: Output directory
        create_pdf: Whether to create combined PDF
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer(csae_model_path, device='cuda')
    
    # Collect statistics
    all_stats = []
    
    print(f"\nAnalyzing {len(feature_ids)} features...")
    print("="*80)
    
    # PDF for combined output
    pdf_path = output_dir / 'combined_analysis.pdf' if create_pdf else None
    
    with PdfPages(pdf_path) if create_pdf else DummyContext() as pdf:
        for i, feature_id in enumerate(tqdm(feature_ids, desc="Processing")):
            # Individual analysis
            save_path = output_dir / f"feature_{feature_id:04d}_analysis.png"
            
            try:
                analyzer.visualize_feature_analysis(
                    feature_id=feature_id,
                    image_path=image_path,
                    top_k=8,
                    save_path=str(save_path)
                )
                
                # Get statistics
                weight_data = analyzer.get_feature_weights(feature_id)
                stats = weight_data['stats']
                stats['feature_id'] = feature_id
                all_stats.append(stats)
                
                # Add to PDF if enabled
                if create_pdf:
                    # Reopen the saved figure and add to PDF
                    img = plt.imread(str(save_path))
                    fig, ax = plt.subplots(figsize=(20, 12))
                    ax.imshow(img)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                
            except Exception as e:
                print(f"\nError analyzing feature {feature_id}: {e}")
                continue
    
    # Save statistics
    stats_path = output_dir / 'feature_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"✓ Analysis complete!")
    print(f"  Individual images: {output_dir}")
    if create_pdf:
        print(f"  Combined PDF: {pdf_path}")
    print(f"  Statistics: {stats_path}")
    print(f"{'='*80}")
    
    # Create summary visualization
    create_summary_plots(all_stats, output_dir)
    
    return all_stats


class DummyContext:
    """Dummy context manager for when PDF is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def savefig(self, *args, **kwargs):
        pass


def create_summary_plots(stats_list, output_dir):
    """
    Create summary visualization across all analyzed features.
    """
    if not stats_list:
        return
    
    print("\nCreating summary plots...")
    
    feature_ids = [s['feature_id'] for s in stats_list]
    sparsities = [s['sparsity'] for s in stats_list]
    non_zeros = [s['non_zero_count'] for s in stats_list]
    abs_maxs = [s['abs_max'] for s in stats_list]
    means = [s['mean'] for s in stats_list]
    stds = [s['std'] for s in stats_list]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Summary Statistics for {len(stats_list)} Analyzed Features',
                fontsize=16, fontweight='bold')
    
    # 1. Sparsity distribution
    axes[0, 0].hist(sparsities, bins=20, color='steelblue', edgecolor='navy', alpha=0.7)
    axes[0, 0].axvline(np.mean(sparsities), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(sparsities):.3f}')
    axes[0, 0].set_xlabel('Sparsity', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Sparsity Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Active channel count
    axes[0, 1].hist(non_zeros, bins=20, color='green', edgecolor='darkgreen', alpha=0.7)
    axes[0, 1].axvline(np.mean(non_zeros), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(non_zeros):.1f}')
    axes[0, 1].set_xlabel('# Active Channels', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Active Channel Distribution', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Max weight magnitude
    axes[0, 2].hist(abs_maxs, bins=20, color='orange', edgecolor='darkorange', alpha=0.7)
    axes[0, 2].axvline(np.mean(abs_maxs), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(abs_maxs):.3f}')
    axes[0, 2].set_xlabel('Max |Weight|', fontsize=11)
    axes[0, 2].set_ylabel('Count', fontsize=11)
    axes[0, 2].set_title('Max Weight Magnitude', fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Feature comparison - Sparsity
    axes[1, 0].bar(range(len(feature_ids)), sparsities, color='steelblue', alpha=0.7)
    axes[1, 0].set_xlabel('Feature Index', fontsize=11)
    axes[1, 0].set_ylabel('Sparsity', fontsize=11)
    axes[1, 0].set_title('Sparsity per Feature', fontweight='bold')
    axes[1, 0].set_xticks(range(0, len(feature_ids), max(1, len(feature_ids)//10)))
    axes[1, 0].set_xticklabels([str(feature_ids[i]) for i in range(0, len(feature_ids), 
                                max(1, len(feature_ids)//10))], rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 5. Scatter: Sparsity vs Active Channels
    scatter = axes[1, 1].scatter(sparsities, non_zeros, c=abs_maxs, cmap='viridis',
                                alpha=0.6, s=100, edgecolor='black')
    axes[1, 1].set_xlabel('Sparsity', fontsize=11)
    axes[1, 1].set_ylabel('# Active Channels', fontsize=11)
    axes[1, 1].set_title('Sparsity vs Active Channels', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Max |Weight|', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Weight statistics
    axes[1, 2].scatter(means, stds, c=non_zeros, cmap='plasma', alpha=0.6, 
                      s=100, edgecolor='black')
    axes[1, 2].axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 2].axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 2].set_xlabel('Mean Weight', fontsize=11)
    axes[1, 2].set_ylabel('Std Weight', fontsize=11)
    axes[1, 2].set_title('Weight Statistics', fontweight='bold')
    cbar2 = plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2])
    cbar2.set_label('# Active', fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(output_dir) / 'summary_statistics.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved to {save_path}")
    plt.close()
    
    # Print text summary
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*60}")
    print(f"Features analyzed: {len(stats_list)}")
    print(f"\nSparsity:")
    print(f"  Mean ± Std: {np.mean(sparsities):.4f} ± {np.std(sparsities):.4f}")
    print(f"  Range: [{np.min(sparsities):.4f}, {np.max(sparsities):.4f}]")
    print(f"  Median: {np.median(sparsities):.4f}")
    print(f"\nActive Channels:")
    print(f"  Mean ± Std: {np.mean(non_zeros):.1f} ± {np.std(non_zeros):.1f}")
    print(f"  Range: [{np.min(non_zeros)}, {np.max(non_zeros)}]")
    print(f"  Median: {np.median(non_zeros):.1f}")
    print(f"\nMax Weight Magnitude:")
    print(f"  Mean ± Std: {np.mean(abs_maxs):.4f} ± {np.std(abs_maxs):.4f}")
    print(f"  Range: [{np.min(abs_maxs):.4f}, {np.max(abs_maxs):.4f}]")
    print(f"  Median: {np.median(abs_maxs):.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Batch analyze multiple CSAE features'
    )
    parser.add_argument('--csae_model', type=str,
                       default='multichannel_csae_resnet50_model.pkl',
                       help='Path to trained CSAE model')
    parser.add_argument('--image_path', type=str,
                       default='data/imagenette/tench/n01440764_1.JPEG',
                       help='Sample image for visualization')
    parser.add_argument('--features', type=int, nargs='+',
                       help='Specific feature IDs to analyze')
    parser.add_argument('--num_random', type=int,
                       help='Number of random features to analyze')
    parser.add_argument('--top_important', type=int,
                       help='Analyze top-k most important features')
    parser.add_argument('--dataset_path', type=str,
                       help='Path to activation dataset (for --top_important)')
    parser.add_argument('--output_dir', type=str,
                       default='batch_analysis',
                       help='Output directory')
    parser.add_argument('--create_pdf', action='store_true',
                       help='Create combined PDF report')
    parser.add_argument('--no_pdf', action='store_true',
                       help='Do not create PDF (faster)')
    
    args = parser.parse_args()
    
    # Determine which features to analyze
    feature_ids = []
    
    if args.features:
        feature_ids = args.features
        print(f"Analyzing {len(feature_ids)} specified features")
    
    elif args.num_random:
        feature_ids = np.random.choice(8192, size=args.num_random, replace=False).tolist()
        print(f"Analyzing {args.num_random} random features")
    
    elif args.top_important:
        if not args.dataset_path:
            print("Error: --dataset_path required for --top_important")
            return
        
        important_features = find_most_important_features(
            args.csae_model,
            args.dataset_path,
            top_k=args.top_important
        )
        feature_ids = [fid for fid, _ in important_features]
    
    else:
        # Default: analyze a few interesting features
        feature_ids = [137, 256, 512, 1024, 2048, 4096]
        print(f"No features specified. Analyzing default set: {feature_ids}")
    
    # Run analysis
    create_pdf = args.create_pdf or not args.no_pdf
    
    analyze_multiple_features(
        feature_ids=feature_ids,
        image_path=args.image_path,
        csae_model_path=args.csae_model,
        output_dir=args.output_dir,
        create_pdf=create_pdf
    )


if __name__ == "__main__":
    main()