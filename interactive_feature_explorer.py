import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from ipywidgets import interact, IntSlider, BoundedIntText, Dropdown, Button, Output, VBox, HBox, Label, Layout, HTML
from IPython.display import display, clear_output, Image as IPImage
from PIL import Image
import joblib
from pathlib import Path
import sys
import os
import io

sys.path.append('.')
from analyze_feature import FeatureAnalyzer
from visualize_multichannel_sae_resnet50 import MultiChannelSAEVisualizerR50
# Import class để tránh lỗi pickle
from run_multichannel_csae_resnet50 import MultiChannelConvSAE

class InteractiveFeatureExplorer:
    """
    Interactive widget-based explorer for CSAE features.
    
    Workflow:
    1. Select an image from dropdown
    2. Show top-16 features extracted from that image (clickable)
    3. Click on a feature to see which channels contribute to it
    """
    
    def __init__(self, 
                 csae_model_path='weights/multichannel_csae_resnet50_model.pkl',
                 original_dir='data/imagenette',
                 visual_dir='output/visualization',
                 top_k_features=16):
        """
        Args:
            csae_model_path: Path to trained model
            original_dir: Path to original image dataset (e.g., data/imagenette)
            visual_dir: Path to image dataset for visualization dropdown (e.g., output/visualization)
            top_k_features: Number of top features to extract per image
        """
        # Initialize both analyzers
        print("Loading Feature Analyzer...")
        try:
            self.feature_analyzer = FeatureAnalyzer(csae_model_path, device='cuda')
        except Exception as e:
            print(f"Error loading model on cuda: {e}")
            print("Trying cpu...")
            self.feature_analyzer = FeatureAnalyzer(csae_model_path, device='cpu')
        
        print("Loading Multi-Channel SAE Visualizer...")
        try:
            self.sae_visualizer = MultiChannelSAEVisualizerR50(
                csae_model_path, 
                device='cuda'
            )
        except Exception as e:
            print(f"Error loading visualizer on cuda: {e}")
            print("Trying cpu...")
            self.sae_visualizer = MultiChannelSAEVisualizerR50(
                csae_model_path,
                device='cpu'
            )
        
        self.original_dir = Path(original_dir)
        self.visual_dir = Path(visual_dir)
        self.top_k_features = top_k_features
        
        # Scan available images
        self.visual_image_paths, self.original_image_paths = self._scan_images()
        print(f"Found {len(self.visual_image_paths)} images in {self.visual_dir}")
        
        if len(self.visual_image_paths) == 0:
            print(f"Warning: No images found in {self.visual_dir}")
            
        # Current state
        self.current_image_idx = 0
        self.current_results = None  # Will store extract_features() results
        self.selected_feature_id = None
        
        # Setup widgets
        self._setup_widgets()
        
        print("✓ Explorer ready!")
    
    def _scan_images(self):
        """
        Scan images in visual directory (.png) and map them to original directory (.JPEG or .jpg).
        Assumes the original filename is the visual filename PLUS the suffix '_features_r50'.
        
        Returns: (list of visual image paths, list of corresponding original image paths)
        """
        visual_files = []
        original_files = []
        
        if not self.visual_dir.exists():
            return [], []
            
        # Scan visual directory only for .png files
        for visual_path in self.visual_dir.rglob('*.png'):
            if visual_path.is_file():
                
                # Get the filename without extension (e.g., 'ILSVRC2012_val_00000213_features_r50')
                visual_stem = visual_path.stem
                
                # Strip the suffix to get the base original file name stem
                # e.g., 'ILSVRC2012_val_00000213_features_r50' -> 'ILSVRC2012_val_00000213'
                if visual_stem.endswith('_features_r50'):
                    original_stem = visual_stem[:-len('_features_r50')]
                else:
                    # If suffix is not present, use the visual stem as is (fallback)
                    original_stem = visual_stem 

                # Get the parent class directory name (e.g., 'n03417042')
                # Assumes the structure is visual_dir/class_name/image_name.png
                class_dir_name = visual_path.parent.name
                
                # Construct the potential original path components
                original_parent = self.original_dir / class_dir_name
                
                found_original = False
                
                # 1. Check for .JPEG (common Imagenet extension)
                original_path = original_parent / f"{original_stem}.JPEG"
                if original_path.exists():
                    visual_files.append(visual_path)
                    original_files.append(original_path)
                    found_original = True
                
                # 2. Check for .jpg (another common Imagenet extension) if .JPEG not found
                if not found_original:
                    original_path = original_parent / f"{original_stem}.jpg"
                    if original_path.exists():
                        visual_files.append(visual_path)
                        original_files.append(original_path)
                        found_original = True
        
        # Sort both lists based on visual_path for consistent ordering
        sorted_pairs = sorted(zip(visual_files, original_files), 
                              key=lambda pair: (pair[0].parent.name, pair[0].name))
        
        return [pair[0] for pair in sorted_pairs], [pair[1] for pair in sorted_pairs]
    
    def _setup_widgets(self):
        """Create interactive widgets."""
        
        # Dropdown to select image
        options = []
        for i, p in enumerate(self.visual_image_paths):
            # Use relative path from visual_dir for display name
            display_name = str(p.relative_to(self.visual_dir))
            options.append((display_name, i))

        self.image_dropdown = Dropdown(
            options=options,
            value=self.current_image_idx,
            description='Select Image:',
            layout=Layout(width='500px'),
            style={'description_width': '120px'}
        )
        
        # Top-K slider
        self.topk_slider = IntSlider(
            value=8,
            min=4,
            max=16,
            step=1,
            description='Show Top-K Ch:',
            continuous_update=False,
            layout=Layout(width='300px'),
            style={'description_width': '120px'}
        )
        
        # Output areas
        self.step1_output = Output()  # Feature extraction visualization
        self.step2_output = Output()  # Channel contribution analysis
        
        # Bind callbacks
        self.image_dropdown.observe(self._on_image_change, names='value')
    
    def _on_image_change(self, change):
        """When user selects a different image, extract and visualize features"""
        with self.step1_output:
            clear_output(wait=True)
            idx = change['new']
            
            # Use original image path for feature extraction
            original_image_path = self.original_image_paths[idx]
            visual_image_path = self.visual_image_paths[idx]
            
            print(f"Processing (Original): {original_image_path}")
            print(f"Display Name (Visual): {visual_image_path.relative_to(self.visual_dir)}")
            print(f"Extracting top-{self.top_k_features} features...")
            
            # Extract features using SAE visualizer
            try:
                self.current_results = self.sae_visualizer.extract_features(
                    str(original_image_path), # Pass the original path to the analyzer
                    top_k=self.top_k_features
                )
                
                # Visualize with clickable features
                self._plot_clickable_features()
                
            except Exception as e:
                print(f"Error extracting features: {e}")
                import traceback
                traceback.print_exc()
            
            # Clear step 2 output when changing image
            self.step2_output.clear_output()
            self.selected_feature_id = None

    def _plot_clickable_features(self):
        """
        Plot feature grid similar to visualize_multichannel_sae_resnet50.py
        with buttons to select features
        """
        if self.current_results is None:
            print("No results to display")
            return
        
        image = self.current_results['image']
        top_features = self.current_results['top_features']
        num_selected_channels = self.current_results['num_selected_channels']
        channel_weights = self.current_results['channel_weights']
        
        n_features = len(top_features)
        n_cols = 8
        n_rows = 1 + (n_features + 3) // 4
        
        fig = plt.figure(figsize=(24, 3.5 * n_rows))
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)
        
        # ===== Row 0: Overview =====
        ax_img = fig.add_subplot(gs[0, 0:2])
        ax_img.imshow(image)
        ax_img.set_title("Input Image", fontsize=12, fontweight='bold')
        ax_img.axis('off')
        
        ax_info = fig.add_subplot(gs[0, 2:4])
        ax_info.axis('off')
        
        top_5_indices = torch.argsort(channel_weights, descending=True)[:5].tolist()
        top_5_scores = [channel_weights[i].item() for i in top_5_indices]
        
        info_text = f"GradCAM Channel Selection:\n"
        info_text += f"  • Selected: {num_selected_channels}/1024 channels\n"
        info_text += f"  • Threshold: 80% cumulative score\n"
        info_text += f"  • Top 5 channels:\n"
        for idx, score in zip(top_5_indices, top_5_scores):
            info_text += f"    #{idx}: {score:.4f}\n"
        
        ax_info.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=ax_info.transAxes)
        
        ax_bar = fig.add_subplot(gs[0, 4:])
        importances = [imp for _, imp, _ in top_features]
        feature_indices = [f"F{idx}" for idx, _, _ in top_features]
        ax_bar.bar(range(len(importances)), importances, color='steelblue', 
                   alpha=0.8, edgecolor='navy')
        ax_bar.set_xlabel('Feature Index', fontsize=10)
        ax_bar.set_ylabel('Importance (sum of activations)', fontsize=10)
        ax_bar.set_title(f'Top-{n_features} CSAE Feature Importance', 
                        fontsize=12, fontweight='bold')
        ax_bar.set_xticks(range(len(importances)))
        ax_bar.set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=8)
        ax_bar.grid(True, alpha=0.3, axis='y')
        
        # ===== Rows 1+: Feature activation maps =====
        for i, (feat_idx, importance, activation_map) in enumerate(top_features):
            row = 1 + i // 4
            col = (i % 4) * 2
            
            ax_feat = fig.add_subplot(gs[row, col:col+2])
            im = ax_feat.imshow(activation_map.numpy(), cmap='hot', 
                               interpolation='bilinear')
            ax_feat.set_title(f"Feature {feat_idx}\nImportance: {importance:.2f}",
                             fontsize=10, fontweight='bold')
            ax_feat.axis('off')
            
            plt.colorbar(im, ax=ax_feat, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Use buttons below to select a feature for channel analysis',
                    fontsize=14, fontweight='bold', y=0.998, color='blue')
        
        plt.tight_layout()
        plt.show()
        
        # Create buttons for each feature
        print("\n" + "="*60)
        print("Select a feature to analyze its channel contributions:")
        print("="*60)
        
        self._create_feature_buttons(top_features)
    
    def _create_feature_buttons(self, top_features):
        """Create a grid of buttons to select features"""
        from ipywidgets import Button, GridBox, Layout
        
        buttons = []
        for feat_idx, importance, _ in top_features:
            btn = Button(
                description=f'Feature {feat_idx}',
                button_style='info',
                tooltip=f'Analyze Feature {feat_idx} (Importance: {importance:.2f})',
                layout=Layout(width='140px', height='35px')
            )
            # Use lambda with default argument to capture feat_idx correctly
            btn.on_click(lambda b, fid=feat_idx: self._on_feature_select(fid))
            buttons.append(btn)
        
        # Arrange buttons in a grid (4 columns)
        grid = GridBox(
            buttons,
            layout=Layout(
                grid_template_columns='repeat(4, 140px)',
                grid_gap='10px',
                justify_content='center'
            )
        )
        
        display(grid)
    
    def _on_feature_select(self, feature_id):
        """Handle feature selection from button"""
        self.selected_feature_id = feature_id
        
        print(f"\n{'='*60}")
        print(f"✓ Selected Feature {feature_id} for channel analysis")
        print(f"{'='*60}")
        
        # Trigger channel analysis
        self._analyze_feature_channels()
    
    def _analyze_feature_channels(self):
        """Analyze which channels contribute to selected feature"""
        if self.selected_feature_id is None or self.current_results is None:
            print("No feature selected")
            return
        
        with self.step2_output:
            clear_output(wait=True)
            
            # Use original image path for feature analysis
            original_image_path = self.original_image_paths[self.image_dropdown.value]
            feature_id = self.selected_feature_id
            top_k = self.topk_slider.value
            
            print(f"="*60)
            print(f"ANALYZING FEATURE {feature_id}")
            print(f"Image: {original_image_path}")
            print(f"="*60)
            
            try:
                # Use feature analyzer to show channel contributions
                self.feature_analyzer.visualize_feature_analysis(
                    feature_id=feature_id,
                    image_path=str(original_image_path),
                    top_k=top_k,
                    save_path=None  # Show inline
                )
                plt.show()
            except Exception as e:
                print(f"Error analyzing feature: {e}")
                import traceback
                traceback.print_exc()

    def display(self):
        """Display the interactive interface."""
        
        # Layout
        step1_header = VBox([
            HTML(value="<h3 style='color: #007bff;'>STEP 1: Select Image and View Top Features</h3>"),
            HTML(value="<p style='font-style: italic;'>Choose an image to extract and visualize its top-16 activated features</p>"),
            self.image_dropdown
        ])
        
        step2_header = VBox([
            HTML(value="<hr style='border: 2px solid #007bff; margin: 30px 0 20px 0;'>"),
            HTML(value="<h3 style='color: #28a745;'>STEP 2: Click Feature to Analyze Channel Contributions</h3>"),
            HTML(value="<p style='font-style: italic;'>Click any feature map above to see which channels contribute to it</p>"),
            HBox([
                Label("Adjust Top-K Channels:", layout=Layout(width='150px')),
                self.topk_slider
            ])
        ])
        
        ui = VBox([
            step1_header,
            self.step1_output,
            step2_header,
            self.step2_output
        ])
        
        display(ui)
        
        # Trigger initial visualization
        if len(self.visual_image_paths) > 0:
            self._on_image_change({'new': self.image_dropdown.value})


# ===== Helper function for Notebook (Updated) =====

def setup_explorer():
    """
    Setup function to call in Jupyter notebook.
    
    Usage in notebook:
        from interactive_feature_explorer import setup_explorer
        explorer = setup_explorer()
    """
    explorer = InteractiveFeatureExplorer(
        csae_model_path='output/weights/multichannel_csae_resnet50_layer3_model.pkl',
        original_dir='data/imagenette',
        visual_dir='output/visualization',
        top_k_features=16
    )
    explorer.display()
    return explorer