"""
Interactive Feature Explorer (Updated with Text Input)

Usage:
    jupyter notebook
    # Then run cells in this script
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
# UPDATED: Import BoundedIntText
from ipywidgets import interact, IntSlider, BoundedIntText, Dropdown, Button, Output, VBox, HBox, Label, Layout
from IPython.display import display, clear_output, Image
import joblib
from pathlib import Path
import sys
import os

sys.path.append('.')
from analyze_feature import FeatureAnalyzer
# Import class để tránh lỗi pickle
from run_multichannel_csae_resnet50 import MultiChannelConvSAE

class InteractiveFeatureExplorer:
    """
    Interactive widget-based explorer for CSAE features.
    Updated to browse existing visualizations and deep-dive into features.
    """
    
    def __init__(self, 
                 csae_model_path='weights/multichannel_csae_resnet50_model.pkl',
                 vis_dir='output/visualization',
                 data_dir='data/imagenette'):
        """
        Args:
            csae_model_path: Path to trained model
            vis_dir: Path to directory containing generated visualizations
            data_dir: Path to original image dataset
        """
        # Initialize analyzer
        print("Loading analyzer...")
        try:
            self.analyzer = FeatureAnalyzer(csae_model_path, device='cuda')
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying cpu...")
            self.analyzer = FeatureAnalyzer(csae_model_path, device='cpu')
        
        self.vis_dir = Path(vis_dir)
        self.data_dir = Path(data_dir)
        
        # 1. Quét các file visualization có sẵn
        self.vis_paths = self._scan_visualization_files()
        print(f"Found {len(self.vis_paths)} visualizations in {self.vis_dir}")
        
        if len(self.vis_paths) == 0:
            print(f"Warning: No visualizations found in {self.vis_dir}")
            
        # Current state
        self.current_feature_id = 0
        self.current_vis_idx = 0
        
        # Setup widgets
        self._setup_widgets()
        
        print("✓ Explorer ready!")
    
    def _scan_visualization_files(self):
        """Tìm tất cả các file .png trong folder visualization"""
        vis_files = []
        if not self.vis_dir.exists():
            return []
            
        # Duyệt qua các subfolder (tên class)
        for class_dir in self.vis_dir.iterdir():
            if class_dir.is_dir():
                # Lấy các file png kết thúc bằng _features_r50.png
                files = list(class_dir.glob('*_features_r50.png'))
                vis_files.extend(files)
        
        # Sắp xếp để dễ tìm
        return sorted(vis_files, key=lambda p: (p.parent.name, p.name))

    def _get_original_image_path(self, vis_path: Path) -> Path:
        """
        2. Map từ file visualization sang file ảnh gốc.
        Ví dụ: output/visualization/tench/n0144_features_r50.png 
            -> data/imagenette/tench/n0144.JPEG
        """
        class_name = vis_path.parent.name
        # Bỏ hậu tố "_features_r50" và đuôi file
        original_stem = vis_path.stem.replace('_features_r50', '')
        
        # Giả định ảnh gốc là .JPEG (của Imagenette)
        # Có thể cần check thêm .jpg hoặc .png nếu dataset khác
        original_path = self.data_dir / class_name / (original_stem + ".JPEG")
        
        if not original_path.exists():
            # Thử tìm đuôi .jpg nếu .JPEG không thấy
            original_path = self.data_dir / class_name / (original_stem + ".jpg")
            
        return original_path

    def _setup_widgets(self):
        """Create interactive widgets."""
        
        # Dropdown chọn ảnh Visualization
        options = []
        for i, p in enumerate(self.vis_paths):
            display_name = f"{p.parent.name}/{p.name}"
            options.append((display_name, i))

        self.vis_dropdown = Dropdown(
            options=options,
            value=self.current_vis_idx,
            description='Select Vis:',
            layout=Layout(width='500px'),
            style={'description_width': '100px'}
        )
        
        # UPDATED: Thay thế IntSlider bằng BoundedIntText (Text box nhập số)
        self.feature_input = BoundedIntText(
            value=self.current_feature_id,
            min=0,
            max=8191,
            step=1,
            description='Enter Feat ID:', # Đổi label cho phù hợp
            continuous_update=False,
            layout=Layout(width='200px'), # Giảm width vì text box nhỏ hơn slider
            style={'description_width': '100px'}
        )
        
        # Top-K slider for analysis (Vẫn giữ slider cho cái này vì số nhỏ)
        self.topk_slider = IntSlider(
            value=8,
            min=4,
            max=16,
            step=1,
            description='Show Top-K Ch:',
            continuous_update=False,
            style={'description_width': '100px'}
        )
        
        # Buttons
        self.analyze_button = Button(
            description='Deep Analyze Feature',
            button_style='primary',
            icon='search',
            layout=Layout(width='200px')
        )
        
        # Output areas
        self.vis_output = Output() 
        self.analysis_output = Output() 
        
        # Bind callbacks
        self.vis_dropdown.observe(self._on_vis_change, names='value')
        self.analyze_button.on_click(self._on_analyze_click)
    
    def _on_vis_change(self, change):
        """Khi chọn ảnh khác từ dropdown, hiển thị ảnh visualization gốc"""
        with self.vis_output:
            clear_output(wait=True)
            idx = change['new']
            vis_path = self.vis_paths[idx]
            
            print(f"Viewing: {vis_path}")
            # Hiển thị ảnh PNG đã generate trước đó
            display(Image(filename=str(vis_path), width=1000))
            
            # Reset analysis output khi đổi ảnh
            self.analysis_output.clear_output()

    def _on_analyze_click(self, button):
        """Thực hiện phân tích sâu feature được chọn"""
        with self.analysis_output:
            clear_output(wait=True)
            
            # Lấy thông tin hiện tại
            vis_idx = self.vis_dropdown.value
            vis_path = self.vis_paths[vis_idx]
            feature_id = self.feature_input.value
            top_k = self.topk_slider.value
            
            # Tìm ảnh gốc
            original_path = self._get_original_image_path(vis_path)
            
            if not original_path.exists():
                print(f"Error: Original image not found at {original_path}")
                return
                
            print(f"="*60)
            print(f"ANALYZING FEATURE {feature_id}")
            print(f"Original Image: {original_path}")
            print(f"="*60)
            
            try:
                # Gọi hàm phân tích từ analyze_feature.py
                self.analyzer.visualize_feature_analysis(
                    feature_id=feature_id,
                    image_path=str(original_path),
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
        
        # Layout giao diện
        header = VBox([
            Label("STEP 1: Select a pre-generated visualization to view Top-16 Features"),
            self.vis_dropdown
        ])
        
        controls = VBox([
            Label("STEP 2: Enter a Feature ID from the image above to analyze Channel Contributions"),
            HBox([self.feature_input, self.topk_slider, self.analyze_button])
        ])
        
        # Cấu trúc:
        # [Dropdown]
        # [Ảnh Visualization to đùng]
        # [Controls chọn Feature ID (Text Box)]
        # [Ảnh phân tích Channel Contribution]
        ui = VBox([
            header,
            self.vis_output, 
            HTML_Separator(), # Đường kẻ phân cách
            controls,
            self.analysis_output
        ])
        
        display(ui)
        
        # Trigger hiển thị ảnh đầu tiên
        self._on_vis_change({'new': self.vis_dropdown.value})

def HTML_Separator():
    from ipywidgets import HTML
    return HTML(value="<hr style='border: 2px solid #007bff; margin: 20px 0;'>")

# ===== Helper function for Notebook =====

def setup_explorer():
    """
    Setup function to call in Jupyter notebook.
    """
    explorer = InteractiveFeatureExplorer(
        csae_model_path='weights/multichannel_csae_resnet50_model.pkl',
        vis_dir='output/visualization',
        data_dir='data/imagenette'
    )
    explorer.display()
    return explorer