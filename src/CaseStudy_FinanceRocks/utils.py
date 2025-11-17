"""Utility functions for EDA."""

import os
import shutil
from pathlib import Path

def setup_plot_directory(filePath: str = "../../../input_files"):
    """Set up the plot directory."""
    plot_dir = os.path.join(filePath, "CaseStudy_FinanceRocks_plots")

    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
        print(f"Folder '{plot_dir}' deleted.")
    print(f"Creating new folder: {plot_dir}")
    os.makedirs(plot_dir, exist_ok=True)

    return plot_dir

def validate_file_path(file_path: str, file_name: str) -> str:
    """Validate that file exists."""
    input_file_full_path = os.path.join(file_path, file_name)
    if not Path(input_file_full_path).is_file():
        raise FileNotFoundError(f"File '{input_file_full_path}' not found.")
    return input_file_full_path
