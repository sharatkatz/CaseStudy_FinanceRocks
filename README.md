# DSBusinessCase

# Automated Exploratory Data Analysis (EDA) Tool

## Overview

This Python script provides a comprehensive automated Exploratory Data Analysis (EDA) tool that loads Parquet datasets, performs detailed analysis, and generates a wide variety of visualizations. The tool is designed for data scientists and analysts who need to quickly understand their data structure, distributions, relationships, and quality issues.

## Features

### Data Analysis
- **Basic Information**: Dataset preview and structure information
- **Missing Values Analysis**: Comprehensive missing value reporting and visualization
- **Summary Statistics**: Detailed statistical summaries for numerical and categorical variables
- **Correlation Analysis**: Correlation matrix for numerical variables

### Visualization Capabilities
- **Distribution Plots**: Histograms and count plots for all variables
- **Box Plots**: For numerical variable distribution analysis
- **Bar Plots**: For categorical variable frequency analysis
- **Heatmaps**: Missing values, correlation matrices, and summary statistics
- **Joint Plots**: Bivariate relationships between numerical variables
- **Pair Plots**: Multi-variable relationships (scatterplot matrices)

### Group Analysis
- **Package-based Analysis**: All analyses can be segmented by a "package" variable
- **Comparative Statistics**: Side-by-side comparisons across different packages

## Requirements

### Python Dependencies
```python
pandas
matplotlib
seaborn
pathlib
```

### Data Format
- Input file must be in Parquet format
- Expected file name: `customer_data.parquet`
- The script looks for a "package" column for segmented analysis (optional)

## Usage

### Basic Execution
```bash
python CaseStudy_FR.py
```

### As a Module
```python
from CaseStudy_FR import ExploratoryDataAnalysis, setup_plot_directory

# Set up plot directory
setup_plot_directory()

# Initialize EDA
eda = ExploratoryDataAnalysis(file_path, "customer_data.parquet")

# Generate all plots and analyses
eda.create_plots()
```

## Output Structure

The tool creates an organized directory structure for all outputs:

```
CaseStudy_FinanceRocks_plots/
├── bar_plots/
├── bar_plots_by_package/
├── box_plots/
├── box_plots_by_package/
├── correlation_matrix/
├── hist_dir/
├── joint_plot_dir/
├── missing_values/
├── missing_values_by_package/
├── pair_plots/
├── summary_statistics/
├── summary_statistics_by_package/
└── combined_summary_statistics_by_package_numeric/
```

## Key Methods

### Core Analysis Methods
- `missing_reports()`: Comprehensive missing value analysis
- `create_plots()`: Main method to generate all visualizations
- `report_missings()`: Detailed missing value reporting
- `visualize_correlation_matrix()`: Correlation heatmap generation

### Visualization Methods
- `create_hist_plots()`: Distribution plots for all variables
- `create_bar_plots()`: Frequency plots for categorical variables
- `create_boxplots()`: Distribution analysis for numerical variables
- `create_joint_plots()`: Bivariate relationship analysis


## Customization
The script can be easily modified to:
- Change plot styles and colors
- Add new types of visualizations
- Modify column selection criteria
- Adjust output directory structure
- Add custom analysis methods

## Output Files

### Visualization Categories
1. **Univariate Analysis**: Histograms, box plots, bar charts
2. **Bivariate Analysis**: Joint plots, correlation matrices
3. **Multivariate Analysis**: Pair plots, combined statistics
4. **Data Quality**: Missing value heatmaps and reports

## Notes

- Do not run autopep8 on this file as it may break formatted print statements
- The script is designed for datasets with mixed data types (numerical and categorical)
- Package-based analysis is optional but provides valuable segmentation insights
- All visualizations are saved as PNG files for easy sharing and documentation

