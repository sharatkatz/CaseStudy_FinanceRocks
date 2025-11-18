#!/usr/bin/python3

"""Main module.
 File: core.py

Summary
 Purpose: simple automated exploratory data analysis (EDA) tool that loads a Parquet dataset,
 prints basic info, reports missing values, and saves a large set of visualizations to disk.
 Entrypoint: if name == "main" — constructs ExploratoryDataAnalysis(filePath, "customer_data.parquet")
 and runs eda.create_plots().

Flow:
1. Import necessary modules and classes.
2. Define global variables for file paths.
3. Set up a temporary directory for storing plots.
4. Define the ExploratoryDataAnalysis class with methods for data analysis and visualization.
5. The module is designed to be imported and used by other scripts or run as a standalone
    script.
6. When run as a standalone script, it initializes the plot directory and performs EDA on the specified dataset.      

 Author: Sharat Sharma
 Date: November 2025

 WARNING: DO NOT RUN autopep8 ON THIS FILE AS IT WILL BREAK THE FORMATTING OF THE
          PRINT STATEMENTS WHICH ARE INTENDED TO BE IN A SPECIFIC FORMAT.

"""

import os
from pathlib import Path
import sys
import shutil
from typing import Optional, List, Callable

import pandas as pd # type: ignore
from matplotlib import pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from pprint import pprint


class ExploratoryDataAnalysis:
    """
    A class for performing exploratory data analysis on customer data.
    This class provides comprehensive data exploration capabilities including:
    - Data loading and validation from parquet files
    - Summary statistics generation (overall and by package)
    - Missing value analysis and visualization
    - Distribution plots (histograms, box plots, bar plots)
    - Correlation analysis
    - Joint plots and pair plots
    - Statistical summaries export to CSV
    Attributes:
        plot_dir (str): Directory path where all generated plots will be saved.
        customer_data (pd.DataFrame): The loaded customer data from the parquet file.
        sub_columns (list): List of columns to analyze, excluding ID columns and 
            columns starting with 'add_'.
        package_var (str): Name of the package variable column for grouping analysis.
        unique_packages (list): List of unique package values in the dataset.
    Methods:
        __init__(filePath, fileName): Initialize the EDA object and load data.
        setup_plot_directory(plot_filepath): Set up directory for saving plots.
        string_and_function(str, func): Helper method to print message and execute function.
        missing_reports(): Generate and display missing value reports.
        create_plots(): Generate all types of plots for the dataset.
        report_missings(): Report missing values in the dataset.
        report_missings_bypackage(): Report missing values grouped by package.
        visualize_missings(): Create heatmap visualization of missing values.
        visualize_missings_bypackage(): Create missing value heatmaps by package.
        create_bar_plots(): Generate bar plots for categorical columns.
        create_barplots_bypackage(): Generate bar plots grouped by package.
        create_boxplots(): Generate box plots for numerical columns.
        create_boxplots_bypackage(): Generate box plots grouped by package.
        create_hist_plots(): Generate histogram plots for all columns.
        create_hist_plots_bypackage(): Generate histograms grouped by package.
        create_joint_plots(): Generate joint plots for pairs of numerical columns.
        create_joint_plots_bypackage(): Generate joint plots grouped by package.
        create_pair_plots(): Generate pair plots for numerical columns.
        create_pair_plots_bypackage(): Generate pair plots grouped by package.
        jointplot_with_package_variable(): Create joint plots with package as x-axis.
        visualize_summary_statistics(): Visualize summary statistics as heatmap.
        visualize_correlation_matrix(): Visualize correlation matrix as heatmap.
        visualize_summary_statistics_bypackage(): Visualize summary stats by package.
        visualize_combined_summary_statistics_by_package_numeric(): Visualize combined 
            numerical summary statistics by package.
        visualize_combined_summary_statistics_by_package(): Visualize all combined 
            summary statistics by package.
        create_summary_statistics_for_all_packages(): Generate summary statistics 
            dictionary for all packages.
        create_combined_summary_statistics_by_package(): Create combined summary 
            statistics DataFrame for all packages.
        create_combined_summary_statistics_by_package_numeric(): Create combined 
            numerical summary statistics by package.
        create_combined_summary_statistics_by_package_categorical(): Create combined 
            categorical summary statistics by package.
        export_combined_summary_statistics_by_package(output_file): Export combined 
            summary statistics to CSV.
        export_combined_summary_statistics_by_package_numeric(output_file): Export 
            numerical summary statistics to CSV.
        export_combined_summary_statistics_by_package_categorical(output_file): Export 
            categorical summary statistics to CSV.
        test_create_plots(): Test method for creating bar plots.
    Raises:
        FileNotFoundError: If the specified data file does not exist.
    Example:
        >>> eda = ExploratoryDataAnalysis(filePath="../data", fileName="customer_data.parquet")
        >>> eda.create_plots()
        >>> eda.export_combined_summary_statistics_by_package("summary_stats.csv")
    """

    def __init__(self, filePath: str="../data", fileName: str="customer_data.parquet"):
        """_summary_

        Args:
            filePath (str): _path to the data file
            fileName (str): _name of the data file
        """
        self.plot_dir: str = self.setup_plot_directory(filePath)
        input_file_full_path = os.path.join(filePath, fileName)
        if Path(input_file_full_path).is_file() is False:
            raise FileNotFoundError(
                f"File '{input_file_full_path}' not found.")

        self.customer_data = pd.read_parquet(input_file_full_path)
        self.sub_columns = [
            _ for _ in self.customer_data.columns if not (
                (_.lower().endswith("_id")) or (
                    _.lower() == "id") or (
                    _.lower().startswith("add_")))]
        self.package_var = "package"
        self.unique_packages = self.customer_data[self.package_var].unique(
        ) if self.package_var in self.customer_data.columns else [None]

        print(f"\nThe set of columns to analyze: {self.sub_columns}\n")
        print("Here under is a preview of the data:\n")
        print(self.customer_data, "\n\n")

        print("Here is the info of the data:\n")
        print(self.customer_data.info(), "\n\n")

        self.missing_reports()
        self.export_combined_summary_statistics_by_package(
            "combined_summary_statistics_by_package.csv"
        )

    def setup_plot_directory(self, plot_filepath: str = "../plots") -> str:
        """Set up the plot directory by deleting existing one and creating a new one."""
        plot_dir: str = os.path.join(plot_filepath, "CaseStudy_FinanceRocks_plots")

        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)
            print(f"Folder '{plot_dir}' deleted.")
        print(f"Creating new folder: {plot_dir}")
        os.makedirs(plot_dir, exist_ok=True)
        return plot_dir


    def string_and_function(self, str: str, func: Callable):
        """
        Execute a callable while printing a formatted header section comprised of a
        separator line and a descriptive message.

        This helper is useful for visually separating logical blocks of output when
        running a sequence of functions (e.g., in exploratory / diagnostic scripts).

        Parameters
        ----------
        str : str
            A descriptive title or message to print above the function's output.
            Note: the parameter name shadows the built-in 'str' type; consider
            renaming (e.g., 'message' or 'title') for clarity.
        func : Callable
            A zero-argument callable to be invoked. Its return value is intentionally
            ignored; only its side effects (such as printing) matter.

        Side Effects
        ------------
        Prints two leading blank lines, a horizontal separator line (80 dashes),
        the provided message, and then any output produced by the callable.

        Returns
        -------
        None
            Always returns None; the callable's result is discarded.

        Raises
        ------
        Any exception raised inside the provided callable will propagate.

        Example
        -------
        def show_stats():
            print("Records:", 128)
            print("Mean:", 42.7)

        self.string_and_function("Data Summary", show_stats)
        """
        print("\n\n")
        print("-" * 80)
        print(str, "\n")
        _ = func()
        return None

    def missing_reports(self):
        """
        Generate and display a suite of missing-value diagnostics.
        This convenience method sequentially:
          1. Prints a header and runs `report_missings` (overall missing value summary).
          2. Prints a header and runs `report_missings_bypackage` (missing summary grouped by package).
          3. Prints a header and runs `visualize_missings` (global missingness visualization).
          4. Prints a header and runs `visualize_missings_bypackage` (missingness visualization grouped by package).
        Each step delegates to `self.string_and_function(header: str, func: Callable)` which is expected
        to handle the display of the header string followed by execution of the provided function.
        Assumptions:
          - The instance implements:
              - report_missings()
              - report_missings_bypackage()
              - visualize_missings()
              - visualize_missings_bypackage()
              - string_and_function(string: str, func: Callable)
          - These called methods perform their own printing / plotting side effects.
        Side Effects:
          - Produces console output (textual reports).
          - Produces visual output (e.g., plots) if the visualization methods generate them.
        Returns:
          None
            The method is used purely for its side effects.
        Raises:
          AttributeError: If any of the expected reporting/visualization methods are missing.
          Exception: Propagates any exception raised within the called reporting/visualization functions.
        Notes:
          Use this as a high-level "one-shot" diagnostic to inspect data completeness both globally and
          segmented by package. For programmatic access to the underlying data, call the individual
          component methods directly instead of this aggregated runner.
        """
        strings = [
            "Here is the missing values report:\n",  # 6
            "Here is the missing values report by package:\n",  # 7
            "Visualizing missing values:\n",  # 8
            "Visualizing missing values by package:\n",  # 9
        ]

        functions = [
            self.report_missings,  # 6
            self.report_missings_bypackage,  # 7
            self.visualize_missings,  # 8
            self.visualize_missings_bypackage,  # 9
        ]

        for string, func in zip(strings, functions):
            self.string_and_function(string, func)

        return None

    def create_plots(self):
        """Generate and dispatch a predefined sequence of exploratory plots and reports.
        This convenience method loops over an ordered list of (label, callable) pairs
        and delegates plotting/reporting work to the helper method
        `self.string_and_function(label, func)`. Each callable is expected to
        produce either a visualization (e.g., histogram, bar plot, box plot, correlation
        matrix) or a missing-value / summary statistics report for the dataset held
        by the instance.
        The sequence currently includes (in order):
        1. Histograms of numeric variables.
        2. Bar plots of categorical/count-like variables.
        3. Box plots of numeric variables overall.
        4. Box plots grouped by package.
        5. Bar plots grouped by package.
        6. Missing values report (global).
        7. Missing values report grouped by package.
        8. Visual (matrix/map) of missing values (global).
        9. Visual (matrix/map) of missing values grouped by package.
        10. Summary statistics visualization (global).
        11. Correlation matrix visualization.
        12. Summary statistics visualization grouped by package.
        13. Combined summary statistics by package (numeric only).
        Side Effects:
        - Produces and (likely) saves or displays multiple figures/reports.
        - May log or print the status string preceding each generated artifact.
        Returns:
            None
        Raises:
            Any exception propagated from the individual plot/report generation
            methods or from `self.string_and_function` (e.g., data validation errors,
            plotting library errors).
        Notes:
        - The ordering in both the `strings` and `functions` lists must remain aligned.
          Adding/removing a plot requires modifying both lists consistently.
        - This method does not perform error handling around individual calls; consider
          wrapping `self.string_and_function` internally if partial completion is desired.
        """
        strings = [
            "Create hist plots:\n",  # 1
            "Creating bar plots:\n",  # 2
            "Creating box plots:\n",  # 3
            "Creating box plots by package:\n",  # 4
            "Creating bar plots by package:\n",  # 5
            "Here is the missing values report:\n",  # 6
            "Here is the missing values report by package:\n",  # 7
            "Visualizing missing values:\n",  # 8
            "Visualizing missing values by package:\n",  # 9
            "Visualizing summary statistics:\n",  # 10
            "Visualizing correlation matrix:\n",  # 11
            "Visualizing summary statistics by package:\n",  # 12
            # 13
            "Visualizing combined summary statistics by package (numerical):\n",
        ]

        functions = [
            self.create_hist_plots,  # 1
            self.create_bar_plots,  # 2
            self.create_boxplots,  # 3
            self.create_boxplots_bypackage,  # 4
            self.create_barplots_bypackage,  # 5
            self.report_missings,  # 6
            self.report_missings_bypackage,  # 7
            self.visualize_missings,  # 8
            self.visualize_missings_bypackage,  # 9
            self.visualize_summary_statistics,  # 10
            self.visualize_correlation_matrix,  # 11
            self.visualize_summary_statistics_bypackage,  # 12
            self.visualize_combined_summary_statistics_by_package_numeric,  # 13
        ]

        for string, func in zip(strings, functions):
            self.string_and_function(string, func)

        return None

    def report_missings(self):
        """Report missing values in the dataset."""
        missing_report = self.customer_data.isnull().sum()
        missing_report = missing_report[missing_report > 0]
        print("Missing Values Report:")
        pprint(missing_report.to_dict())
        return missing_report

    def report_missings_bypackage(self):
        """Report missing values in the dataset by package."""
        if self.package_var not in self.customer_data.columns:
            print(
                f"Package variable '{self.package_var}' not found in data. Skipping missing report by package.")
            return None

        for one_package in self.unique_packages:
            data_subset = self.customer_data.loc[
                self.customer_data[self.package_var] == one_package]
            missing_report = data_subset.isnull().sum()
            missing_report = missing_report[missing_report > 0]
            print(f"Missing Values Report for Package: {one_package}")
            pprint(missing_report.to_dict())
            print("\n")

        return None

    def visualize_missings_bypackage(self):
        """Visualize missing values in the dataset by package."""
        if self.package_var not in self.customer_data.columns:
            print(
                f"Package variable '{self.package_var}' not found in data. Skipping missing visualization by package."
            )
            return None

        for one_package in self.unique_packages:
            data_subset = self.customer_data.loc[
                self.customer_data[self.package_var] == one_package]

            plt.figure(figsize=(12, 8))
            sns.heatmap(
                data_subset.isnull(),
                cbar=False,
                cmap='viridis')
            plt.title(f'Missing Values Heatmap for Package: {one_package}')
            plt.tight_layout()

            miss_plot_dir = os.path.join(
                self.plot_dir, "missing_values_by_package")
            os.makedirs(miss_plot_dir, exist_ok=True)
            save_path = os.path.join(
                miss_plot_dir, f"missing_values_heatmap_{one_package}.png")
            plt.savefig(save_path)
            plt.close()  # Close the figure to free memory

            print(
                f"Saved missing values heatmap for package '{one_package}': {save_path}")

        return None

    def visualize_missings(self):
        """Visualize missing values in the dataset."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.customer_data.isnull(),
            cbar=False,
            cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.tight_layout()

        miss_plot_dir = os.path.join(self.plot_dir, "missing_values")
        os.makedirs(miss_plot_dir, exist_ok=True)
        save_path = os.path.join(miss_plot_dir, "missing_values_heatmap.png")
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory

        print(f"Saved missing values heatmap: {save_path}\n\n")
        return None

    def create_bar_plots(self):
        """Create and save bar plots for each categorical column."""
        # Iterate through each column and create a bar plot
        for column in self.sub_columns:
            if pd.api.types.is_categorical_dtype(
                    self.customer_data[column]) or pd.api.types.is_object_dtype(
                    self.customer_data[column]):
                plt.figure(figsize=(12, 10))
                sns.countplot(x=self.customer_data[column])
                plt.title(f'Bar Plot of {column}')
                plt.xlabel(column)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()  # Adjust plot to prevent labels from overlapping

                bar_plot_dir = os.path.join(self.plot_dir, "bar_plots")
                os.makedirs(bar_plot_dir, exist_ok=True)
                save_path = os.path.join(
                    bar_plot_dir, f"{column}_bar_plot.png")
                plt.savefig(save_path)
                plt.cla()
                plt.clf()
                plt.close()  # Close the figure to free memory

                print(f"Saved bar plot for column: {save_path}")

    def create_barplots_bypackage(self):
        """Create and save bar plots for each categorical column by package."""
        # Iterate through each column and create a bar plot
        for one_package in self.unique_packages:
            for column in self.sub_columns:
                if pd.api.types.is_categorical_dtype(
                        self.customer_data[column]) or pd.api.types.is_object_dtype(
                        self.customer_data[column]):
                    plt.figure(figsize=(12, 10))
                    sns.countplot(
                        x=self.customer_data.loc[
                            self.customer_data[self.package_var] == one_package, column])
                    plt.title(
                        f'Bar Plot of {column} for Package: {one_package}')
                    plt.xlabel(column)
                    plt.ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()  # Adjust plot to prevent labels from overlapping

                    bar_plot_dir = os.path.join(
                        self.plot_dir, "bar_plots_by_package")
                    os.makedirs(bar_plot_dir, exist_ok=True)
                    save_path = os.path.join(
                        bar_plot_dir, f"{one_package}__{column}_bar_plot.png")
                    plt.savefig(save_path)
                    plt.cla()
                    plt.clf()
                    plt.close()  # Close the figure to free memory

                    print(f"Saved bar plot for column: {save_path}")

        plt.close('all')  # Close all open figures
        print("All bar plots by package have been saved.\n\n")
        return None

    def create_boxplots_bypackage(self):
        """Create and save box plots for each numerical column by package."""
        # Iterate through each column and create a box plot
        for one_package in self.unique_packages:
            for column in self.sub_columns:
                if pd.api.types.is_numeric_dtype(self.customer_data[column]):
                    plt.figure(figsize=(12, 10))
                    sns.boxplot(
                        y=self.customer_data.loc[
                            self.customer_data[self.package_var] == one_package, column])
                    plt.title(
                        f'Box Plot of {column} for Package: {one_package}')
                    plt.ylabel(column)
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()  # Adjust plot to prevent labels from overlapping

                    box_plot_dir = os.path.join(
                        self.plot_dir, "box_plots_by_package")
                    os.makedirs(box_plot_dir, exist_ok=True)
                    save_path = os.path.join(
                        box_plot_dir, f"{one_package}__{column}_box_plot.png")
                    plt.savefig(save_path)
                    plt.cla()
                    plt.clf()
                    plt.close()  # Close the figure to free memory

                    print(f"Saved box plot for column: {save_path}")

        plt.close('all')  # Close all open figures
        print("All box plots by package have been saved.\n\n")
        return None

    def create_boxplots(self):
        """Create and save box plots for each numerical column."""
        # Iterate through each column and create a box plot
        for column in self.sub_columns:
            if pd.api.types.is_numeric_dtype(self.customer_data[column]):
                plt.figure(figsize=(12, 10))
                sns.boxplot(y=self.customer_data[column])
                plt.title(f'Box Plot of {column}')
                plt.ylabel(column)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()  # Adjust plot to prevent labels from overlapping

                box_plot_dir = os.path.join(self.plot_dir, "box_plots")
                os.makedirs(box_plot_dir, exist_ok=True)
                save_path = os.path.join(
                    box_plot_dir, f"{column}_box_plot.png")
                plt.savefig(save_path)
                plt.cla()
                plt.clf()
                plt.close()  # Close the figure to free memory

                print(f"Saved box plot for column: {save_path}")

        plt.close('all')  # Close all open figures
        print("All box plots have been saved.\n\n")
        return None

    def visualize_summary_statistics(self):
        """Visualize summary statistics for numerical columns."""
        numeric_columns = [
            col for col in self.sub_columns if pd.api.types.is_numeric_dtype(
                self.customer_data[col])]

        summary_stats = self.customer_data[numeric_columns].describe().T

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            summary_stats,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            cbar=True)
        plt.title('Summary Statistics Heatmap')
        plt.tight_layout()

        summary_plot_dir = os.path.join(self.plot_dir, "summary_statistics")
        os.makedirs(summary_plot_dir, exist_ok=True)
        save_path = os.path.join(
            summary_plot_dir,
            "summary_statistics_heatmap.png")
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory

        print(f"Saved summary statistics heatmap: {save_path}\n\n")
        return None

    def visualize_correlation_matrix(self):
        """Visualize correlation matrix for numerical columns."""
        numeric_columns = [
            col for col in self.sub_columns if pd.api.types.is_numeric_dtype(
                self.customer_data[col])]

        corr_matrix = self.customer_data[numeric_columns].corr()

        plt.figure(figsize=(20, 15))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap='coolwarm',
            cbar=True)
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout()

        corr_plot_dir = os.path.join(self.plot_dir, "correlation_matrix")
        os.makedirs(corr_plot_dir, exist_ok=True)
        save_path = os.path.join(
            corr_plot_dir,
            "correlation_matrix_heatmap.png")
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory

        print(f"Saved correlation matrix heatmap: {save_path}\n\n")
        return None

    def visualize_summary_statistics_bypackage(self):
        """Visualize summary statistics for numerical columns by package."""
        if self.package_var not in self.customer_data.columns:
            print(
                f"Package variable '{self.package_var}' not found in data. Skipping summary statistics visualization by package."
            )
            return None

        numeric_columns = [
            col for col in self.sub_columns if pd.api.types.is_numeric_dtype(
                self.customer_data[col])]

        for one_package in self.unique_packages:
            data_subset = self.customer_data.loc[
                self.customer_data[self.package_var] == one_package]
            summary_stats = data_subset[numeric_columns].describe().T

            plt.figure(figsize=(12, 10))
            sns.heatmap(
                summary_stats,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                cbar=True)
            plt.title(f'Summary Statistics Heatmap for Package: {one_package}')
            plt.tight_layout()

            summary_plot_dir = os.path.join(
                self.plot_dir, "summary_statistics_by_package")
            os.makedirs(summary_plot_dir, exist_ok=True)
            save_path = os.path.join(
                summary_plot_dir,
                f"summary_statistics_heatmap_{one_package}.png")
            plt.savefig(save_path)
            plt.close()  # Close the figure to free memory

            print(
                f"Saved summary statistics heatmap for package '{one_package}': {save_path}")

        return None

    def create_hist_plots(self):
        """Create and save distribution plots for each column."""
        # Iterate through each column and create a distribution plot
        for column in self.sub_columns:
            # Create a new figure for each plot for better separation
            plt.figure(figsize=(8, 5))

            if pd.api.types.is_numeric_dtype(self.customer_data[column]):
                # For numerical columns, use a histogram with KDE
                sns.histplot(self.customer_data[column], kde=True)
                plt.title(f'Distribution of {column} (Numerical)')
                plt.xticks(rotation=45, ha='right')
                plt.xlabel(column)
                plt.ylabel('Frequency / Density')
            else:
                # For categorical columns, use a count plot
                sns.countplot(x=self.customer_data[column])
                plt.title(f'Distribution of {column} (Categorical)')
                plt.xlabel(column)
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Count')

            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()  # Adjust plot to prevent labels from overlapping

            hist_plot_dir = os.path.join(self.plot_dir, "hist_dir")
            os.makedirs(hist_plot_dir, exist_ok=True)
            save_path = os.path.join(
                hist_plot_dir, f"{column}_distribution.png")
            plt.savefig(save_path)
            plt.cla()
            plt.clf()
            plt.close()  # Close the figure to free memory

            print(f"Saved hist plot for column: {save_path}")

        plt.close('all')  # Close all open figures
        print("All hist plots have been saved.\n\n")
        return None

    def create_hist_plots_bypackage(self):
        """Create and save distribution plots for each column by package."""
        # Iterate through each column and create a distribution plot
        for one_package in self.unique_packages:
            for column in self.sub_columns:
                # Create a new figure for each plot for better separation
                plt.figure(figsize=(12, 10))

                data_subset = self.customer_data.loc[
                    self.customer_data[self.package_var] == one_package, column]

                if pd.api.types.is_numeric_dtype(data_subset):
                    # For numerical columns, use a histogram with KDE
                    sns.histplot(data_subset, kde=True)
                    plt.title(
                        f'Distribution of {column} (Numerical) for Package: {one_package}')
                    plt.xticks(rotation=45, ha='right')
                    plt.xlabel(column)
                    plt.ylabel('Frequency / Density')
                else:
                    # For categorical columns, use a count plot
                    sns.countplot(x=data_subset)
                    plt.title(
                        f'Distribution of {column} (Categorical) for Package: {one_package}')
                    plt.xlabel(column)
                    plt.xticks(rotation=45)
                    plt.ylabel('Count')

                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()  # Adjust plot to prevent labels from overlapping

                hist_plot_dir = os.path.join(
                    self.plot_dir, "hist_dir_by_package")
                os.makedirs(hist_plot_dir, exist_ok=True)
                save_path = os.path.join(
                    hist_plot_dir, f"{one_package}__{column}_distribution.png")
                plt.savefig(save_path)
                # This function clears the current active axis within the
                # current figure.
                plt.cla()
                plt.clf()  # This function clears the entire current figure.
                plt.close()  # Close the figure to free memory

                print(f"Saved hist plot for column: {save_path}")

        print("All hist plots by package have been saved.\n\n")
        return None

    def create_joint_plots_bypackage(self):
        """Create and save joint plots for each pair of numerical columns."""
        numeric_columns = [
            col for col in self.sub_columns if pd.api.types.is_numeric_dtype(
                self.customer_data[col])]

        # Iterate through each pair of numerical columns and create a joint
        # plot
        for one_package in self.unique_packages:
            for i in range(len(numeric_columns)):
                for j in range(i + 1, len(numeric_columns)):
                    col_x = numeric_columns[i]
                    col_y = numeric_columns[j]

                    # Create a new figure for each plot
                    plt.figure(figsize=(12, 10))

                    g = sns.JointGrid(
                        x=self.customer_data.loc[self.customer_data[self.package_var] == one_package, col_x],
                        y=self.customer_data.loc[self.customer_data[self.package_var] == one_package, col_y]
                    )
                    g.plot_joint(sns.scatterplot, s=100, alpha=.5)
                    g.plot_marginals(sns.histplot, kde=False)
                    g.ax_joint.tick_params(
                        axis='x', labelrotation=45, ha='right')
                    plt.suptitle(f'Joint Plot of {col_x} vs {col_y}', y=0.9)
                    plt.tight_layout()  # Adjust plot to prevent labels from overlapping

                    joint_plot_dir = os.path.join(
                        self.plot_dir, "joint_plot_dir")
                    os.makedirs(joint_plot_dir, exist_ok=True)
                    save_path = os.path.join(
                        joint_plot_dir, f"{one_package}__{col_x}_vs_{col_y}_jointplot.png")
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close()  # Close the figure to free memory
                    plt.close('all')  # Close all open figures
                    print(f"Saved joint plot for columns: {save_path}")

        print("All joint plots have been saved.\n\n")
        return None

    def create_joint_plots(self):
        """Create and save joint plots for each pair of numerical columns."""
        numeric_columns = [
            col for col in self.sub_columns if pd.api.types.is_numeric_dtype(
                self.customer_data[col])]

        # Iterate through each pair of numerical columns and create a joint
        # plot
        for i in range(len(numeric_columns)):
            for j in range(i + 1, len(numeric_columns)):
                col_x = numeric_columns[i]
                col_y = numeric_columns[j]

                # Create a new figure for each plot
                plt.figure(figsize=(12, 10))

                g = sns.JointGrid(
                    x=self.customer_data[col_x],
                    y=self.customer_data[col_y]
                )
                g.plot_joint(sns.scatterplot, s=100, alpha=.5)
                g.plot_marginals(sns.histplot, kde=False)
                plt.suptitle(f'Joint Plot of {col_x} vs {col_y}')
                plt.xticks(rotation=45)
                plt.tight_layout()  # Adjust plot to prevent labels from overlapping

                joint_plot_dir = os.path.join(
                    self.plot_dir, "joint_plot_dir")
                os.makedirs(joint_plot_dir, exist_ok=True)
                save_path = os.path.join(
                    joint_plot_dir, f"{col_x}_vs_{col_y}_jointplot.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()  # Close the figure to free memory
                print(f"Saved joint plot for columns: {save_path}")

        print("All joint plots have been saved.\n\n")
        return None

    def create_pair_plots(self):
        """Create and save pair plots for numerical columns."""
        numeric_columns = [
            col for col in self.sub_columns if pd.api.types.is_numeric_dtype(
                self.customer_data[col])]

        plt.figure(figsize=(12, 10))  # Create a new figure for the pair plot
        sns.pairplot(
            self.customer_data,
            vars=numeric_columns,
            diag_kind='kde')
        plt.suptitle('Pair Plot of Numerical Columns', y=0.9)
        plt.tight_layout()  # Adjust plot to prevent labels from overlapping

        pair_plot_dir = os.path.join(self.plot_dir, "pair_plots")
        os.makedirs(pair_plot_dir, exist_ok=True)
        save_path = os.path.join(pair_plot_dir, "pair_plot.png")
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory

        print(f"Saved pair plot: {save_path}\n\n")
        return None

    def create_pair_plots_bypackage(self):
        """Create and save pair plots for numerical columns by package."""
        numeric_columns = [
            col for col in self.sub_columns if pd.api.types.is_numeric_dtype(
                self.customer_data[col])]

        if self.package_var not in self.customer_data.columns:
            print(
                f"Package variable '{self.package_var}' not found in data. Skipping pair plot creation.")
            return None

        for one_package in self.unique_packages:
            # Create a new figure for the pair plot
            plt.figure(figsize=(12, 10))
            sns.pairplot(
                self.customer_data.loc[self.customer_data[self.package_var] == one_package],
                vars=numeric_columns,
                diag_kind='kde')
            plt.suptitle('Pair Plot of Numerical Columns by Package', y=0.9)
            plt.tight_layout()  # Adjust plot to prevent labels from overlapping
            pair_plot_dir = os.path.join(self.plot_dir, "pair_plots_bypackage")
            os.makedirs(pair_plot_dir, exist_ok=True)
            save_path = os.path.join(
                pair_plot_dir, f"pair_plot_by_package_{one_package}.png")
            plt.savefig(save_path)
            plt.close()  # Close the figure to free memory
            print(f"Saved pair plot by package '{one_package}': {save_path}")

        print(f"Saved pair plot by package: {save_path}\n\n")
        return None

    def jointplot_with_package_variable(self):
        """Create and save joint plot for two columns with package variable."""
        if self.package_var not in self.customer_data.columns:
            print(
                f"Package variable '{self.package_var}' not found in data. Skipping joint plot creation.")
            return None

        col_x = self.package_var
        numeric_columns = [
            col for col in self.sub_columns if pd.api.types.is_numeric_dtype(
                self.customer_data[col])]

        for col_y in numeric_columns:
            plt.figure(figsize=(12, 10))

            g = sns.JointGrid(
                x=self.customer_data[col_x],
                y=self.customer_data[col_y]
            )
            g.plot_joint(sns.scatterplot, s=100, alpha=.5)
            g.plot_marginals(sns.histplot, kde=False)
            plt.suptitle(f'Joint Plot of {col_x} vs {col_y}')
            g.ax_joint.tick_params(axis='x', labelrotation=45)
            plt.tight_layout()  # Adjust plot to prevent labels from overlapping

            joint_plot_dir = os.path.join(
                self.plot_dir, "joint_plot_dir_by_package_variable")
            os.makedirs(joint_plot_dir, exist_ok=True)
            save_path = os.path.join(
                joint_plot_dir, f"{col_x}_vs_{col_y}_jointplot.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            print(f"Saved joint plot for columns: {save_path}")

        print("All joint plots by package variable have been saved.\n\n")
        return None

    def create_summary_statistics_for_all_packages(self):
        """Create summary statistics for all packages."""
        if self.package_var not in self.customer_data.columns:
            print(
                f"Package variable '{self.package_var}' not found in data. Skipping summary statistics creation.")
            return None

        summary_stats_dict = {}

        for one_package in self.unique_packages:
            data_subset = self.customer_data.loc[
                self.customer_data[self.package_var] == one_package]
            summary_stats = data_subset.describe(include='all').T
            summary_stats_dict[one_package] = summary_stats

            print(
                f"Summary Statistics for Package: {one_package}\n")
            print(summary_stats, "\n\n")

        return summary_stats_dict

    def create_combined_summary_statistics_by_package_numeric(self):
        """Create combined summary statistics for all packages (numerical only)."""
        numeric_columns = [
            col for col in self.sub_columns if pd.api.types.is_numeric_dtype(
                self.customer_data[col])]
        # create summary statistics table of numerical variables in customer
        # data by package
        combined_summary_stats = pd.DataFrame()
        for one_package in self.unique_packages:
            data_subset = self.customer_data.loc[
                self.customer_data[self.package_var] == one_package, numeric_columns]
            summary_stats = data_subset.describe().T
            summary_stats['package'] = one_package
            combined_summary_stats = pd.concat(
                [combined_summary_stats, summary_stats], axis=0)
        print("Combined Summary Statistics (Numerical) by Package:\n")
        print(combined_summary_stats, "\n\n")
        return combined_summary_stats

    def create_combined_summary_statistics_by_package_categorical(self):
        """Create combined summary statistics for all packages (categorical only)."""
        categorical_columns = [
            col for col in self.sub_columns if pd.api.types.is_categorical_dtype(
                self.customer_data[col]) or pd.api.types.is_object_dtype(
                self.customer_data[col])]
        # create summary statistics table of categorical variables in customer
        # data by package
        combined_summary_stats = pd.DataFrame()
        for one_package in self.unique_packages:
            data_subset = self.customer_data.loc[self.customer_data[self.package_var]
                                                 == one_package, categorical_columns]
            summary_stats = data_subset.describe().T
            summary_stats['package'] = one_package
            combined_summary_stats = pd.concat(
                [combined_summary_stats, summary_stats], axis=0)
        print("Combined Summary Statistics (Categorical) by Package:\n")
        print(combined_summary_stats, "\n\n")
        return combined_summary_stats

    def create_combined_summary_statistics_by_package(self):
        """Create combined summary statistics for all packages."""
        # create summary statistics table of all variables in customer data by package
        # and combine the results into a single DataFrame
        if self.package_var not in self.customer_data.columns:
            print(
                f"Package variable '{self.package_var}' not found in data. Skipping combined summary statistics creation."
            )
            return None

        combined_summary_stats = pd.DataFrame()

        for one_package in self.unique_packages:
            data_subset = self.customer_data.loc[
                self.customer_data[self.package_var] == one_package]
            summary_stats = data_subset.describe(include='all').T
            summary_stats['package'] = one_package
            combined_summary_stats = pd.concat(
                [combined_summary_stats, summary_stats], axis=0)

        print("Combined Summary Statistics by Package:\n")
        print(combined_summary_stats, "\n\n")

        return combined_summary_stats

    def visualize_combined_summary_statistics_by_package_numeric(self):
        """Visualize combined summary statistics for all packages (numerical only)."""
        combined_summary_stats = self.create_combined_summary_statistics_by_package_numeric()
        if combined_summary_stats is None or combined_summary_stats.empty:
            print("No combined summary statistics to visualize.")
            return None

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            combined_summary_stats.select_dtypes(include=['number']),
            annot=True,
            cmap='coolwarm',
            cbar=True)
        plt.title('Combined Summary Statistics Heatmap by Package (Numerical)')
        plt.tight_layout()

        summary_plot_dir = os.path.join(
            self.plot_dir, "combined_summary_statistics_by_package_numeric")
        os.makedirs(summary_plot_dir, exist_ok=True)
        save_path = os.path.join(
            summary_plot_dir,
            "combined_summary_statistics_heatmap_by_package_numeric.png")
        plt.savefig(save_path)
        plt.close()  # Close the figure to free memory

        print(
            f"Saved combined summary statistics heatmap by package (numerical): {save_path}\n\n")
        return None

    def visualize_combined_summary_statistics_by_package(self):
        """Visualize combined summary statistics for all packages."""
        # sns.heatmap capture with try and except - ValueError: zero-size array
        # to reduction operation fmin which has no identity
        combined_summary_stats = self.create_combined_summary_statistics_by_package()
        if combined_summary_stats is None or combined_summary_stats.empty:
            print("No combined summary statistics to visualize.")
            return None

        plt.figure(figsize=(12, 10))
        try:
            sns.heatmap(
                combined_summary_stats.select_dtypes(include=['number']),
                annot=True,
                cmap='coolwarm',
                cbar=True)
            plt.title('Combined Summary Statistics Heatmap by Package')
            plt.tight_layout()
        except ValueError as e:
            print(f"Caught a ValueError: {e}")
            print("This likely means the input data for the heatmap was empty or contained only non-numeric/NaN values.")
        else:
            summary_plot_dir = os.path.join(
                self.plot_dir, "combined_summary_statistics_by_package")
            os.makedirs(summary_plot_dir, exist_ok=True)
            save_path = os.path.join(
                summary_plot_dir,
                "combined_summary_statistics_heatmap_by_package.png")
            plt.savefig(save_path)
        finally:
            plt.cla()
            plt.clf()
            plt.close()  # Close the figure to free memory

        print(
            f"Saved combined summary statistics heatmap by package: {save_path}\n\n")
        return None

    def export_combined_summary_statistics_by_package_numeric(
            self, output_file: str):
        """Export combined summary statistics for all packages (numerical only) to a CSV file."""
        combined_summary_stats = self.create_combined_summary_statistics_by_package_numeric()
        if combined_summary_stats is None or combined_summary_stats.empty:
            print("No combined summary statistics to export.")
            return None

        output_path = os.path.join(self.plot_dir, output_file)
        combined_summary_stats.to_csv(output_path, index=True)
        print(
            f"Exported combined summary statistics by package (numerical) to: {output_path}\n\n")
        return None

    def export_combined_summary_statistics_by_package_categorical(
            self, output_file: str):
        """Export combined summary statistics for all packages (categorical only) to a CSV file."""
        combined_summary_stats = self.create_combined_summary_statistics_by_package_categorical()
        if combined_summary_stats is None or combined_summary_stats.empty:
            print("No combined summary statistics to export.")
            return None

        output_path = os.path.join(self.plot_dir, output_file)
        combined_summary_stats.to_csv(output_path, index=True)
        print(
            f"Exported combined summary statistics by package (categorical) to: {output_path}\n\n")
        return None

    def export_combined_summary_statistics_by_package(self, output_file: str):
        """Export combined summary statistics for all packages to a CSV file."""
        combined_summary_stats = self.create_combined_summary_statistics_by_package()
        if combined_summary_stats is None or combined_summary_stats.empty:
            print("No combined summary statistics to export.")
            return None

        output_path = os.path.join(self.plot_dir, output_file)
        combined_summary_stats.to_csv(output_path, index=True)
        print(
            f"Exported combined summary statistics by package to: {output_path}\n\n")
        return None

    def test_create_plots(self):
        """Create and save distribution plots for each column."""

        print("\n\n")
        print("Creating bar plots:\n")
        _ = self.create_bar_plots()

        return None


if __name__ == "__main__":
    import traceback
    try:
        fileName = "customer_data.parquet"
        eda = ExploratoryDataAnalysis(fileName=fileName)
        eda.create_plots()
        print("\n\n")
        print("EDA execution completed successfully.")
    except Exception as e:
        print("An error occurred during EDA execution:")
        traceback.print_exc()
        sys.exit(1)

# PYTEST_FIX_VERIFICATION_1763396399
