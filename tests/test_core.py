#!/usr/bin/env python

"""
This file is used to test the ExploratoryDataAnalysis class.

Summary:
    This file is used to test the ExploratoryDataAnalysis class.
    It is used to test the report_missings method, create_hist_plots method,
    create_barplots_bypackage method, and visualize_missings method.

Flow:
    1. Import the ExploratoryDataAnalysis class from the CaseStudy_FinanceRocks module.
    2. Import the required modules and classes.
    3. Create a fixture for the ExploratoryDataAnalysis class.
    4. Create test cases for the report_missings method.
    5. Run the tests.

Usage:
    pytest test_core.py

Author: Sharat Sharma
Date: November 2025
"""


from CaseStudy_FinanceRocks import ExploratoryDataAnalysis
import sys
import os
import pytest  # type: ignore
import pandas  # type: ignore
import numpy as np  # type: ignore
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import matplotlib
import seaborn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestReportMissings:
    """Test cases for the report_missings method."""
    # -----------------------------------------------------------------------------
    # This fixture creates a mock instance of the ExploratoryDataAnalysis class
    # with a predefined customer_data DataFrame containing missing values.
    # This allows us to test the report_missings method in isolation,
    # without relying on external files or data sources.

    # -----------------------------------------------------------------------------
    # The patch function is used to temporarily replace certain parts of the code
    # with mock objects during testing, allowing isolation of the code under test.
    # The Mock class is used to create mock objects that simulate the behavior of real objects.
    # The patch function is being used here to mock out file system interactions and data loading functions
    # within the ExploratoryDataAnalysis class. Specifically, it mocks:
    """
    1. os.path.join: This is used in the class to construct file paths.
    By mocking it, we prevent any actual file path operations.
    2. pathlib.Path: This is used to handle file paths in a more object-oriented way.
    Mocking it prevents any real file system access.
    3. pandas.read_parquet: This function is used to read Parquet files into pandas DataFrames.
    By mocking it, we avoid loading any real data from disk.
    """
    # This allows us to create a controlled test environment where we can define the data directly
    # in the test, ensuring that our tests are not dependent on external files or data sources.
    # usage:
    # with patch('os.path.join'), \
    #      patch('pathlib.Path'), \
    #      patch('pandas.read_parquet'):
    #         # test code here
    #   # --------------------------------------------------------------------
    #     -----------------------------------------------------------------------------
    # The Mock class is used to create mock objects that simulate the behavior of real objects.
    # In this case, we create a mock instance of the ExploratoryDataAnalysis
    # class.

    @pytest.fixture
    def mock_eda_instance(self):
        """Create a mock EDA instance with test data."""
        with patch('os.path.join'), \
                patch('pathlib.Path'), \
                patch('pandas.read_parquet'):

            eda = Mock(spec=ExploratoryDataAnalysis)
            eda.customer_data = pandas.DataFrame({
                'col1': [1, 2, np.nan, 4],
                'col2': [np.nan, np.nan, 3, 4],
                'col3': [1, 2, 3, 4],
                'col4': [np.nan, np.nan, np.nan, np.nan]
            })
            eda.report_missings = ExploratoryDataAnalysis.report_missings.__get__(
                eda)
            return eda

    def test_report_missings_returns_series(self, mock_eda_instance):
        """Test that report_missings returns a pandas Series."""
        result = mock_eda_instance.report_missings()
        assert isinstance(result, pandas.Series)

    def test_report_missings_contains_only_columns_with_missing_values(
            self, mock_eda_instance):
        """Test that only columns with missing values are included."""
        result = mock_eda_instance.report_missings()
        assert 'col3' not in result.index
        assert 'col1' in result.index
        assert 'col2' in result.index
        assert 'col4' in result.index

    def test_report_missings_correct_counts(self, mock_eda_instance):
        """Test that missing value counts are correct."""
        result = mock_eda_instance.report_missings()
        assert result['col1'] == 1
        assert result['col2'] == 2
        assert result['col4'] == 4

    def test_report_missings_no_missing_values(self):
        """Test when there are no missing values."""
        with patch('os.path.join'), \
                patch('pathlib.Path'), \
                patch('pandas.read_parquet'):

            eda = Mock(spec=ExploratoryDataAnalysis)
            eda.customer_data = pandas.DataFrame({
                'col1': [1, 2, 3, 4],
                'col2': [5, 6, 7, 8]
            })
            eda.report_missings = ExploratoryDataAnalysis.report_missings.__get__(
                eda)

            result = eda.report_missings()
            assert len(result) == 0
            assert isinstance(result, pandas.Series)

    def test_report_missings_empty_dataframe(self):
        """Test with an empty dataframe."""
        with patch('os.path.join'), \
                patch('pathlib.Path'), \
                patch('pandas.read_parquet'):

            eda = Mock(spec=ExploratoryDataAnalysis)
            eda.customer_data = pandas.DataFrame()
            eda.report_missings = ExploratoryDataAnalysis.report_missings.__get__(
                eda)

            result = eda.report_missings()
            assert len(result) == 0
            assert isinstance(result, pandas.Series)


class TestVisualization:

    def test_create_hist_plots(self):
        """Simplified debug test without extensive prints."""
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pandas.read_parquet') as mock_read_parquet:
                # Use a REAL DataFrame to avoid MagicMock issues
                real_df = pandas.DataFrame({
                    'col1': [1, 2, 3, 4, 5],
                    'col2': ['A', 'B', 'A', 'C', 'B'],
                })
                mock_read_parquet.return_value = real_df

                # Mock file system operations
                with patch('os.path.exists', return_value=False):
                    with patch('os.makedirs'):
                        with patch('shutil.rmtree'):
                            # Mock visualization
                            with patch('matplotlib.pyplot.savefig'):
                                with patch('matplotlib.pyplot.close'):
                                    with patch('matplotlib.pyplot.figure'):
                                        with patch('seaborn.histplot') as mock_histplot:
                                            with patch('seaborn.countplot') as mock_countplot:
                                                # Mock the methods called in
                                                # __init__ that might cause
                                                # issues
                                                with patch.object(ExploratoryDataAnalysis, 'missing_reports'):
                                                    with patch.object(ExploratoryDataAnalysis, 'export_combined_summary_statistics_by_package'):
                                                        try:
                                                            eda = ExploratoryDataAnalysis(
                                                                '/fake/path', 'data.parquet')
                                                            eda.create_hist_plots()
                                                            mock_histplot.assert_called()
                                                            mock_countplot.assert_called()
                                                            print(
                                                                "✅ SUCCESS: Test passed!")
                                                        except Exception as e:
                                                            print(
                                                                f"❌ ERROR: {e}")
                                                            print(
                                                                f"❌ ERROR TYPE: {type(e).__name__}")
                                                            import traceback
                                                            traceback.print_exc()
                                                            raise

    def test_create_boxplots(self):
        """Test box plot creation."""
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pandas.read_parquet') as mock_read_parquet:
                real_df = pandas.DataFrame({
                    'numeric_col1': [1, 2, 3, 4, 5],
                    'numeric_col2': [10, 20, 30, 40, 50],
                    'package': ['pkg1', 'pkg2', 'pkg1', 'pkg2', 'pkg1']
                })
                mock_read_parquet.return_value = real_df

                with patch('os.path.exists', return_value=False):
                    with patch('os.makedirs'):
                        with patch('shutil.rmtree'):
                            with patch('matplotlib.pyplot.savefig'):
                                with patch('matplotlib.pyplot.close'):
                                    with patch('matplotlib.pyplot.figure'):
                                        with patch('seaborn.boxplot') as mock_boxplot:
                                            with patch.object(ExploratoryDataAnalysis, 'missing_reports'):
                                                with patch.object(ExploratoryDataAnalysis, 'export_combined_summary_statistics_by_package'):
                                                    try:
                                                        eda = ExploratoryDataAnalysis(
                                                            '/fake/path', 'data.parquet')
                                                        eda.create_boxplots()
                                                        mock_boxplot.assert_called()
                                                    except Exception as e:
                                                        print(f"❌ ERROR: {e}")
                                                        print(
                                                            f"❌ ERROR TYPE: {type(e).__name__}")
                                                        import traceback
                                                        traceback.print_exc()
                                                        raise

    def test_create_boxplots_bypackage(self):
        """Test box plot creation by package."""
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pandas.read_parquet') as mock_read_parquet:
                # Use real DataFrame
                real_df = pandas.DataFrame({
                    # Include some None values for missing data
                    'col1': [1, 2, 3, None, 5],
                    'col2': [10, None, 30, 40, 50],
                    'package': ['pkg1', 'pkg2', 'pkg1', 'pkg2', 'pkg1']
                })
                mock_read_parquet.return_value = real_df

                with patch('os.path.exists', return_value=False):
                    with patch('os.makedirs'):
                        with patch('shutil.rmtree'):
                            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                                with patch('matplotlib.pyplot.close'):
                                    with patch('matplotlib.pyplot.figure'):
                                        with patch('seaborn.boxplot') as mock_boxplot:
                                            with patch.object(ExploratoryDataAnalysis, 'missing_reports'):
                                                with patch.object(ExploratoryDataAnalysis, 'export_combined_summary_statistics_by_package'):
                                                    eda = ExploratoryDataAnalysis(
                                                        '/fake/path', 'data.parquet')
                                                    print(
                                                        f"Package var: {eda.package_var}")
                                                    print(
                                                        f"Unique packages: {
                                                            eda.unique_packages}")
                                                    print(
                                                        f"Package in columns: {
                                                            'package' in eda.customer_data.columns}")
                                                    # Call the method
                                                    eda.create_boxplots_bypackage()
                                                    print(
                                                        f"Boxplot called: {
                                                            mock_boxplot.called}")
                                                    print(
                                                        f"Savefig called: {
                                                            mock_savefig.called}")
                                                    if mock_boxplot.called:
                                                        print(
                                                            f"Boxplot call args: {
                                                                mock_boxplot.call_args}")
                                        mock_boxplot.assert_called()

    def test_create_bar_plots(self):
        """Test bar plot creation."""
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pandas.read_parquet') as mock_read_parquet:
                real_df = pandas.DataFrame({
                    'category_col': ['A', 'B', 'A', 'C', 'B', 'A'],
                    'numeric_col': [1, 2, 3, 4, 5, 6],
                    'package': ['pkg1', 'pkg2', 'pkg1', 'pkg2', 'pkg1', 'pkg2']
                })
                mock_read_parquet.return_value = real_df

                with patch('os.path.exists', return_value=False):
                    with patch('os.makedirs'):
                        with patch('shutil.rmtree'):
                            with patch('matplotlib.pyplot.savefig'):
                                with patch('matplotlib.pyplot.close'):
                                    with patch('matplotlib.pyplot.figure'):
                                        with patch('seaborn.countplot') as mock_countplot:
                                            with patch.object(ExploratoryDataAnalysis, 'missing_reports'):
                                                with patch.object(ExploratoryDataAnalysis, 'export_combined_summary_statistics_by_package'):
                                                    try:
                                                        eda = ExploratoryDataAnalysis(
                                                            '/fake/path', 'data.parquet')
                                                        eda.create_bar_plots()
                                                    except Exception as e:
                                                        print(f"❌ ERROR: {e}")
                                                        print(
                                                            f"❌ ERROR TYPE: {type(e).__name__}")
                                                        import traceback
                                                        traceback.print_exc()
                                                        raise
                                            mock_countplot.assert_called()

    def test_create_barplots_bypackage(self):
        """Test bar plot creation by package."""
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pandas.read_parquet') as mock_read_parquet:
                real_df = pandas.DataFrame({
                    'category_col': ['A', 'B', 'A', 'C'],
                    'numeric_col': [1, 2, 3, 4],
                    'package': ['pkg1', 'pkg2', 'pkg1', 'pkg2']
                })
                mock_read_parquet.return_value = real_df

                with patch('os.path.exists', return_value=False):
                    with patch('os.makedirs'):
                        with patch('shutil.rmtree'):
                            with patch('matplotlib.pyplot.savefig'):
                                with patch('matplotlib.pyplot.close'):
                                    with patch('matplotlib.pyplot.figure'):
                                        with patch('seaborn.countplot') as mock_countplot:
                                            with patch('pandas.api.types.is_categorical_dtype') as mock_is_categorical:
                                                with patch('pandas.api.types.is_object_dtype') as mock_is_object:
                                                    mock_is_categorical.side_effect = lambda x: x == 'category_col'
                                                    mock_is_object.side_effect = lambda x: x == 'category_col'

                                                    with patch.object(ExploratoryDataAnalysis, 'missing_reports'):
                                                        with patch.object(ExploratoryDataAnalysis, 'export_combined_summary_statistics_by_package'):
                                                            eda = ExploratoryDataAnalysis(
                                                                '/fake/path', 'data.parquet')

                                                            # SAFER DEBUG:
                                                            # Check data types
                                                            # individually
                                                            print(
                                                                "Column type analysis:")
                                                            for col in eda.sub_columns:
                                                                col_data = eda.customer_data[col]
                                                                print(
                                                                    f"  {col}: dtype={
                                                                        col_data.dtype}, object_dtype={
                                                                        pandas.api.types.is_object_dtype(col_data)}, categorical_dtype={
                                                                        pandas.api.types.is_categorical_dtype(col_data)}")

                                                            # Call the method
                                                            eda.create_barplots_bypackage()
                                                            mock_countplot.assert_called()

    def test_visualize_missings(self):
        """Test missing values visualization."""
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pandas.read_parquet') as mock_read_parquet:
                # Create mock dataframe
                mock_df = MagicMock()
                mock_df.columns = ['col1', 'col2']
                mock_df.isnull.return_value = pandas.DataFrame({
                    'col1': [False, True, False],
                    'col2': [True, False, True]
                })
                mock_read_parquet.return_value = mock_df

                with patch('os.path.join', return_value='/fake/path/data.parquet'):
                    with patch('os.makedirs'):
                        with patch('matplotlib.pyplot.savefig'):
                            with patch('shutil.rmtree'):
                                with patch('matplotlib.pyplot.close'):
                                    with patch('matplotlib.pyplot.figure'):
                                        # ← SPECIFIC function
                                        with patch('seaborn.heatmap') as mock_heatmap:
                                            eda = ExploratoryDataAnalysis(
                                                '/fake/path', 'data.parquet')
                                            eda.visualize_missings()
                                        # Verify visualization was called
                                    mock_heatmap.assert_called()

    def test_visualize_missings_bypackage(self):
        """Test missing values visualization by package."""
        with patch('pathlib.Path.is_file', return_value=True):
            with patch('pandas.read_parquet') as mock_read_parquet:
                # Use real DataFrame
                real_df = pandas.DataFrame({
                    # Include some None values for missing data
                    'col1': [1, 2, 3, None, 5],
                    'col2': [10, None, 30, 40, 50],
                    'package': ['pkg1', 'pkg2', 'pkg1', 'pkg2', 'pkg1']
                })
                mock_read_parquet.return_value = real_df

                with patch('os.path.exists', return_value=False):
                    with patch('os.makedirs'):
                        with patch('shutil.rmtree'):
                            with patch('matplotlib.pyplot.savefig') as mock_savefig:
                                with patch('matplotlib.pyplot.close'):
                                    with patch('matplotlib.pyplot.figure'):
                                        with patch('seaborn.heatmap') as mock_heatmap:
                                            with patch.object(ExploratoryDataAnalysis, 'missing_reports'):
                                                with patch.object(ExploratoryDataAnalysis, 'export_combined_summary_statistics_by_package'):
                                                    eda = ExploratoryDataAnalysis(
                                                        '/fake/path', 'data.parquet')

                                                    print(
                                                        f"Package var: {eda.package_var}")
                                                    print(
                                                        f"Unique packages: {
                                                            eda.unique_packages}")
                                                    print(
                                                        f"Package in columns: {
                                                            'package' in eda.customer_data.columns}")

                                                    # Call the method
                                                    eda.visualize_missings_bypackage()

                                                    print(
                                                        f"Heatmap called: {
                                                            mock_heatmap.called}")
                                                    print(
                                                        f"Savefig called: {
                                                            mock_savefig.called}")
                                                    if mock_heatmap.called:
                                                        print(
                                                            f"Heatmap call args: {
                                                                mock_heatmap.call_args}")

                                        mock_heatmap.assert_called()
