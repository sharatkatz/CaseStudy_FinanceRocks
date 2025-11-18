import sys
print("=== Debugging imports ===")

# Import the package
import CaseStudy_FinanceRocks
print(f"Package file: {CaseStudy_FinanceRocks.__file__}")

# Check what's available
print(f"Available in package: {[x for x in dir(CaseStudy_FinanceRocks) if not x.startswith('_')]}")

# Try to import the class
try:
    from CaseStudy_FinanceRocks import ExploratoryDataAnalysis
    print("✓ ExploratoryDataAnalysis imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")

# Check __init__.py content
import inspect
print(f"__init__.py location: {inspect.getfile(CaseStudy_FinanceRocks)}")
