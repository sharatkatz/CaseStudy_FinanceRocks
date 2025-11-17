import sys

print("=== IMPORT DIAGNOSTICS ===")
print(f"Python path: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    print("\n1. Testing package import...")
    import CaseStudy_FinanceRocks
    print(f"   ✓ Package location: {CaseStudy_FinanceRocks.__file__}")
    
    print("\n2. Checking package contents...")
    print(f"   Package attributes: {[x for x in dir(CaseStudy_FinanceRocks) if not x.startswith('_')]}")
    
    print("\n3. Testing core module import...")
    from CaseStudy_FinanceRocks import core
    print("   ✓ Core module imported")
    
    print("\n4. Testing class import...")
    from CaseStudy_FinanceRocks.core import ExploratoryDataAnalysis
    print("   ✓ ExploratoryDataAnalysis class imported")
    
    print("\n5. Testing class attributes...")
    print(f"   Class: {ExploratoryDataAnalysis}")
    print(f"   Doc: {ExploratoryDataAnalysis.__doc__}")
    
    print("\n🎉 ALL IMPORTS SUCCESSFUL!")
    
except ImportError as e:
    print(f"❌ IMPORT FAILED: {e}")
    print(f"Python path entries:")
    for path in sys.path:
        print(f"   {path}")
except Exception as e:
    print(f"❌ OTHER ERROR: {e}")
