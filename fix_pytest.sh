#!/bin/bash
echo "=== COMPLETE PYTEST FIX ==="

echo "1. Cleaning caches..."
rm -rf .pytest_cache/ __pycache__/ build/ dist/ *.egg-info/
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

echo "2. Reinstalling package..."
pip uninstall CaseStudy_FinanceRocks -y 2>/dev/null
pip install -e . >/dev/null 2>&1

echo "3. Clearing Python cache..."
python -c "
import sys
modules_to_remove = [k for k in list(sys.modules.keys()) if 'CaseStudy' in k]
for mod in modules_to_remove:
    del sys.modules[mod]
print(f'Removed {len(modules_to_remove)} cached modules')
"

echo "4. Adding test marker..."
echo "" >> src/CaseStudy_FinanceRocks/core.py
echo "# PYTEST_FIX_VERIFICATION_$(date +%s)" >> src/CaseStudy_FinanceRocks/core.py

echo "5. Running pytest..."
python -m pytest tests/ -v --tb=short --cache-clear
