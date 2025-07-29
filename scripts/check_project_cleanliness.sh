#!/bin/bash

# Project Cleanliness Check Script
# This script checks for common issues that might affect project cleanliness

echo "🔍 NeMo2 Qwen2.5B Japanese Fine-tuning Project Cleanliness Check"
echo "================================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check for temporary files
echo -e "\n1. Checking for temporary files..."
TEMP_FILES=$(find . -type f \( -name "*.tmp" -o -name "*.temp" -o -name "*.log" -o -name "*.cache" -o -name ".DS_Store" -o -name "*.swp" -o -name "*.swo" -o -name "*~" \) -not -path "./.git/*" 2>/dev/null | wc -l)
if [ "$TEMP_FILES" -eq 0 ]; then
    echo -e "${GREEN}✓ No temporary files found${NC}"
else
    echo -e "${YELLOW}⚠ Found $TEMP_FILES temporary files${NC}"
    find . -type f \( -name "*.tmp" -o -name "*.temp" -o -name "*.log" -o -name "*.cache" -o -name ".DS_Store" -o -name "*.swp" -o -name "*.swo" -o -name "*~" \) -not -path "./.git/*" 2>/dev/null | head -10
fi

# 2. Check for large files not in .gitignore
echo -e "\n2. Checking for large files (>10MB)..."
LARGE_FILES=$(find . -type f -size +10M -not -path "./.git/*" -exec ls -lh {} \; 2>/dev/null | wc -l)
if [ "$LARGE_FILES" -eq 0 ]; then
    echo -e "${GREEN}✓ No large files found${NC}"
else
    echo -e "${YELLOW}⚠ Found $LARGE_FILES large files (>10MB)${NC}"
    find . -type f -size +10M -not -path "./.git/*" -exec ls -lh {} \; 2>/dev/null | head -5
fi

# 3. Check for __pycache__ directories
echo -e "\n3. Checking for __pycache__ directories..."
PYCACHE_DIRS=$(find . -type d -name "__pycache__" -not -path "./.git/*" 2>/dev/null | wc -l)
if [ "$PYCACHE_DIRS" -eq 0 ]; then
    echo -e "${GREEN}✓ No __pycache__ directories found${NC}"
else
    echo -e "${YELLOW}⚠ Found $PYCACHE_DIRS __pycache__ directories${NC}"
fi

# 4. Check git status
echo -e "\n4. Checking git status..."
if [ -d .git ]; then
    UNTRACKED=$(git ls-files --others --exclude-standard | wc -l)
    MODIFIED=$(git diff --name-only | wc -l)
    STAGED=$(git diff --cached --name-only | wc -l)
    
    if [ "$UNTRACKED" -eq 0 ] && [ "$MODIFIED" -eq 0 ] && [ "$STAGED" -eq 0 ]; then
        echo -e "${GREEN}✓ Working directory is clean${NC}"
    else
        echo -e "${YELLOW}⚠ Git status:${NC}"
        [ "$UNTRACKED" -gt 0 ] && echo "  - $UNTRACKED untracked files"
        [ "$MODIFIED" -gt 0 ] && echo "  - $MODIFIED modified files"
        [ "$STAGED" -gt 0 ] && echo "  - $STAGED staged files"
    fi
else
    echo -e "${RED}✗ Not a git repository${NC}"
fi

# 5. Check for empty directories
echo -e "\n5. Checking for empty directories..."
EMPTY_DIRS=$(find . -type d -empty -not -path "./.git/*" 2>/dev/null | wc -l)
if [ "$EMPTY_DIRS" -eq 0 ]; then
    echo -e "${GREEN}✓ No empty directories found${NC}"
else
    echo -e "${YELLOW}⚠ Found $EMPTY_DIRS empty directories${NC}"
fi

# 6. Check Python code quality (if flake8 is installed)
echo -e "\n6. Checking Python code quality..."
if command_exists flake8; then
    FLAKE8_ERRORS=$(flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics 2>/dev/null || echo "0")
    if [ "$FLAKE8_ERRORS" = "0" ]; then
        echo -e "${GREEN}✓ No critical Python syntax errors found${NC}"
    else
        echo -e "${RED}✗ Found Python syntax errors${NC}"
    fi
else
    echo -e "${YELLOW}⚠ flake8 not installed, skipping Python quality check${NC}"
fi

# 7. Check for TODO/FIXME comments
echo -e "\n7. Checking for TODO/FIXME comments..."
TODO_COUNT=$(grep -r "TODO\|FIXME" . --include="*.py" --include="*.sh" --include="*.md" --exclude-dir=".git" 2>/dev/null | wc -l)
if [ "$TODO_COUNT" -eq 0 ]; then
    echo -e "${GREEN}✓ No TODO/FIXME comments found${NC}"
else
    echo -e "${YELLOW}⚠ Found $TODO_COUNT TODO/FIXME comments${NC}"
    grep -r "TODO\|FIXME" . --include="*.py" --include="*.sh" --include="*.md" --exclude-dir=".git" 2>/dev/null | head -5
fi

# 8. Check if README exists
echo -e "\n8. Checking documentation..."
if [ -f "README.md" ]; then
    echo -e "${GREEN}✓ README.md exists${NC}"
else
    echo -e "${RED}✗ README.md not found${NC}"
fi

# Summary
echo -e "\n================================================================"
echo "Cleanliness check complete!"
echo ""
echo "To clean up temporary files, run:"
echo "  find . -type f \\( -name '*.tmp' -o -name '*.temp' -o -name '*.log' -o -name '.DS_Store' \\) -delete"
echo ""
echo "To remove __pycache__ directories, run:"
echo "  find . -type d -name '__pycache__' -exec rm -rf {} +"
echo "" 