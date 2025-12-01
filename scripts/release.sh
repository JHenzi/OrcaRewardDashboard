#!/bin/bash

# Release Script
# Exports training data, commits changes, and pushes to repository

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory (handles symlinks)
# Follow symlinks to get actual script location
if [ -L "${BASH_SOURCE[0]}" ]; then
    # If script is a symlink, get the actual path
    SCRIPT_DIR="$( cd "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" && pwd )"
else
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
fi

# Project root is parent of scripts directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root (all paths are relative to here)
cd "$PROJECT_ROOT"

echo -e "${GREEN}🚀 Starting release process...${NC}\n"

# Step 1: Export training data
echo -e "${YELLOW}Step 1: Exporting training data...${NC}"
python scripts/export_training_data.py

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Export failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Export complete${NC}\n"

# Step 2: Check for changes
echo -e "${YELLOW}Step 2: Checking for changes...${NC}"
if [ -z "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}⚠️  No changes to commit${NC}"
    exit 0
fi

# Show what will be committed
echo -e "${YELLOW}Changes to be committed:${NC}"
git status --short
echo ""

# Step 3: Git add
echo -e "${YELLOW}Step 3: Staging changes...${NC}"
git add .

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Git add failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Changes staged${NC}\n"

# Step 4: Get commit message
echo -e "${YELLOW}Step 4: Commit message${NC}"
echo -e "Enter commit message (or press Enter for default):"
read -r COMMIT_MSG

if [ -z "$COMMIT_MSG" ]; then
    # Default message with timestamp
    COMMIT_MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${YELLOW}Using default message: ${COMMIT_MSG}${NC}"
fi

# Step 5: Commit
echo -e "\n${YELLOW}Step 5: Committing changes...${NC}"
git commit -m "$COMMIT_MSG"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Commit failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Changes committed${NC}\n"

# Step 6: Push
echo -e "${YELLOW}Step 6: Pushing to remote...${NC}"
read -p "Push to remote? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Push failed!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Pushed to remote${NC}\n"
else
    echo -e "${YELLOW}⚠️  Skipped push${NC}\n"
fi

echo -e "${GREEN}🎉 Release complete!${NC}"

