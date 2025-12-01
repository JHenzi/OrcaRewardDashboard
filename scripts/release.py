#!/usr/bin/env python3
"""
Release Script

Exports training data, commits changes, and pushes to repository.
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

# Colors for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def print_step(step_num, message):
    """Print a step message."""
    print(f"\n{Colors.YELLOW}Step {step_num}: {message}{Colors.NC}")


def print_success(message):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.NC}")


def print_error(message):
    """Print error message."""
    print(f"{Colors.RED}❌ {message}{Colors.NC}")


def print_warning(message):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")


def get_project_root():
    """Get project root directory, handling symlinks."""
    # Get the directory of this script (even if it's a symlink)
    if os.path.islink(__file__):
        # If script is a symlink, resolve it
        script_path = os.path.realpath(__file__)
    else:
        script_path = __file__
    
    script_dir = os.path.dirname(os.path.abspath(script_path))
    # Project root is parent of scripts directory
    project_root = os.path.dirname(script_dir)
    return project_root


def run_command(cmd, check=True, capture_output=False, cwd=None):
    """Run a shell command."""
    if cwd is None:
        cwd = get_project_root()
    
    try:
        if capture_output:
            result = subprocess.run(
                cmd,
                shell=True,
                check=check,
                capture_output=True,
                text=True,
                cwd=cwd
            )
            return result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=True, check=check, cwd=cwd)
            return None
    except subprocess.CalledProcessError as e:
        if check:
            print_error(f"Command failed: {cmd}")
            print_error(f"Error: {e}")
            sys.exit(1)
        return None


def export_training_data(cutoff_date=None):
    """Export training data to committed location (latest)."""
    print_step(1, "Exporting training data (latest, will be committed)...")
    
    project_root = get_project_root()
    export_script = os.path.join(project_root, "scripts", "export_training_data.py")
    
    cmd = ["python3", export_script]
    if cutoff_date:
        cmd.extend(["--cutoff-date", cutoff_date])
    # Don't use --archive, so it exports to latest location (committed)
    
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        print_success("Export complete (to training_data/export/)")
        return True
    except subprocess.CalledProcessError:
        print_error("Export failed!")
        return False


def check_git_changes():
    """Check if there are changes to commit."""
    print_step(2, "Checking for changes...")
    
    project_root = get_project_root()
    
    # Check if we're in a git repo
    if not os.path.exists(os.path.join(project_root, ".git")):
        print_error("Not a git repository!")
        return False
    
    # Check for changes
    output = run_command("git status --porcelain", capture_output=True)
    
    if not output:
        print_warning("No changes to commit")
        return False
    
    # Show changes
    print(f"\n{Colors.BLUE}Changes to be committed:{Colors.NC}")
    run_command("git status --short", check=False)
    return True


def stage_changes():
    """Stage all changes."""
    print_step(3, "Staging changes...")
    
    run_command("git add .")
    print_success("Changes staged")


def get_commit_message(default=None):
    """Get commit message from user."""
    print_step(4, "Commit message")
    
    if default:
        print(f"Default message: {Colors.YELLOW}{default}{Colors.NC}")
        response = input("Enter commit message (or press Enter for default): ").strip()
    else:
        response = input("Enter commit message: ").strip()
    
    if not response:
        if default:
            return default
        else:
            # Generate default with timestamp
            return f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    return response


def commit_changes(message):
    """Commit changes."""
    print_step(5, "Committing changes...")
    
    run_command(f'git commit -m "{message}"')
    print_success("Changes committed")


def push_changes(force=False, skip_prompt=False):
    """Push changes to remote."""
    print_step(6, "Pushing to remote...")
    
    if not skip_prompt:
        response = input("Push to remote? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print_warning("Skipped push")
            return False
    
    cmd = "git push"
    if force:
        cmd += " --force"
    
    try:
        run_command(cmd)
        print_success("Pushed to remote")
        return True
    except SystemExit:
        print_error("Push failed!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Release script: export data, commit, and push")
    parser.add_argument(
        '--skip-export',
        action='store_true',
        help='Skip training data export'
    )
    parser.add_argument(
        '--cutoff-date',
        type=str,
        help='Cutoff date for export (ISO format)'
    )
    parser.add_argument(
        '--message', '-m',
        type=str,
        help='Commit message (skips prompt)'
    )
    parser.add_argument(
        '--skip-push',
        action='store_true',
        help='Skip push (just commit)'
    )
    parser.add_argument(
        '--force-push',
        action='store_true',
        help='Force push (use with caution!)'
    )
    parser.add_argument(
        '--auto-push',
        action='store_true',
        help='Auto-push without prompt'
    )
    
    args = parser.parse_args()
    
    # Change to project root for all operations
    project_root = get_project_root()
    os.chdir(project_root)
    
    print(f"{Colors.GREEN}🚀 Starting release process...{Colors.NC}")
    print(f"{Colors.BLUE}Working directory: {project_root}{Colors.NC}")
    
    # Step 1: Export training data
    if not args.skip_export:
        if not export_training_data(args.cutoff_date):
            sys.exit(1)
    else:
        print_warning("Skipping export (--skip-export)")
    
    # Step 2: Check for changes
    if not check_git_changes():
        print_warning("Nothing to commit")
        sys.exit(0)
    
    # Step 3: Stage changes
    stage_changes()
    
    # Step 4: Get commit message
    default_msg = f"Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if args.message:
        commit_msg = args.message
    else:
        commit_msg = get_commit_message(default=default_msg)
    
    # Step 5: Commit
    commit_changes(commit_msg)
    
    # Step 6: Push
    if not args.skip_push:
        push_changes(force=args.force_push, skip_prompt=args.auto_push)
    else:
        print_warning("Skipped push (--skip-push)")
    
    print(f"\n{Colors.GREEN}🎉 Release complete!{Colors.NC}")


if __name__ == "__main__":
    main()

