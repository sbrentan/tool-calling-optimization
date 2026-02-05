#!/usr/bin/env python3
"""
Script to update output_dir in YAML configuration files.

Usage:
    python scripts/update_output_dir.py <yaml_dir> <new_output_dir>
    
Examples:
    python scripts/update_output_dir.py real_experiments/139_experiments_cloud/plan experiments/139_experiments_cloud/results
    python scripts/update_output_dir.py experiments/plan experiments/results
    
    # To remove output_dir entirely (use default from config.py):
    python scripts/update_output_dir.py real_experiments/139_experiments_cloud/plan --remove
"""

import argparse
import sys
from pathlib import Path


def update_output_dir_in_file(
    file_path: Path, 
    new_output_dir: str | None, 
    dry_run: bool = False,
    add_if_missing: bool = True
) -> bool:
    """
    Update or remove the output_dir field in a YAML file.
    
    Args:
        file_path: Path to the YAML file
        new_output_dir: New output directory path, or None to remove the field
        dry_run: If True, don't actually modify files
        add_if_missing: If True, add output_dir if not present in file
        
    Returns:
        True if file was modified, False otherwise
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    modified = False
    new_lines = []
    found_output_dir = False
    
    for line in lines:
        # Check if this is the output_dir line
        if line.strip().startswith("output_dir:"):
            found_output_dir = True
            if new_output_dir is None:
                # Remove the line entirely
                modified = True
                continue
            else:
                # Replace with new value
                # Preserve indentation
                indent = len(line) - len(line.lstrip())
                new_line = " " * indent + f"output_dir: {new_output_dir}"
                if new_line != line:
                    modified = True
                    line = new_line
        new_lines.append(line)
    
    # Add output_dir at the end if not found and add_if_missing is True
    if not found_output_dir and add_if_missing and new_output_dir is not None:
        # Add a blank line if the file doesn't end with one
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(f"output_dir: {new_output_dir}")
        # Ensure file ends with newline
        if new_lines[-1]:
            new_lines.append("")
        modified = True
    
    if modified and not dry_run:
        file_path.write_text("\n".join(new_lines), encoding="utf-8")
    
    return modified


def main():
    parser = argparse.ArgumentParser(
        description="Update output_dir in YAML configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/update_output_dir.py real_experiments/139_experiments_cloud/plan experiments/139_experiments_cloud/results
    python scripts/update_output_dir.py experiments/plan experiments/results --dry-run
    python scripts/update_output_dir.py real_experiments/139_experiments_cloud/plan --remove
        """
    )
    parser.add_argument("yaml_dir", help="Directory containing YAML files to update")
    parser.add_argument("new_output_dir", nargs="?", help="New output_dir value")
    parser.add_argument("--remove", action="store_true", help="Remove output_dir field entirely (uses default)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without modifying files")
    parser.add_argument("--no-add-if-missing", action="store_true", help="Don't add output_dir if not present (default: add it)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.remove and not args.new_output_dir:
        parser.error("Either provide new_output_dir or use --remove")
    
    if args.remove and args.new_output_dir:
        parser.error("Cannot use --remove with a new_output_dir value")
    
    yaml_dir = Path(args.yaml_dir)
    if not yaml_dir.exists():
        print(f"Error: Directory does not exist: {yaml_dir}")
        sys.exit(1)
    
    # Find all YAML files
    yaml_files = list(yaml_dir.glob("*.yaml")) + list(yaml_dir.glob("*.yml"))
    
    if not yaml_files:
        print(f"No YAML files found in {yaml_dir}")
        sys.exit(1)
    
    print(f"Found {len(yaml_files)} YAML files in {yaml_dir}")
    
    if args.dry_run:
        print("DRY RUN - no files will be modified\n")
    
    new_output_dir = None if args.remove else args.new_output_dir
    action = "Removing" if args.remove else f"Setting to '{new_output_dir}'"
    add_if_missing = not args.no_add_if_missing
    
    modified_count = 0
    for yaml_file in sorted(yaml_files):
        was_modified = update_output_dir_in_file(
            yaml_file, 
            new_output_dir, 
            dry_run=args.dry_run,
            add_if_missing=add_if_missing
        )
        if was_modified:
            modified_count += 1
            status = "[WOULD MODIFY]" if args.dry_run else "[MODIFIED]"
            print(f"  {status} {yaml_file.name}")
        else:
            print(f"  [UNCHANGED] {yaml_file.name}")
    
    print(f"\n{action} output_dir in {modified_count}/{len(yaml_files)} files")
    
    if args.dry_run and modified_count > 0:
        print("\nRun without --dry-run to apply changes")


if __name__ == "__main__":
    main()
