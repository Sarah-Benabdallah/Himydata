#!/usr/bin/env python
"""
Product Clustering Project - Main Entry Point

This script serves as the central entry point for the Product Clustering project,
providing access to both exploration and clustering functionality through a unified interface.
"""

import os
import argparse
import sys
# Add the project root to the Python path to enable imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
from scripts.run_exploration import run_exploration, run_exploration_interactive
from scripts.run_clustering import run_clustering, run_clustering_interactive


def display_main_menu():
    """Display the main menu for the project"""
    print("\n" + "=" * 60)
    print(" PRODUCT CLUSTERING PROJECT ".center(60, "="))
    print("=" * 60)
    print("1. Run Data Exploration")
    print("2. Run Clustering Analysis")
    print("3. Run Complete Pipeline (Exploration + Clustering)")
    print("0. Exit")
    print("-" * 60)

    choice = input("Enter your choice (0-3): ")
    return choice


def run_interactive_mode():
    """Run the project in interactive mode with a main menu"""
    combined_data = None

    while True:
        choice = display_main_menu()

        if choice == '0':
            print("\nExiting program. Goodbye!")
            break

        elif choice == '1':
            # Run data exploration
            print("\nStarting data exploration...")

            # Ask for interactive mode
            interactive = input("Run exploration in interactive mode? (y/n, default: y): ").strip().lower()
            if interactive != 'n':
                combined_data = run_exploration_interactive()
            else:
                combined_data = run_exploration()

        elif choice == '2':
            # Run clustering analysis
            print("\nStarting clustering analysis...")

            # If we don't have combined data, check if it exists
            if combined_data is None:
                default_path = 'results/combined_data.csv'
                if os.path.exists(default_path):
                    use_existing = input(
                        f"Use existing combined data from {default_path}? (y/n, default: y): ").strip().lower()
                    if use_existing != 'n':
                        print(f"Using existing data from {default_path}")
                    else:
                        # Ask for custom path
                        custom_path = input("Enter path to combined data file: ").strip()
                        if os.path.exists(custom_path):
                            default_path = custom_path
                            print(f"Using data from {custom_path}")
                        else:
                            print(f"File not found: {custom_path}")
                            print("Please run data exploration first to generate the combined data.")
                            continue
                else:
                    print("Combined data not found. Please run data exploration first.")
                    continue

            # Ask for interactive mode
            interactive = input("Run clustering in interactive mode? (y/n, default: y): ").strip().lower()
            if interactive != 'n':
                clustering_results = run_clustering_interactive()
            else:
                if combined_data is not None:
                    clustering_results, best_method = run_clustering(combined_data)
                else:
                    clustering_results, best_method = run_clustering()

        elif choice == '3':
            # Run complete pipeline
            print("\nRunning complete pipeline (exploration + clustering)...")

            # Ask for interactive mode for each step
            interactive_exp = input("Run exploration in interactive mode? (y/n, default: n): ").strip().lower()
            if interactive_exp == 'y':
                combined_data = run_exploration_interactive()
            else:
                combined_data = run_exploration()

            interactive_clust = input("Run clustering in interactive mode? (y/n, default: n): ").strip().lower()
            if interactive_clust == 'y':
                clustering_results = run_clustering_interactive()
            else:
                clustering_results, best_method = run_clustering(combined_data)

            print("\nComplete pipeline execution finished.")

        else:
            print("\nInvalid choice. Please select a number between 0 and 3.")

        # Pause before returning to menu
        input("\nPress Enter to return to main menu...")


def run_automated_pipeline(mode, interactive=False, data_path=None):
    """Run the pipeline in automated mode"""
    combined_data = None

    if mode in ('exploration', 'full'):
        # Run exploration
        print("\nRunning data exploration...")
        if interactive:
            combined_data = run_exploration_interactive()
        else:
            combined_data = run_exploration()

    if mode in ('clustering', 'full'):
        # Run clustering
        print("\nRunning clustering analysis...")
        if interactive:
            run_clustering_interactive()
        else:
            if mode == 'clustering' and data_path:
                # Load combined data from specified path
                from scripts.run_clustering import load_combined_data
                try:
                    custom_data = load_combined_data(data_path)
                    clustering_results, best_method = run_clustering(custom_data)
                except Exception as e:
                    print(f"Error loading data from {data_path}: {e}")
                    print("Running clustering with default data...")
                    clustering_results, best_method = run_clustering()
            else:
                # Use combined data from exploration or default path
                clustering_results, best_method = run_clustering(combined_data)


def main():
    """Main entry point for the Product Clustering project"""
    parser = argparse.ArgumentParser(
        description='Product Clustering Project - Analysis and Clustering Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the complete pipeline in interactive mode with menu
  python main.py --interactive

  # Run only the data exploration step
  python main.py --mode exploration

  # Run only the clustering step with custom data file
  python main.py --mode clustering --data path/to/data.csv

  # Run the complete pipeline in automated mode
  python main.py --mode full

  # Run data exploration in interactive mode
  python main.py --mode exploration --interactive
        """
    )

    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode with main menu')
    parser.add_argument('--mode', choices=['exploration', 'clustering', 'full'],
                        default='full',
                        help='Which part of the pipeline to run')
    parser.add_argument('--data', type=str,
                        help='Path to combined data file (for clustering mode)')

    args = parser.parse_args()

    if args.interactive and args.mode == 'full':
        # Run in fully interactive mode with main menu
        run_interactive_mode()
    else:
        # Run in automated mode with specific parameters
        run_automated_pipeline(args.mode, args.interactive, args.data)


if __name__ == "__main__":
    main()