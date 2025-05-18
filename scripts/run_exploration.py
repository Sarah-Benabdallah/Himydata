# scripts/run_exploration.py
import os
import argparse
from src.data_processing.loading import setup_data_paths, load_product_info, count_timeseries_files, \
    load_timeseries_file
from src.data_processing.preprocessing import clean_product_info, clean_timeseries_data, fix_time_column, \
    find_target_column, analyze_product_info
from src.feature_engineering.feature_extraction import extract_timeseries_features, process_all_timeseries, \
    combine_with_product_info, save_combined_data
from src.visualization.exploratory_visualization import visualize_product_info, plot_timeseries, plot_distribution, \
    compare_multiple_timeseries, analyze_process_patterns


def display_menu():
    """Display the interactive menu"""
    print("\n" + "=" * 50)
    print("PRODUCTS EXPLORATION MENU")
    print("=" * 50)
    print("1. Explore product information")
    print("2. Analyze a single time series")
    print("3. Compare multiple time series")
    print("4. Process all time series and extract features")
    print("5. Combine features with product information")
    print("6. Save combined data for clustering")
    print("7. Run complete exploration pipeline")
    print("0. Exit")
    print("-" * 50)

    choice = input("Enter your choice (0-7): ")
    return choice


def explore_product_info(paths):
    """Explore and visualize product information"""
    print("\nExploring product information...")

    # Load product information
    product_info = load_product_info(paths['product_info_file'])

    # Clean product information
    product_info_clean = clean_product_info(product_info)

    # Analyze product information
    product_analysis = analyze_product_info(product_info_clean)

    # Visualize product information
    visualize_product_info(product_info_clean)

    return product_info_clean, product_analysis


def analyze_single_timeseries(paths):
    """Analyze a single time series file"""
    # List available time series files
    timeseries_files = count_timeseries_files(paths['timeseries_dir'])
    file_basenames = [os.path.basename(f) for f in timeseries_files]

    # Let user choose a file
    print("\nAvailable time series files:")
    for i, filename in enumerate(file_basenames[:10]):
        print(f"{i + 1}. {filename}")

    if len(file_basenames) > 10:
        print(f"... and {len(file_basenames) - 10} more files")

    while True:
        file_choice = input("\nEnter file number or filename (default: 1): ").strip()

        # Default to first file
        if not file_choice:
            file_choice = "1"

        # Handle numeric choice
        if file_choice.isdigit():
            idx = int(file_choice) - 1
            if 0 <= idx < len(file_basenames):
                selected_file = file_basenames[idx]
                break
            else:
                print("Invalid selection. Please try again.")
        # Handle filename input
        elif file_choice in file_basenames:
            selected_file = file_choice
            break
        else:
            print("Invalid selection. Please try again.")

    # Load and process the selected file
    sample_file = os.path.join(paths['timeseries_dir'], selected_file)
    print(f"\nLoading file: {selected_file}")
    sample_ts = load_timeseries_file(sample_file)

    # Clean and process
    sample_ts_clean = clean_timeseries_data(sample_ts)
    sample_ts_clean = fix_time_column(sample_ts_clean)

    # Find target column
    target_column = find_target_column(sample_ts_clean)

    if target_column:
        # Visualize
        plot_timeseries(sample_ts_clean, target_column, f"File: {selected_file}")
        plot_distribution(sample_ts_clean, target_column)

        # Extract features for this file
        features = extract_timeseries_features(sample_ts_clean[target_column],
                                               os.path.splitext(selected_file)[0])

        print("\nExtracted features:")
        for key, value in features.items():
            if key != 'product_id':
                print(f"{key}: {value:.4f}")

    return sample_ts_clean, target_column


def compare_timeseries(paths):
    """Compare multiple time series"""
    # List available time series files
    timeseries_files = count_timeseries_files(paths['timeseries_dir'])
    file_basenames = [os.path.basename(f) for f in timeseries_files]
    product_ids = [os.path.splitext(f)[0] for f in file_basenames]

    print("\nSelect products to compare (max 6 recommended):")
    print("Available options:")
    print("1. Use default comparison set (18N079, 18P001, 18R001, 18S001)")
    print("2. Specify products manually")
    print("3. Select random products")

    choice = input("Enter your choice (1-3): ").strip()

    if choice == '1':
        # Use default set
        products_to_compare = ['18N079', '18P001', '18R001', '18S001']
    elif choice == '2':
        # Manual selection
        print("\nEnter product IDs separated by commas (e.g., 18N079,18P001)")
        print("Available product IDs:", ", ".join(product_ids[:10]), "...")

        product_input = input("Products to compare: ").strip()
        products_to_compare = [p.strip() for p in product_input.split(',')]

        # Validate selections
        valid_products = [p for p in products_to_compare if p in product_ids]
        if len(valid_products) == 0:
            print("No valid products selected. Using default set.")
            products_to_compare = ['18N079', '18P001', '18R001', '18S001']
        else:
            products_to_compare = valid_products
    elif choice == '3':
        # Random selection
        import random
        max_products = min(6, len(product_ids))
        count = input(f"How many products to compare (1-{max_products})? ").strip()

        try:
            count = int(count)
            count = max(1, min(count, max_products))
            products_to_compare = random.sample(product_ids, count)
        except ValueError:
            print("Invalid input. Selecting 4 random products.")
            products_to_compare = random.sample(product_ids, 4)
    else:
        # Default to default set
        print("Invalid choice. Using default comparison set.")
        products_to_compare = ['18N079', '18P001', '18R001', '18S001']

    # Run comparison
    print(f"\nComparing {len(products_to_compare)} products: {', '.join(products_to_compare)}")
    comparison_results = compare_multiple_timeseries(
        timeseries_dir=paths['timeseries_dir'],
        product_ids=products_to_compare
    )

    # Analyze process patterns
    if comparison_results:
        analyze_process_patterns(comparison_results)

    return comparison_results


def process_all_series(paths, product_analysis):
    """Process all time series and extract features"""
    # Get product IDs from analysis
    all_product_ids = product_analysis['product_ids']

    print("\nProcess time series options:")
    print(f"1. Process all time series files ({len(all_product_ids)} files)")
    print("2. Process a sample of time series files")
    print("3. Specify maximum number of files to process")

    choice = input("Enter your choice (1-3): ").strip()

    if choice == '1':
        # Process all files
        product_ids = all_product_ids
        print(f"\nProcessing all {len(product_ids)} time series files...")
    elif choice == '2':
        # Process sample
        import random
        sample_size = input("Enter sample size (default: 50): ").strip()

        try:
            size = int(sample_size) if sample_size else 50
            size = min(size, len(all_product_ids))
            product_ids = random.sample(all_product_ids, size)
        except ValueError:
            print("Invalid input. Using default sample size of 50.")
            size = min(50, len(all_product_ids))
            product_ids = random.sample(all_product_ids, size)

        print(f"\nProcessing {len(product_ids)} randomly selected time series files...")
    elif choice == '3':
        # Specify max
        max_input = input(f"Maximum files to process (1-{len(all_product_ids)}): ").strip()

        try:
            max_files = int(max_input)
            max_files = min(max_files, len(all_product_ids))
            product_ids = all_product_ids[:max_files]
        except ValueError:
            print("Invalid input. Processing all files.")
            product_ids = all_product_ids

        print(f"\nProcessing {len(product_ids)} time series files...")
    else:
        # Default to processing all
        print("Invalid choice. Processing all files.")
        product_ids = all_product_ids
        print(f"\nProcessing all {len(product_ids)} time series files...")

    # Process the selected files
    timeseries_features = process_all_timeseries(
        timeseries_dir=paths['timeseries_dir'],
        product_ids=product_ids,
        target_column='Pinceur Sup Mesure de courant'
    )

    return timeseries_features


def run_exploration_interactive():
    """Run the data exploration pipeline interactively"""
    print("Starting interactive data exploration...")

    # Setup paths
    paths = setup_data_paths('./data')

    # Initialize variables for sharing between functions
    product_info_clean = None
    product_analysis = None
    timeseries_features = None
    combined_data = None

    while True:
        choice = display_menu()

        if choice == '0':
            print("\nExiting exploration. Goodbye!")
            break

        elif choice == '1':
            # Explore product information
            product_info_clean, product_analysis = explore_product_info(paths)

        elif choice == '2':
            # Analyze a single time series
            sample_ts_clean, target_column = analyze_single_timeseries(paths)

        elif choice == '3':
            # Compare multiple time series
            comparison_results = compare_timeseries(paths)

        elif choice == '4':
            # Process all time series
            if product_analysis is None:
                print("\nProduct analysis not available. Analyzing product information first...")
                product_info_clean, product_analysis = explore_product_info(paths)

            timeseries_features = process_all_series(paths, product_analysis)

        elif choice == '5':
            # Combine features with product information
            if timeseries_features is None:
                print("\nNo features available. Processing time series first...")
                if product_analysis is None:
                    product_info_clean, product_analysis = explore_product_info(paths)
                timeseries_features = process_all_series(paths, product_analysis)

            if product_info_clean is None:
                print("\nLoading product information first...")
                product_info_clean, product_analysis = explore_product_info(paths)

            print("\nCombining features with product information...")
            combined_data = combine_with_product_info(timeseries_features, product_info_clean)

            # Display a sample of the combined data
            print("\nSample of combined data:")
            print(combined_data.head())

        elif choice == '6':
            # Save combined data
            if combined_data is None:
                print("\nNo combined data available. Creating it first...")

                if timeseries_features is None:
                    print("Processing time series...")
                    if product_analysis is None:
                        product_info_clean, product_analysis = explore_product_info(paths)
                    timeseries_features = process_all_series(paths, product_analysis)

                if product_info_clean is None:
                    print("Loading product information...")
                    product_info_clean, product_analysis = explore_product_info(paths)

                print("Combining features with product information...")
                combined_data = combine_with_product_info(timeseries_features, product_info_clean)

            # Ask for output location
            output_dir = input("\nEnter output directory (default: ./results): ").strip()
            if not output_dir:
                output_dir = './results'

            # Save the data
            saved_path = save_combined_data(combined_data, output_dir)
            print(f"Combined data saved to {saved_path}")

        elif choice == '7':
            # Run complete pipeline
            run_exploration()

        else:
            print("\nInvalid choice. Please select a number between 0 and 7.")

        # Pause before returning to menu
        input("\nPress Enter to continue...")

    return combined_data


def run_exploration():
    """Run the data exploration pipeline"""
    print("Starting data exploration...")

    # Setup paths
    paths = setup_data_paths('./data')

    # Load and clean product information
    product_info = load_product_info(paths['product_info_file'])
    product_info_clean = clean_product_info(product_info)

    # Analyze product information
    product_analysis = analyze_product_info(product_info_clean)

    # Visualize product information
    visualize_product_info(product_info_clean)

    # Load and explore sample time series
    timeseries_files = count_timeseries_files(paths['timeseries_dir'])
    sample_file = os.path.join(paths['timeseries_dir'], '18N078.txt')
    sample_ts = load_timeseries_file(sample_file)
    sample_ts_clean = clean_timeseries_data(sample_ts)
    sample_ts_clean = fix_time_column(sample_ts_clean)

    # Find target column in sample time series
    target_column = find_target_column(sample_ts_clean)
    if target_column:
        plot_timeseries(sample_ts_clean, target_column, "Sample")
        plot_distribution(sample_ts_clean, target_column)

    # Compare multiple time series
    products_to_compare = ['18N079', '18P001', '18R001', '18S001']
    comparison_results = compare_multiple_timeseries(
        timeseries_dir=paths['timeseries_dir'],
        product_ids=products_to_compare
    )

    # Analyze process patterns
    analyze_process_patterns(comparison_results)

    # Process all time series and extract features
    all_product_ids = product_analysis['product_ids']
    timeseries_features = process_all_timeseries(
        timeseries_dir=paths['timeseries_dir'],
        product_ids=all_product_ids,
        target_column='Pinceur Sup Mesure de courant'
    )

    # Combine with product information
    combined_data = combine_with_product_info(timeseries_features, product_info_clean)

    # Save combined data for clustering
    saved_path = save_combined_data(combined_data)

    print(f"Data exploration complete. Combined data saved to {saved_path}")
    return combined_data


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Product Data Exploration')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode with menu')

    args = parser.parse_args()

    if args.interactive:
        # Run interactive exploration
        run_exploration_interactive()
    else:
        # Run full pipeline
        run_exploration()