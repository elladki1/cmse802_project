"""
Author: Rana Elladki
Date: 10/15/2025
Description: Main CLI for Molecular Descriptor Extraction Pipeline
"""

import argparse
import time
import os
import copy
import pandas as pd
from multiprocessing import Pool, cpu_count

# internal imports
from parse_xyz import parse_xyz_folder 
import extract_desc
import eda

def prompt_if_missing(args):
    """Prompt the user for missing arguments interactively."""
    if not args.data_dir:
        args.data_dir = input("Enter path to data directory (e.g., ./sample_data): ").strip()
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory '{args.data_dir}' does not exist.")

    if not args.output:
        args.output = input("Enter path to output directory (default: ./results/processed_data): ").strip() or "/mnt/home/elladki1/CMSE802/Final_Project/cmse802_project/results"

    if args.top is None:
        try:
            args.top = int(input("Enter number of top correlated features to keep (default: 5): ").strip() or 5)
        except ValueError:
            args.top = 5

    return args


def main():
    parser = argparse.ArgumentParser(
        description="Molecular descriptor extraction and preprocessing pipeline."
    )
    # REMOVE DEFAULT BEFORE TURNING IN 
    parser.add_argument("--data-dir", "-d", help="Path to directory containing .xyz files", default="/mnt/home/elladki1/CMSE802/Final_Project/cmse802_project/data/raw_data")
    parser.add_argument("--output", "-o", help="Directory to save processed data and results", default="/mnt/home/elladki1/CMSE802/Final_Project/cmse802_project/results")
    parser.add_argument("--top", "-t", type=int, help="Number of top correlated features to keep", default=5)

    args = parser.parse_args()
    args = prompt_if_missing(args)

    start_time = time.time()
    print("Parsing raw data...")
    # parse through raw data and extract into pd df 
    # first checks if data has been parsed before and saved for performance optimization
    df = parse_xyz_folder(args.data_dir, as_dataframe=True)
    mols_arr = extract_desc.get_small_molec(df)
    coord_arr = extract_desc.get_atom_coords(df)
    # future improvement to account for BWs if the count is bigger than 0 but this database doesnt have any
    bw_dict, count = extract_desc.compute_boltzmann_weights(df)
    # build descriptor dict for all molecules
    print("Computing descriptors...")
    with Pool(processes=cpu_count()) as pool:
        all_desc = pool.map(extract_desc.process_molecule, zip(mols_arr, coord_arr))
    
    # index it with SMILES
    desc_df = pd.DataFrame(all_desc).set_index("SMILES")

    # perform pearson coefficient correlation analysis to get a dataframe of only the top features
    print("Performing Pearson correlation analysis...")
    reduced_df = eda.pearson_correlation(copy.deepcopy(desc_df), top_count=args.top)
    # filter outliers
    print("Filtering outliers...")
    filtered_df = eda.filter_outliers(copy.deepcopy(reduced_df))

    # log transform skewed data
    print("Log transforming skewed features...")
    log_df = eda.log_transform(copy.deepcopy(filtered_df))
    print(log_df.tail())

    # split the data into training/validation/test sets
    print("Splitting data for ML...")
    X_train, X_val, X_test, y_train, y_val, y_test = eda.split_data(log_df)

    
    print(f"Pipeline complete in {time.time() - start_time:.2f} seconds")



if __name__ == "__main__":
    main()
