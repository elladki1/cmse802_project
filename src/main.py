"""
Author: Rana Elladki
Date: 10/15/2025
Description: Main CLI for Molecular Descriptor Extraction Pipeline.

    This script parses molecular XYZ files, computes descriptors, performs
    feature reduction, trains machine learning models (linear regression and 
    artificial neural network), and evaluates model perfomance.

Notes
-----
Calling file for entire pipeline. It calls:
- parse_xyz to load raw molecular structures
- extract_desc to compute molecular descriptors
- eda for correlation filtering, outlier removal, log-transform, and scaling
- train_lr and train_nn for ML training and validation
- stats for plotting and SHAP analysis

Usage
-----
Run from command line:
    python -m src.main --data_dir ./data/raw_data --output ./results --top 20
"""

import argparse
import time
import os
import copy
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# internal imports
from .parse_xyz import parse_xyz_folder 
from . import extract_desc
from . import eda
from . import train_lr
from . import train_nn
from . import stats

def prompt_if_missing(args):
    """Prompt the user for missing arguments interactively."""
    if not args.data_dir:
        args.data_dir = input("Enter path to data directory (e.g., ./sample_data): ").strip()
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory '{args.data_dir}' does not exist.")

    if not args.output:
        args.output = input("Enter path to output directory (e.g., ./results/): ").strip() 
    if not os.path.exists(args.output):
        raise FileNotFoundError(f"Results directory '{args.output}' does not exist.")

    if args.top is None:
        try:
            args.top = int(input("Enter number of top correlated features to keep (default: 20): ").strip() or 20)
        except ValueError:
            args.top = 20

    return args


def main():
    parser = argparse.ArgumentParser(
        description="Molecular descriptor extraction and preprocessing pipeline."
    )
     
    parser.add_argument("--data_dir", "-d", help="Path to directory containing .xyz files")
    parser.add_argument("--output", "-o", help="Directory to save processed data and results")
    parser.add_argument("--top", "-t", type=int, help="Number of top correlated features to keep")

    args = parser.parse_args()
    args = prompt_if_missing(args)

    start_time = time.time()
    print("Parsing raw data...")
    # parse through raw data and extract into pd df 
    # first checks if data has been parsed before and saved for performance optimization
    df = parse_xyz_folder(args.data_dir, as_dataframe=True)
    
    mols_arr = extract_desc.get_small_molec(df)
    coord_arr = extract_desc.get_atom_coords(df)
    # create dictionaries whose keys are the molecule IDs
    mol_dict = {m['ID']: m for m in mols_arr}
    coord_dict = {c['ID']: c for c in coord_arr}
    # iterate through mol_dict and pair it with the appropriate coordinates from coord_dict
    paired = [(mol_dict[i], coord_dict[i]) for i in mol_dict.keys() if i in coord_dict]

    # future improvement to account for BWs if the count is bigger than 0 but this database doesnt have any
    bw_dict, count = extract_desc.compute_boltzmann_weights(df)
    # build descriptor dict for all molecules
    print("Computing descriptors...")
    with Pool(processes=cpu_count()) as pool:
        all_desc = pool.map(extract_desc.process_molecule, paired)
    pool.close()
    pool.join()
    
    # index it with SMILES
    desc_df = pd.DataFrame(all_desc).set_index("SMILES")
    
    # perform pearson coefficient correlation analysis to get a dataframe of only the top features
    print("Performing Pearson correlation analysis...")
    reduced_df = eda.pearson_correlation(copy.deepcopy(desc_df), outdir=args.output, top_count=args.top)
    
    # filter outliers
    print("Filtering outliers...")
    filtered_df = eda.filter_outliers(copy.deepcopy(reduced_df), outdir=args.output)
    
    # split the data into training/validation/test sets
    print("Splitting data for ML...")
    X_train, X_val, X_test, y_train, y_val, y_test = eda.split_data(filtered_df)

    # log transform skewed features to compare logged vs not logged 
    print("Log transforming skewed features...")
    X_train_logged = eda.log_transform(copy.deepcopy(X_train), outdir=args.output, skew_thresh=0.7)
    X_val_logged = eda.log_transform(copy.deepcopy(X_val), skew_thresh=0.7, draw=0)
    X_test_logged = eda.log_transform(copy.deepcopy(X_test), skew_thresh=0.7, draw=0)
    y_train_logged = copy.deepcopy(y_train)
    y_val_logged = copy.deepcopy(y_val)
    y_test_logged = copy.deepcopy(y_test)
    
    # dict to store validation set stats
    val_stats = {}
    # dict to store test set stats
    test_stats = {}

    # train linear regression and test on validation set
    lr = train_lr.train_lr(X_train, y_train)
    # test linear model on validation set
    y_pred_val, report = train_lr.test_lr(lr, X_val, y_val)
    print("Done Predicting LR Validation Set...")
    val_stats["LR"] = report
    
    
    # train linear regression for logged data and test on validation set
    lr_logged = train_lr.train_lr(X_train_logged, y_train_logged)
    # test linear model on logged validation set
    y_pred_val_log, report = train_lr.test_lr(lr_logged, X_val_logged, y_val_logged)
    print("Done Predicting LR Logged Validation Set...")
    val_stats["LR Log"] = report
    
    # use the same model for the testing set
    y_pred_test, report = train_lr.test_lr(lr, X_test, y_test)
    print("Done Predicting LR Test Set...")
    test_stats["LR"] = report

    # use the same model for the testing set
    y_pred_test_log, report = train_lr.test_lr(lr_logged, X_test_logged, y_test_logged)
    print("Done Predicting LR Logged Test Set...")
    test_stats["LR Log"] = report
    
    # train artificial neural network 
    ann_gs = train_nn.train_val_ann(X_train, X_val, y_train, y_val)
    y_pred_ann, report = train_nn.test_ann(ann_gs, X_val, y_val)
    print("Done Predicting Ann Validation Set...")
    val_stats["ANN"] = report

    # train ANN on logged set
    ann_gs_log = train_nn.train_val_ann(X_train_logged, X_val_logged, y_train_logged, y_val_logged)
    y_pred_ann_log, report = train_nn.test_ann(ann_gs_log, X_val_logged, y_val_logged)
    print("Done Predicting ANN Logged Validation Set...")
    val_stats["ANN Log"] = report
    
    # train ANN on a scaled set
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(copy.deepcopy(X_train)), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.fit_transform(copy.deepcopy(X_val)), columns=X_val.columns)
    y_train_scaled = copy.deepcopy(y_train)
    y_val_scaled = copy.deepcopy(y_val)
    ann_gs_scaled = train_nn.train_val_ann(X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled)
    y_pred_ann_scaled, report = train_nn.test_ann(ann_gs_scaled, X_val_scaled, y_val_scaled)
    print("Done Predicting ANN Scaled Validation Set...")
    val_stats["ANN Scaled"] = report
    
    # test ANN model on test set
    y_pred_test_ann, report = train_nn.test_ann(ann_gs, X_test, y_test)
    print("Done Predicting ANN Test Set...")
    test_stats["ANN"] = report

    # test log ANN model on logged test set
    y_pred_ann_test_log, report = train_nn.test_ann(ann_gs_log, X_test_logged, y_test_logged)
    print("Done Predicting ANN Logged Test Set...")
    test_stats["ANN Log"] = report
    
    # test scaled ANN model on test set 
    # scale test set
    X_test_scaled = pd.DataFrame(scaler.fit_transform(copy.deepcopy(X_test)), columns=X_test.columns)
    y_test_scaled = copy.deepcopy(y_test)
    y_pred_ann_test_scale, report = train_nn.test_ann(ann_gs_scaled, X_test_scaled, y_test_scaled)
    print("Done Predicting ANN Scaled Test Set...")
    test_stats["ANN Scaled"] = report

    # Print table of validation and testing statics 
    print(stats.create_table(val_stats, test_stats))

    # Compare predicted to actual RG2 in validation and test sets with LR 
    lr_set = [
        (y_val, y_pred_val, "Valid"),
        (y_val_logged, y_pred_val_log, "Valid Logged"),
        (y_test, y_pred_test, "Test"),
        (y_test_logged, y_pred_test_log, "Test Logged")
    ]
    #Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    # Plot each dataset 
    for ax, (x, y, label) in zip(axes, lr_set):
        stats.plot_true_pred(ax, x, y, label)
    plt.tight_layout()
    fig.suptitle("Linear Regression (LR) Performance", fontsize=16)
    plt.subplots_adjust(top=0.92)  # adjust to make room for suptitle
    outfile = os.path.join(args.output, f"lr_pred_vs_true.png")
    plt.savefig(outfile, dpi=300)
    plt.close()

    # Compare predicted to actual RG2 in validation and test sets with ANN
    ann_set = [
        (y_val, y_pred_ann, "Valid"),
        (y_val_logged, y_pred_ann_log, "Valid Logged"),
        (y_val_scaled, y_pred_ann_scaled, "Valid Scaled"),
        (y_test, y_pred_test_ann, "Test"),
        (y_test_logged, y_pred_ann_test_log, "Test Logged"),
        (y_test_scaled, y_pred_ann_test_scale, "Test Scaled")
        ]
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15,8))
    axes = axes.flatten()
    # Plot each dataset
    for ax, (x, y, label) in zip(axes, ann_set):
        stats.plot_true_pred(ax, x, y, label)
    plt.tight_layout()
    fig.suptitle("Artificial Neural Network (ANN) Performance", fontsize=16)
    plt.subplots_adjust(top=0.92)  # adjust to make room for suptitle
    outfile = os.path.join(args.output, f"ann_pred_vs_true.png")
    plt.savefig(outfile, dpi=300)
    plt.close()

    # Evaluate the impact of the physicochemical descriptors on RG2
    # Generate coefficient values plots for linear model for best R^2
    outfile = os.path.join(args.output, f"lr_coef_wghts.png")
    stats.lr_coef_weights(lr_logged, X_train.columns, outfile)

    # Perform SHAP analysis for artificial neural network model with best R^2
    outfile = os.path.join(args.output, f"ann_shap.png")
    stats.ann_shap_analysis(ann_gs_scaled, X_train_scaled, X_test_scaled, outfile)
    print("Best params for scaled ANN:", ann_gs_scaled.best_params_)

    print(f"Run complete in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
