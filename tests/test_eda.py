"""
Author: Rana Elladki
Date: 10/20/2025
Description: Unittests for /src/eda.py
"""

import unittest
import os
import copy
import numpy as np
import pandas as pd
import random
from src.eda import pearson_correlation, filter_outliers, log_transform, split_data

# Specify random seed for reporducability in testing
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# Create a dictionary to be used for testing
DF = pd.DataFrame({
    "ID": range(1, 11),
    "RingCount": random.choices(range(0, 8), k=10),
    "Kappa2": np.random.uniform(-27, 8, 10),
    "SPS": random.sample(range(0, 108), 10),
    "WI": np.random.uniform(0, 120, 10),
    "Z2": np.random.uniform(0, 124, 10),
    "RG2": np.random.uniform(0, 13, 10)
})

# Test directory for output figures
TEST_DIR = os.path.join(os.path.dirname(__file__), "test_results")

class TestPearsonCorr(unittest.TestCase):
    def test_top_count(self):
        """Test that top count respects defaults and user specification"""
        # Check reduced dataframe has only 4 columns (3 for top features + 1 for RG2)
        reduced_df = pearson_correlation(DF, outdir=TEST_DIR, top_count=3)
        self.assertEqual(len(reduced_df.columns), 4)

        # Check that if top_count not specfied, uses maximum available features or default top count
        # This should use maximum=5 (column count = 6) available features 
        # because only 5 feature columns are available
        all_feat_df = pearson_correlation(DF, outdir=TEST_DIR)
        self.assertEqual(len(all_feat_df.columns), len(DF.columns)-1)
    
    def test_contents(self):
        """Check contents of reduced dataframe"""
        reduced_df = pearson_correlation(DF, outdir=TEST_DIR)
        
        # Check returned DF does not contain ID column
        self.assertNotIn("ID", reduced_df)
        
        # Check returned DF contains RG2
        self.assertIn("RG2", reduced_df)

        # Manually test corr to check if features sorted correctly
        test_df = copy.deepcopy(DF)
        test_df = test_df.drop(["ID"], axis=1)
        corr_pear = test_df.corr(method='pearson')
        rg2_corr = corr_pear["RG2"].drop("RG2").abs().sort_values(ascending=False)
        strong_corr = rg2_corr.index[:5]
        res_df = test_df[strong_corr.tolist() + ["RG2"]]

        self.assertEqual(len(reduced_df.columns), len(res_df.columns))
        pd.testing.assert_index_equal(reduced_df.columns, res_df.columns)

    def test_plots(self):
        """Check if output plots are/aren't in specified directory"""
        # Check if heatmap of 5 features is present
        pearson_correlation(DF, outdir=TEST_DIR)
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "corr_heatmap_top_5_feats.png")))

        # Check that heatmap of 4 features is not present
        pearson_correlation(DF, top_count=4, draw=0) 
        self.assertFalse(os.path.exists(os.path.join(TEST_DIR, "corr_heatmap_top_4_feats.png"))) 

        # Check that error is raised if draw=1 but no output dir is provided
        with self.assertRaises(ValueError):
            pearson_correlation(DF, outdir=None, draw=1)  

class TestFilterOutliers(unittest.TestCase):
    def test_outlier_filtering(self):
        # start with dataframe after PCCA has been performed
        test_df = pearson_correlation(DF, outdir=TEST_DIR, top_count=5)
        filtered_df = filter_outliers(test_df, outdir=TEST_DIR)
        
        # Check columns available are the same even after filtering 
        pd.testing.assert_index_equal(test_df.columns, filtered_df.columns)

        # Check values remaining are within limits
        for feat in ["WI", "Kappa2", "Z2"]:
            feat_min = DF[feat].quantile(0.001)
            feat_max = DF[feat].quantile(0.999)
            self.assertTrue(filtered_df[feat].min() >= feat_min)
            self.assertTrue(filtered_df[feat].max() <= feat_max)
    
    def test_plots(self):
        """Check if output plots are/aren't in specified directory"""
        # Check if histograms of filtered and unfiltered features are present
        filter_outliers(DF, outdir=TEST_DIR)
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "hist_unfiltered.png")))
        self.assertTrue(os.path.exists(os.path.join(TEST_DIR, "hist_filtered.png")))

        # Check that error is raised if draw=1 but no output dir is provided
        with self.assertRaises(ValueError):
            filter_outliers(DF, outdir=None, draw=1)  

class TestLogTransform(unittest.TestCase):
    def test_log_transform(self):
        # Add a very skewed column
        test_df = copy.deepcopy(DF)
        test_df["SkewedCol"] = np.array([1, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        # Do NOT log target col
        logged_df = copy.deepcopy(test_df)
        logged_df = log_transform(logged_df, exclude="RG2",draw=0)
        # Check log transform not applied to RG2 col
        pd.testing.assert_series_equal(test_df["RG2"], logged_df["RG2"])
        # Check to see log transform applied to SkewedCol
        self.assertFalse((logged_df["SkewedCol"] == test_df["SkewedCol"]).all())
        # Check no plots were generated
        self.assertFalse(os.path.exists(os.path.join(TEST_DIR, "hist_logged.png")))

class TestSplitData(unittest.TestCase):
    def test_data_split(self):
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(DF)
        # Check the length of X_test dataframe columns is 1 less than original DF
        self.assertEqual(len(X_test.columns), len(DF.columns)-1)
        # Check that X_test and X_val have the same length
        self.assertEqual(len(X_test), len(X_val))
        # Check that X_ dataframes have the same number of columns
        self.assertEqual(len(X_test.columns), len(X_train.columns))
        
        # Check that RG2 not in X
        for X in [X_train, X_val, X_test]:
            self.assertNotIn("RG2", X.columns)

        # Check that the sum of rows of X_ dataframes equals DF rows 
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), len(DF))
        
        # Check that y_ only has RG2 column
        for y in [y_train, y_val, y_test]:
            self.assertEqual(y.name, "RG2")
        
        # Check taht the sum of rows of y_ equals DF rows
        self.assertEqual(len(y_train) + len(y_val) + len(y_test), len(DF))
        
        