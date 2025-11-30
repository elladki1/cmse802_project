"""
Author: Rana Elladki
Date: 10/20/2025
Description: Unittests for /src/stats.py
"""

import unittest
from prettytable import PrettyTable
from src.stats import populate_table, create_table

# It is not necessary to create units tests for the other functions, because they
# are used for plotting purposes which cannot be easily validated. 

class TestPopTable(unittest.TestCase):
    def test_populate_table(self):
        # Create test dict
        test_dict = {
            "LR Log": {
                "R2" : 0.81, "RMSE": 0.45, "MSE": 0.201, "MAE": 0.322, "PC": 0.89
            },
            "ANN" : {
                "R2" : 0.90, "RMSE": 0.339, "MSE": 0.115, "MAE": 0.204, "PC": 0.91
            }
        }
        table = PrettyTable(["Model", "Scaling", "R²", "RMSE", "MSE", "MAE", "PC"])
        test_table = populate_table(test_dict, table)
        
        # Check two rows where added
        # Pretty table header isn't counted
        self.assertEqual(len(test_table._rows), 2)

        # Check contents of first (0th) row
        row_0 = test_table._rows[0]
        # Check the first key containing name and scaling is split correctly
        self.assertEqual(row_0[0], "Linear Regression (LR)")
        self.assertEqual(row_0[1], "Log")
        # Check a couple statistics
        self.assertEqual(row_0[2], "0.810")
        self.assertEqual(row_0[5], "0.322")

        # Check contents of second (1st) row
        row_1 = test_table._rows[1]
        # Check the scaling is empty "--" key in dict does not specify any
        self.assertEqual(row_1[1], "--")
    
    def test_missing_table(self):
        test_dict = {
            # Missing RMSE and MAE
            "LR Log": {
                "R2" : 0.81,  "MSE": 0.201,  "PC": 0.89
            },
            # Missing RMSE
            "ANN" : {
                "R2" : 0.90,  "MSE": 0.115, "MAE": 0.204, "PC": 0.91
            }
        }
        table = PrettyTable(["Model", "Scaling", "R²", "RMSE", "MSE", "MAE", "PC"])
        test_table = populate_table(test_dict, table)
        
        row_0 = test_table._rows[0]
        row_1 = test_table._rows[1]
        
        # Check 7 columns still printed even if data missing
        self.assertEqual(len(row_0), 7)
        self.assertEqual(len(row_1), 7)

        # Check RMSE entires are "--"
        self.assertEqual(row_0[3], "--")
        self.assertEqual(row_1[3], "--")

        # Check other entries still still present
        self.assertEqual(row_0[2], "0.810")
        self.assertEqual(row_1[6], "0.910")

class TestCreateTable(unittest.TestCase):
    def test_create_table(self):
        valid_dict = {
            "LR Log": {
                "R2" : 0.81, "RMSE": 0.45, "MSE": 0.201, "MAE": 0.322, "PC": 0.89
            }
        }
        test_dict = {
            "ANN" : {
                "R2" : 0.90, "RMSE": 0.339, "MSE": 0.115, "MAE": 0.204, "PC": 0.91
            }
        }

        table = create_table(valid_dict, test_dict)

        # Check there are 4 rows
        # 1 separation row (validation set)
        # 1 data row (validation model)
        # 1 separation row (testing set)
        # 1 data row (testing model)
        self.assertEqual(len(table._rows), 4)

        # Check separator rows exist
        self.assertEqual(table._rows[0][0], "--- VALIDATION SET ---")
        self.assertEqual(table._rows[2][0], "--- TESTING SET ---")