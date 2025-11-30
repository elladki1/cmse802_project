"""
Author: Rana Elladki
Date: 10/20/2025
Description: Unittests for /src/parse_xyz.py
"""

import unittest
import os
import pandas as pd
from src.parse_xyz import safe_float, parse_xyz_file, parse_xyz_folder

TEST_DIR = os.path.join(os.path.dirname(__file__), "test_data")
EXP_1 = os.path.join(TEST_DIR, "example1.xyz")
EXP_2 = os.path.join(TEST_DIR, "example2.xyz")
FAKE_DIR = os.path.join(os.path.dirname(__file__), "fake_dir")

class TestSafeFloat(unittest.TestCase):
    def test_safe_float(self):
        self.assertEqual(safe_float("2.19"), 2.19)
    
    def test_fortran_float(self):
        self.assertAlmostEqual(safe_float("2.1997*^-6"), 2.1997e-6)
    
    def test_alpha_str(self):
        with self.assertRaises(ValueError):
            safe_float("apple")
    
class TestParseXYZFile(unittest.TestCase):
    def test_parse_ex1(self):
        mol = parse_xyz_file(EXP_1)

        self.assertIsInstance(mol, dict)
        self.assertEqual(mol.keys(), {"id", "natoms", "properties", "atoms", 
                                      "frequencies", "smiles", "inchi"})
        self.assertEqual(mol["inchi"]["gdb9"], "InChI=1S/CH4/h1H4")
    
    def test_parse_ex2(self):
        mol = parse_xyz_file(EXP_2)

        self.assertIn("G", mol["properties"])
        self.assertIsInstance(mol["frequencies"], list)
        self.assertEqual(mol["natoms"], 4)

class TestParseXYZFolder(unittest.TestCase):
    def test_dir_access(self):
        with self.assertRaises(FileNotFoundError):
            parse_xyz_folder(FAKE_DIR, check_cache=False)
    
    def test_parse_to_lst(self):
        mols_lst = parse_xyz_folder(TEST_DIR, as_dataframe=False, check_cache=False)

        self.assertIsInstance(mols_lst, list)
        self.assertEqual(len(mols_lst), 3)
    
    def test_parse_to_df(self):
        mols_df = parse_xyz_folder(TEST_DIR, check_cache=False)

        self.assertIsInstance(mols_df, pd.DataFrame)

    def test_cache_file(self):
        cache_path = os.path.join(TEST_DIR, "test_cache.parquet")

        # first call will create cache file
        df_nocache = parse_xyz_folder(TEST_DIR, cache_file=cache_path, check_cache=False)
        self.assertTrue(os.path.exists(cache_path))

        # second call will use cache file
        df_cache = parse_xyz_folder(TEST_DIR, cache_file=cache_path, check_cache=True)
        # https://www.mungingdata.com/pandas/unit-testing-pandas/
        pd.testing.assert_frame_equal(df_nocache, df_cache)
