"""
Author: Rana Elladki
Date: 10/20/2025
Description: Unittests for /src/extract_desc.py
"""

import unittest
import os
import numpy as np
import pandas as pd
from rdkit.Chem import GetPeriodicTable
from src.parse_xyz import parse_xyz_folder
from src.extract_desc import get_small_molec, get_atom_coords, compute_rg2, \
    compute_boltzmann_weights, build_desc_dict, process_molecule

# Skip unit testing for get_weiner_index, get_randic_index, and get_zagreb_index
# because they do not test logic, rather just call Mordred library functions
# Skip unit testing for get_rdkit_desc because function acts as an rdkit wrapper with no new logic

TEST_DIR = os.path.join(os.path.dirname(__file__), "test_data")
DF = parse_xyz_folder(TEST_DIR, check_cache=False)
PTABLE = GetPeriodicTable()

class TestSmallMolec(unittest.TestCase):
    def test_small_mol(self):
        small_mols = get_small_molec(DF)
        # test dir contains large molecule. Check that it is not included 
        self.assertEqual(len(small_mols), 2)
        # Large molecule is docosane so check that SMILES not in small_mols lst 
        smiles_lst = [i["SMILES"] for i in small_mols]
        self.assertNotIn("CCCCCCCCCCCCCCCCCCCCCC", smiles_lst)
    
    def test_cutoff(self):
        small_mols = get_small_molec(DF, 400)
        for mol in small_mols:
            # test all molecules in small_mols have a MW smaller than the cutoff
            self.assertLessEqual(mol["MW"], 400)

class TestAtomCoordsExtract(unittest.TestCase):
    def test_get_atom_coords(self):
        coords_lst = get_atom_coords(DF)

        # Check that the size of coords_list is the same as the dataframe
        self.assertEqual(len(coords_lst), len(DF))

        for molec in coords_lst:
            # each atom in the molecule in coords list should be a dictionary
            atom = molec["COORDS"][0]
            self.assertIsInstance(atom, dict)
        
            # check all coord elements are present in coords: element, x, y, z, and charge
            self.assertIn("element", atom)
            self.assertIn("x", atom)
            self.assertIn("y", atom)
            self.assertIn("z", atom)
            self.assertIn("charge", atom)

class TestRG2Calc(unittest.TestCase):
    def test_rg2_one_mol(self):
        """Check that RG2 is 0 for single molecule"""
        molecule ={'ID': 1, 'COORDS': [{'element': 'C', 'x': -0.013, 'y': 1.09, 'z': 0.01, 'charge': -0.54}]}
        self.assertAlmostEqual(compute_rg2(molecule), 0.0)

    def test_rg2_small_mol(self):
        """Check RG2 computed correctly for small molecule""" 
        molecule ={'ID': 2, 'COORDS': [
            {'element': 'C', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'charge': 0.0},
            {'element': 'H', 'x': 0.50, 'y': 0.0, 'z': 0.50, 'charge': 0.0}]} 
        coords=molecule['COORDS']
        masses = np.array([PTABLE.GetAtomicWeight(atom['element']) for atom in coords])
        positions = np.array([[atom['x'], atom['y'], atom['z']] for atom in coords])
        tot_mass = masses.sum()
        com = np.average(positions, axis=0, weights=masses)
        rg2 = np.sum(masses*np.sum((positions - com)**2, axis=1))/tot_mass
        
        self.assertAlmostEqual(compute_rg2(molecule), rg2)

    def test_return_type(self):
        """Check RG2 returns float"""
        molecule ={'ID': 2, 'COORDS': [
            {'element': 'C', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'charge': 0.0},
            {'element': 'H', 'x': 0.50, 'y': 0.0, 'z': 0.50, 'charge': 0.0}]} 
        self.assertIsInstance(compute_rg2(molecule), float)

class TestBoltzmannWeights(unittest.TestCase):
    def test_single_conformer(self):
        """Check if only one conformer present, count of conformers is 0 and BW is 1.0"""
        df = pd.DataFrame({"id": [1, 2],
                            "smiles_gdb9": ["C", "N"],
                            "G": [-40.498597, -56.544961]})
        bw_dict, count = compute_boltzmann_weights(df)
        self.assertEqual(count, 0)
        self.assertEqual(bw_dict['C'][0]["BW"], 1.0)

    def test_multiple_conformers(self):
        """Check if multiple conformers present for 1 molecule"""
        df = pd.DataFrame({"id": [1, 2, 3],
                            "smiles_gdb9": ["C", "N", "C"],
                            "G": [-40.50, -56.54, -40.501 ]})
        bw_dict, count = compute_boltzmann_weights(df)
        self.assertEqual(count, 1)

        # check bw_dict contains both C conformers
        self.assertEqual(len(bw_dict["C"]), 2)
        
        # check BW weights for C add to 1
        tot_weight = sum(i["BW"] for i in bw_dict['C'])
        self.assertAlmostEqual(tot_weight, 1.0)

class TestBuildDescDict(unittest.TestCase):
    def test_smiles(self):
        """Check SMILES only input returns descriptors without RG2"""
        smiles = 'CCC'
        desc = build_desc_dict(smiles)
        
        # Check rdKit descriptors are included
        self.assertIn("Kappa2", desc)
        self.assertIn("NumRotatableBonds", desc)

        # Check computed desciriptors are included
        self.assertIn("WI", desc)
        self.assertIn("Z1", desc)

        # Check RG2 not included since coords not passed
        self.assertNotIn("RG2", desc)
    
    def test_coords(self):
        """Check coords only input returns just RG2"""
        mol_coords ={'ID': 2, 'COORDS': [
            {'element': 'C', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'charge': 0.0},
            {'element': 'H', 'x': 0.50, 'y': 0.0, 'z': 0.50, 'charge': 0.0}]} 
        desc = build_desc_dict(coords=mol_coords)

        # Check RG2 included
        self.assertIn("RG2", desc)

        # Check rdKit descriptors and computed descriptors not included
        self.assertNotIn("Kappa2", desc)
        self.assertNotIn("Z2", desc)
    
    def test_invalid_smiles(self):
        """Raise value error if SMILES invalid"""
        with self.assertRaises(ValueError):
            build_desc_dict(smiles="FAKESMILE")

class TestProcessMolecule(unittest.TestCase):
    def test_process_molecule(self):
        molecule = {"ID": 1, "SMILES": "C"}
        coords ={'ID': 1, 'COORDS': [
            {'element': 'C', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'charge': 0.0},
            {'element': 'H', 'x': 0.50, 'y': 0.0, 'z': 0.50, 'charge': 0.0}]} 
        desc = process_molecule((molecule, coords))

        # Check output is dict
        self.assertIsInstance(desc, dict)

        # Check ID and SMILES are correct
        self.assertEqual(desc["ID"], 1)
        self.assertEqual(desc["SMILES"], "C")

        # Check rdKit descriptors are included
        self.assertIn("Kappa2", desc)

        # Check computer descriptors are included 
        self.assertIn("Z1", desc)

        # Check that RG2 is included
        self.assertIn("RG2", desc)
