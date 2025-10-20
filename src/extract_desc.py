"""
Author: Rana Elladki
Date: 10/01/2025
Description: Utilities for extracting molecular descriptors from SMILES strings 
using RDKit, Mordred, and related chemistry libraries.

Provides functions to parse molecules, compute topological indices,
Boltzmann weights, and radius of gyration squared (Rg²).
"""
from parse_xyz import parse_xyz_folder
from rdkit import Chem
from rdkit.Chem import Descriptors, GetPeriodicTable
from mordred import WienerIndex, ZagrebIndex
from multiprocessing import Pool, cpu_count
import numpy as np
import time
import pandas as pd


ptable = GetPeriodicTable()

def get_small_molec(df, mw_cutoff=300):
    """
    Extract small molecules (based on molecular weight) from a DataFrame using 
    SMILES strings.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with SMILES strings and molecular properties.
    mw_cutoff : int, default=300
        Molecular weights cutoff to define small molecules.
    
    Returns
    -------
    list of dict
        Each dictionary contains:
        - 'ID' : int
            Molecule ID.
        - 'SMILES' : str
            SMILES string for the molecule.
        - 'MW' : float
            Molecular weight of the molecule.
    
    Examples
    --------
    >>> mols_arr = get_small_molecules(df)
    >>> mols[0]['MW']
    58.12
    """
    small_mols = []
    for row in df.itertuples(index=False):
        smiles = row.smiles_gdb9
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol_weight = Descriptors.MolWt(mol)
        if mol_weight <= mw_cutoff:
            small_mols.append({
                "ID": row.id,
                "SMILES": smiles,
                "MW": mol_weight
            })

    return small_mols

def get_atom_coords(df):
    """
    Extract atomic coordinates from each molecule in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data with SMILES strings and molecular properties.

    Returns
    -------
    list of dict
        Each dictionary contains:
        - 'ID' : int
            Molecule ID.
        - 'COORDS' : list of dict
            Atomic coordinates and element information.
    
    Examples
    --------
    >>> coord_arr = get_atom_coords(df)
    >>> coords[0]['COORDS'][0]['element']
    'C'
    """
    return [{"ID": row.id, "COORDS": row.atoms} for row in df.itertuples(index=False)]

def get_wiener_idx(mol):
    """
    Calculate the Wiender index for a molecule using Mordred. 

    The Wiener index is the sum of all shortest path distances between atom
    pairs in the molecule. 

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule object.

    Returns
    -------
    int
        Wiener index of the molecule.
    
    Examples
    --------
    >>> mol = MolFromSmiles("CC(C)C")  # Isobutane
    >>> get_wiener_index(mol)
    9
    """
    wiener_index = WienerIndex.WienerIndex()
    return wiener_index(mol)
        

def get_randic_index(mol):
    """
    Calculate the Randic (connectivity) index of a molecule from SMILES.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule object.

    Returns
    -------
    float
        Randic index of the molecule
    
    Examples
    --------
    >>> mol = MolFromSmiles("CC(C)C")  # Isobutane
    >>> get_randic_index(mol)
    1.732
    """
    randic_idx = 0.0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtom()
        j = bond.GetEndAtom()
        d_i = i.GetDegree()
        d_j = j.GetDegree()
        if d_i == 0 or d_j == 0:
            continue  # skip isolated atoms
        randic_idx += 1 / np.sqrt(d_i * d_j)

    return randic_idx

def get_zagreb_index(mol):
    """
    Calculate the first and second Zagreb index for a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Molecule object.

    Returns
    -------
    tuple of int
        (Z1, Z2) where:
        - Z1 : first Zagreb index
        - Z2 : second Zagreb index
    
    Examples
    --------
    >>> mol = MolFromSmiles("CCCC")  # Butane
    >>> get_zagreb_index(mol)
    (12, 9)
    """
    zagreb_idx1 = ZagrebIndex.ZagrebIndex(1)
    zagreb_idx2 = ZagrebIndex.ZagrebIndex(2)
    z1 = zagreb_idx1(mol)
    z2 = zagreb_idx2(mol)
    return z1, z2

def compute_rg2(molecule):
    """
    Compute Rg^2 for one molecule entry of your dataset.

    Parameters
    ----------
    molecule: dict 
        Dictionary containing:
        - 'ID' : int
            Molecule ID.
        - 'COORDS' : list of dict
            Atomic data with keys 'element', 'x', 'y', 'z'.
    
    Returns 
    -------
    float
        Radius of gyration squared.

    Examples
    --------
    >>> molecule = {'ID': 1, 'COORDS': array([
            {'charge': -0.535689, 'element': 'C', 'x': -0.0126981359, 'y': 1.0858041578, 'z': 0.0080009958}, 
            {'charge': 0.133921, 'element': 'H', 'x': 0.002150416, 'y': -0.0060313176, 'z': 0.0019761204}, 
            {'charge': 0.133922, 'element': 'H', 'x': 1.0117308433, 'y': 1.4637511618, 'z': 0.0002765748}, 
            {'charge': 0.133923, 'element': 'H', 'x': -0.540815069, 'y': 1.4475266138, 'z': -0.8766437152}, 
            {'charge': 0.133923, 'element': 'H', 'x': -0.5238136345, 'y': 1.4379326443, 'z': 0.9063972942}], 
            dtype=object)}
    >>> compute_rg2(molecule)
    0.0
    """
    coords = molecule['COORDS']
    
    masses = np.array([ptable.GetAtomicWeight(atom['element']) for atom in coords])
    positions = np.array([[atom['x'], atom['y'], atom['z']] for atom in coords])
    
    total_mass = masses.sum()
    com = np.average(positions, axis=0, weights=masses)
    
    rg2 = np.sum(masses * np.sum((positions - com)**2, axis=1)) / total_mass
    return rg2

def compute_boltzmann_weights(df, energy="G", temp=298.15):
    """
    Compute Botzmann weights for conformers of molecules based on Gibbs energy.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing energies and SMILES strings.
    energy : str, default="G"
        Column name containing Gibbs free energy (in Hartree).
    temp : float, default=298.15
        Temperature in Kelvin.
    
    Returns
    -------
    tuple
        (bw_dict, count) where:
        - bw_dict : dict
            Mapping of SMILES → list of {'ID', 'BW'} weights.
        - count : int
            Number of molecules with multiple conformers. 
            If count greater than 1, RG2 will be calculated differently.
    
    Examples
    --------
    >>> bw_dict, count = compute_boltzmann_weights(df)
    >>> list(bw_dict.keys())[0]
    'CCO'
    """
    # Gibbs energy given in Hartree --> convert to kJ/mol
    h_to_kj_mol = 2625.50
    R = 8.314e-3 # gas constant

    bw_dict = {}
    count = 0
    # Group by SMILES
    for smiles, group in df.groupby("smiles_gdb9"):
        energies = group[energy].values
        ids = group["id"].values


        # Shift energies to improve numerical stability
        min_energy = np.min(energies)
        shifted_energies = (energies - min_energy)*h_to_kj_mol

        # Compute Boltzmann factors
        boltz_factors = np.exp(-shifted_energies / (R * temp))
        total = np.sum(boltz_factors)
        weights = boltz_factors / total

        # Store results
        bw_dict[smiles] = [{"ID": int(i), "BW": float(w)} for i, w in zip(ids, weights)]

        if len(group) > 1:
            count += 1
    return bw_dict, count

def get_rdkit_desc(desc_dict, mol):
    """
    Compute all available RDKit molecular descriptors for a given molecule.

    Parameters
    ----------
    desc_dict : dict
        Dictionary to store descriptors.
    mol : rdkit.Chem.Mol
        Molecule object.

    Returns
    -------
    dict
        Updated dictionary containing RDKit descriptor values.
    """
    for desc, fnc in Descriptors.descList:
        try:
            desc_dict[desc] = fnc(mol)
        except:
            desc_dict[desc] = None
    return desc_dict 

def build_desc_dict(smiles=None, coords=None):
    """
    Build a dictionary of descriptors for a molecule
    Parameters
    ----------
    smiles : str, optional
        SMILES string representation of the molecule.
    coords : dict, optional
        Dictionary of atomic coordinates and element types.

    Returns
    -------
    dict
        Molecular descriptors including RDKit, topological,
        and structural (Rg²) properties.

    Examples
    --------
    >>> build_desc_dict(smiles="CCO")
    {'MolWt': 46.07, ..., 'WI': 9, 'RI': 1.732,...}
    """
    desc_dict = {}

    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        desc_dict = get_rdkit_desc(desc_dict, mol)
        desc_dict["WI"] = get_wiener_idx(mol)
        desc_dict["RI"] = get_randic_index(mol)
        z1, z2 = get_zagreb_index(mol)
        desc_dict["Z1"] = z1
        desc_dict["Z2"] = z2
        
    if coords is not None:
        desc_dict["RG2"] = compute_rg2(coords)
    
    return desc_dict

def process_molecule(args):
    """
    Helper function for multiprocessing descriptor computation.
    AI was used for assistance on this function

    Parameters
    ----------
    args : tuple
        Tuple of (mol_entry, coord_entry) where:
        - mol_entry : dict
            Molecule info containing 'ID' and 'SMILES'.
        - coord_entry : dict
            Coordinate info containing 'COORDS'.

    Returns
    -------
    dict
        Combined descriptor dictionary for the molecule.
    """
    mol_entry, coord_entry = args
    desc = build_desc_dict(mol_entry["SMILES"], coord_entry)
    desc["ID"] = mol_entry["ID"]
    desc["SMILES"] = mol_entry["SMILES"]
    
    return desc
    