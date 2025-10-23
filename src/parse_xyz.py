"""
Author: Rana Elladki
Date: 10/01/2025
Description: Parse through xyz files containing molecules from QM9 database and 
                saves into pandas dataframe
"""

import os
import pandas as pd

def safe_float(x):
    """
    Convert a string to float, handling Fortran-style scientific notation.

    Parameters 
    ----------
    x: str
      Fortran-style scientific notation string to convert into float

    Returns
    -------
    float 

    Examples
    --------
    >>> num = safe_float('2.1997*^-6')
    >>> num 
    2.1997e-6
    """
    if isinstance(x, str) and "*^" in x:
        x = x.replace("*^", "e")
    try:
        return float(x)
    except:
        raise ValueError(f"Cannot convert '{x}' to float.")


def parse_xyz_file(filename):
    """
    Parse a QM9-style .xyz file to extract information into a dictionary.

    Parameters
    ----------
    filename: str
            File containing a molecule and its relevant data
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'id': molecule index (int)
        - 'natoms': number of atoms (int)
        - 'properties': molecular properties (dict)
        - 'atoms': list of atom dicts with keys ['element', 'x', 'y', 'z', 'charge']
        - 'frequencies': list of floats
        - 'smiles': dict with keys ['gdb9', 'relaxed']
        - 'inchi': dict with keys ['gdb9', 'relaxed']
    
    Examples
    --------
    >>> processed_dict = parse_xyz_file('filename.xyz')
    """
    properties_list = [
        "tag", "index", "A", "B", "C", "mu", "alpha", "homo", "lumo",
        "gap", "r2", "zpve", "U0", "U", "H", "G", "Cv"
    ]
    
    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    natoms = int(lines[0])  # number of atoms

    # properties
    props_values = lines[1].split()
    properties = {
        properties_list[i]: safe_float(props_values[i]) if i >= 2 else props_values[i]
        for i in range(len(properties_list))
    }

    mol_id = int(props_values[1])  # molecule ID (index)

    # atoms
    atoms = []
    for line in lines[2:2+natoms]:
        parts = line.split()
        atoms.append({
            "element": parts[0],
            "x": safe_float(parts[1]),
            "y": safe_float(parts[2]),
            "z": safe_float(parts[3]),
            "charge": safe_float(parts[4])
        })

    # frequencies
    frequencies = [safe_float(x) for x in lines[2 + natoms].split()]

    # SMILES
    smiles_parts = lines[3 + natoms].split()
    smiles = {"gdb9": smiles_parts[0], "relaxed": smiles_parts[1]}

    # InChI
    inchi_parts = lines[4 + natoms].split()
    inchi = {"gdb9": inchi_parts[0], "relaxed": inchi_parts[1]}

    return {
        "id": mol_id,
        "natoms": natoms,
        "properties": properties,
        "atoms": atoms,
        "frequencies": frequencies,
        "smiles": smiles,
        "inchi": inchi
    }


def parse_xyz_folder(folder, as_dataframe=True, cache_file="xyz_data.parquet", check_cache=True):
    """
    Parse all .xyz files in a spcified folder and save formatted output into list 
        or pandas dataframe. Uses caching for speed. If check_cache=True, checks
        if the data has been parsed and save into a parquet file before pasting 
        through entire directory again.
    
    Parameters
    ----------
    folder: str
        Folder containing .xyz files.
    as_dataframe: bool, default=True
        Whether to return data as a pandas dataframe
    cache_file: str, default='xyz_data.parquet'
        Path to cache file for storing parsed data.
    check_cache: bool, default=True
        If True, load from cache if available.
    
    Returns
    -------
    pandas.DataFrame or list
        Preprocessed molecular data ready for modeling. If `as_dataframe` is False,
        returns a list of dictionaries.

    Example
    -------
    >>> folder = '/raw_data/'
    >>> df = parse_xyz_folder(folder, as_datafram=True, cache_file='xyz.parquet',
                check_cache=True)
    """
    cache_path = os.path.join(folder, cache_file)
    # If folder does not exist, raise FileNotFoundError
    if not os.path.exists(folder):
        raise FileNotFoundError(f"The folder '{folder}' does not exist or is not accessible.")

    # If cache exists and we allow cache, just load
    if check_cache and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path} ...")
        return pd.read_parquet(cache_path) if as_dataframe else pd.read_parquet(cache_path).to_dict("records")

    # Otherwise, parse all files fresh
    print(f"Parsing all XYZ files in {folder} ...")
    molecules = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".xyz"):
            filepath = os.path.join(folder, fname)
            molecules.append(parse_xyz_file(filepath))

    if as_dataframe:
        # Flatten into DataFrame
        data = []
        for mol in molecules:
            row = {
                "id": mol["id"],
                "natoms": mol["natoms"],
                "smiles_gdb9": mol["smiles"]["gdb9"],
                "smiles_relaxed": mol["smiles"]["relaxed"],
                "inchi_gdb9": mol["inchi"]["gdb9"],
                "inchi_relaxed": mol["inchi"]["relaxed"],
                "atoms": mol["atoms"],               # nested list of dicts
                "frequencies": mol["frequencies"],   # list of floats
            }
            row.update(mol["properties"])
            data.append(row)

        df = pd.DataFrame(data)

        # Save to cache
        print(f"Saving parsed data to {cache_path} ...")
        df.to_parquet(cache_path, index=False)

        return df

    return molecules
