"""
Author: Rana Elladki
Data: 10/01/2025
Description: Extract molecular descriptors of SMILES using RDkit
"""
from parse_xyz import parse_xyz_folder
from rdkit import Chem
from rdkit.Chem import Descriptors, GetPeriodicTable
from mordred import WienerIndex, ZagrebIndex
from multiprocessing import Pool, cpu_count
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import copy

ptable = GetPeriodicTable()

def get_small_molec(df, mw_cutoff=300):
    """
    Extract small molecules (based on molecular weight) from dataframe using SMILES.
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
    Extract atom coordinates from dataframe rows.
    """
    return [{"ID": row.id, "COORDS": row.atoms} for row in df.itertuples(index=False)]

def get_wiener_idx(mol):
    """
    Calculate the Wiender Index for molecules. WienerIndex calculation based on: https://chem.libretexts.org/Courses/Intercollegiate_Courses/Cheminformatics/05%3A_5._Quantitative_Structure_Property_Relationships/5.05%3A_Python_Assignment
    """
    wiener_index = WienerIndex.WienerIndex()
    return wiener_index(mol)
        

def get_randic_index(mol):
    """
    Calculate the Randic (connectivity) index of a molecule from SMILES.
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
    zagreb_idx1 = ZagrebIndex.ZagrebIndex(1)
    zagreb_idx2 = ZagrebIndex.ZagrebIndex(2)
    z1 = zagreb_idx1(mol)
    z2 = zagreb_idx2(mol)
    return z1, z2

def compute_rg2(molecule):
    """
    Compute Rg^2 for one molecule entry of your dataset.
    molecule: dict with {'ID': int, 'COORDS': numpy array of dicts with 'element','x','y','z'}
    i need to find a better way to do this dictionary
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
    Compute Botzmann weights for molecules with the same SMILES, using Gibbs free energy
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
    for desc, fnc in Descriptors.descList:
        try:
            desc_dict[desc] = fnc(mol)
        except:
            desc_dict[desc] = None
    return desc_dict 

def build_desc_dict(smiles=None, coords=None):
    """
    Build a dictionary of descriptors for a molecule
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
    When called, it processes molecule to use parallelism to speed up getting descriptors from RDkit
    I used AI to help with this function
    """
    mol_entry, coord_entry = args
    desc = build_desc_dict(mol_entry["SMILES"], coord_entry)
    desc["ID"] = mol_entry["ID"]
    desc["SMILES"] = mol_entry["SMILES"]
    
    return desc

def split_data(df, rand_state=42):
    
    # drop RG2 because that is what we want to predict
    X = df.drop(["RG2"], axis=1)
    y = df["RG2"]

    # split into training/validation and test set (80:20)
    X_train_val, X_test, y_train_val, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=rand_state)
    
    # split training/validation set into separate training and validation sets (75:25)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=rand_state)
    
    data_split = (X_train, X_val, X_test, y_train, y_val, y_test)

    return data_split

def pearson_correlation(df, target='RG2', top_count=20):
    """ 
    Perform pearson correlation coefficient analysis to find discriptors with impact on RG2
    """
    # drop IDs because that is not a feature 
    df = df.drop(["ID"], axis=1)
    # get the correlation matrix using pearson correlation coefficient analysis
    corr_pearson = df.corr(method='pearson')
    # get the rg2 correlation column and remove rg2's correaltion with itself, because it is just 1
    rg2_corr = corr_pearson[target].drop(target).abs().sort_values(ascending=False)
    # get top 20 features
    strong_corr = rg2_corr.index[:top_count]
    # reduced descriptor dataframe
    reduced_df = df[strong_corr.tolist() + [target]]
    # get a heat map
    plt.figure(figsize=(20,20))
    sns.heatmap(reduced_df.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.savefig(f"corr_heatmap_top_{top_count}_feats.png", dpi=300)
    return reduced_df

def filter_outliers(df, q_min=0.001, q_max=0.999):
    """ This function is inspired from Day13-inclass """
    # histogram of data before filtering 
    df.hist(figsize=(10,10), bins=50)
    plt.tight_layout()  # make sure subplots don't overlap
    plt.savefig("hist_unfiltered.png", dpi=300)

    # define a percentile value range for features
    perc_range = pd.DataFrame([df.quantile(q=q_min, axis=0), df.quantile(q=q_max, axis=0)])
    
    for feat in df.columns[:-1]:
        # these are small molecules, so most will have NumRotatableBonds=0 which will end up getting excluded, so don't remove outliers because data correctly mostly zeros
        # include features that have values greater than or equal to bottom 0.1% of data
        df = df[df[feat] >= perc_range[feat].iloc[0,]]
        # include features that have values less than or equal to top 99% of data
        df = df[df[feat] <= perc_range[feat].iloc[1,]]
    
    df.hist(figsize=(10,10), bins=50)
    plt.tight_layout()  # make sure subplots don't overlap
    plt.savefig("hist_filtered.png", dpi=300)
    
    return df

def log_transform(df, exclude=None, skew_thresh=1.0):
    """Applies log transform to heavily skewed data in datafram"""
    # calculate skewness for each column
    skewness = df[df.columns].skew()
    for col in df.columns:
        if exclude!=None and col in exclude:
            continue
        if skewness[col] >= skew_thresh and (df[col] > 0).all():
            df[col] = np.log2(df[col])
    df.hist(figsize=(10,10), bins=50)
    plt.tight_layout()  # make sure subplots don't overlap
    plt.savefig("hist_logged.png", dpi=300)
    return df

def main():
    start_time = time.time()
    # filepath containing raw data
    fp = "/mnt/home/elladki1/CMSE802/Final_Project/cmse802_project/data/raw_data"

    # parse through raw data and extract into pd df 
    # first checks if data has been parsed before and saved for performance optimization
    df = parse_xyz_folder(fp, as_dataframe=True)
    print("Done processing raw data...")
    mols_arr = get_small_molec(df)
    print("Done retrieving molecules...")
    coord_arr = get_atom_coords(df)
    print("Done retrieving atom coords...")
    # future improvement to account for BWs if the count is bigger than 0 but this database doesnt have any
    bw_dict,count = compute_boltzmann_weights(df)
    print("Done calculating Boltzmann weights...")

    # build descriptor dict for all molecules
    with Pool(processes=cpu_count()) as pool:
        all_desc = pool.map(process_molecule, zip(mols_arr, coord_arr))

    # index it with the SMILES 
    desc_df = pd.DataFrame(all_desc).set_index("SMILES")

    # get a dataframe of only the top features
    reduce = copy.deepcopy(desc_df)
    reduced_df = pearson_correlation(reduce,top_count=5)
    print("Done performing Pearson correlation coefficient analysis...")

    # filter outliers 
    fltr = copy.deepcopy(reduced_df)
    filter_df = filter_outliers(fltr)
    print("Done filtering outliers...")

    # log transform skewed columns
    skwd = copy.deepcopy(filter_df)
    log_df = log_transform(skwd)
    print("Done logging skewed data...")
    

    # split the data into training/validation/test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(log_df, rand_state=42)
    print("Done splitting data set...")
    
    

    end_time = time.time()
    print("Time(s):", end_time-start_time)
    return all_desc

if __name__ == "__main__":
    main()
    